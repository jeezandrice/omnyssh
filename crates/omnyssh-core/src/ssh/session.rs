//! Async SSH session management via russh.
//!
//! Provides [`SshSession`] — a thin wrapper around a russh client handle that
//! supports connecting, executing commands, and graceful disconnect.
//! Authentication order: identity file → SSH agent → failure.
//!
//! Hosts with a `ProxyJump` are reached through their bastions: each hop is
//! connected and authenticated in turn, and the next hop rides a
//! `direct-tcpip` channel opened on the previous one (the `ssh -J` model).
//!
//! Connection and command timeouts are enforced:
//! - Connect timeout: 10 seconds (per hop)
//! - Command timeout: 30 seconds

use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context};
use async_trait::async_trait;
use russh::client::{self, Handle};
use russh::ChannelMsg;
use tokio::time;

use crate::ssh::client::Host;

// ---------------------------------------------------------------------------
// russh Handler implementation
// ---------------------------------------------------------------------------

/// Shared russh client handler used by every native SSH path (metrics, SFTP,
/// terminal).
///
/// Verifies the server's host key against `~/.ssh/known_hosts`.
/// Unknown hosts are recorded on first connection (trust on first use);
/// changed keys are rejected.
pub(crate) struct KnownHostsHandler {
    /// Hostname used for known_hosts lookup.
    host: String,
    /// Port used for known_hosts lookup.
    port: u16,
}

#[async_trait]
impl client::Handler for KnownHostsHandler {
    type Error = russh::Error;

    async fn check_server_key(
        &mut self,
        server_public_key: &russh::keys::key::PublicKey,
    ) -> Result<bool, Self::Error> {
        match russh::keys::check_known_hosts(&self.host, self.port, server_public_key) {
            // Key is in known_hosts and matches.
            Ok(true) => Ok(true),
            // Host not seen before — record the key (trust on first use) so a
            // later key change is detected, then accept. Recording is
            // best-effort: a connection must not fail just because
            // known_hosts is unwritable.
            Ok(false) => {
                tracing::warn!(
                    host = %self.host,
                    port = self.port,
                    "Accepting unknown host key for {} (Trust On First Use)", self.host
                );
                match russh::keys::known_hosts::learn_known_hosts(
                    &self.host,
                    self.port,
                    server_public_key,
                ) {
                    Ok(()) => tracing::info!(
                        host = %self.host,
                        port = self.port,
                        "recorded new host key in known_hosts"
                    ),
                    Err(e) => tracing::warn!(
                        host = %self.host,
                        error = %e,
                        "could not record host key in known_hosts"
                    ),
                }
                Ok(true)
            }
            // A previously recorded key changed — refuse; possible MITM.
            Err(russh::keys::Error::KeyChanged { .. }) => {
                tracing::warn!(
                    host = %self.host,
                    port = self.port,
                    "server key mismatch in known_hosts — possible MITM attack, refusing connection"
                );
                Ok(false)
            }
            // Unreadable or corrupt known_hosts — fail closed rather than
            // accept an unverified key.
            Err(e) => {
                tracing::warn!(
                    host = %self.host,
                    error = %e,
                    "known_hosts check failed; refusing connection"
                );
                Ok(false)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SshConnection
// ---------------------------------------------------------------------------

/// An authenticated russh connection to one host, plus the jump-host
/// connections it is tunnelled through (empty for a direct connection).
///
/// The bastion handles are owned for the whole lifetime of the connection so
/// the chain outlives nothing it carries. Teardown runs the other way: the
/// target's session task holds the `direct-tcpip` stream of the hop below it,
/// so dropping this struct closes the target first and cascades outward.
/// Derefs to the target's [`Handle`], so callers open channels on it exactly as
/// before.
pub(crate) struct SshConnection {
    handle: Handle<KnownHostsHandler>,
    /// Bastions, nearest-first. Never used directly — kept alive by ownership.
    _jumps: Vec<Handle<KnownHostsHandler>>,
}

impl std::ops::Deref for SshConnection {
    type Target = Handle<KnownHostsHandler>;

    fn deref(&self) -> &Self::Target {
        &self.handle
    }
}

// ---------------------------------------------------------------------------
// SshSession
// ---------------------------------------------------------------------------

/// An authenticated SSH session ready for command execution.
///
/// Holds the russh client handle for the duration of its lifetime.
/// Drop → the connection is cleaned up by russh's internal tasks.
///
/// Wrapped in Arc to allow sharing across multiple operations (discovery + metrics).
#[derive(Clone)]
pub struct SshSession {
    handle: Arc<SshConnection>,
}

impl SshSession {
    /// Connect and authenticate to `host`.
    ///
    /// Authentication is attempted in order:
    /// 1. SSH agent (unix only, via `SSH_AUTH_SOCK`).
    /// 2. Identity file specified in the host config (`identity_file`).
    /// 3. Default key files (`~/.ssh/id_ed25519`, `id_rsa`, etc.).
    /// 4. Password (if provided in host config).
    ///
    /// A host with a `ProxyJump` is reached through its bastion chain; each hop
    /// authenticates the same way.
    ///
    /// Returns an error when no method succeeds or the connection times out.
    ///
    /// # Errors
    /// - Connection timeout (> 10 s per hop)
    /// - Authentication failure
    /// - Network error
    /// - An unresolvable `ProxyJump` chain (cycle or too many hops)
    pub async fn connect(host: &Host) -> anyhow::Result<Self> {
        Ok(Self {
            handle: Arc::new(connect_and_auth(host).await?),
        })
    }

    /// Execute a shell command on the remote host and return its stdout.
    ///
    /// A new SSH channel is opened for each call so sessions can be
    /// reused across multiple commands. The remote exit status is ignored —
    /// use [`SshSession::run_command_checked`] when it carries the result.
    ///
    /// # Errors
    /// Returns an error on channel failure or if the command times out (30 s).
    pub async fn run_command(&self, cmd: &str) -> anyhow::Result<String> {
        Ok(self.exec(cmd).await?.0)
    }

    /// Like [`SshSession::run_command`] but returns an error when the remote
    /// command exits with a non-zero status. Use for `test`-style probes whose
    /// exit code is the answer (e.g. `sudo -n true`).
    ///
    /// # Errors
    /// As [`SshSession::run_command`], plus a non-zero remote exit status.
    pub async fn run_command_checked(&self, cmd: &str) -> anyhow::Result<String> {
        let (output, status) = self.exec(cmd).await?;
        match status {
            // A missing exit status is treated as success — failing a command
            // that likely worked is worse than missing a rare edge case.
            None | Some(0) => Ok(output),
            Some(code) => Err(anyhow!("remote command exited with status {code}")),
        }
    }

    /// Opens a channel, runs `cmd`, and returns its stdout and exit status.
    async fn exec(&self, cmd: &str) -> anyhow::Result<(String, Option<u32>)> {
        let mut channel = self
            .handle
            .channel_open_session()
            .await
            .context("open SSH channel")?;

        channel.exec(true, cmd).await.context("exec SSH command")?;

        time::timeout(Duration::from_secs(30), collect_output(&mut channel))
            .await
            .map_err(|_| anyhow!("command timed out (30 s): {}", cmd))?
            .context("read command output")
    }

    /// Opens a new SSH channel, requests the SFTP subsystem, and returns the
    /// channel as an async stream suitable for [`russh_sftp::client::SftpSession::new`].
    ///
    /// The `SshSession` **must** remain alive for the entire lifetime of the
    /// SFTP session — dropping it closes the underlying TCP connection.
    ///
    /// # Errors
    /// Returns an error if the channel cannot be opened or if the server rejects
    /// the SFTP subsystem request.
    pub async fn open_sftp_channel(
        &self,
    ) -> anyhow::Result<russh::ChannelStream<russh::client::Msg>> {
        let channel = self
            .handle
            .channel_open_session()
            .await
            .context("open SFTP session channel")?;
        channel
            .request_subsystem(true, "sftp")
            .await
            .context("request SFTP subsystem")?;
        Ok(channel.into_stream())
    }

    /// Gracefully close the SSH connection.
    pub async fn disconnect(self) {
        let _ = self
            .handle
            .disconnect(russh::Disconnect::ByApplication, "", "en")
            .await;
    }
}

// ---------------------------------------------------------------------------
// Connection + authentication
// ---------------------------------------------------------------------------

/// Per-hop budget for the TCP connect and SSH handshake.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Connect to `host`, verify its host key, and authenticate — through the
/// host's `ProxyJump` chain when it has one.
///
/// Shared by [`SshSession::connect`] (metrics/SFTP) and the terminal so every
/// native SSH path honors the same keys, agent, passwords, known_hosts policy,
/// and bastions.
///
/// # Errors
/// Connection timeout (> 10 s per hop), host-key rejection, authentication
/// failure, or an unresolvable `ProxyJump` chain.
pub(crate) async fn connect_and_auth(host: &Host) -> anyhow::Result<SshConnection> {
    let chain = jump_chain(host).await?;
    let config = client_config();

    // Walk the bastions outward: the first is reached directly, every later one
    // through its predecessor. The target then rides the last hop.
    let mut jumps: Vec<Handle<KnownHostsHandler>> = Vec::with_capacity(chain.len());
    for hop in &chain {
        let handle = match jumps.last() {
            None => connect_direct(&config, hop).await,
            Some(via) => connect_tunnelled(&config, via, hop).await,
        }
        .map_err(|e| anyhow!("ProxyJump via '{}' failed: {e:#}", hop.name))?;
        jumps.push(handle);
    }

    let handle = match (jumps.last(), chain.last()) {
        (Some(via), Some(last)) => connect_tunnelled(&config, via, host)
            .await
            .map_err(|e| anyhow!("connecting via '{}' failed: {e:#}", last.name))?,
        _ => connect_direct(&config, host).await?,
    };

    Ok(SshConnection {
        handle,
        _jumps: jumps,
    })
}

/// Wall-clock budget one [`SshSession::connect`] needs for `host`: the per-hop
/// connect timeout once for every bastion in its `ProxyJump` chain, plus the
/// target.
///
/// Callers that wrap the connect in a timeout of their own must scale it by
/// this — a fixed budget trips on a bastion chain before the connection has had
/// the time [`connect_and_auth`] is entitled to.
pub(crate) async fn connect_budget(host: &Host) -> Duration {
    // A chain that fails to resolve costs nothing to connect; the caller's own
    // attempt reports why.
    let hops = jump_chain(host).await.map_or(0, |chain| chain.len());
    CONNECT_TIMEOUT * (hops as u32 + 1)
}

/// The shared russh client configuration (timeouts + keepalives).
fn client_config() -> Arc<client::Config> {
    // russh 0.46 implements the NIST ECDH algorithms but intentionally omits
    // them from its default preference list. A number of still-supported
    // network appliances (including Cisco IOS 15.x) advertise NIST ECDH plus
    // SHA-1 finite-field KEX, but not curve25519 or group14-sha256. Advertise
    // the ECDH algorithms explicitly so those peers can negotiate a modern KEX
    // without enabling any SHA-1 exchange method on the client.
    let mut preferred = russh::Preferred::default();
    let mut kex = preferred.kex.to_vec();
    for (offset, algorithm) in [
        russh::kex::ECDH_SHA2_NISTP256,
        russh::kex::ECDH_SHA2_NISTP384,
        russh::kex::ECDH_SHA2_NISTP521,
    ]
    .into_iter()
    .enumerate()
    {
        if !kex.contains(&algorithm) {
            // Keep curve25519 first, then the interoperable ECDH choices,
            // followed by russh's existing finite-field and extension entries.
            kex.insert(2 + offset, algorithm);
        }
    }
    preferred.kex = std::borrow::Cow::Owned(kex);

    Arc::new(client::Config {
        preferred,
        // No inactivity timeout: russh skips resetting it on the iteration that
        // sends a keepalive, so a peer that never answers `keepalive@openssh.com`
        // (common in appliance SSH stacks) was torn down after 30 s even while
        // its commands still ran. Liveness stays bounded by `keepalive_max`.
        keepalive_interval: Some(Duration::from_secs(15)),
        keepalive_max: 3,
        ..Default::default()
    })
}

/// Resolves `host`'s `ProxyJump` into the hops to connect before it.
///
/// The jump aliases are looked up in the merged host list, so a bastion defined
/// elsewhere in `~/.ssh/config` (or in `hosts.toml`) contributes its own
/// HostName/User/Port/IdentityFile. Loading is skipped entirely for the common
/// no-`ProxyJump` case.
///
/// # Errors
/// An unreadable host list, or a chain that cannot be resolved. Both fail the
/// connection: resolving a bastion alias against nothing would fall back to
/// dialling the alias as a hostname, which is a different machine.
async fn jump_chain(host: &Host) -> anyhow::Result<Vec<Host>> {
    if crate::ssh::jump::jump_value(host).is_none() {
        return Ok(Vec::new());
    }
    // load_all_hosts() is blocking file I/O — keep it off the async worker.
    let known = tokio::task::spawn_blocking(crate::config::load_all_hosts)
        .await
        .context("host list load panicked")?
        .map_err(|e| anyhow!("could not load hosts for ProxyJump resolution: {e:#}"))?;

    let chain = crate::ssh::jump::resolve_chain(host, &known)?;
    tracing::debug!(
        host = %host.name,
        via = %chain.iter().map(|h| h.name.as_str()).collect::<Vec<_>>().join(" -> "),
        "resolved ProxyJump chain"
    );
    Ok(chain)
}

/// Opens a TCP connection to `host` and authenticates.
async fn connect_direct(
    config: &Arc<client::Config>,
    host: &Host,
) -> anyhow::Result<Handle<KnownHostsHandler>> {
    let addr = format!("{}:{}", host.hostname, host.port);
    let handle = time::timeout(
        CONNECT_TIMEOUT,
        client::connect(Arc::clone(config), addr, known_hosts_handler(host)),
    )
    .await
    .map_err(|_| anyhow!("SSH connection timed out (10 s)"))?
    .context("SSH connection failed")?;

    finish_auth(handle, host).await
}

/// Reaches `host` through the already-connected bastion `via`: a `direct-tcpip`
/// channel on the bastion carries a second SSH session to the target, which is
/// verified and authenticated in its own right.
async fn connect_tunnelled(
    config: &Arc<client::Config>,
    via: &Handle<KnownHostsHandler>,
    host: &Host,
) -> anyhow::Result<Handle<KnownHostsHandler>> {
    // The originator address is informational; ssh(1) reports the loopback it
    // forwards from, and servers only log it.
    //
    // Timed out like the handshake it precedes: the bastion answers only once
    // its own connect() to the target resolves, so a firewalled target would
    // otherwise park the caller for the bastion's whole SYN budget.
    let channel = time::timeout(
        CONNECT_TIMEOUT,
        via.channel_open_direct_tcpip(host.hostname.clone(), host.port as u32, "127.0.0.1", 0),
    )
    .await
    .map_err(|_| anyhow!("SSH connection timed out (10 s)"))?
    .with_context(|| format!("open tunnel to {}:{}", host.hostname, host.port))?;

    let handle = time::timeout(
        CONNECT_TIMEOUT,
        client::connect_stream(
            Arc::clone(config),
            channel.into_stream(),
            known_hosts_handler(host),
        ),
    )
    .await
    .map_err(|_| anyhow!("SSH connection timed out (10 s)"))?
    .context("SSH connection failed")?;

    finish_auth(handle, host).await
}

/// The host-key verifier for `host`. The lookup uses the target's own
/// hostname/port even over a tunnel, so `known_hosts` entries match what an
/// `ssh -J` would record.
fn known_hosts_handler(host: &Host) -> KnownHostsHandler {
    KnownHostsHandler {
        host: host.hostname.clone(),
        port: host.port,
    }
}

/// Authenticates `handle` as `host`, converting a refusal into an error.
async fn finish_auth(
    mut handle: Handle<KnownHostsHandler>,
    host: &Host,
) -> anyhow::Result<Handle<KnownHostsHandler>> {
    if !authenticate(&mut handle, host).await? {
        return Err(anyhow!("SSH authentication failed for {}", host.name));
    }
    Ok(handle)
}

// ---------------------------------------------------------------------------
// Authentication helpers
// ---------------------------------------------------------------------------

async fn authenticate(handle: &mut Handle<KnownHostsHandler>, host: &Host) -> anyhow::Result<bool> {
    let user = host.user.clone();

    // 1. Try SSH agent first — it handles passphrase-protected keys and is the
    //    most common auth method for non-interactive clients.
    #[cfg(unix)]
    {
        if try_agent_auth(handle, &user).await.unwrap_or(false) {
            return Ok(true);
        }
    }

    // 2. Try explicit identity_file from host config.
    if let Some(key_path) = &host.identity_file {
        let path = expand_tilde(key_path);
        if try_key_auth(handle, &user, &path).await.unwrap_or(false) {
            return Ok(true);
        }
    }

    // 3. Try default key files — mirrors what the `ssh` binary does when no
    //    -i flag is given. Skips files that don't exist.
    for key_path in default_key_paths() {
        if key_path.exists() {
            let path_str = key_path.to_string_lossy().into_owned();
            if try_key_auth(handle, &user, &path_str)
                .await
                .unwrap_or(false)
            {
                return Ok(true);
            }
        }
    }

    // 4. Try password authentication if provided.
    //    Password auth is NOT recommended for production use but is required for
    //    the initial connection before setting up key-based auth.
    if let Some(password) = &host.password {
        if try_password_auth(handle, &user, password)
            .await
            .unwrap_or(false)
        {
            tracing::info!(
                host = %host.name,
                "Connected via password authentication — consider setting up SSH key"
            );
            return Ok(true);
        }
    }

    Ok(false)
}

/// Returns the standard default SSH private key paths in priority order.
fn default_key_paths() -> Vec<std::path::PathBuf> {
    let Some(home) = dirs::home_dir() else {
        return vec![];
    };
    let ssh = home.join(".ssh");
    [
        "id_ed25519",
        "id_rsa",
        "id_ecdsa",
        "id_ecdsa_sk",
        "id_ed25519_sk",
    ]
    .iter()
    .map(|name| ssh.join(name))
    .collect()
}

async fn try_key_auth(
    handle: &mut Handle<KnownHostsHandler>,
    user: &str,
    key_path: &str,
) -> anyhow::Result<bool> {
    // load_secret_key is synchronous (file I/O) — offload to blocking pool.
    let path = key_path.to_string();
    let key_pair = tokio::task::spawn_blocking(move || {
        russh::keys::load_secret_key(&path, None).with_context(|| format!("load key from {path}"))
    })
    .await
    .context("spawn_blocking panicked")??;

    let ok = handle
        .authenticate_publickey(user, Arc::new(key_pair))
        .await
        .context("authenticate_publickey")?;
    Ok(ok)
}

#[cfg(unix)]
async fn try_agent_auth(
    handle: &mut Handle<KnownHostsHandler>,
    user: &str,
) -> anyhow::Result<bool> {
    use russh::keys::agent::client::AgentClient;

    let mut agent = AgentClient::connect_env()
        .await
        .context("connect to SSH agent")?;

    let identities = agent
        .request_identities()
        .await
        .context("request agent identities")?;

    for pubkey in identities {
        let (agent_back, result) = handle.authenticate_future(user, pubkey, agent).await;
        agent = agent_back;
        match result {
            Ok(true) => return Ok(true),
            Ok(false) => continue,
            Err(_) => continue,
        }
    }
    Ok(false)
}

/// Try password-based authentication.
///
/// # Errors
/// Returns an error if the authentication attempt fails.
async fn try_password_auth(
    handle: &mut Handle<KnownHostsHandler>,
    user: &str,
    password: &str,
) -> anyhow::Result<bool> {
    let ok = handle
        .authenticate_password(user, password)
        .await
        .context("authenticate with password")?;
    Ok(ok)
}

// ---------------------------------------------------------------------------
// Output collection
// ---------------------------------------------------------------------------

async fn collect_output(
    channel: &mut russh::Channel<russh::client::Msg>,
) -> anyhow::Result<(String, Option<u32>)> {
    let mut buf = Vec::new();
    let mut exit_status = None;
    loop {
        match channel.wait().await {
            Some(ChannelMsg::Data { ref data }) => {
                buf.extend_from_slice(data);
            }
            Some(ChannelMsg::ExtendedData { .. }) => {
                // stderr — discard to avoid corrupting stdout-only parser input
            }
            Some(ChannelMsg::ExitStatus { exit_status: code }) => {
                // Record it but keep reading: trailing stdout may still arrive
                // before the channel is closed.
                exit_status = Some(code);
            }
            Some(ChannelMsg::Eof) => {
                // Continue reading — ExitStatus may arrive after Eof.
            }
            Some(ChannelMsg::Close) | None => break,
            _ => {}
        }
    }
    // Use .lines() semantics: replace \r\n → \n for cross-platform safety.
    let raw = String::from_utf8_lossy(&buf);
    let normalised: String = raw.lines().flat_map(|l| [l, "\n"]).collect();
    Ok((normalised, exit_status))
}

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

fn expand_tilde(path: &str) -> String {
    if path.starts_with("~/") || path == "~" {
        if let Some(home) = dirs::home_dir() {
            return path.replacen('~', &home.to_string_lossy(), 1);
        }
    }
    path.to_string()
}

#[cfg(test)]
mod tests {
    use super::client_config;

    #[test]
    fn client_config_advertises_nist_ecdh_kex() {
        let config = client_config();
        for algorithm in [
            russh::kex::ECDH_SHA2_NISTP256,
            russh::kex::ECDH_SHA2_NISTP384,
            russh::kex::ECDH_SHA2_NISTP521,
        ] {
            assert!(
                config.preferred.kex.contains(&algorithm),
                "missing KEX algorithm {}",
                algorithm.as_ref()
            );
        }
    }
}
