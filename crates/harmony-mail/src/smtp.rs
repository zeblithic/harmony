//! Sans-I/O SMTP state machine.
//!
//! Pure `(State, Event) -> (State, Vec<Action>)` transitions with no networking,
//! no tokio, no async. The caller (I/O layer) feeds [`SmtpEvent`]s and executes
//! the returned [`SmtpAction`]s.

use std::net::IpAddr;

use crate::message::ADDRESS_HASH_LEN;

// ── Configuration ────────────────────────────────────────────────────

/// Gateway configuration for SMTP sessions.
#[derive(Debug, Clone)]
pub struct SmtpConfig {
    /// The domain this gateway serves (used in greeting and EHLO response).
    pub domain: String,
    /// The MX hostname advertised in DNS.
    pub mx_host: String,
    /// Maximum accepted message size in bytes.
    pub max_message_size: usize,
    /// Maximum number of recipients per message.
    pub max_recipients: usize,
}

// ── State ────────────────────────────────────────────────────────────

/// SMTP session state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SmtpState {
    /// TCP connection accepted, no data exchanged yet.
    Connected,
    /// 220 greeting sent, awaiting EHLO.
    GreetingSent,
    /// EHLO accepted, ready for MAIL FROM.
    Ready,
    /// MAIL FROM accepted, awaiting RCPT TO.
    MailFromReceived,
    /// At least one valid RCPT TO accepted, awaiting DATA or more RCPT TO.
    RcptToReceived,
    /// DATA command accepted, receiving message body.
    DataReceiving,
    /// Session closed.
    Closed,
}

// ── Commands ─────────────────────────────────────────────────────────

/// Parsed SMTP command (produced by the parser in smtp_parse.rs, consumed here).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SmtpCommand {
    Ehlo { domain: String },
    MailFrom { address: String },
    RcptTo { address: String },
    Data,
    Quit,
    Rset,
    Noop,
    StartTls,
}

// ── Events ───────────────────────────────────────────────────────────

/// Events fed into the state machine.
#[derive(Debug, Clone)]
pub enum SmtpEvent {
    /// New TCP connection established.
    Connected { peer_ip: IpAddr, tls: bool },
    /// A parsed SMTP command.
    Command(SmtpCommand),
    /// TLS upgrade completed.
    TlsCompleted,
    /// Complete message data received (after DATA, dot-terminated).
    DataComplete(Vec<u8>),
    /// Result from Harmony address resolution.
    HarmonyResolved {
        local_part: String,
        identity: Option<[u8; ADDRESS_HASH_LEN]>,
    },
}

// ── Actions ──────────────────────────────────────────────────────────

/// Actions emitted by the state machine for the I/O layer to execute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SmtpAction {
    /// Send a single-line SMTP response (e.g. `250 OK`).
    SendResponse(u16, String),
    /// Send a multi-line EHLO response with structured capabilities.
    ///
    /// The I/O layer formats this as `250-{greeting}\r\n` followed by
    /// `250-{cap}\r\n` for each capability except the last, which uses `250 {cap}\r\n`.
    SendEhloResponse {
        greeting: String,
        capabilities: Vec<String>,
    },
    /// Initiate TLS upgrade.
    StartTls,
    /// Check SPF for the sender domain.
    CheckSpf {
        sender_domain: String,
        peer_ip: IpAddr,
    },
    /// Resolve a local part to a Harmony identity.
    ResolveHarmonyAddress { local_part: String, domain: String },
    /// Message accepted — deliver to Harmony network.
    DeliverToHarmony {
        recipients: Vec<[u8; ADDRESS_HASH_LEN]>,
        data: Vec<u8>,
    },
    /// Close the connection.
    Close,
}

// ── Session ──────────────────────────────────────────────────────────

/// The SMTP session state machine.
pub struct SmtpSession {
    /// Current protocol state.
    pub state: SmtpState,
    config: SmtpConfig,
    peer_ip: Option<IpAddr>,
    tls: bool,
    ehlo_domain: Option<String>,
    mail_from: Option<String>,
    resolved_recipients: Vec<[u8; ADDRESS_HASH_LEN]>,
    /// RCPT TO waiting for Harmony address resolution.
    pending_rcpt: Option<String>,
    data: Vec<u8>,
}

impl SmtpSession {
    /// Create a new SMTP session with the given configuration.
    ///
    /// Initial state is [`SmtpState::Connected`].
    pub fn new(config: SmtpConfig) -> Self {
        Self {
            state: SmtpState::Connected,
            config,
            peer_ip: None,
            tls: false,
            ehlo_domain: None,
            mail_from: None,
            resolved_recipients: Vec::new(),
            pending_rcpt: None,
            data: Vec::new(),
        }
    }

    /// Feed an event into the state machine and return the resulting actions.
    pub fn handle(&mut self, event: SmtpEvent) -> Vec<SmtpAction> {
        match event {
            SmtpEvent::Connected { peer_ip, tls } => self.handle_connected(peer_ip, tls),
            SmtpEvent::Command(cmd) => self.handle_command(cmd),
            SmtpEvent::TlsCompleted => self.handle_tls_completed(),
            SmtpEvent::DataComplete(bytes) => self.handle_data_complete(bytes),
            SmtpEvent::HarmonyResolved {
                local_part,
                identity,
            } => self.handle_harmony_resolved(&local_part, identity),
        }
    }

    // ── Event handlers ───────────────────────────────────────────────

    fn handle_connected(&mut self, peer_ip: IpAddr, tls: bool) -> Vec<SmtpAction> {
        if self.state != SmtpState::Connected {
            return vec![];
        }
        self.peer_ip = Some(peer_ip);
        self.tls = tls;
        self.state = SmtpState::GreetingSent;
        vec![SmtpAction::SendResponse(
            220,
            format!("{} ESMTP Harmony Mail", self.config.mx_host),
        )]
    }

    fn handle_command(&mut self, cmd: SmtpCommand) -> Vec<SmtpAction> {
        match cmd {
            SmtpCommand::Ehlo { domain } => self.handle_ehlo(domain),
            SmtpCommand::MailFrom { address } => self.handle_mail_from(address),
            SmtpCommand::RcptTo { address } => self.handle_rcpt_to(address),
            SmtpCommand::Data => self.handle_data(),
            SmtpCommand::Quit => self.handle_quit(),
            SmtpCommand::Rset => self.handle_rset(),
            SmtpCommand::Noop => vec![SmtpAction::SendResponse(250, "OK".to_string())],
            SmtpCommand::StartTls => self.handle_starttls(),
        }
    }

    fn handle_ehlo(&mut self, domain: String) -> Vec<SmtpAction> {
        // RFC 5321 §4.1.4: EHLO is valid from any post-greeting state and acts
        // as an implicit RSET. Reject only from Connected (no greeting yet),
        // DataReceiving (mid-transfer), and Closed.
        match self.state {
            SmtpState::Connected | SmtpState::DataReceiving | SmtpState::Closed => {
                return vec![SmtpAction::SendResponse(
                    503,
                    "Bad sequence of commands".to_string(),
                )];
            }
            SmtpState::GreetingSent => {}
            // Implicit RSET for any post-EHLO state.
            _ => {
                self.mail_from = None;
                self.resolved_recipients.clear();
                self.pending_rcpt = None;
                self.data.clear();
            }
        }

        self.ehlo_domain = Some(domain);
        self.state = SmtpState::Ready;

        let mut capabilities = vec![
            format!("SIZE {}", self.config.message_size_display()),
            "8BITMIME".to_string(),
        ];
        if !self.tls {
            capabilities.push("STARTTLS".to_string());
        }

        vec![SmtpAction::SendEhloResponse {
            greeting: format!("{} Hello", self.config.domain),
            capabilities,
        }]
    }

    fn handle_mail_from(&mut self, address: String) -> Vec<SmtpAction> {
        if self.state != SmtpState::Ready {
            return vec![SmtpAction::SendResponse(
                503,
                "Bad sequence of commands".to_string(),
            )];
        }
        let sender_domain = extract_domain(&address);
        self.mail_from = Some(address);
        self.state = SmtpState::MailFromReceived;

        let mut actions = vec![SmtpAction::SendResponse(250, "OK".to_string())];
        // Skip SPF for null sender (MAIL FROM:<>) — RFC 5321 §4.5.5 bounce/DSN.
        if !sender_domain.is_empty() {
            if let Some(peer_ip) = self.peer_ip {
                actions.push(SmtpAction::CheckSpf {
                    sender_domain,
                    peer_ip,
                });
            }
        }
        actions
    }

    fn handle_rcpt_to(&mut self, address: String) -> Vec<SmtpAction> {
        if self.state != SmtpState::MailFromReceived && self.state != SmtpState::RcptToReceived {
            return vec![SmtpAction::SendResponse(
                503,
                "Bad sequence of commands".to_string(),
            )];
        }

        // Guard: reject if a previous RCPT TO resolution is still in flight.
        // The I/O layer must feed HarmonyResolved before the next RCPT TO.
        if self.pending_rcpt.is_some() {
            return vec![SmtpAction::SendResponse(
                421,
                "4.7.0 Previous recipient resolution pending".to_string(),
            )];
        }

        if self.resolved_recipients.len() >= self.config.max_recipients {
            return vec![SmtpAction::SendResponse(
                452,
                "Too many recipients".to_string(),
            )];
        }

        let (local_part, domain) = split_address(&address);
        self.pending_rcpt = Some(local_part.clone());

        vec![SmtpAction::ResolveHarmonyAddress {
            local_part,
            domain,
        }]
    }

    fn handle_data(&mut self) -> Vec<SmtpAction> {
        if self.state != SmtpState::RcptToReceived {
            return vec![SmtpAction::SendResponse(
                503,
                "Bad sequence of commands".to_string(),
            )];
        }
        self.state = SmtpState::DataReceiving;
        vec![SmtpAction::SendResponse(
            354,
            "Start mail input".to_string(),
        )]
    }

    fn handle_quit(&mut self) -> Vec<SmtpAction> {
        self.state = SmtpState::Closed;
        vec![
            SmtpAction::SendResponse(221, "Bye".to_string()),
            SmtpAction::Close,
        ]
    }

    fn handle_rset(&mut self) -> Vec<SmtpAction> {
        self.mail_from = None;
        self.resolved_recipients.clear();
        self.pending_rcpt = None;
        self.data.clear();
        // RFC 5321 §4.1.1.5: RSET resets the mail transaction, not the session.
        // Only advance to Ready if EHLO has already been completed.
        match self.state {
            SmtpState::Connected | SmtpState::GreetingSent => {
                // Pre-EHLO: keep current state, don't let client skip handshake.
            }
            _ => {
                self.state = SmtpState::Ready;
            }
        }
        vec![SmtpAction::SendResponse(250, "OK".to_string())]
    }

    fn handle_starttls(&mut self) -> Vec<SmtpAction> {
        if self.tls {
            return vec![SmtpAction::SendResponse(
                503,
                "TLS already active".to_string(),
            )];
        }
        vec![
            SmtpAction::SendResponse(220, "Ready to start TLS".to_string()),
            SmtpAction::StartTls,
        ]
    }

    fn handle_tls_completed(&mut self) -> Vec<SmtpAction> {
        self.tls = true;
        // After STARTTLS, the client must re-issue EHLO.
        self.state = SmtpState::GreetingSent;
        self.ehlo_domain = None;
        self.mail_from = None;
        self.resolved_recipients.clear();
        self.pending_rcpt = None;
        self.data.clear();
        vec![]
    }

    fn handle_data_complete(&mut self, bytes: Vec<u8>) -> Vec<SmtpAction> {
        if self.state != SmtpState::DataReceiving {
            return vec![];
        }
        let recipients = std::mem::take(&mut self.resolved_recipients);
        let actions = vec![
            SmtpAction::DeliverToHarmony {
                recipients,
                data: bytes,
            },
            SmtpAction::SendResponse(250, "OK".to_string()),
        ];

        // Reset for next message on the same connection.
        self.mail_from = None;
        self.pending_rcpt = None;
        self.data.clear();
        self.state = SmtpState::Ready;

        actions
    }

    fn handle_harmony_resolved(
        &mut self,
        _local_part: &str,
        identity: Option<[u8; ADDRESS_HASH_LEN]>,
    ) -> Vec<SmtpAction> {
        self.pending_rcpt = None;
        match identity {
            Some(hash) => {
                self.resolved_recipients.push(hash);
                self.state = SmtpState::RcptToReceived;
                vec![SmtpAction::SendResponse(250, "OK".to_string())]
            }
            None => vec![SmtpAction::SendResponse(
                550,
                "5.1.1 User not found".to_string(),
            )],
        }
    }
}

// ── Helpers ──────────────────────────────────────────────────────────

impl SmtpConfig {
    fn message_size_display(&self) -> usize {
        self.max_message_size
    }
}

/// Extract the domain from an email address (everything after the last `@`).
fn extract_domain(address: &str) -> String {
    address
        .rsplit_once('@')
        .map(|(_, domain)| domain.to_string())
        .unwrap_or_default()
}

/// Split an email address into (local_part, domain).
fn split_address(address: &str) -> (String, String) {
    match address.rsplit_once('@') {
        Some((local, domain)) => (local.to_string(), domain.to_string()),
        None => (address.to_string(), String::new()),
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    fn test_config() -> SmtpConfig {
        SmtpConfig {
            domain: "harmony.example.com".to_string(),
            mx_host: "mail.harmony.example.com".to_string(),
            max_message_size: 10 * 1024 * 1024,
            max_recipients: 100,
        }
    }

    fn connected_session() -> SmtpSession {
        let mut session = SmtpSession::new(test_config());
        session.handle(SmtpEvent::Connected {
            peer_ip: IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)),
            tls: false,
        });
        session
    }

    fn ready_session() -> SmtpSession {
        let mut session = connected_session();
        session.handle(SmtpEvent::Command(SmtpCommand::Ehlo {
            domain: "sender.example.com".to_string(),
        }));
        session
    }

    #[test]
    fn connected_sends_greeting() {
        let mut session = SmtpSession::new(test_config());
        let actions = session.handle(SmtpEvent::Connected {
            peer_ip: IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
            tls: false,
        });

        assert_eq!(session.state, SmtpState::GreetingSent);
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            SmtpAction::SendResponse(code, msg) => {
                assert_eq!(*code, 220);
                assert!(
                    msg.contains("mail.harmony.example.com"),
                    "greeting should contain domain: {msg}"
                );
            }
            other => panic!("expected SendResponse, got {other:?}"),
        }
    }

    #[test]
    fn ehlo_after_greeting() {
        let mut session = connected_session();
        let actions = session.handle(SmtpEvent::Command(SmtpCommand::Ehlo {
            domain: "sender.example.com".to_string(),
        }));

        assert_eq!(session.state, SmtpState::Ready);
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            SmtpAction::SendEhloResponse {
                greeting,
                capabilities,
            } => {
                assert!(
                    greeting.contains("Hello"),
                    "greeting should contain Hello: {greeting}"
                );
                let caps = capabilities.join(" ");
                assert!(caps.contains("SIZE"), "should list SIZE: {caps}");
                assert!(caps.contains("8BITMIME"), "should list 8BITMIME: {caps}");
                assert!(
                    caps.contains("STARTTLS"),
                    "should list STARTTLS when not TLS: {caps}"
                );
            }
            other => panic!("expected SendEhloResponse, got {other:?}"),
        }
    }

    #[test]
    fn ehlo_after_ready_resets_transaction() {
        // RFC 5321 §4.1.4: EHLO at any point acts as implicit RSET.
        let mut session = ready_session();
        session.handle(SmtpEvent::Command(SmtpCommand::MailFrom {
            address: "alice@sender.example.com".to_string(),
        }));
        assert_eq!(session.state, SmtpState::MailFromReceived);

        // Re-issue EHLO mid-transaction.
        let actions = session.handle(SmtpEvent::Command(SmtpCommand::Ehlo {
            domain: "new-sender.example.com".to_string(),
        }));

        assert_eq!(session.state, SmtpState::Ready);
        assert!(session.mail_from.is_none(), "EHLO should clear mail_from");
        assert_eq!(actions.len(), 1);
        assert!(
            matches!(&actions[0], SmtpAction::SendEhloResponse { .. }),
            "expected SendEhloResponse, got {:?}",
            actions[0]
        );
    }

    #[test]
    fn rset_before_ehlo_preserves_state() {
        // RFC 5321 §4.1.1.5: RSET before EHLO must not advance past GreetingSent.
        let mut session = connected_session();
        assert_eq!(session.state, SmtpState::GreetingSent);

        let actions = session.handle(SmtpEvent::Command(SmtpCommand::Rset));

        assert_eq!(
            actions[0],
            SmtpAction::SendResponse(250, "OK".to_string())
        );
        assert_eq!(
            session.state,
            SmtpState::GreetingSent,
            "RSET before EHLO should not advance to Ready"
        );

        // Client must still issue EHLO to proceed.
        let actions = session.handle(SmtpEvent::Command(SmtpCommand::MailFrom {
            address: "alice@example.com".to_string(),
        }));
        assert_eq!(
            actions[0],
            SmtpAction::SendResponse(503, "Bad sequence of commands".to_string()),
            "MAIL FROM should be rejected before EHLO"
        );
    }

    #[test]
    fn mail_from_after_ready() {
        let mut session = ready_session();
        let actions = session.handle(SmtpEvent::Command(SmtpCommand::MailFrom {
            address: "alice@sender.example.com".to_string(),
        }));

        assert_eq!(session.state, SmtpState::MailFromReceived);
        assert!(actions.len() >= 2, "expected at least 2 actions, got {actions:?}");

        // First action: 250 OK
        assert_eq!(
            actions[0],
            SmtpAction::SendResponse(250, "OK".to_string())
        );

        // Second action: CheckSpf
        match &actions[1] {
            SmtpAction::CheckSpf {
                sender_domain,
                peer_ip: _,
            } => {
                assert_eq!(sender_domain, "sender.example.com");
            }
            other => panic!("expected CheckSpf, got {other:?}"),
        }
    }

    #[test]
    fn rcpt_to_triggers_resolve() {
        let mut session = ready_session();
        session.handle(SmtpEvent::Command(SmtpCommand::MailFrom {
            address: "alice@sender.example.com".to_string(),
        }));

        let actions = session.handle(SmtpEvent::Command(SmtpCommand::RcptTo {
            address: "bob@harmony.example.com".to_string(),
        }));

        assert_eq!(actions.len(), 1);
        match &actions[0] {
            SmtpAction::ResolveHarmonyAddress { local_part, domain } => {
                assert_eq!(local_part, "bob");
                assert_eq!(domain, "harmony.example.com");
            }
            other => panic!("expected ResolveHarmonyAddress, got {other:?}"),
        }
    }

    #[test]
    fn rcpt_to_unknown_recipient_rejects() {
        let mut session = ready_session();
        session.handle(SmtpEvent::Command(SmtpCommand::MailFrom {
            address: "alice@sender.example.com".to_string(),
        }));
        session.handle(SmtpEvent::Command(SmtpCommand::RcptTo {
            address: "unknown@harmony.example.com".to_string(),
        }));

        // Simulate resolution failure.
        let actions = session.handle(SmtpEvent::HarmonyResolved {
            local_part: "unknown".to_string(),
            identity: None,
        });

        assert_eq!(actions.len(), 1);
        match &actions[0] {
            SmtpAction::SendResponse(code, msg) => {
                assert_eq!(*code, 550);
                assert!(
                    msg.contains("User not found"),
                    "550 response should mention user not found: {msg}"
                );
            }
            other => panic!("expected SendResponse(550, ...), got {other:?}"),
        }
        // State should NOT advance to RcptToReceived.
        assert_eq!(session.state, SmtpState::MailFromReceived);
    }

    #[test]
    fn data_command_accepted_after_rcpt() {
        let mut session = ready_session();
        session.handle(SmtpEvent::Command(SmtpCommand::MailFrom {
            address: "alice@sender.example.com".to_string(),
        }));
        session.handle(SmtpEvent::Command(SmtpCommand::RcptTo {
            address: "bob@harmony.example.com".to_string(),
        }));
        // Resolve the recipient successfully.
        session.handle(SmtpEvent::HarmonyResolved {
            local_part: "bob".to_string(),
            identity: Some([0xBB; ADDRESS_HASH_LEN]),
        });
        assert_eq!(session.state, SmtpState::RcptToReceived);

        let actions = session.handle(SmtpEvent::Command(SmtpCommand::Data));

        assert_eq!(session.state, SmtpState::DataReceiving);
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            SmtpAction::SendResponse(code, msg) => {
                assert_eq!(*code, 354);
                assert!(
                    msg.contains("Start mail input"),
                    "354 response should say start mail input: {msg}"
                );
            }
            other => panic!("expected SendResponse(354, ...), got {other:?}"),
        }
    }

    #[test]
    fn quit_from_any_state() {
        // Test QUIT from Connected state.
        let mut session = SmtpSession::new(test_config());
        let actions = session.handle(SmtpEvent::Command(SmtpCommand::Quit));
        assert_eq!(session.state, SmtpState::Closed);
        assert_eq!(actions.len(), 2);
        assert_eq!(
            actions[0],
            SmtpAction::SendResponse(221, "Bye".to_string())
        );
        assert_eq!(actions[1], SmtpAction::Close);

        // Test QUIT from Ready state.
        let mut session = ready_session();
        let actions = session.handle(SmtpEvent::Command(SmtpCommand::Quit));
        assert_eq!(session.state, SmtpState::Closed);
        assert_eq!(actions.len(), 2);
        assert_eq!(
            actions[0],
            SmtpAction::SendResponse(221, "Bye".to_string())
        );
        assert_eq!(actions[1], SmtpAction::Close);
    }

    #[test]
    fn command_out_of_order_rejected() {
        // DATA before MAIL FROM (from Ready state).
        let mut session = ready_session();
        let actions = session.handle(SmtpEvent::Command(SmtpCommand::Data));

        assert_eq!(actions.len(), 1);
        match &actions[0] {
            SmtpAction::SendResponse(code, msg) => {
                assert_eq!(*code, 503);
                assert!(
                    msg.contains("Bad sequence"),
                    "503 response should mention bad sequence: {msg}"
                );
            }
            other => panic!("expected SendResponse(503, ...), got {other:?}"),
        }
        // State should remain Ready.
        assert_eq!(session.state, SmtpState::Ready);
    }

    #[test]
    fn null_sender_skips_spf() {
        // MAIL FROM:<> is the RFC 5321 §4.5.5 null sender for bounces/DSNs.
        // Should NOT emit CheckSpf with an empty domain.
        let mut session = ready_session();
        let actions = session.handle(SmtpEvent::Command(SmtpCommand::MailFrom {
            address: String::new(),
        }));

        assert_eq!(session.state, SmtpState::MailFromReceived);
        // Should only have the 250 OK — no CheckSpf action.
        assert_eq!(actions.len(), 1);
        assert_eq!(
            actions[0],
            SmtpAction::SendResponse(250, "OK".to_string())
        );
    }

    #[test]
    fn concurrent_rcpt_to_rejected() {
        // Second RCPT TO while first resolution is pending should be rejected.
        let mut session = ready_session();
        session.handle(SmtpEvent::Command(SmtpCommand::MailFrom {
            address: "alice@sender.example.com".to_string(),
        }));

        // First RCPT TO — starts resolution.
        session.handle(SmtpEvent::Command(SmtpCommand::RcptTo {
            address: "bob@harmony.example.com".to_string(),
        }));
        assert!(session.pending_rcpt.is_some());

        // Second RCPT TO before resolution completes — should be rejected.
        let actions = session.handle(SmtpEvent::Command(SmtpCommand::RcptTo {
            address: "carol@harmony.example.com".to_string(),
        }));

        assert_eq!(actions.len(), 1);
        match &actions[0] {
            SmtpAction::SendResponse(code, _) => {
                assert_eq!(*code, 421, "should reject with 421 temporary failure");
            }
            other => panic!("expected SendResponse(421, ...), got {other:?}"),
        }
    }
}
