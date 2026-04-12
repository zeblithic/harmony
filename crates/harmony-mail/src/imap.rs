//! Sans-I/O IMAP state machine.
//!
//! Pure `(State, Event) -> (State, Vec<Action>)` transitions with no networking,
//! no tokio, no async. The caller (I/O layer) feeds [`ImapEvent`]s and executes
//! the returned [`ImapAction`]s.

use crate::imap_parse::{
    FetchAttribute, ImapCommand, ImapTaggedCommand, SearchKey, SequenceSet, StatusItem,
    StoreOperation,
};

// ── Configuration ────────────────────────────────────────────────────

/// Gateway configuration for IMAP sessions.
#[derive(Debug, Clone)]
pub struct ImapConfig {
    /// The domain this gateway serves.
    pub domain: String,
    /// Whether TLS is currently active on this connection.
    pub tls_active: bool,
    /// Maximum authentication failures before disconnect.
    pub max_auth_failures: u32,
}

// ── State ────────────────────────────────────────────────────────────

/// IMAP session state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImapState {
    /// TCP connection accepted, no data exchanged yet.
    Connected,
    /// Greeting sent. Only CAPABILITY, STARTTLS, LOGIN, AUTHENTICATE, LOGOUT, NOOP valid.
    NotAuthenticated { auth_failures: u32 },
    /// Authenticated. Mailbox commands available.
    Authenticated { username: String },
    /// A mailbox is selected. Message-level commands available.
    Selected {
        username: String,
        mailbox: SelectedMailbox,
    },
    /// IDLE mode active. Awaiting DONE or notifications.
    Idling {
        tag: String,
        username: String,
        mailbox: SelectedMailbox,
    },
    /// STARTTLS negotiation in progress.
    TlsNegotiating,
    /// Session terminated.
    Logout,
}

/// State of a selected mailbox (carried in `Selected` and `Idling` states).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectedMailbox {
    pub name: String,
    pub uid_validity: u32,
    pub uid_next: u32,
    pub total_messages: u32,
    pub recent_count: u32,
    pub read_only: bool,
}

// ── Events ───────────────────────────────────────────────────────────

/// Events fed into the state machine.
#[derive(Debug, Clone)]
pub enum ImapEvent {
    /// TCP connection established.
    Connected,
    /// A parsed IMAP command.
    Command(ImapTaggedCommand),
    /// TLS upgrade completed.
    TlsCompleted,
    /// Authentication result from I/O layer.
    AuthResult { success: bool, username: String },
    /// Mailbox state loaded from storage.
    MailboxLoaded(MailboxSnapshot),
    /// STORE/flag update completed.
    StoreComplete {
        /// (sequence_num, updated flags) pairs.
        updated: Vec<(u32, Vec<String>)>,
    },
    /// EXPUNGE completed.
    ExpungeComplete {
        /// Sequence numbers of expunged messages (descending order for safe removal).
        expunged_seqnums: Vec<u32>,
    },
    /// SEARCH results.
    SearchComplete { results: Vec<u32> },
    /// COPY/MOVE completed.
    CopyComplete {
        /// (source_uid, dest_uid) pairs.
        uid_mapping: Vec<(u32, u32)>,
    },
    /// IDLE: client sent DONE.
    IdleDone,
    /// IDLE: new messages arrived.
    IdleNotify { new_exists: u32 },
}

/// Snapshot of mailbox state returned from storage.
#[derive(Debug, Clone)]
pub struct MailboxSnapshot {
    pub name: String,
    pub uid_validity: u32,
    pub uid_next: u32,
    pub total_messages: u32,
    pub recent_count: u32,
    pub unseen_count: u32,
    pub first_unseen: Option<u32>,
    pub read_only: bool,
}

// ── Actions ──────────────────────────────────────────────────────────

/// Status of a tagged IMAP response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResponseStatus {
    Ok,
    No,
    Bad,
}

/// Actions emitted by the state machine for the I/O layer to execute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImapAction {
    /// Send an untagged response (e.g., `* OK ...`, `* 5 EXISTS`).
    SendUntagged(String),
    /// Send a tagged response (e.g., `A001 OK SELECT completed`).
    SendTagged {
        tag: String,
        status: ResponseStatus,
        code: Option<String>,
        text: String,
    },
    /// Send a continuation request (e.g., `+ Ready`).
    SendContinuation(String),
    /// Authenticate credentials.
    Authenticate { username: String, password: String },
    /// Load mailbox state from storage.
    SelectMailbox { name: String, read_only: bool },
    /// Fetch message data.
    FetchMessages {
        sequence_set: SequenceSet,
        attributes: Vec<FetchAttribute>,
        uid_mode: bool,
    },
    /// Update flags.
    StoreFlags {
        sequence_set: SequenceSet,
        operation: StoreOperation,
        flags: Vec<String>,
        uid_mode: bool,
        silent: bool,
    },
    /// Expunge deleted messages.
    Expunge,
    /// Execute SEARCH.
    Search {
        criteria: Vec<SearchKey>,
        uid_mode: bool,
    },
    /// Copy/move messages.
    CopyMessages {
        sequence_set: SequenceSet,
        destination: String,
        uid_mode: bool,
        is_move: bool,
    },
    /// List mailboxes matching pattern.
    ListMailboxes { reference: String, pattern: String },
    /// Get STATUS for a mailbox.
    GetStatus {
        mailbox: String,
        items: Vec<StatusItem>,
    },
    /// Create a mailbox.
    CreateMailbox { name: String },
    /// Delete a mailbox.
    DeleteMailbox { name: String },
    /// Subscribe to a mailbox.
    SubscribeMailbox { name: String },
    /// Unsubscribe from a mailbox.
    UnsubscribeMailbox { name: String },
    /// Enter IDLE: subscribe to notifications.
    StartIdle { tag: String },
    /// Exit IDLE.
    StopIdle,
    /// Initiate TLS upgrade.
    StartTls,
    /// Close the connection.
    Close,
}

// ── Session ──────────────────────────────────────────────────────────

/// The IMAP session state machine.
pub struct ImapSession {
    pub state: ImapState,
    config: ImapConfig,
    /// Tag of the pending command (for async responses).
    /// Tag of the pending command (for async responses).
    /// Public so the I/O layer can consume it for actions that complete synchronously
    /// in the I/O layer (LIST, STATUS, CREATE, DELETE, etc.).
    pub pending_tag: Option<String>,
}

impl ImapSession {
    pub fn new(config: ImapConfig) -> Self {
        Self {
            state: ImapState::Connected,
            config,
            pending_tag: None,
        }
    }

    /// Feed an event into the state machine and return the resulting actions.
    pub fn handle(&mut self, event: ImapEvent) -> Vec<ImapAction> {
        match event {
            ImapEvent::Connected => self.handle_connected(),
            ImapEvent::Command(cmd) => self.handle_command(cmd),
            ImapEvent::TlsCompleted => self.handle_tls_completed(),
            ImapEvent::AuthResult { success, username } => {
                self.handle_auth_result(success, &username)
            }
            ImapEvent::MailboxLoaded(snapshot) => self.handle_mailbox_loaded(snapshot),
            ImapEvent::StoreComplete { updated } => self.handle_store_complete(updated),
            ImapEvent::ExpungeComplete { expunged_seqnums } => {
                self.handle_expunge_complete(expunged_seqnums)
            }
            ImapEvent::SearchComplete { results } => self.handle_search_complete(results),
            ImapEvent::CopyComplete { uid_mapping } => self.handle_copy_complete(uid_mapping),
            ImapEvent::IdleDone => self.handle_idle_done(),
            ImapEvent::IdleNotify { new_exists } => self.handle_idle_notify(new_exists),
        }
    }

    // ── Event handlers ───────────────────────────────────────────────

    fn handle_connected(&mut self) -> Vec<ImapAction> {
        if self.state != ImapState::Connected {
            return vec![];
        }
        self.state = ImapState::NotAuthenticated { auth_failures: 0 };
        vec![ImapAction::SendUntagged(format!(
            "OK [CAPABILITY {}] Harmony Mail IMAP ready",
            self.capability_string()
        ))]
    }

    fn handle_command(&mut self, cmd: ImapTaggedCommand) -> Vec<ImapAction> {
        let tag = cmd.tag;

        // Commands valid in any state (except Logout and TlsNegotiating)
        match &self.state {
            ImapState::Logout | ImapState::Connected => {
                return vec![ImapAction::SendTagged {
                    tag,
                    status: ResponseStatus::Bad,
                    code: None,
                    text: "invalid state".to_string(),
                }];
            }
            ImapState::TlsNegotiating => {
                return vec![ImapAction::SendTagged {
                    tag,
                    status: ResponseStatus::Bad,
                    code: None,
                    text: "TLS negotiation in progress".to_string(),
                }];
            }
            ImapState::Idling { .. } => {
                // During IDLE, only DONE is valid (handled via IdleDone event, not Command)
                return vec![ImapAction::SendTagged {
                    tag,
                    status: ResponseStatus::Bad,
                    code: None,
                    text: "in IDLE mode, send DONE to exit".to_string(),
                }];
            }
            _ => {}
        }

        match cmd.command {
            // ── Any-state commands ──────────────────────────────────
            ImapCommand::Capability => {
                let caps = self.capability_string();
                vec![
                    ImapAction::SendUntagged(format!("CAPABILITY {caps}")),
                    ImapAction::SendTagged {
                        tag,
                        status: ResponseStatus::Ok,
                        code: None,
                        text: "CAPABILITY completed".to_string(),
                    },
                ]
            }
            ImapCommand::Noop => {
                vec![ImapAction::SendTagged {
                    tag,
                    status: ResponseStatus::Ok,
                    code: None,
                    text: "NOOP completed".to_string(),
                }]
            }
            ImapCommand::Logout => {
                self.state = ImapState::Logout;
                vec![
                    ImapAction::SendUntagged(
                        "BYE Harmony Mail IMAP server closing connection".to_string(),
                    ),
                    ImapAction::SendTagged {
                        tag,
                        status: ResponseStatus::Ok,
                        code: None,
                        text: "LOGOUT completed".to_string(),
                    },
                    ImapAction::Close,
                ]
            }

            // ── Not-authenticated commands ──────────────────────────
            ImapCommand::StartTls => self.handle_starttls(tag),
            ImapCommand::Login { username, password } => self.handle_login(tag, username, password),
            ImapCommand::AuthenticatePlain { initial_response } => {
                self.handle_authenticate_plain(tag, initial_response)
            }

            // ── Authenticated commands ──────────────────────────────
            ImapCommand::Select { mailbox } => self.handle_select(tag, mailbox, false),
            ImapCommand::Examine { mailbox } => self.handle_select(tag, mailbox, true),
            ImapCommand::List { reference, pattern } => self.handle_list(tag, reference, pattern),
            ImapCommand::Status { mailbox, items } => self.handle_status(tag, mailbox, items),
            ImapCommand::Create { mailbox } => self.handle_create(tag, mailbox),
            ImapCommand::Delete { mailbox } => self.handle_delete(tag, mailbox),
            ImapCommand::Subscribe { mailbox } => self.handle_subscribe(tag, mailbox),
            ImapCommand::Unsubscribe { mailbox } => self.handle_unsubscribe(tag, mailbox),

            // ── Selected commands ───────────────────────────────────
            ImapCommand::Close => self.handle_close(tag),
            ImapCommand::Unselect => self.handle_unselect(tag),
            ImapCommand::Expunge => self.handle_expunge(tag),
            ImapCommand::Fetch {
                sequence_set,
                attributes,
                uid,
            } => self.handle_fetch(tag, sequence_set, attributes, uid),
            ImapCommand::Store {
                sequence_set,
                operation,
                flags,
                uid,
            } => self.handle_store(tag, sequence_set, operation, flags, uid),
            ImapCommand::Search { criteria, uid } => self.handle_search(tag, criteria, uid),
            ImapCommand::Copy {
                sequence_set,
                mailbox,
                uid,
            } => self.handle_copy(tag, sequence_set, mailbox, uid, false),
            ImapCommand::Move {
                sequence_set,
                mailbox,
                uid,
            } => self.handle_copy(tag, sequence_set, mailbox, uid, true),
            ImapCommand::Idle => self.handle_idle(tag),
        }
    }

    // ── Command handlers ────────────────────────────────────────────

    fn handle_starttls(&mut self, tag: String) -> Vec<ImapAction> {
        if !matches!(self.state, ImapState::NotAuthenticated { .. }) {
            return vec![self.bad_tag(tag, "STARTTLS only valid before authentication")];
        }
        if self.config.tls_active {
            return vec![self.bad_tag(tag, "TLS already active")];
        }
        self.state = ImapState::TlsNegotiating;
        self.pending_tag = Some(tag.clone());
        vec![
            ImapAction::SendTagged {
                tag,
                status: ResponseStatus::Ok,
                code: None,
                text: "Begin TLS negotiation".to_string(),
            },
            ImapAction::StartTls,
        ]
    }

    fn handle_login(&mut self, tag: String, username: String, password: String) -> Vec<ImapAction> {
        if !matches!(self.state, ImapState::NotAuthenticated { .. }) {
            return vec![self.bad_tag(tag, "already authenticated")];
        }
        if !self.config.tls_active {
            return vec![ImapAction::SendTagged {
                tag,
                status: ResponseStatus::No,
                code: Some("PRIVACYREQUIRED".to_string()),
                text: "LOGIN requires TLS".to_string(),
            }];
        }
        self.pending_tag = Some(tag);
        vec![ImapAction::Authenticate { username, password }]
    }

    fn handle_authenticate_plain(
        &mut self,
        tag: String,
        initial_response: Option<String>,
    ) -> Vec<ImapAction> {
        if !matches!(self.state, ImapState::NotAuthenticated { .. }) {
            return vec![self.bad_tag(tag, "already authenticated")];
        }
        if !self.config.tls_active {
            return vec![ImapAction::SendTagged {
                tag,
                status: ResponseStatus::No,
                code: Some("PRIVACYREQUIRED".to_string()),
                text: "AUTHENTICATE requires TLS".to_string(),
            }];
        }

        match initial_response {
            Some(ref data) => {
                // Decode SASL PLAIN: base64 of \0username\0password
                match decode_sasl_plain(data) {
                    Some((username, password)) => {
                        self.pending_tag = Some(tag);
                        vec![ImapAction::Authenticate { username, password }]
                    }
                    None => vec![ImapAction::SendTagged {
                        tag,
                        status: ResponseStatus::No,
                        code: None,
                        text: "invalid SASL PLAIN data".to_string(),
                    }],
                }
            }
            None => {
                // Need continuation for credentials
                self.pending_tag = Some(tag);
                vec![ImapAction::SendContinuation(String::new())]
            }
        }
    }

    fn handle_select(&mut self, tag: String, mailbox: String, read_only: bool) -> Vec<ImapAction> {
        match &self.state {
            ImapState::Authenticated { .. } | ImapState::Selected { .. } => {}
            _ => return vec![self.bad_tag(tag, "not authenticated")],
        }
        self.pending_tag = Some(tag);
        vec![ImapAction::SelectMailbox {
            name: mailbox,
            read_only,
        }]
    }

    fn handle_list(&mut self, tag: String, reference: String, pattern: String) -> Vec<ImapAction> {
        if !self.is_authenticated() {
            return vec![self.bad_tag(tag, "not authenticated")];
        }
        self.pending_tag = Some(tag);
        vec![ImapAction::ListMailboxes { reference, pattern }]
    }

    fn handle_status(
        &mut self,
        tag: String,
        mailbox: String,
        items: Vec<StatusItem>,
    ) -> Vec<ImapAction> {
        if !self.is_authenticated() {
            return vec![self.bad_tag(tag, "not authenticated")];
        }
        self.pending_tag = Some(tag);
        vec![ImapAction::GetStatus { mailbox, items }]
    }

    fn handle_create(&mut self, tag: String, mailbox: String) -> Vec<ImapAction> {
        if !self.is_authenticated() {
            return vec![self.bad_tag(tag, "not authenticated")];
        }
        self.pending_tag = Some(tag);
        vec![ImapAction::CreateMailbox { name: mailbox }]
    }

    fn handle_delete(&mut self, tag: String, mailbox: String) -> Vec<ImapAction> {
        if !self.is_authenticated() {
            return vec![self.bad_tag(tag, "not authenticated")];
        }
        self.pending_tag = Some(tag);
        vec![ImapAction::DeleteMailbox { name: mailbox }]
    }

    fn handle_subscribe(&mut self, tag: String, mailbox: String) -> Vec<ImapAction> {
        if !self.is_authenticated() {
            return vec![self.bad_tag(tag, "not authenticated")];
        }
        self.pending_tag = Some(tag);
        vec![ImapAction::SubscribeMailbox { name: mailbox }]
    }

    fn handle_unsubscribe(&mut self, tag: String, mailbox: String) -> Vec<ImapAction> {
        if !self.is_authenticated() {
            return vec![self.bad_tag(tag, "not authenticated")];
        }
        self.pending_tag = Some(tag);
        vec![ImapAction::UnsubscribeMailbox { name: mailbox }]
    }

    fn handle_close(&mut self, tag: String) -> Vec<ImapAction> {
        match &self.state {
            ImapState::Selected { username, mailbox } => {
                let username = username.clone();
                let _was_readonly = mailbox.read_only;
                self.state = ImapState::Authenticated { username };
                // CLOSE implicitly expunges (if not read-only)
                // For simplicity in v1.1, we just close without expunge
                vec![ImapAction::SendTagged {
                    tag,
                    status: ResponseStatus::Ok,
                    code: None,
                    text: "CLOSE completed".to_string(),
                }]
            }
            _ => vec![self.bad_tag(tag, "no mailbox selected")],
        }
    }

    fn handle_unselect(&mut self, tag: String) -> Vec<ImapAction> {
        match &self.state {
            ImapState::Selected { username, .. } => {
                let username = username.clone();
                self.state = ImapState::Authenticated { username };
                vec![ImapAction::SendTagged {
                    tag,
                    status: ResponseStatus::Ok,
                    code: None,
                    text: "UNSELECT completed".to_string(),
                }]
            }
            _ => vec![self.bad_tag(tag, "no mailbox selected")],
        }
    }

    fn handle_expunge(&mut self, tag: String) -> Vec<ImapAction> {
        match &self.state {
            ImapState::Selected { mailbox, .. } if !mailbox.read_only => {
                self.pending_tag = Some(tag);
                vec![ImapAction::Expunge]
            }
            ImapState::Selected { .. } => vec![ImapAction::SendTagged {
                tag,
                status: ResponseStatus::No,
                code: Some("READ-ONLY".to_string()),
                text: "mailbox is read-only".to_string(),
            }],
            _ => vec![self.bad_tag(tag, "no mailbox selected")],
        }
    }

    fn handle_fetch(
        &mut self,
        tag: String,
        sequence_set: SequenceSet,
        attributes: Vec<FetchAttribute>,
        uid_mode: bool,
    ) -> Vec<ImapAction> {
        if !self.is_selected() {
            return vec![self.bad_tag(tag, "no mailbox selected")];
        }
        self.pending_tag = Some(tag);
        vec![ImapAction::FetchMessages {
            sequence_set,
            attributes,
            uid_mode,
        }]
    }

    fn handle_store(
        &mut self,
        tag: String,
        sequence_set: SequenceSet,
        operation: StoreOperation,
        flags: Vec<String>,
        uid_mode: bool,
    ) -> Vec<ImapAction> {
        match &self.state {
            ImapState::Selected { mailbox, .. } if !mailbox.read_only => {
                let silent = matches!(
                    operation,
                    StoreOperation::SetSilent
                        | StoreOperation::AddSilent
                        | StoreOperation::RemoveSilent
                );
                self.pending_tag = Some(tag);
                vec![ImapAction::StoreFlags {
                    sequence_set,
                    operation,
                    flags,
                    uid_mode,
                    silent,
                }]
            }
            ImapState::Selected { .. } => vec![ImapAction::SendTagged {
                tag,
                status: ResponseStatus::No,
                code: Some("READ-ONLY".to_string()),
                text: "mailbox is read-only".to_string(),
            }],
            _ => vec![self.bad_tag(tag, "no mailbox selected")],
        }
    }

    fn handle_search(
        &mut self,
        tag: String,
        criteria: Vec<SearchKey>,
        uid_mode: bool,
    ) -> Vec<ImapAction> {
        if !self.is_selected() {
            return vec![self.bad_tag(tag, "no mailbox selected")];
        }
        self.pending_tag = Some(tag);
        vec![ImapAction::Search { criteria, uid_mode }]
    }

    fn handle_copy(
        &mut self,
        tag: String,
        sequence_set: SequenceSet,
        mailbox: String,
        uid_mode: bool,
        is_move: bool,
    ) -> Vec<ImapAction> {
        if !self.is_selected() {
            return vec![self.bad_tag(tag, "no mailbox selected")];
        }
        if is_move {
            if let ImapState::Selected {
                mailbox: ref sel, ..
            } = self.state
            {
                if sel.read_only {
                    return vec![ImapAction::SendTagged {
                        tag,
                        status: ResponseStatus::No,
                        code: Some("READ-ONLY".to_string()),
                        text: "mailbox is read-only".to_string(),
                    }];
                }
            }
        }
        self.pending_tag = Some(tag);
        vec![ImapAction::CopyMessages {
            sequence_set,
            destination: mailbox,
            uid_mode,
            is_move,
        }]
    }

    fn handle_idle(&mut self, tag: String) -> Vec<ImapAction> {
        if !self.is_selected() {
            return vec![self.bad_tag(tag, "no mailbox selected")];
        }
        match std::mem::replace(&mut self.state, ImapState::Logout) {
            ImapState::Selected { username, mailbox } => {
                self.state = ImapState::Idling {
                    tag: tag.clone(),
                    username,
                    mailbox,
                };
                vec![
                    ImapAction::SendContinuation("idling".to_string()),
                    ImapAction::StartIdle { tag },
                ]
            }
            other => {
                self.state = other;
                vec![self.bad_tag(tag, "no mailbox selected")]
            }
        }
    }

    // ── Async result handlers ───────────────────────────────────────

    fn handle_tls_completed(&mut self) -> Vec<ImapAction> {
        if self.state != ImapState::TlsNegotiating {
            return vec![];
        }
        self.config.tls_active = true;
        self.state = ImapState::NotAuthenticated { auth_failures: 0 };
        vec![]
    }

    fn handle_auth_result(&mut self, success: bool, username: &str) -> Vec<ImapAction> {
        let tag = self.pending_tag.take().unwrap_or_else(|| "?".to_string());

        if success {
            self.state = ImapState::Authenticated {
                username: username.to_string(),
            };
            vec![ImapAction::SendTagged {
                tag,
                status: ResponseStatus::Ok,
                code: Some(format!("CAPABILITY {}", self.capability_string())),
                text: "LOGIN completed".to_string(),
            }]
        } else {
            let failures = match &mut self.state {
                ImapState::NotAuthenticated { auth_failures } => {
                    *auth_failures += 1;
                    *auth_failures
                }
                _ => 0,
            };

            if failures >= self.config.max_auth_failures {
                self.state = ImapState::Logout;
                vec![
                    ImapAction::SendUntagged("BYE too many authentication failures".to_string()),
                    ImapAction::SendTagged {
                        tag,
                        status: ResponseStatus::No,
                        code: None,
                        text: "authentication failed".to_string(),
                    },
                    ImapAction::Close,
                ]
            } else {
                vec![ImapAction::SendTagged {
                    tag,
                    status: ResponseStatus::No,
                    code: None,
                    text: "authentication failed".to_string(),
                }]
            }
        }
    }

    fn handle_mailbox_loaded(&mut self, snapshot: MailboxSnapshot) -> Vec<ImapAction> {
        let tag = self.pending_tag.take().unwrap_or_else(|| "?".to_string());
        let username = self.current_username().to_string();

        let selected = SelectedMailbox {
            name: snapshot.name.clone(),
            uid_validity: snapshot.uid_validity,
            uid_next: snapshot.uid_next,
            total_messages: snapshot.total_messages,
            recent_count: snapshot.recent_count,
            read_only: snapshot.read_only,
        };

        self.state = ImapState::Selected {
            username,
            mailbox: selected,
        };

        let access = if snapshot.read_only {
            "READ-ONLY"
        } else {
            "READ-WRITE"
        };
        let cmd_name = if snapshot.read_only {
            "EXAMINE"
        } else {
            "SELECT"
        };

        let mut actions = vec![
            ImapAction::SendUntagged(format!("{} EXISTS", snapshot.total_messages)),
            ImapAction::SendUntagged(format!("{} RECENT", snapshot.recent_count)),
            ImapAction::SendUntagged(
                "FLAGS (\\Answered \\Flagged \\Deleted \\Seen \\Draft)".to_string(),
            ),
            ImapAction::SendUntagged(
                "OK [PERMANENTFLAGS (\\Answered \\Flagged \\Deleted \\Seen \\Draft \\*)] permanent flags".to_string(),
            ),
            ImapAction::SendUntagged(format!(
                "OK [UIDVALIDITY {}] UIDs valid",
                snapshot.uid_validity
            )),
            ImapAction::SendUntagged(format!(
                "OK [UIDNEXT {}] predicted next UID",
                snapshot.uid_next
            )),
        ];

        if let Some(first) = snapshot.first_unseen {
            actions.push(ImapAction::SendUntagged(format!(
                "OK [UNSEEN {first}] first unseen message"
            )));
        }

        actions.push(ImapAction::SendTagged {
            tag,
            status: ResponseStatus::Ok,
            code: Some(access.to_string()),
            text: format!("{cmd_name} completed"),
        });

        actions
    }

    fn handle_store_complete(&mut self, updated: Vec<(u32, Vec<String>)>) -> Vec<ImapAction> {
        let tag = self.pending_tag.take().unwrap_or_else(|| "?".to_string());
        let mut actions = Vec::new();

        // Send untagged FETCH responses for updated flags (unless SILENT)
        for (seqnum, flags) in &updated {
            let flags_str = if flags.is_empty() {
                "()".to_string()
            } else {
                format!("({})", flags.join(" "))
            };
            actions.push(ImapAction::SendUntagged(format!(
                "{seqnum} FETCH (FLAGS {flags_str})"
            )));
        }

        actions.push(ImapAction::SendTagged {
            tag,
            status: ResponseStatus::Ok,
            code: None,
            text: "STORE completed".to_string(),
        });
        actions
    }

    fn handle_expunge_complete(&mut self, expunged_seqnums: Vec<u32>) -> Vec<ImapAction> {
        let tag = self.pending_tag.take().unwrap_or_else(|| "?".to_string());
        let mut actions = Vec::new();

        for seqnum in &expunged_seqnums {
            actions.push(ImapAction::SendUntagged(format!("{seqnum} EXPUNGE")));
        }

        // Update selected mailbox state
        if let ImapState::Selected {
            ref mut mailbox, ..
        } = self.state
        {
            mailbox.total_messages = mailbox
                .total_messages
                .saturating_sub(expunged_seqnums.len() as u32);
        }

        actions.push(ImapAction::SendTagged {
            tag,
            status: ResponseStatus::Ok,
            code: None,
            text: "EXPUNGE completed".to_string(),
        });
        actions
    }

    fn handle_search_complete(&mut self, results: Vec<u32>) -> Vec<ImapAction> {
        let tag = self.pending_tag.take().unwrap_or_else(|| "?".to_string());
        let nums: Vec<String> = results.iter().map(|n| n.to_string()).collect();
        vec![
            ImapAction::SendUntagged(format!("SEARCH {}", nums.join(" "))),
            ImapAction::SendTagged {
                tag,
                status: ResponseStatus::Ok,
                code: None,
                text: "SEARCH completed".to_string(),
            },
        ]
    }

    fn handle_copy_complete(&mut self, uid_mapping: Vec<(u32, u32)>) -> Vec<ImapAction> {
        let tag = self.pending_tag.take().unwrap_or_else(|| "?".to_string());

        let code = if !uid_mapping.is_empty() {
            let src_uids: Vec<String> = uid_mapping.iter().map(|(s, _)| s.to_string()).collect();
            let dst_uids: Vec<String> = uid_mapping.iter().map(|(_, d)| d.to_string()).collect();
            Some(format!(
                "COPYUID {} {} {}",
                0,
                src_uids.join(","),
                dst_uids.join(",")
            ))
        } else {
            None
        };

        vec![ImapAction::SendTagged {
            tag,
            status: ResponseStatus::Ok,
            code,
            text: "COPY completed".to_string(),
        }]
    }

    fn handle_idle_done(&mut self) -> Vec<ImapAction> {
        match std::mem::replace(&mut self.state, ImapState::Logout) {
            ImapState::Idling {
                tag,
                username,
                mailbox,
            } => {
                self.state = ImapState::Selected { username, mailbox };
                vec![
                    ImapAction::StopIdle,
                    ImapAction::SendTagged {
                        tag,
                        status: ResponseStatus::Ok,
                        code: None,
                        text: "IDLE completed".to_string(),
                    },
                ]
            }
            other => {
                self.state = other;
                vec![]
            }
        }
    }

    fn handle_idle_notify(&mut self, new_exists: u32) -> Vec<ImapAction> {
        match &mut self.state {
            ImapState::Idling { mailbox, .. } => {
                mailbox.total_messages = new_exists;
                vec![ImapAction::SendUntagged(format!("{new_exists} EXISTS"))]
            }
            _ => vec![],
        }
    }

    // ── Helpers ─────────────────────────────────────────────────────

    fn capability_string(&self) -> String {
        let mut caps = vec!["IMAP4rev2"];

        match &self.state {
            ImapState::NotAuthenticated { .. } | ImapState::Connected => {
                if !self.config.tls_active {
                    caps.push("STARTTLS");
                    caps.push("LOGINDISABLED");
                } else {
                    caps.push("AUTH=PLAIN");
                }
            }
            _ => {
                caps.push("IDLE");
                caps.push("MOVE");
                caps.push("UNSELECT");
                caps.push("UIDPLUS");
            }
        }

        caps.join(" ")
    }

    fn is_authenticated(&self) -> bool {
        matches!(
            self.state,
            ImapState::Authenticated { .. } | ImapState::Selected { .. }
        )
    }

    fn is_selected(&self) -> bool {
        matches!(self.state, ImapState::Selected { .. })
    }

    fn current_username(&self) -> &str {
        match &self.state {
            ImapState::Authenticated { username } => username,
            ImapState::Selected { username, .. } => username,
            ImapState::Idling { username, .. } => username,
            _ => "",
        }
    }

    fn bad_tag(&self, tag: String, text: &str) -> ImapAction {
        ImapAction::SendTagged {
            tag,
            status: ResponseStatus::Bad,
            code: None,
            text: text.to_string(),
        }
    }
}

// ── SASL PLAIN decoding ─────────────────────────────────────────────

/// Decode SASL PLAIN: base64(\0username\0password).
/// Returns (username, password) on success.
fn decode_sasl_plain(data: &str) -> Option<(String, String)> {
    use base64::Engine;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(data.trim())
        .ok()?;

    // Format: [authzid] \0 authcid \0 passwd
    let mut parts = bytes.splitn(3, |&b| b == 0);
    let _authzid = parts.next()?; // ignored
    let username = parts.next()?;
    let password = parts.next()?;

    let username = std::str::from_utf8(username).ok()?.to_string();
    let password = std::str::from_utf8(password).ok()?.to_string();

    if username.is_empty() {
        return None;
    }

    Some((username, password))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imap_parse::parse_command;
    use base64::Engine;

    fn config_tls() -> ImapConfig {
        ImapConfig {
            domain: "q8.fyi".to_string(),
            tls_active: true,
            max_auth_failures: 3,
        }
    }

    fn config_no_tls() -> ImapConfig {
        ImapConfig {
            domain: "q8.fyi".to_string(),
            tls_active: false,
            max_auth_failures: 3,
        }
    }

    fn connected_session(config: ImapConfig) -> ImapSession {
        let mut s = ImapSession::new(config);
        s.handle(ImapEvent::Connected);
        s
    }

    fn authenticated_session() -> ImapSession {
        let mut s = connected_session(config_tls());
        let cmd = parse_command("A001 LOGIN alice secret").unwrap();
        s.handle(ImapEvent::Command(cmd));
        s.handle(ImapEvent::AuthResult {
            success: true,
            username: "alice".to_string(),
        });
        s
    }

    fn selected_session() -> ImapSession {
        let mut s = authenticated_session();
        let cmd = parse_command("A002 SELECT INBOX").unwrap();
        s.handle(ImapEvent::Command(cmd));
        s.handle(ImapEvent::MailboxLoaded(MailboxSnapshot {
            name: "INBOX".to_string(),
            uid_validity: 12345,
            uid_next: 10,
            total_messages: 5,
            recent_count: 1,
            unseen_count: 2,
            first_unseen: Some(3),
            read_only: false,
        }));
        s
    }

    // ── Greeting ────────────────────────────────────────────────────

    #[test]
    fn greeting_sent_on_connect() {
        let mut s = ImapSession::new(config_tls());
        let actions = s.handle(ImapEvent::Connected);
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            ImapAction::SendUntagged(msg) => {
                assert!(msg.contains("OK"));
                assert!(msg.contains("CAPABILITY"));
                assert!(msg.contains("IMAP4rev2"));
            }
            _ => panic!("expected SendUntagged"),
        }
    }

    #[test]
    fn greeting_advertises_starttls_without_tls() {
        let mut s = ImapSession::new(config_no_tls());
        let actions = s.handle(ImapEvent::Connected);
        let msg = match &actions[0] {
            ImapAction::SendUntagged(m) => m,
            _ => panic!(),
        };
        assert!(msg.contains("STARTTLS"));
        assert!(msg.contains("LOGINDISABLED"));
        assert!(!msg.contains("AUTH=PLAIN"));
    }

    #[test]
    fn greeting_advertises_auth_with_tls() {
        let mut s = ImapSession::new(config_tls());
        let actions = s.handle(ImapEvent::Connected);
        let msg = match &actions[0] {
            ImapAction::SendUntagged(m) => m,
            _ => panic!(),
        };
        assert!(msg.contains("AUTH=PLAIN"));
        assert!(!msg.contains("STARTTLS"));
        assert!(!msg.contains("LOGINDISABLED"));
    }

    // ── CAPABILITY ──────────────────────────────────────────────────

    #[test]
    fn capability_command() {
        let mut s = connected_session(config_tls());
        let cmd = parse_command("A001 CAPABILITY").unwrap();
        let actions = s.handle(ImapEvent::Command(cmd));
        assert_eq!(actions.len(), 2);
        assert!(matches!(&actions[0], ImapAction::SendUntagged(m) if m.contains("IMAP4rev2")));
        assert!(matches!(
            &actions[1],
            ImapAction::SendTagged {
                status: ResponseStatus::Ok,
                ..
            }
        ));
    }

    // ── NOOP ────────────────────────────────────────────────────────

    #[test]
    fn noop_succeeds() {
        let mut s = connected_session(config_tls());
        let cmd = parse_command("A001 NOOP").unwrap();
        let actions = s.handle(ImapEvent::Command(cmd));
        assert_eq!(actions.len(), 1);
        assert!(matches!(
            &actions[0],
            ImapAction::SendTagged {
                status: ResponseStatus::Ok,
                ..
            }
        ));
    }

    // ── LOGOUT ──────────────────────────────────────────────────────

    #[test]
    fn logout_closes_connection() {
        let mut s = connected_session(config_tls());
        let cmd = parse_command("A001 LOGOUT").unwrap();
        let actions = s.handle(ImapEvent::Command(cmd));
        assert!(actions
            .iter()
            .any(|a| matches!(a, ImapAction::SendUntagged(m) if m.contains("BYE"))));
        assert!(actions.iter().any(|a| matches!(a, ImapAction::Close)));
        assert_eq!(s.state, ImapState::Logout);
    }

    // ── LOGIN ───────────────────────────────────────────────────────

    #[test]
    fn login_requires_tls() {
        let mut s = connected_session(config_no_tls());
        let cmd = parse_command("A001 LOGIN alice secret").unwrap();
        let actions = s.handle(ImapEvent::Command(cmd));
        assert!(matches!(&actions[0], ImapAction::SendTagged {
            status: ResponseStatus::No,
            code: Some(c),
            ..
        } if c == "PRIVACYREQUIRED"));
    }

    #[test]
    fn login_success() {
        let mut s = connected_session(config_tls());
        let cmd = parse_command("A001 LOGIN alice secret").unwrap();
        let actions = s.handle(ImapEvent::Command(cmd));
        assert!(matches!(&actions[0], ImapAction::Authenticate { .. }));

        let actions = s.handle(ImapEvent::AuthResult {
            success: true,
            username: "alice".to_string(),
        });
        assert!(matches!(
            &actions[0],
            ImapAction::SendTagged {
                status: ResponseStatus::Ok,
                ..
            }
        ));
        assert!(
            matches!(s.state, ImapState::Authenticated { ref username } if username == "alice")
        );
    }

    #[test]
    fn login_failure_increments_counter() {
        let mut s = connected_session(config_tls());

        for i in 0..2 {
            let cmd = parse_command("A001 LOGIN alice wrong").unwrap();
            s.handle(ImapEvent::Command(cmd));
            let actions = s.handle(ImapEvent::AuthResult {
                success: false,
                username: "alice".to_string(),
            });
            assert!(
                matches!(
                    &actions[0],
                    ImapAction::SendTagged {
                        status: ResponseStatus::No,
                        ..
                    }
                ),
                "attempt {i} should fail"
            );
            assert!(
                matches!(s.state, ImapState::NotAuthenticated { auth_failures } if auth_failures == (i + 1) as u32)
            );
        }
    }

    #[test]
    fn login_failure_disconnects_after_max() {
        let mut s = connected_session(config_tls());

        for _ in 0..3 {
            let cmd = parse_command("A001 LOGIN alice wrong").unwrap();
            s.handle(ImapEvent::Command(cmd));
            s.handle(ImapEvent::AuthResult {
                success: false,
                username: "alice".to_string(),
            });
        }
        assert_eq!(s.state, ImapState::Logout);
    }

    // ── AUTHENTICATE PLAIN ──────────────────────────────────────────

    #[test]
    fn authenticate_plain_with_initial_response() {
        let mut s = connected_session(config_tls());
        // base64 of \0alice\0secret
        let b64 = base64::engine::general_purpose::STANDARD.encode(b"\0alice\0secret");
        let line = format!("A001 AUTHENTICATE PLAIN {b64}");
        let cmd = parse_command(&line).unwrap();
        let actions = s.handle(ImapEvent::Command(cmd));
        assert!(
            matches!(&actions[0], ImapAction::Authenticate { username, password }
            if username == "alice" && password == "secret")
        );
    }

    #[test]
    fn authenticate_plain_without_initial_response() {
        let mut s = connected_session(config_tls());
        let cmd = parse_command("A001 AUTHENTICATE PLAIN").unwrap();
        let actions = s.handle(ImapEvent::Command(cmd));
        assert!(matches!(&actions[0], ImapAction::SendContinuation(_)));
    }

    // ── STARTTLS ────────────────────────────────────────────────────

    #[test]
    fn starttls_transitions_to_negotiating() {
        let mut s = connected_session(config_no_tls());
        let cmd = parse_command("A001 STARTTLS").unwrap();
        let actions = s.handle(ImapEvent::Command(cmd));
        assert!(actions.iter().any(|a| matches!(a, ImapAction::StartTls)));
        assert_eq!(s.state, ImapState::TlsNegotiating);
    }

    #[test]
    fn starttls_rejected_when_tls_active() {
        let mut s = connected_session(config_tls());
        let cmd = parse_command("A001 STARTTLS").unwrap();
        let actions = s.handle(ImapEvent::Command(cmd));
        assert!(matches!(
            &actions[0],
            ImapAction::SendTagged {
                status: ResponseStatus::Bad,
                ..
            }
        ));
    }

    #[test]
    fn tls_completed_returns_to_not_authenticated() {
        let mut s = connected_session(config_no_tls());
        let cmd = parse_command("A001 STARTTLS").unwrap();
        s.handle(ImapEvent::Command(cmd));
        s.handle(ImapEvent::TlsCompleted);
        assert!(matches!(s.state, ImapState::NotAuthenticated { .. }));
        assert!(s.config.tls_active);
    }

    // ── SELECT ──────────────────────────────────────────────────────

    #[test]
    fn select_requires_authentication() {
        let mut s = connected_session(config_tls());
        let cmd = parse_command("A001 SELECT INBOX").unwrap();
        let actions = s.handle(ImapEvent::Command(cmd));
        assert!(matches!(
            &actions[0],
            ImapAction::SendTagged {
                status: ResponseStatus::Bad,
                ..
            }
        ));
    }

    #[test]
    fn select_sends_mailbox_info() {
        let mut s = authenticated_session();
        let cmd = parse_command("A002 SELECT INBOX").unwrap();
        let actions = s.handle(ImapEvent::Command(cmd));
        assert!(
            matches!(&actions[0], ImapAction::SelectMailbox { name, read_only: false } if name == "INBOX")
        );

        let actions = s.handle(ImapEvent::MailboxLoaded(MailboxSnapshot {
            name: "INBOX".to_string(),
            uid_validity: 100,
            uid_next: 5,
            total_messages: 3,
            recent_count: 1,
            unseen_count: 2,
            first_unseen: Some(2),
            read_only: false,
        }));

        assert!(actions
            .iter()
            .any(|a| matches!(a, ImapAction::SendUntagged(m) if m.contains("EXISTS"))));
        assert!(actions
            .iter()
            .any(|a| matches!(a, ImapAction::SendUntagged(m) if m.contains("RECENT"))));
        assert!(actions
            .iter()
            .any(|a| matches!(a, ImapAction::SendUntagged(m) if m.contains("UIDVALIDITY"))));
        assert!(actions.iter().any(
            |a| matches!(a, ImapAction::SendTagged { code: Some(c), .. } if c == "READ-WRITE")
        ));
    }

    #[test]
    fn examine_opens_read_only() {
        let mut s = authenticated_session();
        let cmd = parse_command("A002 EXAMINE INBOX").unwrap();
        let actions = s.handle(ImapEvent::Command(cmd));
        assert!(matches!(
            &actions[0],
            ImapAction::SelectMailbox {
                read_only: true,
                ..
            }
        ));
    }

    // ── CLOSE / UNSELECT ────────────────────────────────────────────

    #[test]
    fn close_returns_to_authenticated() {
        let mut s = selected_session();
        let cmd = parse_command("A003 CLOSE").unwrap();
        let actions = s.handle(ImapEvent::Command(cmd));
        assert!(matches!(
            &actions[0],
            ImapAction::SendTagged {
                status: ResponseStatus::Ok,
                ..
            }
        ));
        assert!(matches!(s.state, ImapState::Authenticated { .. }));
    }

    #[test]
    fn unselect_returns_to_authenticated() {
        let mut s = selected_session();
        let cmd = parse_command("A003 UNSELECT").unwrap();
        let actions = s.handle(ImapEvent::Command(cmd));
        assert!(matches!(
            &actions[0],
            ImapAction::SendTagged {
                status: ResponseStatus::Ok,
                ..
            }
        ));
        assert!(matches!(s.state, ImapState::Authenticated { .. }));
    }

    #[test]
    fn close_without_select_fails() {
        let mut s = authenticated_session();
        let cmd = parse_command("A003 CLOSE").unwrap();
        let actions = s.handle(ImapEvent::Command(cmd));
        assert!(matches!(
            &actions[0],
            ImapAction::SendTagged {
                status: ResponseStatus::Bad,
                ..
            }
        ));
    }

    // ── FETCH ───────────────────────────────────────────────────────

    #[test]
    fn fetch_requires_selected() {
        let mut s = authenticated_session();
        let cmd = parse_command("A003 FETCH 1 (FLAGS)").unwrap();
        let actions = s.handle(ImapEvent::Command(cmd));
        assert!(matches!(
            &actions[0],
            ImapAction::SendTagged {
                status: ResponseStatus::Bad,
                ..
            }
        ));
    }

    #[test]
    fn fetch_emits_action() {
        let mut s = selected_session();
        let cmd = parse_command("A003 FETCH 1 (FLAGS UID)").unwrap();
        let actions = s.handle(ImapEvent::Command(cmd));
        assert!(matches!(
            &actions[0],
            ImapAction::FetchMessages {
                uid_mode: false,
                ..
            }
        ));
    }

    #[test]
    fn uid_fetch_sets_uid_mode() {
        let mut s = selected_session();
        let cmd = parse_command("A003 UID FETCH 1:* (FLAGS)").unwrap();
        let actions = s.handle(ImapEvent::Command(cmd));
        assert!(matches!(
            &actions[0],
            ImapAction::FetchMessages { uid_mode: true, .. }
        ));
    }

    // ── STORE ───────────────────────────────────────────────────────

    #[test]
    fn store_emits_action() {
        let mut s = selected_session();
        let cmd = parse_command("A003 STORE 1 +FLAGS (\\Seen)").unwrap();
        let actions = s.handle(ImapEvent::Command(cmd));
        assert!(matches!(
            &actions[0],
            ImapAction::StoreFlags { silent: false, .. }
        ));
    }

    #[test]
    fn store_silent_flag() {
        let mut s = selected_session();
        let cmd = parse_command("A003 STORE 1 +FLAGS.SILENT (\\Seen)").unwrap();
        let actions = s.handle(ImapEvent::Command(cmd));
        assert!(matches!(
            &actions[0],
            ImapAction::StoreFlags { silent: true, .. }
        ));
    }

    #[test]
    fn store_on_readonly_rejected() {
        let mut s = authenticated_session();
        let cmd = parse_command("A002 EXAMINE INBOX").unwrap();
        s.handle(ImapEvent::Command(cmd));
        s.handle(ImapEvent::MailboxLoaded(MailboxSnapshot {
            name: "INBOX".to_string(),
            uid_validity: 100,
            uid_next: 5,
            total_messages: 3,
            recent_count: 0,
            unseen_count: 0,
            first_unseen: None,
            read_only: true,
        }));

        let cmd = parse_command("A003 STORE 1 +FLAGS (\\Seen)").unwrap();
        let actions = s.handle(ImapEvent::Command(cmd));
        assert!(matches!(
            &actions[0],
            ImapAction::SendTagged {
                status: ResponseStatus::No,
                ..
            }
        ));
    }

    #[test]
    fn store_complete_sends_fetch_updates() {
        let mut s = selected_session();
        let cmd = parse_command("A003 STORE 1 +FLAGS (\\Seen)").unwrap();
        s.handle(ImapEvent::Command(cmd));

        let actions = s.handle(ImapEvent::StoreComplete {
            updated: vec![(1, vec!["\\Seen".to_string()])],
        });

        assert!(actions
            .iter()
            .any(|a| matches!(a, ImapAction::SendUntagged(m) if m.contains("FETCH"))));
        assert!(actions.iter().any(|a| matches!(
            a,
            ImapAction::SendTagged {
                status: ResponseStatus::Ok,
                ..
            }
        )));
    }

    // ── EXPUNGE ─────────────────────────────────────────────────────

    #[test]
    fn expunge_emits_action() {
        let mut s = selected_session();
        let cmd = parse_command("A003 EXPUNGE").unwrap();
        let actions = s.handle(ImapEvent::Command(cmd));
        assert!(matches!(&actions[0], ImapAction::Expunge));
    }

    #[test]
    fn expunge_complete_sends_notifications() {
        let mut s = selected_session();
        let cmd = parse_command("A003 EXPUNGE").unwrap();
        s.handle(ImapEvent::Command(cmd));

        let actions = s.handle(ImapEvent::ExpungeComplete {
            expunged_seqnums: vec![3, 1],
        });

        let expunge_msgs: Vec<_> = actions
            .iter()
            .filter(|a| matches!(a, ImapAction::SendUntagged(m) if m.contains("EXPUNGE")))
            .collect();
        assert_eq!(expunge_msgs.len(), 2);
    }

    // ── SEARCH ──────────────────────────────────────────────────────

    #[test]
    fn search_emits_action() {
        let mut s = selected_session();
        let cmd = parse_command("A003 SEARCH UNSEEN").unwrap();
        let actions = s.handle(ImapEvent::Command(cmd));
        assert!(matches!(
            &actions[0],
            ImapAction::Search {
                uid_mode: false,
                ..
            }
        ));
    }

    #[test]
    fn search_complete_sends_results() {
        let mut s = selected_session();
        let cmd = parse_command("A003 SEARCH ALL").unwrap();
        s.handle(ImapEvent::Command(cmd));

        let actions = s.handle(ImapEvent::SearchComplete {
            results: vec![1, 3, 5],
        });

        assert!(actions
            .iter()
            .any(|a| matches!(a, ImapAction::SendUntagged(m) if m.contains("SEARCH 1 3 5"))));
    }

    // ── COPY / MOVE ─────────────────────────────────────────────────

    #[test]
    fn copy_emits_action() {
        let mut s = selected_session();
        let cmd = parse_command("A003 COPY 1:3 Sent").unwrap();
        let actions = s.handle(ImapEvent::Command(cmd));
        assert!(matches!(
            &actions[0],
            ImapAction::CopyMessages { is_move: false, .. }
        ));
    }

    #[test]
    fn move_emits_action() {
        let mut s = selected_session();
        let cmd = parse_command("A003 MOVE 1 Trash").unwrap();
        let actions = s.handle(ImapEvent::Command(cmd));
        assert!(matches!(
            &actions[0],
            ImapAction::CopyMessages { is_move: true, .. }
        ));
    }

    // ── IDLE ────────────────────────────────────────────────────────

    #[test]
    fn idle_enters_idling_state() {
        let mut s = selected_session();
        let cmd = parse_command("A003 IDLE").unwrap();
        let actions = s.handle(ImapEvent::Command(cmd));

        assert!(actions
            .iter()
            .any(|a| matches!(a, ImapAction::SendContinuation(_))));
        assert!(actions
            .iter()
            .any(|a| matches!(a, ImapAction::StartIdle { .. })));
        assert!(matches!(s.state, ImapState::Idling { .. }));
    }

    #[test]
    fn idle_done_returns_to_selected() {
        let mut s = selected_session();
        let cmd = parse_command("A003 IDLE").unwrap();
        s.handle(ImapEvent::Command(cmd));

        let actions = s.handle(ImapEvent::IdleDone);
        assert!(actions.iter().any(|a| matches!(a, ImapAction::StopIdle)));
        assert!(actions.iter().any(|a| matches!(
            a,
            ImapAction::SendTagged {
                status: ResponseStatus::Ok,
                ..
            }
        )));
        assert!(matches!(s.state, ImapState::Selected { .. }));
    }

    #[test]
    fn idle_notify_sends_exists() {
        let mut s = selected_session();
        let cmd = parse_command("A003 IDLE").unwrap();
        s.handle(ImapEvent::Command(cmd));

        let actions = s.handle(ImapEvent::IdleNotify { new_exists: 10 });
        assert!(actions
            .iter()
            .any(|a| matches!(a, ImapAction::SendUntagged(m) if m == "10 EXISTS")));
    }

    #[test]
    fn commands_during_idle_rejected() {
        let mut s = selected_session();
        let cmd = parse_command("A003 IDLE").unwrap();
        s.handle(ImapEvent::Command(cmd));

        let cmd = parse_command("A004 NOOP").unwrap();
        let actions = s.handle(ImapEvent::Command(cmd));
        assert!(matches!(
            &actions[0],
            ImapAction::SendTagged {
                status: ResponseStatus::Bad,
                ..
            }
        ));
    }

    // ── LIST ────────────────────────────────────────────────────────

    #[test]
    fn list_emits_action() {
        let mut s = authenticated_session();
        let cmd = parse_command(r#"A002 LIST "" "*""#).unwrap();
        let actions = s.handle(ImapEvent::Command(cmd));
        assert!(matches!(&actions[0], ImapAction::ListMailboxes { .. }));
    }

    // ── CREATE / DELETE / SUBSCRIBE ─────────────────────────────────

    #[test]
    fn create_emits_action() {
        let mut s = authenticated_session();
        let cmd = parse_command("A002 CREATE Archive").unwrap();
        let actions = s.handle(ImapEvent::Command(cmd));
        assert!(matches!(&actions[0], ImapAction::CreateMailbox { ref name } if name == "Archive"));
    }

    #[test]
    fn delete_emits_action() {
        let mut s = authenticated_session();
        let cmd = parse_command("A002 DELETE Archive").unwrap();
        let actions = s.handle(ImapEvent::Command(cmd));
        assert!(matches!(&actions[0], ImapAction::DeleteMailbox { ref name } if name == "Archive"));
    }

    #[test]
    fn subscribe_emits_action() {
        let mut s = authenticated_session();
        let cmd = parse_command("A002 SUBSCRIBE INBOX").unwrap();
        let actions = s.handle(ImapEvent::Command(cmd));
        assert!(matches!(&actions[0], ImapAction::SubscribeMailbox { .. }));
    }

    // ── State validation ────────────────────────────────────────────

    #[test]
    fn select_from_selected_works() {
        let mut s = selected_session();
        let cmd = parse_command("A004 SELECT Sent").unwrap();
        let actions = s.handle(ImapEvent::Command(cmd));
        assert!(
            matches!(&actions[0], ImapAction::SelectMailbox { ref name, .. } if name == "Sent")
        );
    }

    #[test]
    fn authenticated_commands_rejected_before_auth() {
        let mut s = connected_session(config_tls());
        for line in &["A001 SELECT INBOX", "A001 LIST \"\" \"*\"", "A001 CREATE X"] {
            let cmd = parse_command(line).unwrap();
            let actions = s.handle(ImapEvent::Command(cmd));
            assert!(
                matches!(
                    &actions[0],
                    ImapAction::SendTagged {
                        status: ResponseStatus::Bad,
                        ..
                    }
                ),
                "command {line} should be rejected before auth"
            );
        }
    }

    #[test]
    fn selected_commands_rejected_before_select() {
        let mut s = authenticated_session();
        for line in &[
            "A002 FETCH 1 (FLAGS)",
            "A002 STORE 1 FLAGS (\\Seen)",
            "A002 EXPUNGE",
        ] {
            let cmd = parse_command(line).unwrap();
            let actions = s.handle(ImapEvent::Command(cmd));
            assert!(
                matches!(
                    &actions[0],
                    ImapAction::SendTagged {
                        status: ResponseStatus::Bad,
                        ..
                    } | ImapAction::SendTagged {
                        status: ResponseStatus::No,
                        ..
                    }
                ),
                "command {line} should be rejected before SELECT"
            );
        }
    }

    // ── SASL PLAIN decoding ─────────────────────────────────────────

    #[test]
    fn decode_sasl_plain_valid() {
        use base64::Engine;
        let data = base64::engine::general_purpose::STANDARD.encode(b"\0alice\0secret");
        let (user, pass) = decode_sasl_plain(&data).unwrap();
        assert_eq!(user, "alice");
        assert_eq!(pass, "secret");
    }

    #[test]
    fn decode_sasl_plain_with_authzid() {
        use base64::Engine;
        let data = base64::engine::general_purpose::STANDARD.encode(b"admin\0alice\0secret");
        let (user, pass) = decode_sasl_plain(&data).unwrap();
        assert_eq!(user, "alice");
        assert_eq!(pass, "secret");
    }

    #[test]
    fn decode_sasl_plain_invalid() {
        assert!(decode_sasl_plain("not-base64!!!").is_none());
        use base64::Engine;
        let data = base64::engine::general_purpose::STANDARD.encode(b"\0\0secret");
        assert!(decode_sasl_plain(&data).is_none()); // empty username
    }

    // ── Capability changes ──────────────────────────────────────────

    #[test]
    fn capabilities_change_after_auth() {
        let mut s = connected_session(config_tls());
        let pre_auth = s.capability_string();
        assert!(pre_auth.contains("AUTH=PLAIN"));
        assert!(!pre_auth.contains("IDLE"));

        let cmd = parse_command("A001 LOGIN alice pass").unwrap();
        s.handle(ImapEvent::Command(cmd));
        s.handle(ImapEvent::AuthResult {
            success: true,
            username: "alice".to_string(),
        });

        let post_auth = s.capability_string();
        assert!(!post_auth.contains("AUTH=PLAIN"));
        assert!(post_auth.contains("IDLE"));
        assert!(post_auth.contains("MOVE"));
        assert!(post_auth.contains("UIDPLUS"));
    }
}
