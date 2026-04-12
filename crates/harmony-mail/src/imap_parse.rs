//! IMAP command parser.
//!
//! Parses tagged IMAP commands from complete command strings. The codec
//! (`imap_codec.rs`) handles framing, literal assembly, and yields complete
//! command lines to this parser.
//!
//! Grammar (simplified):
//! ```text
//! command     = tag SP command-name [SP arguments] CRLF
//! tag         = 1*astring-char (excluding "+")
//! arguments   = atom / quoted / literal / list / sequence-set
//! ```

// ── Types ───────────────────────────────────────────────────────────

/// A parsed IMAP command with its tag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImapTaggedCommand {
    pub tag: String,
    pub command: ImapCommand,
}

/// IMAP commands supported in v1.1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImapCommand {
    // Any state
    Capability,
    Noop,
    Logout,

    // Not authenticated
    StartTls,
    Login {
        username: String,
        password: String,
    },
    AuthenticatePlain {
        initial_response: Option<String>,
    },

    // Authenticated
    Select {
        mailbox: String,
    },
    Examine {
        mailbox: String,
    },
    List {
        reference: String,
        pattern: String,
    },
    Status {
        mailbox: String,
        items: Vec<StatusItem>,
    },
    Create {
        mailbox: String,
    },
    Delete {
        mailbox: String,
    },
    Subscribe {
        mailbox: String,
    },
    Unsubscribe {
        mailbox: String,
    },

    // Selected
    Close,
    Unselect,
    Expunge,
    Search {
        criteria: Vec<SearchKey>,
        uid: bool,
    },
    Fetch {
        sequence_set: SequenceSet,
        attributes: Vec<FetchAttribute>,
        uid: bool,
    },
    Store {
        sequence_set: SequenceSet,
        operation: StoreOperation,
        flags: Vec<String>,
        uid: bool,
    },
    Copy {
        sequence_set: SequenceSet,
        mailbox: String,
        uid: bool,
    },
    Move {
        sequence_set: SequenceSet,
        mailbox: String,
        uid: bool,
    },
    Idle,
    // Uid prefix wrapper (dispatched to inner command with uid=true)
    // Not a variant — handled during parsing by setting the uid flag.
}

/// FETCH data item attributes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FetchAttribute {
    Envelope,
    Flags,
    InternalDate,
    Rfc822Size,
    Uid,
    Body,
    BodyStructure,
    BodySection {
        section: Option<String>,
        partial: Option<(u32, u32)>,
        peek: bool,
    },
    Rfc822,
    Rfc822Header,
    Rfc822Text,
}

/// STATUS data items.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatusItem {
    Messages,
    Recent,
    UidNext,
    UidValidity,
    Unseen,
}

/// STORE operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreOperation {
    Set,          // FLAGS
    Add,          // +FLAGS
    Remove,       // -FLAGS
    SetSilent,    // FLAGS.SILENT
    AddSilent,    // +FLAGS.SILENT
    RemoveSilent, // -FLAGS.SILENT
}

/// SEARCH key criteria.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SearchKey {
    All,
    Seen,
    Unseen,
    Flagged,
    Unflagged,
    Answered,
    Unanswered,
    Deleted,
    Undeleted,
    Draft,
    Undraft,
    Recent,
    New, // Recent AND Unseen
    Old, // NOT Recent
    From(String),
    To(String),
    Subject(String),
    Body(String),
    Since(String),  // date string
    Before(String), // date string
    On(String),     // date string
    Larger(u32),
    Smaller(u32),
    Uid(SequenceSet),
    SequenceSet(SequenceSet),
    Not(Box<SearchKey>),
    Or(Box<SearchKey>, Box<SearchKey>),
    Header(String, String),
    Keyword(String),
    Unkeyword(String),
}

/// Sequence set: list of sequence numbers or ranges.
/// `*` is represented as `u32::MAX`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequenceSet {
    pub ranges: Vec<SequenceRange>,
}

/// A single element in a sequence set: either a single number or a range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequenceRange {
    pub start: u32,
    pub end: Option<u32>, // None means single number, Some means start:end
}

// ── Parse errors ────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("missing tag")]
    MissingTag,
    #[error("missing command")]
    MissingCommand,
    #[error("unknown command: {0}")]
    UnknownCommand(String),
    #[error("bad arguments: {0}")]
    BadArguments(String),
    #[error("bad sequence set: {0}")]
    BadSequenceSet(String),
}

// ── Main parser ─────────────────────────────────────────────────────

/// Parse a complete IMAP command line into a tagged command.
pub fn parse_command(line: &str) -> Result<ImapTaggedCommand, ParseError> {
    let line = line.trim_end();
    let (tag, rest) = split_first_word(line).ok_or(ParseError::MissingTag)?;

    if tag.is_empty() || tag == "+" {
        return Err(ParseError::MissingTag);
    }

    let tag = tag.to_string();
    let (verb, args) = split_first_word(rest).ok_or(ParseError::MissingCommand)?;
    let verb_upper = verb.to_uppercase();

    // Handle UID prefix
    if verb_upper == "UID" {
        return parse_uid_command(&tag, args);
    }

    let command = parse_command_verb(&verb_upper, args)?;
    Ok(ImapTaggedCommand { tag, command })
}

fn parse_uid_command(tag: &str, rest: &str) -> Result<ImapTaggedCommand, ParseError> {
    let (verb, args) = split_first_word(rest).ok_or(ParseError::MissingCommand)?;
    let verb_upper = verb.to_uppercase();

    let command = match verb_upper.as_str() {
        "FETCH" => {
            let (seq_str, attr_str) = split_first_word(args).ok_or_else(|| {
                ParseError::BadArguments("FETCH requires sequence and items".into())
            })?;
            let sequence_set = parse_sequence_set(seq_str)?;
            let attributes = parse_fetch_attributes(attr_str)?;
            ImapCommand::Fetch {
                sequence_set,
                attributes,
                uid: true,
            }
        }
        "STORE" => {
            let cmd = parse_store_args(args)?;
            match cmd {
                ImapCommand::Store {
                    sequence_set,
                    operation,
                    flags,
                    ..
                } => ImapCommand::Store {
                    sequence_set,
                    operation,
                    flags,
                    uid: true,
                },
                _ => unreachable!(),
            }
        }
        "SEARCH" => {
            let criteria = parse_search_criteria(args)?;
            ImapCommand::Search {
                criteria,
                uid: true,
            }
        }
        "COPY" => {
            let (seq_str, mailbox) = split_first_word(args).ok_or_else(|| {
                ParseError::BadArguments("COPY requires sequence and mailbox".into())
            })?;
            let sequence_set = parse_sequence_set(seq_str)?;
            let mailbox = parse_astring(mailbox);
            ImapCommand::Copy {
                sequence_set,
                mailbox,
                uid: true,
            }
        }
        "MOVE" => {
            let (seq_str, mailbox) = split_first_word(args).ok_or_else(|| {
                ParseError::BadArguments("MOVE requires sequence and mailbox".into())
            })?;
            let sequence_set = parse_sequence_set(seq_str)?;
            let mailbox = parse_astring(mailbox);
            ImapCommand::Move {
                sequence_set,
                mailbox,
                uid: true,
            }
        }
        "EXPUNGE" => {
            // UID EXPUNGE requires a UID set argument (RFC 4315).
            // Plain EXPUNGE (without UID set handling) would delete ALL
            // \Deleted messages, which is not the correct UID EXPUNGE
            // semantics. Reject until proper UID-set filtering is implemented.
            return Err(ParseError::BadArguments(
                "UID EXPUNGE requires a UID set (not yet supported)".into(),
            ));
        }
        _ => return Err(ParseError::UnknownCommand(format!("UID {verb_upper}"))),
    };

    Ok(ImapTaggedCommand {
        tag: tag.to_string(),
        command,
    })
}

fn parse_command_verb(verb: &str, args: &str) -> Result<ImapCommand, ParseError> {
    match verb {
        // Any state
        "CAPABILITY" => Ok(ImapCommand::Capability),
        "NOOP" => Ok(ImapCommand::Noop),
        "LOGOUT" => Ok(ImapCommand::Logout),

        // Not authenticated
        "STARTTLS" => Ok(ImapCommand::StartTls),
        "LOGIN" => parse_login(args),
        "AUTHENTICATE" => parse_authenticate(args),

        // Authenticated
        "SELECT" => Ok(ImapCommand::Select {
            mailbox: parse_astring(args),
        }),
        "EXAMINE" => Ok(ImapCommand::Examine {
            mailbox: parse_astring(args),
        }),
        "LIST" => parse_list(args),
        "STATUS" => parse_status(args),
        "CREATE" => Ok(ImapCommand::Create {
            mailbox: parse_astring(args),
        }),
        "DELETE" => Ok(ImapCommand::Delete {
            mailbox: parse_astring(args),
        }),
        "SUBSCRIBE" => Ok(ImapCommand::Subscribe {
            mailbox: parse_astring(args),
        }),
        "UNSUBSCRIBE" => Ok(ImapCommand::Unsubscribe {
            mailbox: parse_astring(args),
        }),

        // Selected
        "CLOSE" => Ok(ImapCommand::Close),
        "UNSELECT" => Ok(ImapCommand::Unselect),
        "EXPUNGE" => Ok(ImapCommand::Expunge),
        "IDLE" => Ok(ImapCommand::Idle),
        "SEARCH" => {
            let criteria = parse_search_criteria(args)?;
            Ok(ImapCommand::Search {
                criteria,
                uid: false,
            })
        }
        "FETCH" => {
            let (seq_str, attr_str) = split_first_word(args).ok_or_else(|| {
                ParseError::BadArguments("FETCH requires sequence and items".into())
            })?;
            let sequence_set = parse_sequence_set(seq_str)?;
            let attributes = parse_fetch_attributes(attr_str)?;
            Ok(ImapCommand::Fetch {
                sequence_set,
                attributes,
                uid: false,
            })
        }
        "STORE" => parse_store_args(args),
        "COPY" => {
            let (seq_str, mailbox) = split_first_word(args).ok_or_else(|| {
                ParseError::BadArguments("COPY requires sequence and mailbox".into())
            })?;
            let sequence_set = parse_sequence_set(seq_str)?;
            let mailbox = parse_astring(mailbox);
            Ok(ImapCommand::Copy {
                sequence_set,
                mailbox,
                uid: false,
            })
        }
        "MOVE" => {
            let (seq_str, mailbox) = split_first_word(args).ok_or_else(|| {
                ParseError::BadArguments("MOVE requires sequence and mailbox".into())
            })?;
            let sequence_set = parse_sequence_set(seq_str)?;
            let mailbox = parse_astring(mailbox);
            Ok(ImapCommand::Move {
                sequence_set,
                mailbox,
                uid: false,
            })
        }
        _ => Err(ParseError::UnknownCommand(verb.to_string())),
    }
}

// ── Argument parsers ────────────────────────────────────────────────

fn parse_login(args: &str) -> Result<ImapCommand, ParseError> {
    let (username, rest) = parse_next_astring(args)
        .ok_or_else(|| ParseError::BadArguments("LOGIN requires username".into()))?;
    let (password, _) = parse_next_astring(rest)
        .ok_or_else(|| ParseError::BadArguments("LOGIN requires password".into()))?;
    Ok(ImapCommand::Login { username, password })
}

fn parse_authenticate(args: &str) -> Result<ImapCommand, ParseError> {
    let (mechanism, rest) = split_first_word(args)
        .ok_or_else(|| ParseError::BadArguments("AUTHENTICATE requires mechanism".into()))?;
    if mechanism.to_uppercase() != "PLAIN" {
        return Err(ParseError::BadArguments(format!(
            "unsupported mechanism: {mechanism}"
        )));
    }
    let initial_response = if rest.trim().is_empty() {
        None
    } else {
        Some(rest.trim().to_string())
    };
    Ok(ImapCommand::AuthenticatePlain { initial_response })
}

fn parse_list(args: &str) -> Result<ImapCommand, ParseError> {
    let (reference, rest) = parse_next_astring(args)
        .ok_or_else(|| ParseError::BadArguments("LIST requires reference".into()))?;
    let (pattern, _) = parse_next_astring(rest)
        .ok_or_else(|| ParseError::BadArguments("LIST requires pattern".into()))?;
    Ok(ImapCommand::List { reference, pattern })
}

fn parse_status(args: &str) -> Result<ImapCommand, ParseError> {
    let (mailbox, rest) = parse_next_astring(args)
        .ok_or_else(|| ParseError::BadArguments("STATUS requires mailbox".into()))?;

    let rest = rest.trim();
    if !rest.starts_with('(') || !rest.ends_with(')') {
        return Err(ParseError::BadArguments(
            "STATUS items must be parenthesized".into(),
        ));
    }
    let inner = &rest[1..rest.len() - 1];
    let mut items = Vec::new();
    for word in inner.split_whitespace() {
        let item = match word.to_uppercase().as_str() {
            "MESSAGES" => StatusItem::Messages,
            "RECENT" => StatusItem::Recent,
            "UIDNEXT" => StatusItem::UidNext,
            "UIDVALIDITY" => StatusItem::UidValidity,
            "UNSEEN" => StatusItem::Unseen,
            other => {
                return Err(ParseError::BadArguments(format!(
                    "unknown STATUS item: {other}"
                )))
            }
        };
        items.push(item);
    }
    Ok(ImapCommand::Status { mailbox, items })
}

fn parse_store_args(args: &str) -> Result<ImapCommand, ParseError> {
    let (seq_str, rest) = split_first_word(args)
        .ok_or_else(|| ParseError::BadArguments("STORE requires sequence".into()))?;
    let sequence_set = parse_sequence_set(seq_str)?;

    let (op_str, flags_str) = split_first_word(rest)
        .ok_or_else(|| ParseError::BadArguments("STORE requires operation".into()))?;
    let operation = match op_str.to_uppercase().as_str() {
        "FLAGS" => StoreOperation::Set,
        "+FLAGS" => StoreOperation::Add,
        "-FLAGS" => StoreOperation::Remove,
        "FLAGS.SILENT" => StoreOperation::SetSilent,
        "+FLAGS.SILENT" => StoreOperation::AddSilent,
        "-FLAGS.SILENT" => StoreOperation::RemoveSilent,
        other => {
            return Err(ParseError::BadArguments(format!(
                "unknown STORE op: {other}"
            )))
        }
    };

    let flags = parse_flag_list(flags_str)?;
    Ok(ImapCommand::Store {
        sequence_set,
        operation,
        flags,
        uid: false,
    })
}

// ── Sequence set parsing ────────────────────────────────────────────

/// Parse an IMAP sequence set string (e.g., "1:*", "1,3,5:7", "*").
pub fn parse_sequence_set(s: &str) -> Result<SequenceSet, ParseError> {
    let s = s.trim();
    if s.is_empty() {
        return Err(ParseError::BadSequenceSet("empty".into()));
    }

    let mut ranges = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            return Err(ParseError::BadSequenceSet("empty element".into()));
        }
        if let Some((start_str, end_str)) = part.split_once(':') {
            let start = parse_seq_number(start_str)?;
            let end = parse_seq_number(end_str)?;
            ranges.push(SequenceRange {
                start,
                end: Some(end),
            });
        } else {
            let num = parse_seq_number(part)?;
            ranges.push(SequenceRange {
                start: num,
                end: None,
            });
        }
    }

    Ok(SequenceSet { ranges })
}

fn parse_seq_number(s: &str) -> Result<u32, ParseError> {
    let s = s.trim();
    if s == "*" {
        Ok(u32::MAX)
    } else {
        s.parse::<u32>()
            .map_err(|_| ParseError::BadSequenceSet(format!("invalid number: {s}")))
    }
}

// ── FETCH attribute parsing ─────────────────────────────────────────

/// Parse FETCH attributes, expanding macros.
pub fn parse_fetch_attributes(s: &str) -> Result<Vec<FetchAttribute>, ParseError> {
    let s = s.trim();

    // Handle macros (atoms, not parenthesized)
    match s.to_uppercase().as_str() {
        "ALL" => {
            return Ok(vec![
                FetchAttribute::Flags,
                FetchAttribute::InternalDate,
                FetchAttribute::Rfc822Size,
                FetchAttribute::Envelope,
            ])
        }
        "FAST" => {
            return Ok(vec![
                FetchAttribute::Flags,
                FetchAttribute::InternalDate,
                FetchAttribute::Rfc822Size,
            ])
        }
        "FULL" => {
            return Ok(vec![
                FetchAttribute::Flags,
                FetchAttribute::InternalDate,
                FetchAttribute::Rfc822Size,
                FetchAttribute::Envelope,
                FetchAttribute::Body,
            ])
        }
        _ => {}
    }

    // Parenthesized list
    let inner = if s.starts_with('(') && s.ends_with(')') {
        &s[1..s.len() - 1]
    } else {
        s
    };

    let mut attrs = Vec::new();
    let mut chars = inner.chars().peekable();
    let mut current = String::new();

    while chars.peek().is_some() {
        current.clear();
        // Skip whitespace
        while chars.peek() == Some(&' ') {
            chars.next();
        }
        if chars.peek().is_none() {
            break;
        }

        // Collect token (may include brackets for BODY[...])
        while let Some(&ch) = chars.peek() {
            if ch == ' ' && !current.contains('[') {
                break;
            }
            if ch == ']' {
                current.push(ch);
                chars.next();
                // Check for <partial> after ]
                if chars.peek() == Some(&'<') {
                    while let Some(&pch) = chars.peek() {
                        current.push(pch);
                        chars.next();
                        if pch == '>' {
                            break;
                        }
                    }
                }
                break;
            }
            current.push(ch);
            chars.next();
        }

        if current.is_empty() {
            continue;
        }

        let attr = parse_single_fetch_attr(&current)?;
        attrs.push(attr);
    }

    if attrs.is_empty() {
        return Err(ParseError::BadArguments("empty FETCH attributes".into()));
    }

    Ok(attrs)
}

fn parse_single_fetch_attr(s: &str) -> Result<FetchAttribute, ParseError> {
    let upper = s.to_uppercase();

    match upper.as_str() {
        "ENVELOPE" => Ok(FetchAttribute::Envelope),
        "FLAGS" => Ok(FetchAttribute::Flags),
        "INTERNALDATE" => Ok(FetchAttribute::InternalDate),
        "RFC822.SIZE" => Ok(FetchAttribute::Rfc822Size),
        "UID" => Ok(FetchAttribute::Uid),
        "BODY" => Ok(FetchAttribute::Body),
        "BODYSTRUCTURE" => Ok(FetchAttribute::BodyStructure),
        "RFC822" => Ok(FetchAttribute::Rfc822),
        "RFC822.HEADER" => Ok(FetchAttribute::Rfc822Header),
        "RFC822.TEXT" => Ok(FetchAttribute::Rfc822Text),
        _ => {
            // BODY[section]<partial> or BODY.PEEK[section]<partial>
            if upper.starts_with("BODY.PEEK[") || upper.starts_with("BODY[") {
                let peek = upper.starts_with("BODY.PEEK[");
                let bracket_start = if peek { 10 } else { 5 };
                // Find the matching ]
                let rest = &s[bracket_start..];
                let bracket_end = rest
                    .find(']')
                    .ok_or_else(|| ParseError::BadArguments(format!("unclosed bracket in: {s}")))?;
                let section = &rest[..bracket_end];
                let section = if section.is_empty() {
                    None
                } else {
                    Some(section.to_string())
                };

                let after_bracket = &rest[bracket_end + 1..];
                let partial = if after_bracket.starts_with('<') && after_bracket.ends_with('>') {
                    let inner = &after_bracket[1..after_bracket.len() - 1];
                    let (offset, count) = inner.split_once('.').ok_or_else(|| {
                        ParseError::BadArguments(format!("bad partial: {after_bracket}"))
                    })?;
                    let offset: u32 = offset.parse().map_err(|_| {
                        ParseError::BadArguments(format!("bad partial offset: {offset}"))
                    })?;
                    let count: u32 = count.parse().map_err(|_| {
                        ParseError::BadArguments(format!("bad partial count: {count}"))
                    })?;
                    Some((offset, count))
                } else {
                    None
                };

                Ok(FetchAttribute::BodySection {
                    section,
                    partial,
                    peek,
                })
            } else {
                Err(ParseError::BadArguments(format!("unknown FETCH attr: {s}")))
            }
        }
    }
}

// ── SEARCH criteria parsing ─────────────────────────────────────────

fn parse_search_criteria(args: &str) -> Result<Vec<SearchKey>, ParseError> {
    let args = args.trim();
    if args.is_empty() {
        return Ok(vec![SearchKey::All]);
    }

    let tokens = tokenize_search(args);
    let mut iter = tokens.iter().map(|s| s.as_str()).peekable();
    let mut keys = Vec::new();

    while iter.peek().is_some() {
        let key = parse_one_search_key(&mut iter)?;
        keys.push(key);
    }

    if keys.is_empty() {
        keys.push(SearchKey::All);
    }
    Ok(keys)
}

fn parse_one_search_key<'a>(
    iter: &mut std::iter::Peekable<impl Iterator<Item = &'a str>>,
) -> Result<SearchKey, ParseError> {
    let token = iter
        .next()
        .ok_or_else(|| ParseError::BadArguments("unexpected end of SEARCH criteria".into()))?;

    match token.to_uppercase().as_str() {
        "ALL" => Ok(SearchKey::All),
        "SEEN" => Ok(SearchKey::Seen),
        "UNSEEN" => Ok(SearchKey::Unseen),
        "FLAGGED" => Ok(SearchKey::Flagged),
        "UNFLAGGED" => Ok(SearchKey::Unflagged),
        "ANSWERED" => Ok(SearchKey::Answered),
        "UNANSWERED" => Ok(SearchKey::Unanswered),
        "DELETED" => Ok(SearchKey::Deleted),
        "UNDELETED" => Ok(SearchKey::Undeleted),
        "DRAFT" => Ok(SearchKey::Draft),
        "UNDRAFT" => Ok(SearchKey::Undraft),
        "RECENT" => Ok(SearchKey::Recent),
        "NEW" => Ok(SearchKey::New),
        "OLD" => Ok(SearchKey::Old),
        "FROM" => {
            let val = next_string_arg(iter, "FROM")?;
            Ok(SearchKey::From(val))
        }
        "TO" => {
            let val = next_string_arg(iter, "TO")?;
            Ok(SearchKey::To(val))
        }
        "SUBJECT" => {
            let val = next_string_arg(iter, "SUBJECT")?;
            Ok(SearchKey::Subject(val))
        }
        "BODY" => {
            let val = next_string_arg(iter, "BODY")?;
            Ok(SearchKey::Body(val))
        }
        "SINCE" => {
            let val = next_string_arg(iter, "SINCE")?;
            Ok(SearchKey::Since(val))
        }
        "BEFORE" => {
            let val = next_string_arg(iter, "BEFORE")?;
            Ok(SearchKey::Before(val))
        }
        "ON" => {
            let val = next_string_arg(iter, "ON")?;
            Ok(SearchKey::On(val))
        }
        "LARGER" => {
            let val = next_u32_arg(iter, "LARGER")?;
            Ok(SearchKey::Larger(val))
        }
        "SMALLER" => {
            let val = next_u32_arg(iter, "SMALLER")?;
            Ok(SearchKey::Smaller(val))
        }
        "UID" => {
            let seq_str = iter
                .next()
                .ok_or_else(|| ParseError::BadArguments("UID requires sequence set".into()))?;
            let seq = parse_sequence_set(seq_str)?;
            Ok(SearchKey::Uid(seq))
        }
        "NOT" => {
            let inner = parse_one_search_key(iter)?;
            Ok(SearchKey::Not(Box::new(inner)))
        }
        "OR" => {
            let a = parse_one_search_key(iter)?;
            let b = parse_one_search_key(iter)?;
            Ok(SearchKey::Or(Box::new(a), Box::new(b)))
        }
        "HEADER" => {
            let field = next_string_arg(iter, "HEADER field")?;
            let value = next_string_arg(iter, "HEADER value")?;
            Ok(SearchKey::Header(field, value))
        }
        "KEYWORD" => {
            let val = next_string_arg(iter, "KEYWORD")?;
            Ok(SearchKey::Keyword(val))
        }
        "UNKEYWORD" => {
            let val = next_string_arg(iter, "UNKEYWORD")?;
            Ok(SearchKey::Unkeyword(val))
        }
        other => {
            // Try as sequence set
            if other
                .chars()
                .all(|c: char| c.is_ascii_digit() || c == ':' || c == ',' || c == '*')
            {
                let seq = parse_sequence_set(other)?;
                Ok(SearchKey::SequenceSet(seq))
            } else {
                Err(ParseError::BadArguments(format!(
                    "unknown SEARCH key: {other}"
                )))
            }
        }
    }
}

fn next_string_arg<'a>(
    iter: &mut impl Iterator<Item = &'a str>,
    context: &str,
) -> Result<String, ParseError> {
    let val = iter
        .next()
        .ok_or_else(|| ParseError::BadArguments(format!("{context} requires an argument")))?;
    Ok(unquote(val))
}

fn next_u32_arg<'a>(
    iter: &mut impl Iterator<Item = &'a str>,
    context: &str,
) -> Result<u32, ParseError> {
    let val = iter
        .next()
        .ok_or_else(|| ParseError::BadArguments(format!("{context} requires a number")))?;
    val.parse::<u32>()
        .map_err(|_| ParseError::BadArguments(format!("{context}: not a number: {val}")))
}

/// Tokenize a search string, respecting quoted strings.
fn tokenize_search(s: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut chars = s.chars().peekable();

    while chars.peek().is_some() {
        // Skip whitespace
        while chars.peek() == Some(&' ') {
            chars.next();
        }
        if chars.peek().is_none() {
            break;
        }

        if chars.peek() == Some(&'"') {
            // Quoted string
            chars.next(); // consume opening quote
            let mut token = String::from("\"");
            while let Some(&ch) = chars.peek() {
                chars.next();
                if ch == '"' {
                    token.push('"');
                    break;
                }
                if ch == '\\' {
                    if let Some(&escaped) = chars.peek() {
                        chars.next();
                        token.push(escaped);
                    }
                } else {
                    token.push(ch);
                }
            }
            tokens.push(token);
        } else {
            // Atom
            let mut token = String::new();
            while let Some(&ch) = chars.peek() {
                if ch == ' ' {
                    break;
                }
                token.push(ch);
                chars.next();
            }
            tokens.push(token);
        }
    }

    tokens
}

// ── Flag list parsing ───────────────────────────────────────────────

fn parse_flag_list(s: &str) -> Result<Vec<String>, ParseError> {
    let s = s.trim();
    let inner = if s.starts_with('(') && s.ends_with(')') {
        &s[1..s.len() - 1]
    } else {
        s
    };

    let mut flags = Vec::new();
    for word in inner.split_whitespace() {
        flags.push(word.to_string());
    }
    Ok(flags)
}

// ── String parsing helpers ──────────────────────────────────────────

/// Parse an IMAP astring (atom or quoted string).
/// Strips quotes if present.
fn parse_astring(s: &str) -> String {
    unquote(s.trim())
}

/// Parse the next astring from the input, returning (value, remaining).
/// Handles quoted strings and atoms.
fn parse_next_astring(s: &str) -> Option<(String, &str)> {
    let s = s.trim_start();
    if s.is_empty() {
        return None;
    }

    if s.starts_with('"') {
        // Quoted string: find matching close quote (handle escapes)
        let mut end = 1;
        let bytes = s.as_bytes();
        while end < bytes.len() {
            if bytes[end] == b'\\' {
                end += 2; // skip escaped char
            } else if bytes[end] == b'"' {
                let value = unquote(&s[..end + 1]);
                let rest = &s[end + 1..];
                return Some((value, rest.trim_start()));
            } else {
                end += 1;
            }
        }
        // Unterminated quote — treat rest as value
        Some((unquote(s), ""))
    } else {
        // Atom: next whitespace-delimited token
        match s.find(' ') {
            Some(pos) => {
                let value = s[..pos].to_string();
                Some((value, &s[pos + 1..]))
            }
            None => Some((s.to_string(), "")),
        }
    }
}

/// Remove quotes from a quoted string, handling escape sequences.
fn unquote(s: &str) -> String {
    let s = s.trim();
    if s.len() >= 2 && s.starts_with('"') && s.ends_with('"') {
        let inner = &s[1..s.len() - 1];
        let mut result = String::with_capacity(inner.len());
        let mut escape = false;
        for ch in inner.chars() {
            if escape {
                result.push(ch);
                escape = false;
            } else if ch == '\\' {
                escape = true;
            } else {
                result.push(ch);
            }
        }
        result
    } else {
        s.to_string()
    }
}

/// Split string at first whitespace, returning (first_word, rest).
fn split_first_word(s: &str) -> Option<(&str, &str)> {
    let s = s.trim_start();
    if s.is_empty() {
        return None;
    }
    match s.find(' ') {
        Some(pos) => Some((&s[..pos], &s[pos + 1..])),
        None => Some((s, "")),
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper
    fn parse(line: &str) -> ImapTaggedCommand {
        parse_command(line).unwrap()
    }

    fn parse_err(line: &str) -> ParseError {
        parse_command(line).unwrap_err()
    }

    // ── Basic commands ──────────────────────────────────────────────

    #[test]
    fn parse_capability() {
        let cmd = parse("A001 CAPABILITY");
        assert_eq!(cmd.tag, "A001");
        assert_eq!(cmd.command, ImapCommand::Capability);
    }

    #[test]
    fn parse_noop() {
        let cmd = parse("tag1 NOOP");
        assert_eq!(cmd.command, ImapCommand::Noop);
    }

    #[test]
    fn parse_logout() {
        let cmd = parse("A003 LOGOUT");
        assert_eq!(cmd.command, ImapCommand::Logout);
    }

    #[test]
    fn parse_starttls() {
        let cmd = parse("A004 STARTTLS");
        assert_eq!(cmd.command, ImapCommand::StartTls);
    }

    // ── Login ───────────────────────────────────────────────────────

    #[test]
    fn parse_login_plain() {
        let cmd = parse("A001 LOGIN alice s3cret");
        assert_eq!(
            cmd.command,
            ImapCommand::Login {
                username: "alice".to_string(),
                password: "s3cret".to_string(),
            }
        );
    }

    #[test]
    fn parse_login_quoted() {
        let cmd = parse(r#"A001 LOGIN "user name" "pass word""#);
        assert_eq!(
            cmd.command,
            ImapCommand::Login {
                username: "user name".to_string(),
                password: "pass word".to_string(),
            }
        );
    }

    #[test]
    fn parse_login_escaped_quotes() {
        let cmd = parse(r#"A001 LOGIN "us\"er" "pa\\ss""#);
        assert_eq!(
            cmd.command,
            ImapCommand::Login {
                username: "us\"er".to_string(),
                password: "pa\\ss".to_string(),
            }
        );
    }

    // ── Authenticate ────────────────────────────────────────────────

    #[test]
    fn parse_authenticate_plain_no_initial() {
        let cmd = parse("A001 AUTHENTICATE PLAIN");
        assert_eq!(
            cmd.command,
            ImapCommand::AuthenticatePlain {
                initial_response: None
            }
        );
    }

    #[test]
    fn parse_authenticate_plain_with_initial() {
        let cmd = parse("A001 AUTHENTICATE PLAIN dGVzdAB0ZXN0AHRlc3Q=");
        assert_eq!(
            cmd.command,
            ImapCommand::AuthenticatePlain {
                initial_response: Some("dGVzdAB0ZXN0AHRlc3Q=".to_string()),
            }
        );
    }

    // ── SELECT / EXAMINE ────────────────────────────────────────────

    #[test]
    fn parse_select() {
        let cmd = parse("A002 SELECT INBOX");
        assert_eq!(
            cmd.command,
            ImapCommand::Select {
                mailbox: "INBOX".to_string()
            }
        );
    }

    #[test]
    fn parse_select_quoted() {
        let cmd = parse(r#"A002 SELECT "My Mailbox""#);
        assert_eq!(
            cmd.command,
            ImapCommand::Select {
                mailbox: "My Mailbox".to_string()
            }
        );
    }

    #[test]
    fn parse_examine() {
        let cmd = parse("A002 EXAMINE INBOX");
        assert_eq!(
            cmd.command,
            ImapCommand::Examine {
                mailbox: "INBOX".to_string()
            }
        );
    }

    // ── LIST ────────────────────────────────────────────────────────

    #[test]
    fn parse_list() {
        let cmd = parse(r#"A003 LIST "" "*""#);
        assert_eq!(
            cmd.command,
            ImapCommand::List {
                reference: String::new(),
                pattern: "*".to_string(),
            }
        );
    }

    #[test]
    fn parse_list_inbox() {
        let cmd = parse(r#"A003 LIST "" INBOX"#);
        assert_eq!(
            cmd.command,
            ImapCommand::List {
                reference: String::new(),
                pattern: "INBOX".to_string(),
            }
        );
    }

    // ── STATUS ──────────────────────────────────────────────────────

    #[test]
    fn parse_status() {
        let cmd = parse("A004 STATUS INBOX (MESSAGES UNSEEN UIDNEXT)");
        assert_eq!(
            cmd.command,
            ImapCommand::Status {
                mailbox: "INBOX".to_string(),
                items: vec![
                    StatusItem::Messages,
                    StatusItem::Unseen,
                    StatusItem::UidNext
                ],
            }
        );
    }

    // ── CREATE / DELETE ─────────────────────────────────────────────

    #[test]
    fn parse_create() {
        let cmd = parse("A005 CREATE Archive");
        assert_eq!(
            cmd.command,
            ImapCommand::Create {
                mailbox: "Archive".to_string()
            }
        );
    }

    #[test]
    fn parse_delete() {
        let cmd = parse("A006 DELETE Archive");
        assert_eq!(
            cmd.command,
            ImapCommand::Delete {
                mailbox: "Archive".to_string()
            }
        );
    }

    // ── CLOSE / UNSELECT / EXPUNGE / IDLE ───────────────────────────

    #[test]
    fn parse_close() {
        let cmd = parse("A007 CLOSE");
        assert_eq!(cmd.command, ImapCommand::Close);
    }

    #[test]
    fn parse_unselect() {
        let cmd = parse("A008 UNSELECT");
        assert_eq!(cmd.command, ImapCommand::Unselect);
    }

    #[test]
    fn parse_expunge() {
        let cmd = parse("A009 EXPUNGE");
        assert_eq!(cmd.command, ImapCommand::Expunge);
    }

    #[test]
    fn parse_idle() {
        let cmd = parse("A010 IDLE");
        assert_eq!(cmd.command, ImapCommand::Idle);
    }

    // ── Sequence sets ───────────────────────────────────────────────

    #[test]
    fn parse_seqset_single() {
        let seq = parse_sequence_set("5").unwrap();
        assert_eq!(
            seq.ranges,
            vec![SequenceRange {
                start: 5,
                end: None
            }]
        );
    }

    #[test]
    fn parse_seqset_range() {
        let seq = parse_sequence_set("1:10").unwrap();
        assert_eq!(
            seq.ranges,
            vec![SequenceRange {
                start: 1,
                end: Some(10)
            }]
        );
    }

    #[test]
    fn parse_seqset_star() {
        let seq = parse_sequence_set("*").unwrap();
        assert_eq!(
            seq.ranges,
            vec![SequenceRange {
                start: u32::MAX,
                end: None
            }]
        );
    }

    #[test]
    fn parse_seqset_range_to_star() {
        let seq = parse_sequence_set("1:*").unwrap();
        assert_eq!(
            seq.ranges,
            vec![SequenceRange {
                start: 1,
                end: Some(u32::MAX)
            }]
        );
    }

    #[test]
    fn parse_seqset_complex() {
        let seq = parse_sequence_set("1,3,5:7,10:*").unwrap();
        assert_eq!(seq.ranges.len(), 4);
        assert_eq!(
            seq.ranges[0],
            SequenceRange {
                start: 1,
                end: None
            }
        );
        assert_eq!(
            seq.ranges[1],
            SequenceRange {
                start: 3,
                end: None
            }
        );
        assert_eq!(
            seq.ranges[2],
            SequenceRange {
                start: 5,
                end: Some(7)
            }
        );
        assert_eq!(
            seq.ranges[3],
            SequenceRange {
                start: 10,
                end: Some(u32::MAX)
            }
        );
    }

    // ── FETCH ───────────────────────────────────────────────────────

    #[test]
    fn parse_fetch_all_macro() {
        let cmd = parse("A001 FETCH 1:* ALL");
        match &cmd.command {
            ImapCommand::Fetch {
                attributes, uid, ..
            } => {
                assert!(!uid);
                assert_eq!(attributes.len(), 4);
                assert!(attributes.contains(&FetchAttribute::Flags));
                assert!(attributes.contains(&FetchAttribute::Envelope));
            }
            _ => panic!("expected Fetch"),
        }
    }

    #[test]
    fn parse_fetch_parenthesized() {
        let cmd = parse("A001 FETCH 1 (FLAGS UID ENVELOPE)");
        match &cmd.command {
            ImapCommand::Fetch { attributes, .. } => {
                assert_eq!(attributes.len(), 3);
                assert!(attributes.contains(&FetchAttribute::Flags));
                assert!(attributes.contains(&FetchAttribute::Uid));
                assert!(attributes.contains(&FetchAttribute::Envelope));
            }
            _ => panic!("expected Fetch"),
        }
    }

    #[test]
    fn parse_fetch_body_section() {
        let cmd = parse("A001 FETCH 1 BODY[HEADER]");
        match &cmd.command {
            ImapCommand::Fetch { attributes, .. } => {
                assert_eq!(attributes.len(), 1);
                assert_eq!(
                    attributes[0],
                    FetchAttribute::BodySection {
                        section: Some("HEADER".to_string()),
                        partial: None,
                        peek: false,
                    }
                );
            }
            _ => panic!("expected Fetch"),
        }
    }

    #[test]
    fn parse_fetch_body_peek_empty_section() {
        let cmd = parse("A001 FETCH 1 BODY.PEEK[]");
        match &cmd.command {
            ImapCommand::Fetch { attributes, .. } => {
                assert_eq!(
                    attributes[0],
                    FetchAttribute::BodySection {
                        section: None,
                        partial: None,
                        peek: true,
                    }
                );
            }
            _ => panic!("expected Fetch"),
        }
    }

    #[test]
    fn parse_fetch_body_partial() {
        let cmd = parse("A001 FETCH 1 BODY[]<0.1024>");
        match &cmd.command {
            ImapCommand::Fetch { attributes, .. } => {
                assert_eq!(
                    attributes[0],
                    FetchAttribute::BodySection {
                        section: None,
                        partial: Some((0, 1024)),
                        peek: false,
                    }
                );
            }
            _ => panic!("expected Fetch"),
        }
    }

    // ── UID prefix ──────────────────────────────────────────────────

    #[test]
    fn parse_uid_fetch() {
        let cmd = parse("A001 UID FETCH 1:* (FLAGS UID)");
        match &cmd.command {
            ImapCommand::Fetch { uid, .. } => assert!(uid),
            _ => panic!("expected Fetch"),
        }
    }

    #[test]
    fn parse_uid_store() {
        let cmd = parse("A001 UID STORE 1 +FLAGS (\\Seen)");
        match &cmd.command {
            ImapCommand::Store {
                uid,
                operation,
                flags,
                ..
            } => {
                assert!(uid);
                assert_eq!(*operation, StoreOperation::Add);
                assert_eq!(flags, &["\\Seen"]);
            }
            _ => panic!("expected Store"),
        }
    }

    #[test]
    fn parse_uid_search() {
        let cmd = parse("A001 UID SEARCH UNSEEN");
        match &cmd.command {
            ImapCommand::Search { uid, criteria } => {
                assert!(uid);
                assert_eq!(criteria, &[SearchKey::Unseen]);
            }
            _ => panic!("expected Search"),
        }
    }

    #[test]
    fn parse_uid_copy() {
        let cmd = parse("A001 UID COPY 1:3 Sent");
        match &cmd.command {
            ImapCommand::Copy { uid, mailbox, .. } => {
                assert!(uid);
                assert_eq!(mailbox, "Sent");
            }
            _ => panic!("expected Copy"),
        }
    }

    // ── STORE ───────────────────────────────────────────────────────

    #[test]
    fn parse_store_set_flags() {
        let cmd = parse("A001 STORE 1 FLAGS (\\Seen \\Flagged)");
        match &cmd.command {
            ImapCommand::Store {
                operation, flags, ..
            } => {
                assert_eq!(*operation, StoreOperation::Set);
                assert_eq!(flags, &["\\Seen", "\\Flagged"]);
            }
            _ => panic!("expected Store"),
        }
    }

    #[test]
    fn parse_store_remove_flags() {
        let cmd = parse("A001 STORE 1:5 -FLAGS (\\Deleted)");
        match &cmd.command {
            ImapCommand::Store {
                operation, flags, ..
            } => {
                assert_eq!(*operation, StoreOperation::Remove);
                assert_eq!(flags, &["\\Deleted"]);
            }
            _ => panic!("expected Store"),
        }
    }

    #[test]
    fn parse_store_silent() {
        let cmd = parse("A001 STORE 1 +FLAGS.SILENT (\\Seen)");
        match &cmd.command {
            ImapCommand::Store { operation, .. } => {
                assert_eq!(*operation, StoreOperation::AddSilent);
            }
            _ => panic!("expected Store"),
        }
    }

    // ── SEARCH ──────────────────────────────────────────────────────

    #[test]
    fn parse_search_unseen() {
        let cmd = parse("A001 SEARCH UNSEEN");
        match &cmd.command {
            ImapCommand::Search { criteria, uid } => {
                assert!(!uid);
                assert_eq!(criteria, &[SearchKey::Unseen]);
            }
            _ => panic!("expected Search"),
        }
    }

    #[test]
    fn parse_search_from() {
        let cmd = parse(r#"A001 SEARCH FROM "alice@example.com""#);
        match &cmd.command {
            ImapCommand::Search { criteria, .. } => {
                assert_eq!(
                    criteria,
                    &[SearchKey::From("alice@example.com".to_string())]
                );
            }
            _ => panic!("expected Search"),
        }
    }

    #[test]
    fn parse_search_not() {
        let cmd = parse("A001 SEARCH NOT SEEN");
        match &cmd.command {
            ImapCommand::Search { criteria, .. } => {
                assert_eq!(criteria, &[SearchKey::Not(Box::new(SearchKey::Seen))]);
            }
            _ => panic!("expected Search"),
        }
    }

    #[test]
    fn parse_search_or() {
        let cmd = parse("A001 SEARCH OR SEEN FLAGGED");
        match &cmd.command {
            ImapCommand::Search { criteria, .. } => {
                assert_eq!(
                    criteria,
                    &[SearchKey::Or(
                        Box::new(SearchKey::Seen),
                        Box::new(SearchKey::Flagged),
                    )]
                );
            }
            _ => panic!("expected Search"),
        }
    }

    // ── COPY / MOVE ─────────────────────────────────────────────────

    #[test]
    fn parse_copy() {
        let cmd = parse("A001 COPY 1:3 Trash");
        match &cmd.command {
            ImapCommand::Copy { mailbox, uid, .. } => {
                assert!(!uid);
                assert_eq!(mailbox, "Trash");
            }
            _ => panic!("expected Copy"),
        }
    }

    #[test]
    fn parse_move() {
        let cmd = parse(r#"A001 MOVE 5 "Archive""#);
        match &cmd.command {
            ImapCommand::Move { mailbox, uid, .. } => {
                assert!(!uid);
                assert_eq!(mailbox, "Archive");
            }
            _ => panic!("expected Move"),
        }
    }

    // ── SUBSCRIBE / UNSUBSCRIBE ─────────────────────────────────────

    #[test]
    fn parse_subscribe() {
        let cmd = parse("A001 SUBSCRIBE INBOX");
        assert_eq!(
            cmd.command,
            ImapCommand::Subscribe {
                mailbox: "INBOX".to_string()
            }
        );
    }

    #[test]
    fn parse_unsubscribe() {
        let cmd = parse("A001 UNSUBSCRIBE Junk");
        assert_eq!(
            cmd.command,
            ImapCommand::Unsubscribe {
                mailbox: "Junk".to_string()
            }
        );
    }

    // ── Error cases ─────────────────────────────────────────────────

    #[test]
    fn parse_error_missing_tag() {
        assert!(matches!(parse_err(""), ParseError::MissingTag));
    }

    #[test]
    fn parse_error_missing_command() {
        assert!(matches!(parse_err("A001"), ParseError::MissingCommand));
    }

    #[test]
    fn parse_error_unknown_command() {
        assert!(matches!(
            parse_err("A001 FOOBAR"),
            ParseError::UnknownCommand(_)
        ));
    }

    #[test]
    fn parse_error_bad_sequence() {
        assert!(matches!(
            parse_err("A001 FETCH abc (FLAGS)"),
            ParseError::BadSequenceSet(_)
        ));
    }

    // ── Case insensitivity ──────────────────────────────────────────

    #[test]
    fn commands_are_case_insensitive() {
        let cmd = parse("a001 capability");
        assert_eq!(cmd.command, ImapCommand::Capability);

        let cmd = parse("A001 select INBOX");
        assert_eq!(
            cmd.command,
            ImapCommand::Select {
                mailbox: "INBOX".to_string()
            }
        );
    }
}
