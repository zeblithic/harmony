//! SMTP command parser.
//!
//! Parses raw SMTP command lines into [`SmtpCommand`] variants.

use crate::error::MailError;
use crate::smtp::SmtpCommand;

/// Parse a raw SMTP command line into an [`SmtpCommand`].
///
/// The input may or may not end with `\r\n`; trailing CRLF is stripped.
/// Command verbs are case-insensitive. Both `EHLO` and `HELO` produce
/// [`SmtpCommand::Ehlo`].
pub fn parse_command(line: &[u8]) -> Result<SmtpCommand, MailError> {
    // Strip trailing \r\n if present.
    let line = match line {
        [head @ .., b'\r', b'\n'] => head,
        other => other,
    };

    let text = std::str::from_utf8(line)
        .map_err(|_| MailError::InvalidUtf8 { field: "command" })?;

    let text = text.trim();
    if text.is_empty() {
        return Err(MailError::UnknownCommand(String::new()));
    }

    // Split into verb and the rest.
    let (verb, rest) = match text.split_once(' ') {
        Some((v, r)) => (v, Some(r)),
        None => (text, None),
    };

    let verb_upper = verb.to_ascii_uppercase();

    match verb_upper.as_str() {
        "EHLO" | "HELO" => {
            let domain = rest
                .map(|s| s.trim().to_string())
                .unwrap_or_default();
            Ok(SmtpCommand::Ehlo { domain })
        }
        "MAIL" => {
            // Expect "FROM:<address> [params]"
            let rest = rest.ok_or_else(|| {
                MailError::UnknownCommand("MAIL".to_string())
            })?;
            let address = parse_mail_from_arg(rest)?;
            Ok(SmtpCommand::MailFrom { address })
        }
        "RCPT" => {
            // Expect "TO:<address>"
            let rest = rest.ok_or_else(|| {
                MailError::UnknownCommand("RCPT".to_string())
            })?;
            let address = parse_rcpt_to_arg(rest)?;
            Ok(SmtpCommand::RcptTo { address })
        }
        "DATA" => Ok(SmtpCommand::Data),
        "QUIT" => Ok(SmtpCommand::Quit),
        "RSET" => Ok(SmtpCommand::Rset),
        "NOOP" => Ok(SmtpCommand::Noop),
        "STARTTLS" => Ok(SmtpCommand::StartTls),
        _ => Err(MailError::UnknownCommand(verb_upper)),
    }
}

/// Parse the argument to `MAIL FROM:`, extracting the address.
///
/// Handles both `FROM:<addr>` and `FROM:addr` forms, and ignores
/// ESMTP parameters after the closing bracket.
fn parse_mail_from_arg(arg: &str) -> Result<String, MailError> {
    let arg = arg.trim();
    // Strip the "FROM:" prefix (case-insensitive).
    let after_from = if arg.len() >= 5 && arg[..5].eq_ignore_ascii_case("FROM:") {
        &arg[5..]
    } else {
        return Err(MailError::UnknownCommand("MAIL".to_string()));
    };

    Ok(extract_address(after_from))
}

/// Parse the argument to `RCPT TO:`, extracting the address.
fn parse_rcpt_to_arg(arg: &str) -> Result<String, MailError> {
    let arg = arg.trim();
    // Strip the "TO:" prefix (case-insensitive).
    let after_to = if arg.len() >= 3 && arg[..3].eq_ignore_ascii_case("TO:") {
        &arg[3..]
    } else {
        return Err(MailError::UnknownCommand("RCPT".to_string()));
    };

    Ok(extract_address(after_to))
}

/// Extract an email address from a string that may be wrapped in angle brackets.
///
/// If angle brackets are present, the content between `<` and `>` is returned.
/// Any ESMTP parameters after the closing `>` are ignored.
/// If no brackets, the first whitespace-delimited token is returned.
fn extract_address(s: &str) -> String {
    let s = s.trim();
    if let Some(start) = s.find('<') {
        if let Some(end) = s[start..].find('>') {
            return s[start + 1..start + end].to_string();
        }
    }
    // No brackets — take up to first whitespace.
    s.split_whitespace()
        .next()
        .unwrap_or("")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_ehlo() {
        let cmd = parse_command(b"EHLO sender.example.com\r\n").unwrap();
        assert_eq!(
            cmd,
            SmtpCommand::Ehlo {
                domain: "sender.example.com".to_string()
            }
        );
    }

    #[test]
    fn parse_helo() {
        let cmd = parse_command(b"HELO sender.example.com\r\n").unwrap();
        assert_eq!(
            cmd,
            SmtpCommand::Ehlo {
                domain: "sender.example.com".to_string()
            }
        );
    }

    #[test]
    fn parse_mail_from() {
        let cmd = parse_command(b"MAIL FROM:<sender@example.com>\r\n").unwrap();
        assert_eq!(
            cmd,
            SmtpCommand::MailFrom {
                address: "sender@example.com".to_string()
            }
        );
    }

    #[test]
    fn parse_mail_from_with_size() {
        let cmd =
            parse_command(b"MAIL FROM:<sender@example.com> SIZE=1024\r\n").unwrap();
        assert_eq!(
            cmd,
            SmtpCommand::MailFrom {
                address: "sender@example.com".to_string()
            }
        );
    }

    #[test]
    fn parse_mail_from_no_brackets() {
        let cmd = parse_command(b"MAIL FROM:sender@example.com\r\n").unwrap();
        assert_eq!(
            cmd,
            SmtpCommand::MailFrom {
                address: "sender@example.com".to_string()
            }
        );
    }

    #[test]
    fn parse_rcpt_to() {
        let cmd = parse_command(b"RCPT TO:<jake_z@q8.fyi>\r\n").unwrap();
        assert_eq!(
            cmd,
            SmtpCommand::RcptTo {
                address: "jake_z@q8.fyi".to_string()
            }
        );
    }

    #[test]
    fn parse_data() {
        let cmd = parse_command(b"DATA\r\n").unwrap();
        assert_eq!(cmd, SmtpCommand::Data);
    }

    #[test]
    fn parse_quit() {
        let cmd = parse_command(b"QUIT\r\n").unwrap();
        assert_eq!(cmd, SmtpCommand::Quit);
    }

    #[test]
    fn parse_rset() {
        let cmd = parse_command(b"RSET\r\n").unwrap();
        assert_eq!(cmd, SmtpCommand::Rset);
    }

    #[test]
    fn parse_noop() {
        let cmd = parse_command(b"NOOP\r\n").unwrap();
        assert_eq!(cmd, SmtpCommand::Noop);
    }

    #[test]
    fn parse_starttls() {
        let cmd = parse_command(b"STARTTLS\r\n").unwrap();
        assert_eq!(cmd, SmtpCommand::StartTls);
    }

    #[test]
    fn parse_case_insensitive() {
        let cmd = parse_command(b"ehlo EXAMPLE.COM\r\n").unwrap();
        assert_eq!(
            cmd,
            SmtpCommand::Ehlo {
                domain: "EXAMPLE.COM".to_string()
            }
        );
    }

    #[test]
    fn parse_null_sender() {
        // RFC 5321 §4.5.5: MAIL FROM:<> is the null sender for bounces/DSNs.
        let cmd = parse_command(b"MAIL FROM:<>\r\n").unwrap();
        assert_eq!(
            cmd,
            SmtpCommand::MailFrom {
                address: String::new()
            }
        );
    }

    #[test]
    fn parse_unknown_command() {
        let result = parse_command(b"VRFY user\r\n");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, MailError::UnknownCommand(ref v) if v == "VRFY"),
            "expected UnknownCommand(\"VRFY\"), got {err:?}"
        );
    }
}
