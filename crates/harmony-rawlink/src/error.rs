//! Error types for the rawlink crate.

use std::fmt;

/// Errors that can occur in rawlink operations.
#[derive(Debug)]
pub enum RawLinkError {
    /// Socket creation or binding failed.
    SocketError(String),
    /// Ring buffer setup failed.
    RingError(String),
    /// Frame encoding/decoding failed.
    FrameError(String),
    /// I/O error from the underlying socket.
    IoError(std::io::Error),
    /// Capability check failed (CAP_NET_RAW missing).
    PermissionDenied(String),
}

impl fmt::Display for RawLinkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SocketError(msg) => write!(f, "socket error: {msg}"),
            Self::RingError(msg) => write!(f, "ring buffer error: {msg}"),
            Self::FrameError(msg) => write!(f, "frame error: {msg}"),
            Self::IoError(e) => write!(f, "I/O error: {e}"),
            Self::PermissionDenied(msg) => write!(f, "permission denied: {msg}"),
        }
    }
}

impl std::error::Error for RawLinkError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::IoError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for RawLinkError {
    fn from(e: std::io::Error) -> Self {
        if e.raw_os_error() == Some(libc::EPERM) || e.raw_os_error() == Some(libc::EACCES) {
            Self::PermissionDenied(
                "AF_PACKET requires CAP_NET_RAW capability (run as root or setcap)".into(),
            )
        } else {
            Self::IoError(e)
        }
    }
}
