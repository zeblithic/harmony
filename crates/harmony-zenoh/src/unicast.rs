//! Sans-I/O unicast routing layer for point-to-point message delivery.
//!
//! [`UnicastRouter`] manages logical channels on a single peer connection,
//! providing an alternative to pub/sub broadcast for messages that are
//! inherently point-to-point (trades, DMs, private interactions).
//!
//! # Wire format
//!
//! Unicast frames use tag `0x03` followed by a 2-byte big-endian channel ID
//! and the payload:
//!
//! ```text
//! [0x03][channel_id: 2 BE][payload: N bytes]
//! ```
//!
//! # Usage
//!
//! ```ignore
//! let mut router = UnicastRouter::new();
//! router.open_channel(channels::TRADE);
//!
//! // Sending
//! let action = router.send(channels::TRADE, payload)?;
//! let frame = UnicastRouter::encode_frame(channels::TRADE, &payload);
//! // ... encrypt frame via Link and send
//!
//! // Receiving
//! let (channel_id, payload) = UnicastRouter::decode_frame(&data)?;
//! let actions = router.handle_event(UnicastEvent::MessageReceived {
//!     channel_id, payload: payload.to_vec(),
//! });
//! ```

use alloc::{sync::Arc, vec, vec::Vec};
#[cfg(not(feature = "std"))]
use hashbrown::HashSet;
#[cfg(feature = "std")]
use std::collections::HashSet;

use crate::error::ZenohError;

/// Logical channel identifier for multiplexing unicast streams.
pub type ChannelId = u16;

/// Well-known channel IDs for the harmony ecosystem.
pub mod channels {
    use super::ChannelId;
    /// Trade negotiation messages between players.
    pub const TRADE: ChannelId = 1;
    /// Direct messages between players.
    pub const DM: ChannelId = 2;
}

/// Wire protocol frame tag for unicast messages.
///
/// Sits alongside `0x01` (control) and `0x02` (pub/sub data) in the
/// Reticulum Link data channel framing.
pub const FRAME_TAG_UNICAST: u8 = 0x03;

/// Minimum unicast frame size: 1 (tag) + 2 (channel_id).
const MIN_FRAME_SIZE: usize = 3;

/// Events from the network to the router.
#[derive(Debug, Clone)]
pub enum UnicastEvent {
    /// A unicast message was received from a peer.
    MessageReceived {
        channel_id: ChannelId,
        payload: Vec<u8>,
    },
}

/// Actions the router wants the caller to perform.
#[derive(Debug, Clone)]
pub enum UnicastAction {
    /// Send a unicast message to the peer.
    /// Caller should encode the frame, encrypt via Link, and send.
    SendMessage {
        channel_id: ChannelId,
        payload: Vec<u8>,
    },
    /// Deliver a received message to the application.
    Deliver {
        channel_id: ChannelId,
        payload: Arc<[u8]>,
    },
}

/// Manages unicast channels for a single peer connection.
///
/// Each channel is identified by a [`ChannelId`]. Only messages on
/// open channels are delivered; messages on unknown channels are
/// silently dropped.
#[derive(Debug, Clone)]
pub struct UnicastRouter {
    channels: HashSet<ChannelId>,
}

impl UnicastRouter {
    /// Create a new router with no open channels.
    pub fn new() -> Self {
        Self {
            channels: HashSet::new(),
        }
    }

    /// Open a channel to send and receive messages on.
    pub fn open_channel(&mut self, channel_id: ChannelId) {
        self.channels.insert(channel_id);
    }

    /// Close a channel. Messages on this channel will be dropped.
    pub fn close_channel(&mut self, channel_id: ChannelId) {
        self.channels.remove(&channel_id);
    }

    /// Check if a channel is open.
    pub fn is_open(&self, channel_id: ChannelId) -> bool {
        self.channels.contains(&channel_id)
    }

    /// Prepare a message for sending on a channel.
    ///
    /// Returns `ChannelNotOpen` if the channel hasn't been opened.
    pub fn send(
        &self,
        channel_id: ChannelId,
        payload: Vec<u8>,
    ) -> Result<UnicastAction, ZenohError> {
        if !self.channels.contains(&channel_id) {
            return Err(ZenohError::ChannelNotOpen(channel_id));
        }
        Ok(UnicastAction::SendMessage {
            channel_id,
            payload,
        })
    }

    /// Handle an inbound unicast message.
    ///
    /// Delivers the message if the channel is open; silently drops it otherwise.
    pub fn handle_event(&self, event: UnicastEvent) -> Vec<UnicastAction> {
        match event {
            UnicastEvent::MessageReceived {
                channel_id,
                payload,
            } => {
                if self.channels.contains(&channel_id) {
                    vec![UnicastAction::Deliver {
                        channel_id,
                        payload: Arc::from(payload),
                    }]
                } else {
                    vec![]
                }
            }
        }
    }

    /// Encode a unicast wire frame: `[0x03][channel_id: 2 BE][payload]`.
    pub fn encode_frame(channel_id: ChannelId, payload: &[u8]) -> Vec<u8> {
        let mut frame = Vec::with_capacity(MIN_FRAME_SIZE + payload.len());
        frame.push(FRAME_TAG_UNICAST);
        frame.extend_from_slice(&channel_id.to_be_bytes());
        frame.extend_from_slice(payload);
        frame
    }

    /// Decode a unicast wire frame. Returns `(channel_id, payload)`.
    ///
    /// Expects the full frame including the `0x03` tag byte.
    pub fn decode_frame(data: &[u8]) -> Result<(ChannelId, &[u8]), ZenohError> {
        if data.len() < MIN_FRAME_SIZE {
            return Err(ZenohError::UnicastFrameTooShort(data.len()));
        }
        if data[0] != FRAME_TAG_UNICAST {
            return Err(ZenohError::InvalidMessageType(data[0]));
        }
        let channel_id = u16::from_be_bytes([data[1], data[2]]);
        let payload = &data[3..];
        Ok((channel_id, payload))
    }
}

impl Default for UnicastRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn send_returns_message_action() {
        let mut router = UnicastRouter::new();
        router.open_channel(channels::TRADE);
        let action = router.send(channels::TRADE, vec![1, 2, 3]).unwrap();
        match action {
            UnicastAction::SendMessage {
                channel_id,
                payload,
            } => {
                assert_eq!(channel_id, channels::TRADE);
                assert_eq!(payload, vec![1, 2, 3]);
            }
            _ => panic!("expected SendMessage"),
        }
    }

    #[test]
    fn send_on_closed_channel_errors() {
        let router = UnicastRouter::new();
        let result = router.send(channels::TRADE, vec![1, 2, 3]);
        assert!(matches!(result, Err(ZenohError::ChannelNotOpen(1))));
    }

    #[test]
    fn handle_event_delivers_for_open_channel() {
        let mut router = UnicastRouter::new();
        router.open_channel(channels::TRADE);
        let actions = router.handle_event(UnicastEvent::MessageReceived {
            channel_id: channels::TRADE,
            payload: vec![42],
        });
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            UnicastAction::Deliver {
                channel_id,
                payload,
            } => {
                assert_eq!(*channel_id, channels::TRADE);
                assert_eq!(&**payload, &[42]);
            }
            _ => panic!("expected Deliver"),
        }
    }

    #[test]
    fn handle_event_drops_for_closed_channel() {
        let router = UnicastRouter::new();
        let actions = router.handle_event(UnicastEvent::MessageReceived {
            channel_id: channels::TRADE,
            payload: vec![42],
        });
        assert!(actions.is_empty());
    }

    #[test]
    fn encode_decode_round_trip() {
        let payload = b"hello unicast";
        let frame = UnicastRouter::encode_frame(channels::DM, payload);
        assert_eq!(frame[0], FRAME_TAG_UNICAST);
        let (channel_id, decoded) = UnicastRouter::decode_frame(&frame).unwrap();
        assert_eq!(channel_id, channels::DM);
        assert_eq!(decoded, payload);
    }

    #[test]
    fn decode_frame_too_short() {
        let result = UnicastRouter::decode_frame(&[0x03]);
        assert!(matches!(result, Err(ZenohError::UnicastFrameTooShort(_))));
    }

    #[test]
    fn decode_frame_wrong_tag_rejected() {
        // A pub/sub frame (tag 0x02) must not be silently misparsed as unicast.
        let result = UnicastRouter::decode_frame(&[0x02, 0x00, 0x01, 0xFF]);
        assert!(matches!(result, Err(ZenohError::InvalidMessageType(0x02))));
    }

    #[test]
    fn decode_frame_empty() {
        let result = UnicastRouter::decode_frame(&[]);
        assert!(matches!(result, Err(ZenohError::UnicastFrameTooShort(0))));
    }

    #[test]
    fn decode_frame_minimal_no_payload() {
        let frame = UnicastRouter::encode_frame(channels::TRADE, &[]);
        let (channel_id, payload) = UnicastRouter::decode_frame(&frame).unwrap();
        assert_eq!(channel_id, channels::TRADE);
        assert!(payload.is_empty());
    }

    #[test]
    fn multiple_channels_independent() {
        let mut router = UnicastRouter::new();
        router.open_channel(channels::TRADE);
        // DM channel NOT opened.

        assert!(router.send(channels::TRADE, vec![1]).is_ok());
        assert!(router.send(channels::DM, vec![1]).is_err());

        let trade_actions = router.handle_event(UnicastEvent::MessageReceived {
            channel_id: channels::TRADE,
            payload: vec![1],
        });
        let dm_actions = router.handle_event(UnicastEvent::MessageReceived {
            channel_id: channels::DM,
            payload: vec![2],
        });
        assert_eq!(trade_actions.len(), 1);
        assert!(dm_actions.is_empty());
    }

    #[test]
    fn open_close_channel() {
        let mut router = UnicastRouter::new();
        assert!(!router.is_open(channels::TRADE));
        router.open_channel(channels::TRADE);
        assert!(router.is_open(channels::TRADE));
        router.close_channel(channels::TRADE);
        assert!(!router.is_open(channels::TRADE));
        assert!(router.send(channels::TRADE, vec![1]).is_err());
    }

    #[test]
    fn well_known_channel_ids() {
        // Ensure well-known IDs are distinct and non-zero.
        assert_ne!(channels::TRADE, 0);
        assert_ne!(channels::DM, 0);
        assert_ne!(channels::TRADE, channels::DM);
    }
}
