# TOU/2U — Social Postcards for Human Reconnection

**Status:** Design approved. Future feature — depends on Wylene, Jain (foundation shipped), Nakaiah.

**Color:** Green (Jain) + Blue (Wylene) — cross-cutting social feature.

**Metaphor:** In a world of mansions with butlers, TOU is the mail system. You write a postcard (Wylene helps), your butler posts it (your Jain sends it), their butler receives it (their Jain screens it), and if it passes the household's rules, it reaches the resident (their Wylene surfaces it). If it doesn't pass, the butler quietly discards it and the resident never has to deal with it.

---

## Overview

**TOU** ("Thinking Of U" / "To You") is a protocol for carrying social signals between users' Jain instances. It enables:

- **Reconnection** — rekindling dormant friendships without the vulnerability of reaching out directly
- **Rituals** — daily good mornings, good nights, gratitude pulses to your closest people
- **Coordination** — helping couples and close friends find time together without being "annoying" about it
- **Ambient warmth** — a background sense that people care about you, without notification fatigue

The core principle: **the receiver's Jain has absolute sovereignty over what gets surfaced.** The sender expresses intent; the receiver's butler decides what to do with it. If the postcard is unwelcome, the sender never even knows it was screened — they just hear silence, which could mean anything.

### Why This Matters

Digital communication has a bravery problem. Reaching out to someone you haven't talked to in years is hard. Telling your partner "I want to spend time with you tonight" is surprisingly hard. Saying "good morning, I'm grateful you're in my life" every day takes emotional energy that most people can't sustain.

TOU removes the bravery requirement. Your Wylene notices the intent (sometimes before you're consciously aware of it), your Jain carries it, their Jain screens it, and if the feeling is mutual, reconnection happens organically. If it isn't mutual, nothing happens — no rejection, no awkwardness, no notification that someone you'd rather not hear from is thinking about you.

The couples case is the trust litmus test: if TOU works for married partners coordinating private plans — the most intimate use case — it works for everything. If Wylene and Jain earn trust at that level, the "old friend reconnection" case is easy by comparison.

### Design Values

- **Consent flows toward the receiver.** The sender can ask for receipts but isn't entitled to them.
- **No third party sees the data.** Postcards are encrypted end-to-end. Matching and filtering happen locally on each user's device.
- **No commercial exploitation.** If both Jains suggest Starbucks for a meetup, it's because it genuinely fits both users' criteria (distance, price, preferences). Nobody paid for placement.
- **Silence is not informative.** The sender cannot distinguish "their Jain burned it" from "they haven't been online" from "they're thinking about it." This is by design.
- **Auditability without surveillance.** The receiver can always review what their Jain filtered, but nobody else can.

---

## Signal Detection (Wylene — Sending Side)

Wylene detects "thinking of someone" through four channels, ranging from explicit to ambient:

### 1. Explicit

User tells Wylene directly: "I've been thinking about Sarah" or "I miss Alex." Easiest case, but also the hardest emotionally — sometimes you don't know you're missing someone until Wylene notices.

### 2. Behavioral

Wylene notices interaction patterns that suggest someone is on the user's mind:

- Lingering on old photos of a person
- Re-reading an old message thread
- Playing music associated with a friendship (user model tracks these associations)
- Searching for someone's name or profile
- Revisiting shared content (trip photos, collaborative playlists)

### 3. Temporal

Calendar and history-based signals:

- Birthday proximity
- Anniversary of a shared event (trip, graduation, wedding)
- "It's been N months since you last communicated"
- Seasonal patterns ("you always call your college friends in October")

### 4. Ambient (Guardian/Watching mode)

Environmental signals requiring sensor access (capability-gated):

- At a restaurant/place you used to visit together
- Hearing a song associated with them in public
- Near their neighborhood or workplace
- Seeing something that matches their interests ("Sarah would love this bookstore")

**Wylene's output:** A structured TOU signal handed to the user's own Jain. Wylene never sends anything across the network — she only talks to her local Jain.

---

## The TOU Postcard (Content Model)

A TOU postcard is a content blob, content-addressed and encrypted like everything else in Harmony:

```rust
/// A social signal sent between Jain instances.
pub struct TouPostcard {
    /// Sender's 128-bit address hash.
    pub from_hash: [u8; 16],
    /// Recipient's 128-bit address hash.
    pub to_hash: [u8; 16],
    /// What kind of social signal this is.
    pub signal_type: TouSignal,
    /// Intensity/eagerness of the signal.
    pub warmth: WarmthLevel,
    /// What receipt the sender is requesting (receiver may decline).
    pub receipt_request: ReceiptRequest,
    /// When this postcard was created.
    pub timestamp: f64,
    /// Optional coordination payload (for couples/close friends with
    /// mutual TOU channel established).
    pub payload: Option<TouPayload>,
}

/// The type of social signal.
pub enum TouSignal {
    /// "I'm thinking of you" — reconnection, missing someone.
    ThinkingOfYou,
    /// Morning ritual — daily warmth.
    GoodMorning,
    /// Evening ritual.
    GoodNight,
    /// Gratitude / "prayer" — thankfulness pulse.
    Gratitude,
    /// Coordination request — "I'm free and want to spend time together."
    /// Requires mutual TOU channel (higher consent tier).
    Coordinate,
}

/// How eager/warm the signal is.
pub enum WarmthLevel {
    /// Gentle background warmth. Default.
    Gentle,
    /// Actively warm — "I really miss you."
    Warm,
    /// Eager — "I'd love to reconnect soon."
    Eager,
}

/// What receipt the sender is requesting.
pub enum ReceiptRequest {
    /// Fire and forget. No information flows back. Default.
    None,
    /// Delivery receipt — "their Jain received it." Guaranteed if requested.
    /// Says nothing about what happened next.
    Delivery,
    /// Read receipt — "they saw it." NOT guaranteed. Receiver's Jain
    /// decides based on standing rules whether to send this back.
    Read,
}
```

**Key design decision:** The base postcard has **no message body**. The signal type IS the message. "ThinkingOfYou" from Jake's address hash is the entire content. This is intentional — it's a postcard, not a letter. The simplicity is the point: one byte of intent, maximum warmth, minimum vulnerability.

The optional `TouPayload` enables richer coordination for close relationships (calendar availability, preference data for meetup suggestions) but requires a bilateral mutual TOU channel to be established first.

---

## Jain-to-Jain Exchange (Network Layer)

### Zenoh Key Expressions

```
tou/postcard/{recipient_hash}    # encrypted postcards → recipient's Jain
tou/receipt/{sender_hash}        # delivery/read receipts back to sender's Jain
tou/channel/{channel_hash}       # mutual TOU channel traffic (couples/close friends)
```

### Encryption

Postcards are encrypted to the recipient's public key using the same Harmony-native cryptographic primitives (ChaCha20-Poly1305). Even the network infrastructure cannot read the postcard contents. Only the recipient's Jain, running on the recipient's device with access to their private key, can decrypt and evaluate.

### Delivery Receipt

When a Jain receives a postcard with `ReceiptRequest::Delivery`, it publishes a delivery receipt to `tou/receipt/{sender_hash}`. This is automatic and network-level — it confirms "your Jain's postcard arrived at their Jain" and nothing more. It does not indicate whether the postcard was surfaced, muted, burned, or is still in the queue.

### Read Receipt

Governed entirely by the receiver's standing Jain filter rules, not per-message. The sender can request one, but the receiver's Jain decides whether to honor it based on pre-configured rules:

- "Never send read receipts" (default)
- "Send read receipts to companions only"
- "Send read receipts during work hours only"
- "Always send read receipts"

**The receiver is never obligated.** You are not entitled to know whether someone read your postcard just because you sent it. That's their information to share when they want.

---

## Receiving Jain's Evaluation (Screening)

The receiving Jain applies filter rules using the same architecture as content filtering (Sensitivity x SocialContext), extended for social signals:

### TOU Filter Rules

```rust
pub struct TouFilterRule {
    /// Match postcards from specific senders or relationship tiers.
    pub sender_filter: SenderFilter,
    /// Match specific signal types.
    pub signal_types: Vec<TouSignal>,
    /// Time windows when this rule applies.
    pub time_window: Option<TimeWindow>,
    /// Maximum frequency from any single sender.
    pub max_frequency: Option<FrequencyLimit>,
    /// What to do with matching postcards.
    pub action: TouFilterAction,
}

pub enum SenderFilter {
    /// Specific blocked identities.
    BlockList(Vec<[u8; 16]>),
    /// Only allow from known contacts at or above this relationship tier.
    MinRelationshipTier(RelationshipTier),
    /// Allow from anyone (open).
    Open,
}

pub enum TouFilterAction {
    /// Delete permanently. Postcard is gone.
    Burn,
    /// Store silently, don't surface. Recoverable via audit.
    Mute,
    /// Welcome but hold for better timing.
    Queue,
    /// Pass to Wylene for organic nudging.
    Surface,
}
```

### Evaluation Inputs

| Input | Evaluated against |
|---|---|
| Sender identity | Block list, mute list, relationship tier |
| Signal type | "I accept GoodMorning from companions only" |
| Timing | "Don't surface anything after 10pm" |
| Frequency | "Max 1 TOU from any person per week" |
| Current social context | "Don't nudge me during work hours" (from Wylene) |
| Warmth level | "Only surface Eager signals from unknowns" |

### Outcomes

- **Burn** — Unwelcome. Delete permanently (if user allows burns) or archive silently.
- **Mute** — Store without surfacing. Recoverable. Good default for "not sure" cases.
- **Queue** — Welcome but bad timing. Hold and re-evaluate when context changes.
- **Surface** — Pass to Wylene for organic presentation.

### The Audit Trail

The user can always review what Jain filtered: "Your butler set aside 3 postcards this month. Want to see what they were?" The user can:

- Review filtered postcards
- Adjust filter rules ("actually, unmute Sarah")
- Recover muted postcards
- Change the burn/mute/queue defaults

This is Jain's existing health report and reconciliation concept extended to social signals. The user has full visibility into their butler's decisions without being burdened by every piece of incoming mail.

---

## Receiving Wylene's Presentation (Nudging)

When Jain surfaces a postcard, Wylene presents it **organically** — it should feel like Wylene's own observation, not a notification from an app.

### Nudge Styles

**Reconnection (ThinkingOfYou):**
- "You haven't talked to Jake in a while. Might be a good day to reach out."
- Does NOT reveal that Jake's Wylene initiated. Feels like Wylene noticed independently.

**Ambient warmth:**
- "3 people who love you thought of you today."
- No names, no action required. Just warmth.

**Ritual (GoodMorning/GoodNight/Gratitude):**
- Woven into the user's morning/evening routine naturally.
- "Sarah, Alex, and Mom sent you good morning warmth."

**Coordination (Coordinate):**
- "You and Sarah are both free Saturday afternoon."
- Requires mutual TOU channel. Richer data, richer nudge.

### Adaptation

The nudge style adapts to:

- **Relationship closeness** — intimate partner gets direct "they want to see you tonight", distant friend gets gentle "might be a good time to reconnect"
- **User preferences** — some people want names, some want anonymous warmth counts
- **Current context** — guardian mode might surface a nudge visually, sleeping mode queues it for later

---

## Standing Signals (Rituals)

Good mornings, good nights, and gratitude pulses are **standing TOU subscriptions**, not individual postcards each day.

```rust
pub struct StandingSignal {
    /// Who receives this signal.
    pub recipients: Vec<[u8; 16]>,
    /// What signal to send.
    pub signal_type: TouSignal,
    /// When to send (cron-like schedule).
    pub schedule: SignalSchedule,
    /// Warmth level for all signals in this subscription.
    pub warmth: WarmthLevel,
}
```

"Every morning, send a GoodMorning signal to my 5 closest people." Jain handles the scheduling and delivery. Their Jain handles whether and when to surface it.

This is **low-bandwidth, high-warmth** — a single byte of intent, not a message thread. The cost of maintaining daily connection drops to near zero for the sender, while the receiver gets a steady pulse of "someone loves you."

---

## The Couples / Coordination Case

Requires a higher consent tier: **a mutual TOU channel**. Both parties explicitly opt in to a bilateral agreement between their Jains that enables coordination postcards with richer data sharing.

### Mutual TOU Channel

```rust
pub struct TouChannel {
    /// Both parties' address hashes.
    pub parties: [[u8; 16]; 2],
    /// What data each party has consented to share.
    pub shared_data: SharedDataConsent,
    /// Channel-specific encryption key (derived via ECDH).
    pub channel_key: ChannelKey,
}

pub struct SharedDataConsent {
    /// Share calendar availability windows (not event details).
    pub calendar_availability: bool,
    /// Share location proximity (not exact location).
    pub location_proximity: bool,
    /// Share preference data for meetup suggestions.
    pub meetup_preferences: bool,
}
```

### The Starbucks Principle

When both Jains have calendar and preference data, they can independently compute meetup suggestions. If both Jains arrive at "Starbucks on 5th" as a good option, it's because:

- It's equidistant between both users' current locations
- It fits both users' price range preferences
- It matches both users' dietary needs
- It's open during a mutual free window

**Nobody paid for that suggestion.** No ad network, no sponsored placement, no "promoted venue." The computation happens locally on each Jain, using only data the users explicitly consented to share. This is what "technology serving humans" looks like.

---

## Color System Integration

| Component | Role in TOU |
|---|---|
| **Wylene (Blue)** | Detects intent (4 channels), presents nudges organically. Never touches the network. |
| **Jain (Green)** | Posts, receives, screens, queues, burns postcards. Owns the filter rules and audit trail. Handles standing signal scheduling. |
| **Nakaiah (Black)** | Encrypts postcards end-to-end. Manages consent tiers for mutual TOU channels. Ensures only the recipient's Jain can read postcards. |
| **Kitri (Red)** | Coordinates standing signals (cron-like scheduling for rituals). Runs behavioral detection workflows for Wylene. |
| **Oluo (Yellow)** | Provides relationship context to Wylene ("when did you last talk to Sarah?", "what music do you associate with Alex?"). |
| **Lyll (White)** | Content-addresses every postcard for auditability. Provides the verifiable audit trail. |

### Zenoh Key Expressions

```
tou/postcard/{recipient_hash}         # encrypted postcards to recipient's Jain
tou/receipt/{sender_hash}             # delivery/read receipts
tou/channel/{channel_hash}            # mutual TOU channel traffic
tou/standing/{sender_hash}            # standing signal configuration (local)
```

---

## Privacy Guarantees

1. **Postcards are end-to-end encrypted.** Only the recipient's Jain can read them.
2. **Filter decisions are local.** No server decides what's spam — your Jain does.
3. **Silence is uninformative.** Senders cannot distinguish burned/muted/queued/unseen.
4. **Read receipts require receiver consent.** Standing rules, not per-message.
5. **Coordination data stays local.** Meetup matching happens on-device, not in the cloud.
6. **Audit is private.** Only the user can review their Jain's filter decisions.
7. **No commercial exploitation.** Suggestions come from user data, not ad networks.

---

## Trust Litmus Test

> If TOU works for married partners coordinating private Saturday plans, it works for everything.

The couples case requires the highest trust: intimate schedule data, location sharing, preference matching — all processed locally by two Jains that the users trust with their most private information. If Harmony earns trust at this level, the "old friend reconnection" case (a single byte of "thinking of you") is trivial by comparison.

This is what makes people say: "Maybe this IS different. Maybe this CAN work for me."

---

## Deferred (Out of Scope)

- Group TOU (family morning rituals, friend group coordination) — bilateral first, group later
- TOU across network boundaries (Reticulum ↔ clearnet bridge)
- ML-based behavioral detection tuning (which signals are most predictive?)
- Perceptual/emotional context detection (Wylene noticing sadness/loneliness)
- TOU analytics for the user ("you reconnected with 4 people this month")
- Integration with external calendar services (Google Calendar, Apple Calendar)
- Cross-device TOU channel management (which device owns the channel key?)

---

## Open Questions

1. **Postcard TTL** — Should postcards expire? "I was thinking of you 6 months ago" might be less useful than "I'm thinking of you now." But a queued postcard that gets surfaced late could still be meaningful.
2. **Warmth escalation** — If I send ThinkingOfYou(Gentle) weekly for a month with no response, should Wylene suggest escalating to Warm? Or is persistent silence a signal to stop?
3. **Mutual detection without PSI** — The original idea included Private Set Intersection for mutual-interest-only revelation. The Jain-to-Jain model is simpler but one-directional. Is there a hybrid where mutual interest amplifies the signal on both sides?
4. **Standing signal fatigue** — If 20 people send me GoodMorning daily, does that become noise? Jain could aggregate ("12 people sent warmth this morning") but that loses the individual connection.
5. **Relationship tier bootstrap** — How does Jain know someone is a "companion" vs "acquaintance"? User-declared? Inferred from communication patterns? Both?
