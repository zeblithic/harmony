# Design Note: Seamless Context Compaction — Continuous Experience Across Context Windows

**Status:** Tracking / exploratory
**Date:** 2026-04-08
**Related:** harmony-ct87 (custom model), latent reasoning research

## Problem

LLM context windows are discrete: each session starts fresh, each window has a hard limit. Users experience jarring discontinuities — the model "forgets" everything between sessions. Current mitigations (RAG, system prompts with summaries, conversation history injection) are bolted on, lossy, and the user is often aware of the seams.

**Goal:** Make the context window invisible to the user. Turn discrete sessions into a continuous flow of experiences, thoughts, and information over time.

## Proposed Architecture

A context window structured into four zones, with an automatic compaction feedback loop:

```
+------------------------------------------------------------------+
| 1. LONG STORY ARC (compacted history)                            |
|    - Accumulated summary of all prior sessions                   |
|    - Mixed fidelity: recent sessions more detailed, older faded  |
|    - Injected at the start of every new context window            |
|    - Analogous to long-term memory / episodic narrative           |
+------------------------------------------------------------------+
| 2. IMMEDIATE CONTEXT (current situation)                         |
|    - Current prompt / input / signal / trigger / sensors / state  |
|    - The "present moment" — what the user just said, current     |
|      system state, active task, environment                      |
|    - Full fidelity, uncompacted                                  |
+------------------------------------------------------------------+
| 3. WORKING / REASONING (active processing)                       |
|    - Model's working memory during this interaction              |
|    - Chain of thought, latent reasoning, tool use, intermediate  |
|      results, scratch space                                      |
|    - Where the actual "thinking" happens                         |
+------------------------------------------------------------------+
| 4. RESULT (output)                                               |
|    - The model's response / action / output for this interaction |
+------------------------------------------------------------------+
| 5. COMPACTION (end-of-window feedback)                           |
|    - Automatic summarization of this session                     |
|    - Folds into the Long Story Arc for the next session          |
|    - "What happened? What did we learn? What changed?"           |
|    - This is the bridge that turns discrete windows into         |
|      continuous experience                                       |
+------------------------------------------------------------------+
              |
              v
        Next session's "Long Story Arc" = previous arc + compaction
```

## Key Design Questions

### Where does this mechanism live?

- **In the model itself?** The model could have native support for structured context zones — special tokens or architectural features that delineate the zones. The compaction step could be a specialized generation mode (like COCONUT's latent reasoning, but for summarization).
- **In the harness/orchestration layer on top?** A wrapper system that manages the context window, triggers compaction at the right time, and injects the compacted arc into the next session. The model is a standard LLM that doesn't know about zones — the orchestrator manages everything.
- **Hybrid?** The model understands zone structure (via training on zone-structured data), but the orchestrator manages timing, compaction triggers, and arc injection.

### Compaction timing

- **After every interaction:** Most computationally expensive, but highest fidelity. The arc is always up-to-date. Every response includes a "compact and carry forward" step.
- **At context window boundaries:** Only compact when approaching the window limit. More efficient, but the arc update is bursty — large gaps between compactions mean more information to summarize at once.
- **Adaptive:** Compact when the working/reasoning section grows past a threshold, or when the model detects a natural "chapter break" in the conversation.

### Compaction quality

- **Lossy by definition:** Compaction is compression. What gets kept vs discarded?
- **Recency bias:** Recent interactions should be more detailed in the arc. Older sessions fade to key facts and decisions. This mirrors Chronos temporal decay — could we reuse the tier/TTL framework for context compaction?
- **Semantic importance:** Some interactions are high-signal (user corrects the model, makes a key decision, introduces a new requirement). Others are low-signal (routine Q&A, status checks). The compaction should weight by importance, not just recency.

### Interaction with latent reasoning

If the model reasons in embedding space (COCONUT/VL-JEPA), the compaction could also happen in embedding space:
- Instead of summarizing the session as text and injecting text into the next arc, produce a compacted embedding that captures the session's semantic content.
- This embedding is injected into the next session's Long Story Arc zone via a mechanism similar to Engram injection.
- The model "remembers" prior sessions in embedding space, not token space — potentially higher fidelity and lower token cost.

### Interaction with Engram

- Could the Long Story Arc be stored as Engram entries? Each session's compaction produces Engram-formatted embeddings that are added to a "personal memory" Engram table.
- Chronos tiers could apply: recent sessions = tier 4 (Regular, 30-day TTL), key decisions = tier 2 (NearEternal, 10-year TTL).
- The model would "remember" prior sessions the same way it "remembers" facts — via hash lookup into the personal Engram table.

## Computational Cost Analysis

**After-every-interaction compaction:**
- Extra forward pass(es) for summarization after each response
- For a 0.5B model: ~2-5 seconds extra per interaction on a 4090
- On edge devices (RPi5): potentially 10-30 seconds — may be prohibitive for interactive use
- Could be done asynchronously (compact in background while user reads response)

**Worth it?** Unclear without testing. The improved continuity might justify the cost for certain use cases (long-running agents, persistent assistants). For quick Q&A, probably not.

## Next Steps

1. File as a bead for exploration once model training infrastructure is stable
2. Consider whether this is a model-level or harness-level feature first
3. Prototype the harness-level approach (cheapest to test) with the current model
4. Evaluate whether embedding-space compaction (vs text summarization) is feasible at 0.5B scale
5. Research existing work on "memory-augmented transformers" with persistent state across sessions (MemTRM, Infini-attention, etc.)
