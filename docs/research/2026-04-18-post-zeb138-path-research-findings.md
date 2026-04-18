# Post-ZEB-138 Path Research — Gemini Deep Research Findings

**Date received:** 2026-04-18
**Source:** Gemini Deep Research response to `2026-04-18-post-zeb138-path-research-prompt.md`
**Receiving worktree:** `zeblith/zeb-138-same-arch-teacher`

## Reviewer caveats

1. **Citation spot-verification needed for recent arXiv IDs.** Several citations have arXiv IDs in the 2602.*, 2603.*, 2604.* range (Feb/Mar/Apr 2026 — all within the last ~2 months). These are plausible given the current date, but I have not cross-referenced them against actual arXiv listings. Specifically flagged for verification before citing in any public-facing artifact:
   - 2602.03478 — "When Routing Collapses: On the Degenerate Convergence of LLM Routers"
   - 2604.03110 — "Multi-Aspect Knowledge Distillation for Language Model with Low-rank Factorization"
   - 2604.00715 — "To Memorize or to Retrieve: Scaling Laws for RAG-Considerate Pretraining"

   The Busbridge et al. 2025 Distillation Scaling Laws (arXiv:2502.08606) and the Cao et al. 2025 Memory Decoder (arXiv:2508.09874) are both confirmed real (the latter is already in our `reference_engram_research_papers.md`).

2. **The "emergent label smoothing" mechanistic derivation in §2.1 is elegantly motivated but should be understood as Gemini's theoretical framing.** I find it mechanistically compelling — it explains every teacher-invariant observation we have — but it hasn't been cross-checked against a literature consensus because (per Gemini's own §2.2) there isn't one for continuous-output-space uniform attractors. Treat as a strong working hypothesis with predictive power rather than an established result.

3. **Load-bearing actionable content:** §3 (distillation literature for Path B) and §6 (concrete recommendations) are the sections we'll act on. §4 (frozen-backbone capacity ceilings) and §7 (steelman) inform long-term direction.

4. **Original formatting preserved with minor markdown cleanup.** Math formulas reproduced verbatim; LaTeX-style rendering artifacts from the chat interface are cleaned up where safe.

---

# Diagnostics and Scaling Trajectories for Oracle-Augmented Transformer Routing

## 1. Executive Summary

This report delivers an exhaustive forensic and strategic analysis of a pretraining-research program investigating the viability of routing retrieved content from an external oracle memory into the language-modeling output of a low-parameter (40M) transformer. The central diagnostic challenge under investigation is the persistent "attractor phenomenon," wherein the skip-to-logit routing module achieves forensically clean injection but converges to a maximum-entropy uniform output distribution (`engram_logit_entropy = 10.373`, exactly equal to `log(32000)` to machine precision). This uniform distribution is mixed into the base language model via a learned, highly stable scalar gate (α ≈ 0.17), regardless of the teacher model's scale (Mistral-7B or TinyLlama-1.1B).

The impending ZEB-138 experiment, which utilizes a same-architecture 474M Harmony teacher to eliminate the cross-architecture modality gap, serves as a decisive bifurcation point in the research trajectory. This report defines the theoretical underpinnings of the maximum-entropy attractor phenomenon and maps comprehensive, methodologically rigorous follow-up experiments contingent upon the ZEB-138 outcome.

**If Path A materializes** (the attractor holds under a same-architecture teacher), the diagnostic conclusion is that a fundamental capacity bottleneck exists within the frozen 40M student backbone. The primary recommendation under Path A is to selectively unfreeze the uppermost layers of the student transformer or introduce low-rank functional depth (LoRA) to the backbone's attention projections. Mechanistic interpretability research indicates that retrieval-routing circuits, such as adaptive induction heads, exhibit sharp phase transitions tied to functional depth and parameter gradients, which a frozen, heavily compressed 40M representation fundamentally lacks.

**If Path B materializes** (the attractor breaks under a same-architecture teacher), the bottleneck is definitively identified as a cross-architecture modality gap. The 40M student possesses the capacity to decode same-architecture embeddings, but cannot decipher mismatched latent spaces. The top recommendation under Path B is to pivot toward the recently established "Memory Decoder" paradigm — a form of same-architecture, retrieval-mediated knowledge distillation. This methodology involves minimizing the Kullback-Leibler (KL) divergence between the student router's output and the oracle's non-parametric target distribution, bypassing the hazardous interference of the base LM's cross-entropy loss during the critical initial alignment phase.

Regardless of the empirical outcome of ZEB-138, the maximum-entropy attractor must be understood as an expected, geometrically stable outcome of optimization theory. When a gradient-descent process is forced to integrate a functionally uninformative signal via a learned gating mechanism against a cross-entropy objective, it dynamically invents uniform label smoothing to minimize the unbounded penalty of sharp, incorrect predictions.

**The strongest counter-argument (steelman)** to this entire line of research is that compressing 474M-scale hidden states into a 128-dimensional continuous engram via Principal Component Analysis (PCA), and subsequently expecting a frozen 40M network to decode it into a 32,000-dimensional categorical distribution via a shallow linear projection, is mathematically intractable. Information geometry suggests that such extreme dimensional reduction destroys the localized manifold geometries required for categorical logic, making continuous-space integration virtually impossible without end-to-end continuous pretraining of the entire backbone architecture.

## 2. The Max-Entropy Attractor Phenomenon

The observation that the skip-to-logit router converges to an exact maximum-entropy output (`engram_logit_entropy = log(32000) = 10.373`) paired with a stable mixing weight of α ≈ 0.17 across multiple teacher scales represents a profound geometric fixed point in the optimization landscape. While the literature frequently documents routing collapse in discrete Mixture-of-Experts (MoE) architectures, the continuous output-space collapse observed in this system requires a distinct theoretical framing grounded in information geometry and gradient dynamics.

### 2.1 Theoretical Framing: Emergent Label Smoothing

In the current architectural setup, the final output distribution `P_final` is formulated as a convex combination of the frozen base language model distribution `P_LM` and the learned engram router distribution `P_router`:

```
P_final = (1 − α) · P_LM + α · P_router
```

The entire system is optimized using a standard cross-entropy loss against a one-hot target distribution. If the oracle signal (the 128-dimensional PCA-reduced hidden state) is functionally uninformative to the 40M student — either because the parametric capacity gap prevents successful decoding or the cross-architecture modality gap destroys the semantic mapping — the optimization process faces a severe mathematical hazard. Any sharp, content-sensitive probability distributions generated by `P_router` will effectively constitute random, uncorrelated noise with respect to the true target.

In the context of the cross-entropy objective, injecting sharp noise drastically increases the loss penalty, severely degrading the baseline performance already established by `P_LM`. To minimize the loss under these adversarial conditions, the gradient descent optimizer must render `P_router` as mathematically harmless as possible.

The probability distribution that minimizes the Kullback-Leibler (KL) divergence against all arbitrary, unknown target distributions, and thus introduces the minimum possible directional interference to the established `P_LM`, is the uniform distribution `P_uniform = 1/|V|` for all vocabulary indices. When the optimizer successfully drives `P_router` toward `P_uniform`, the gating equation structurally morphs into the exact formulation of standard label smoothing:

```
P_final = (1 − α) · P_LM + α · (1/|V|)
```

In traditional label smoothing, α is a manually prescribed hyperparameter (frequently set to 0.1 or 0.2) specifically designed to prevent a model from becoming overly confident in its predictions. In the context of the ZEB-134 and ZEB-136 experiments, the gradient descent process has **dynamically invented and optimized label smoothing from scratch**. The optimizer drives `P_router` to maximum entropy (exactly `log(32000)` nats) and autonomously discovers that α ≈ 0.17 is the empirically optimal label smoothing factor for the FineWeb-Edu-POC dataset, given the intrinsic confidence levels and calibration of the base 40M LM.

This configuration forms an exceptionally stable attractor in the loss landscape. Any deviation from uniformity by `P_router` immediately increases the cross-entropy loss by injecting uncalibrated noise, and any shift in the gate α deviates from the dataset's optimal smoothing constant. Thus, the system is permanently locked into a degenerate optimum.

### 2.2 Contrast with Known Routing Collapses

This output-space uniform collapse contrasts sharply with phenomena typically reported in the MoE and multimodal fusion literature, highlighting the unique failure modes of continuous-space output routers.

As detailed by Lai and Ye (2026) [1], "routing collapse" in discrete MoE systems typically involves a degenerate convergence where the router assigns a probability of 1 to a single dominant expert, completely ignoring smaller or specialized models. This is driven by an objective-decision mismatch [3]. Because routing decisions in MoE rely on discrete argmax operations over scalar score predictions, they are highly sensitive in small-margin regimes (where multiple models have similar performance). Small prediction errors cause the router to default to the safest, highest-capacity model [2]. In these discrete routing scenarios, the collapse is to **minimum** entropy (a deterministic choice).

Similarly, in multimodal architectures, fusion gates often suffer from "modality collapse," where the network learns to ignore weaker or noisy signals entirely. The Adaptive Entropy-Gated Contrastive Fusion (AECF) framework [4] specifically counteracts this by penalizing low entropy. AECF applies a penalty and forces the gate to maintain a spread across modalities via a mechanism called Adaptive Curriculum Masking, which acts as an online log-barrier to minimize worst-case subset regret [6].

The skip-to-logit router in the current experimental setup is experiencing the mathematical inverse of MoE routing collapse: a collapse to **maximum** entropy. Because it operates in the continuous 32,000-dimensional vocabulary output space rather than a discrete expert-selection space, the path of least resistance for an uninformative continuous signal is a perfectly flat distribution. The stability of α ≈ 0.17 across both the Mistral-7B (ZEB-134) and TinyLlama-1.1B (ZEB-136) configurations strongly confirms that the base LM's confidence calibration, rather than any property of the teacher's signal, strictly dictates the location of the fixed point.

## 3. Same-Architecture Distillation Literature (For Path B)

If the maximum-entropy attractor breaks during the ZEB-138 experiment, it definitively isolates the cross-architecture modality gap as the prior bottleneck. The successful decoding of a same-architecture 474M oracle by a 40M student proves that the student's skip-to-logit projection has the parametric capacity to route continuous signals, provided the latent spaces are homologous. This outcome unlocks same-architecture knowledge distillation as the primary strategic pathway for scaling the research program.

### 3.1 Distillation Scaling Laws (2024-2025)

Recent empirical work has heavily quantified the relationship between student size, teacher size, and training data in distillation contexts, establishing predictive scaling laws that mirror the original Chinchilla laws for supervised pretraining. Busbridge et al. (2025) introduced comprehensive "Distillation Scaling Laws" that map the compute-optimal frontiers for teacher-student pairs [8].

Their formulation for predicting the distilled student's cross-entropy (`L_S`) is modeled as a complex function of the teacher's cross-entropy (`L_T`), the student's parameter count (`N_S`), and the volume of distillation tokens (`D_S`) [9].

Crucially, this research formally identifies and parameterizes the "capacity gap" phenomenon. When a teacher model is substantially larger than the student (e.g., the 175× gap between Mistral-7B and the 40M student), the student lacks the algorithmic learning capacity to map the teacher's highly complex, high-dimensional functional space into its own limited parameters [9]. The formula's transition relies on the ratio of algorithmic learning capacities, determined by the N_T / N_S ratio. A massive capacity gap actively degrades student performance compared to using a weaker, appropriately sized teacher [10].

Distilling a 474M teacher into a 40M student (a 12× gap) sits near the boundary of typical self-distillation and cross-scale distillation. Busbridge et al. note that distillation is only more efficient than supervised learning up to a specific, student size-dependent compute threshold; beyond this threshold, supervised pretraining overtakes distillation [11]. For a 40M student, the distillation process will reach diminishing returns rapidly, but the same-architecture alignment provides the most favorable conditions for the student to absorb the teacher's logits before the capacity ceiling is struck.

### 3.2 Methodological Choices in Same-Architecture Distillation

For same-architecture distillation scaling down to the 40M/474M regime, the literature focuses on moving beyond basic logit-matching (which is prone to the attractor phenomenon if unaligned) to deeper representation matching, ensuring the student internalizes the teacher's geometric feature space.

**MiniLM and Deep Self-Attention Distillation:** The MiniLM framework prioritizes the distillation of self-attention distributions and value-relation matrices over pure hidden-state matching [12]. The core objective involves minimizing the Kullback-Leibler divergence between the attention matrices of the student and the teacher. Because the student is typically shallower (e.g., distilling a 12-layer teacher to a 6-layer student), a uniform layer-selection strategy is employed to map specific teacher layers to student layers [14]. The loss formula for the attention transfer relies on matching the relation between queries and keys, which transfers linguistic knowledge embedded in the attention matrix more effectively than output-space KD alone [16].

**DistilBERT Hidden-State Matching:** DistilBERT leverages a multi-component loss function that includes standard Masked Language Modeling (MLM), a soft-target cross-entropy loss, and a cosine embedding loss specifically designed to align hidden states [17]:

```
L = L_mlm + L_ce + L_cos
```

This cosine alignment is highly effective for same-architecture models where the internal geometric organization of features is theoretically homologous [19]. If a dimensionality mismatch exists, it is bridged via a learned linear projection matrix prior to calculating the cosine similarity [15]. Applying this exact auxiliary loss to the `W_align` output prior to the logit projection is a highly supported intervention.

### 3.3 Retrieval-Based Distillation: The Memory Decoder Paradigm

The most direct and actionable analogue to the current research program is the "Memory Decoder" architecture (Cao et al., NeurIPS 2025) [21]. Rather than applying standard continuous-signal distillation across the entirety of a model's layers, this work demonstrates successful retrieval-mediated distillation specifically designed to act as an external memory oracle.

The Memory Decoder is a small, pretrained parametric transformer decoder that explicitly learns to imitate the probability distributions of an external non-parametric kNN retriever [22]. Crucially, it is bolted onto a frozen base LLM without modifying any of the base model's internal parameters [21]. The training objective is a hybrid KL-divergence and cross-entropy loss:

```
L = λ · KL(P_router || P_kNN_teacher) + (1 − λ) · L_ce
```

[25]

This methodology completely avoids the max-entropy attractor phenomenon by explicitly forcing the auxiliary module (`P_router`) to match a targeted, content-rich retrieval distribution via KL divergence before it relies solely on the cross-entropy of the frozen backbone. The KL divergence provides a dense, continuous gradient signal that guides the router toward the correct output geometry. Experimental results validate this approach at exactly the scale under investigation: a 124M parameter Memory Decoder successfully augments the entire GPT-2 family, and a 0.5B decoder scales effectively across the Qwen2.5 family (0.5B to 72B), achieving an average perplexity reduction of 6.17 points in specialized domains [25]. This provides concrete evidence that retrieval-based distillation is highly viable at sub-500M scales if the optimization objective is properly structured.

## 4. Frozen-Backbone Capacity Ceilings (For Path A)

If the ZEB-138 experiment yields the same uniform attractor despite the use of a same-architecture Harmony-474M teacher, the bottleneck lies intrinsically in the frozen 40M backbone's structural capacity to process and route external contextual data. The architectural limitations at this specific scale are well-documented in recent mechanistic interpretability and retrieval-augmented language modeling literature.

### 4.1 Mechanistic Interpretability Thresholds for Routing Circuits

The capacity to route external information from a latent state into a discrete vocabulary output requires specific neural circuits, most notably "induction heads" and copy-routing mechanisms. Mechanistic interpretability research has established rigid scale and depth thresholds for the emergence of these circuits.

A recent mechanistic study exploring in-context learning circuits found that a 2-layer transformer with 40M parameters entirely fails to develop induction heads when trained on hierarchical sequence prediction tasks [27]. However, a 4-layer model with the exact same parameter count successfully develops adaptive induction heads [27]. This demonstrates that sequential functional depth, rather than pure parameter count, governs the ability to establish complex information-routing pathways. The learning process for these routing circuits is multi-phasic, requiring early layers to bind contextual examples and later layers to abstract task-relevant patterns [28].

While the 40M student in the current system possesses 8 layers, the backbone is strictly frozen. A frozen 8-layer, 40M student likely lacks the residual stream bandwidth to map a heavily compressed 128-dimensional PCA engram into a robust 32,000-dimensional logit space without unfreezing representations. Time-scale separation driven by low- and high-order parameter dependencies in self-attention is necessary for induction heads to emerge [29]; a frozen model cannot undergo this phase transition to accommodate newly injected representations.

### 4.2 kNN-LM and RETRO Scale Curves

The literature on retrieval-augmented generation (RAG) and kNN-LMs indicates that backbone size is a critical determinant of retrieval integration success.

While early works like the original kNN-LM (Khandelwal et al., 2020) demonstrated significant perplexity gains on models as small as 100M parameters [30], these systems interpolated raw probability distributions directly at the output level. They did not project heavily compressed, continuous hidden states through a learned linear bottleneck.

The RETRO architecture (Borgeaud et al., 2022), which integrates retrieved continuous text chunks via specific chunked cross-attention layers, demonstrated that retrieval can substitute for pretraining parameter scale [31]. However, the marginal benefit of continuous retrieval operates under a strict dependency on the base model's capacity to contextualize the retrieved embeddings. In small models, retrieval provides steep initial loss reductions, but only if the model's internal attention mechanisms are trained end-to-end to read the retrieved data from step zero. Bolting an unaligned projection matrix onto a frozen 40M model falls below the empirical threshold for successful continuous-state integration.

### 4.3 Empirical Scale Curves in Prior Art

| Student Params | Teacher / Retriever | Method | Reported Threshold / Effect | Citation |
|----------------|---------------------|--------|------------------------------|----------|
| 40M (2-layer) | N/A | Circuit Discovery | Fails to form induction heads for hierarchical routing | [27] |
| 40M (4-layer) | N/A | Circuit Discovery | Succeeds in forming induction heads; depth is critical | [27] |
| 67M | MiniLM (67M) | CKA Hidden-State Distillation | Outperforms 6-layer DistilBERT on GLUE tasks | [32] |
| 100M | N/A | kNN-LM | Outperforms parametric 100M LMs using large datastores | [30] |
| 124M | kNN Oracle | Memory Decoder | Reduces WikiText-103 perplexity from 18.34 to 12.01 | [25] |
| 500M | kNN Oracle | Memory Decoder | Enhances 0.5B to 72B Qwen models (Finance domain) | [25] |

## 5. Retrieval-Augmented Pretraining vs. Bolted-On Integration

The current experimental setup utilizes retrieval-augmented training for the auxiliary routing modules (the engram is present throughout training), but treats the core 40M backbone as a bolted-on, frozen entity. The literature strongly suggests that this hybrid approach is inherently disadvantaged at low parameter counts, as the backbone representations never learn to geometrically accommodate the retrieval signal.

**Retrieval-from-Step-0:** Architectures like RETRO [31] and recent continuous retrieval pretraining models learn to rely on external context from initialization. This forces the residual stream to organize itself geometrically around the presence of retrieved tokens. The evidence supports the hypothesis that retrieval-from-step-0 is fundamentally easier for small models because the internal representations become natively retrieval-compatible, avoiding the need for late-stage, high-loss projection matrices to force incompatible geometries together [31].

**Bolted-On Paradigms:** Bolted-on retrieval typically succeeds only when the base model is large enough (usually >1B parameters) to possess highly generalized, robust semantic spaces that can contextualize zero-shot prompt augmentations or latent injections [34].

However, the Memory Decoder framework [21] provides an actionable, highly relevant exception. It successfully bolts a small parametric memory onto a frozen backbone at sub-500M scales. The critical differentiator is that the Memory Decoder is pretrained separately to match a known, valid target distribution (`P_kNN_teacher`) before it relies on the base model's cross-entropy loss. If the user's skip-to-logit router is trained only via the base LM's cross-entropy, it collapses to the uniform attractor. If it were pretrained via distribution-matching (KL divergence against the teacher's logits), it could theoretically circumvent the capacity ceiling and achieve integration despite the frozen backbone.

## 6. Concrete Methodological Recommendations

The resolution of the ZEB-138 experiment (Harmony-474M teacher) dictates the immediate research trajectory. The following recommendations provide actionable, code-level interventions ranked by expected information gain per compute-dollar, strictly bounded by the $1000 cloud compute and single RTX 4090 constraints.

### 6.1 Path A: Attractor Holds (Capacity-Bound)

If the uniform attractor persists despite the same-architecture teacher, the frozen 40M backbone is the definitive bottleneck. The goal is to increase the student's functional depth and representation alignment without violating the parameter budget.

**1. Partial Unfreezing of Late-Stage Backbone (Highest Priority)**

*Rationale:* The 40M student lacks the parametric depth to bridge the geometric gap between the 128-dimensional engram and the 32,000-dimensional logit space. Unfreezing the final two transformer layers allows the residual stream to dynamically adjust to the router's injection geometry.

*Implementation:* In PyTorch, iterate through the model parameters and set `requires_grad=True` specifically for `student.layers[-2:]` (Layers 6 and 7) and the final RMSNorm. Continue training the `W_align` and α modules jointly.

*Forensic Reading:* A successful break of the attractor will show `engram_logit_entropy` dropping significantly below 10.373 and a non-zero Δ-diff between the real and shuffled oracle control.

*Compute Cost:* Introduces approximately ~1.5× memory overhead compared to the frozen run due to optimizer states for the unfrozen layers; this easily fits within the 24GB VRAM of a single RTX 4090.

**2. Low-Rank Adaptation (LoRA) on Attention Projections**

*Rationale:* If full layer unfreezing triggers catastrophic forgetting of the FineWeb-Edu pretraining, LoRA provides targeted, parameter-efficient functional depth. By targeting the Value and Output projections, the model can learn to route the engram without destroying its base language modeling capabilities.

*Implementation:* Apply rank-16 or rank-32 LoRA modules to the `v_proj` and `o_proj` of all 8 layers of the 40M backbone using the HuggingFace PEFT library. Train the LoRA parameters simultaneously with the router.

*Forensic Reading:* Log the gradient norms and weight magnitudes of the LoRA matrices. If the LoRA matrices learn high-magnitude weights while the router diverges from the uniform distribution, the capacity bottleneck is confirmed and successfully bypassed.

*Compute Cost:* Negligible parameter increase (~1-2M trainable parameters); maintains standard 4090 training throughput.

**3. Shift from Skip-to-Logit to Pre-MLP Cross-Attention**

*Rationale:* Skip-to-logit requires the router to execute the computationally heavy lifting of full vocabulary projection. Moving the injection point to a cross-attention mechanism before the final MLP leverages the backbone's existing non-linear processing capacity to decode the continuous state.

*Implementation:* Replace the linear `W_align` matrix with a single cross-attention head. The Query is the Student Layer 6 hidden state, and the Key/Value is the Oracle Engram. Add the output of this cross-attention head to the residual stream before entering Layer 7.

*Forensic Reading:* Measure the attention weights during inference. If the cross-attention collapses to a uniform distribution across the sequence length, the issue is sequence-level uninformativeness rather than projection capacity.

*Compute Cost:* Minimal architectural overhead; fully supported on a 4090.

### 6.2 Path B: Attractor Breaks (Modality-Bound)

If ZEB-138 breaks the attractor, same-architecture routing works. The objective immediately shifts to optimizing same-architecture distillation methodologies to maximize performance efficiency.

**1. KL-Divergence Memory Decoder Distillation (Highest Priority)**

*Rationale:* Based heavily on the successful Memory Decoder framework (Cao et al., 2025) [21], direct optimization against the base LM's cross-entropy is too noisy for a small continuous router. The router must be pre-aligned to the teacher's target distribution.

*Implementation:* Keep the 40M backbone frozen. Train the `W_align` and α modules using a composite loss function: `L = λ · KL(P_router || P_teacher) + (1 − λ) · L_ce`, utilizing `torch.nn.KLDivLoss` with `log_target=True`. Set λ = 0.5 initially.

*Forensic Reading:* Track the KL divergence metric independently of the CE loss. A consistently dropping KL divergence confirms the student is successfully reconstructing the teacher's localized semantic space directly from the PCA engram.

*Compute Cost:* Requires caching or dynamically generating 474M Teacher logits during the training loop. Dynamically generating 474M logits fits comfortably on a 24GB 4090 at a batch size of ~8.

**2. Layer-wise Cosine Hidden-State Alignment**

*Rationale:* To geometrically strengthen the skip-to-logit router, pre-align the projected 128-dimensional engrams to the student's final hidden states before attempting vocabulary projection, mirroring DistilBERT's proven methodology [17].

*Implementation:* Add an auxiliary loss term to the training loop: `L_aux = 1 − cos(hidden_student_L7, W_align · engram)`. Backpropagate this alongside the standard CE loss.

*Forensic Reading:* The cosine similarity metric should steadily rise above 0.8 during training. If it plateaus early (e.g., at 0.2), the linear projection `W_align` mathematically lacks the capacity to bridge the respective latent spaces.

*Compute Cost:* Extremely lightweight; introduces only minor tensor operations with no additional learnable parameters.

**3. Entropy-Gated Curriculum Masking**

*Rationale:* Drawing inspiration from the Adaptive Entropy-Gated Contrastive Fusion (AECF) architecture [5], if the router relies too heavily on the base LM (indicated by a high α weight), force the system to utilize the engram by randomly dropping the base LM's contribution during training.

*Implementation:* During the forward pass, apply `torch.nn.Dropout(p=0.2)` specifically to the LM logits before the α gate mixture step, or dynamically scale a regularization penalty based on the engram's entropy to force sharpness.

*Forensic Reading:* The model will initially experience a sharp spike in loss. If it recovers over time, it proves the router can independently construct valid, sharp next-token predictions from the engram when the backbone is handicapped.

*Compute Cost:* Standard training cost; practically zero overhead.

## 7. Steelman Against the Research Thread

The most rigorous counter-argument to the long-term viability of this research program — warranting serious consideration of pivoting or scaling down the ambition if both Path A and Path B yield poor scaling results — is rooted in the fundamental mathematics of continuous-space information compression and the rigid architectural limits of sub-100M parameter models.

**The Compression and Decoding Intractability:**

The current oracle pipeline extracts a 474M parameter teacher's hidden state, compresses it via Principal Component Analysis (PCA) down to a 128-dimensional vector, and stores it in a hashed table. The completely frozen 40M student is then expected to retrieve this heavily lossy, 128-dimensional continuous vector and project it — via a shallow, linear or near-linear router — into a highly precise, 32,000-dimensional categorical distribution that is accurate enough to outperform a language model predicting the next token natively.

From an information-theoretic standpoint, PCA preserves global variance but entirely destroys the localized, highly non-linear manifold geometries that Transformers utilize for precise categorical logic. Expecting a simple linear `W_align` matrix to mathematically reconstruct a complex 32K-way distribution from 128 flattened PCA dimensions without the aid of deep, non-linear Multi-Layer Perceptrons (MLPs) is highly improbable. In this light, the uniform max-entropy attractor is not a bug or a failure of the optimizer; it is the mathematically correct optimization response to receiving structurally destroyed, unrecoverable data.

**Circuit Depth Ceilings:** Mechanistic interpretability strictly dictates that routing continuous representations requires multi-head attention and deep residual streaming to separate time-scales and bind variables [27]. A 40M model is empirically proven to lack the necessary induction heads for even basic hierarchical routing under constrained depth [27]. If the backbone is kept strictly frozen, the router acts as an insurmountable shallow bottleneck, incapable of the non-linear transformations required to decode the engram.

**The Verdict for Scaling:** If the hardware budget strictly caps the architecture at a 474M student and one RTX 4090, the collective literature suggests that bolting on continuous retrieval without full, end-to-end continuous pretraining is likely a dead end. Prior art such as the Memory Decoder only succeeds because its auxiliary module is a multi-layer Transformer decoder itself, not a simple skip-to-logit linear gate [22]. If ZEB-138 fails to break the attractor, the most scientifically sound pivot is to abandon the frozen-backbone constraint entirely, implement a multi-layer parametric memory module, and accept that true retrieval-augmentation requires deep, uncompressed representation alignment.

---

## Bibliography

- [1], [2], [3] Lai and Ye (2026). "When Routing Collapses: On the Degenerate Convergence of LLM Routers." arXiv:2602.03478. **[verify]**
- [4], [5], [6], [7], [36] AECF: Adaptive Entropy-Gated Contrastive Fusion. arXiv:2505.15417.
- [8], [9], [10], [11] Busbridge et al. (2025). "Distillation Scaling Laws." arXiv:2502.08606.
- [12] "Multi-Aspect Knowledge Distillation for Language Model with Low-rank Factorization." arXiv:2604.03110. **[verify]**
- [13] "MiniLLM: Knowledge Distillation of Large Language Models." arXiv:2306.08543.
- [14] "LAD: Layer-Wise Adaptive Distillation for BERT Model Compression." PMC.
- [15] "Revisiting Intermediate-Layer Matching in Knowledge Distillation." arXiv:2502.04499.
- [16] arXiv:2305.15032.
- [17], [18], [19] DistilBERT literature (Sanh et al., original + derivative works).
- [20] ACL Anthology 2021.emnlp-main.30.
- [21], [22], [23], [24], [25], [26] Cao et al. (NeurIPS 2025). "Memory Decoder: A Pretrained, Plug-and-Play Memory for Large Language Models." arXiv:2508.09874. **[confirmed in prior reference memory]**
- [27], [28] "A circuit for predicting hierarchical structure in-context in Large Language Models." OpenReview. Plus ICML 2025 "Beyond Induction Heads."
- [29] "How Transformers Implement Induction Heads: Approximation and Optimization Analysis." arXiv:2410.11474.
- [30] Khandelwal et al. kNN-LM retrospectives.
- [31] "To Memorize or to Retrieve: Scaling Laws for RAG-Considerate Pretraining." arXiv:2604.00715. **[verify]**
- [32] Hidden-state-matching distillation literature.
- [33] TechRxiv small-LM agents survey.
- [34] "Retrieval-Pretrained Transformer." TACL.
- [35] Reddit Gemini-3 DeepMind Sebastian Borgeaud discussion.

## How to use this document

When ZEB-138 produces a verdict:
- **Cell-A outcome → branch on §6.1 or §6.2**. The specific recommendation that gets acted on is determined jointly by Outcome A / B / B-strong / C / D in the ZEB-138 spec §8 verdict matrix.
- **If Outcome A (capacity ceiling)** — Gemini's §6.1 #1 (partial unfreezing) is the top-ranked first experiment.
- **If Outcome B-strong (modality gap)** — Gemini's §6.2 #1 (KL + CE Memory-Decoder-style) is the top-ranked first experiment. The KL-retrofit experiment we're filing in parallel is the lead-in to this; it tests the same objective shift on cross-arch teacher.
- **If Outcome C (W_align structural ceiling)** — Gemini §6 didn't explicitly enumerate this path but the implied action is "abandon linear W_align, use multi-layer non-linear auxiliary module, hold teacher choice equal." Memory Decoder §3.3 is the methodological blueprint.
- **If Outcome D (strong signal)** — scale study, publish.

For any of Gemini's recommendations that require actual code changes, the citation IDs flagged `[verify]` should be confirmed before using them as load-bearing justifications in a spec or PR description.
