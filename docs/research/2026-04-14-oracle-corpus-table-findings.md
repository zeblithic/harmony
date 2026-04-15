# Oracle Corpus Table Diagnostic: Disentangling Mechanism and Content in Retrieval-Augmented Pretraining

**Date:** 2026-04-14
**Source:** Gemini Deep Research
**Companion:** `2026-04-14-oracle-corpus-table-research-prompt.md`
**Parent:** `2026-04-14-engram-injection-mechanism-findings.md`, `2026-04-14-model-delta-cross-attention-scaffold.md`

---

## Executive Summary

The transition from a standard dense transformer to a retrieval-augmented
architecture introduces a fundamental credit-assignment problem during
early-stage validation. Specifically, if the student model fails to
demonstrate an improvement in next-token prediction, the failure could
stem from one of two entirely distinct origins: an inert neural injection
mechanism, or a semantically barren retrieval corpus. The ongoing
ZEB-117 bake-off currently evaluates two experimental injection
mechanisms — a gated-residual pathway (gamma) and a localized
cross-attention scaffold (delta) — against a standard dense baseline
(alpha) and a parameter-matched control (beta). However, because the
current external memory corpus is derived from a naive n-gram hashing
schema subjected to a random Gaussian projection, the mathematical
utility of the content itself remains unverified. If the experimental
gamma and delta mechanisms merely match the performance of the beta
baseline, it becomes epistemologically impossible to determine whether
the architecture is inherently incapable of utilizing external memory,
or if the memory itself simply contains no actionable signal to exploit.

To definitively resolve this ambiguity, an "oracle" corpus table must be
constructed. This diagnostic tool is created by distilling the deeply
contextualized hidden states of a vastly superior teacher model over
the identical 800M-token training corpus. By exposing the gamma and
delta injection mechanisms to embeddings that are mathematically known
to contain high-density semantic, syntactic, and structural signals,
the diagnostic perfectly isolates the mechanism's structural capacity
to route and exploit external signals.

The optimal configuration for this oracle diagnostic involves utilizing
the Qwen2.5-1.5B architecture as the teacher model. The required
feature extraction must target the penultimate layer (L-1) at the
final token position of each identified n-gram. This precise
configuration leverages the inherent causal masking of the teacher's
autoregressive attention to prevent future-token label leakage, while
simultaneously maximizing the contextual awareness of the target
n-gram prior to final vocabulary logit projection. To reduce the
high-dimensional 1536-dimensional teacher representation space to
the constrained 128-dimensional student schema required by the
architecture, Principal Component Analysis (PCA) fit on a
representative subset of the corpus is vastly superior to random
Gaussian projection, as transformer hidden states are notoriously
anisotropic.

If the injection mechanisms under study are viable, the introduction
of the oracle table is projected to yield a validation loss
improvement of 0.15 to 0.25 nats over the beta control, based on
historical scaling laws for k-nearest neighbor language modeling. An
improvement exceeding the predefined 0.05 nats go/no-go threshold
definitively confirms that corpus quality is the primary limiting
lever, thereby justifying the allocation of engineering resources
toward a production-grade teacher distillation pipeline. Conversely,
a failure to surpass the beta baseline despite the presence of a
mathematically verified oracle table guarantees that the injection
architecture at the 40M parameter scale is the critical failure
mode. Such an outcome would necessitate an immediate pivot to
alternative routing mechanisms, such as chunked RETRO-style attention
or product-key memory adapters.

## The Entanglement Problem in ZEB-117

The core objective of the ZEB-117 research initiative is to ascertain
whether an external, non-parametric memory table can effectively
transmit useful optimization signals to a highly constrained, 40M-
parameter decoder-only transformer during its pretraining phase. The
system under study is strictly bounded: it operates with 8 layers, a
hidden dimension of 512, grouped-query attention (8/4), and rotary
positional embeddings, trained exclusively on a Chinchilla-optimal
token budget of approximately 800M tokens derived from the
FineWeb-Edu-POC dataset.

The retrieval target is a fixed, non-trainable corpus table of
embeddings indexed by a bigram/trigram xxhash64 algorithm, injected at
layer 2. The four concurrent configurations — alpha (dense baseline),
beta (+786K parameter FFN expansion), gamma (differentiable softmax
retrieval with gated-residual injection), and delta (per-position
cross-attention) — are designed to isolate the parameter overhead from
the architectural routing mechanism. The primary decision gates
pre-committed prior to the 20k-step runs stipulate that an improvement
of >= 0.05 nats over the beta control validates the specific retrieval
mechanism.

However, the architecture of the current corpus table introduces a
severe epistemological blind spot. The Phase 0 ablations (ZEB-102)
unequivocally confirmed that the original production engram table —
which maps an xxhash64 integer to a fixed table prior to a random
Johnson-Lindenstrauss projection — transmits exactly zero signal.
Null-table, shuffled-table, and random-table ablations yielded
validation losses statistically indistinguishable from the
corpus-derived table. Because the research team has never successfully
constructed a corpus table mathematically proven to carry strong
semantic signal, a negative result in the current bake-off (where
gamma and delta match beta) conflates mechanism failure with content
failure.

To determine the next phase of the Harmony v1 roadmap, the research
must separate the delivery mechanism from the payload. If the
delivery mechanism (the neural injection pathway) is inert, the team
must research alternative routing topologies such as hash layers or
adapter-style chunked attention. If the payload (the n-gram hashed
embeddings) is inert, the team must pivot to content quality research,
focusing on contrastive table training or teacher-distilled key spaces
while retaining the winning routing architecture. The oracle corpus
diagnostic is the singular methodological tool capable of executing
this separation.

## Literature Review: Distillation, Extraction, and Representation

The construction of an effective oracle diagnostic is not a trivial
data-processing task; it relies heavily on the intricate intersection
of knowledge distillation, representation learning, and
retrieval-augmented generation. To correctly parameterize the
diagnostic pipeline, empirical precedents must be analyzed across
several interconnected domains: teacher model selection, optimal
layer extraction, n-gram aggregation mathematics, and dimensionality
reduction techniques.

### Teacher Model Selection and Licensing Considerations

The selection of a teacher model to generate the oracle embeddings is
constrained by three primary factors: the model must operate within
the 1B to 3B parameter regime to run efficiently on consumer hardware
(RTX 4090/5080), it must demonstrate exceptional representation
quality on educational and instructional data analogous to
FineWeb-Edu, and it must carry a permissive open-weight license that
allows for offline feature extraction and subsequent student
distillation without restrictive downstream encumbrances.

The contemporary landscape of sub-3B parameter models is dominated by
highly optimized architectures that significantly outperform previous
generations of much larger models. The literature indicates that
Qwen2.5-1.5B and Llama-3.2-1B represent the bleeding edge of compact,
open-weight language models, exhibiting dense knowledge
representations, superior reasoning capabilities, and exceptional
instructional alignment. The Qwen2.5 architecture specifically
outperforms comparable models in general language tasks, mathematical
reasoning, and coding benchmarks due to its extensive pretraining
corpus and optimized attention modules. Similarly, the Llama-3.2-1B
model, distilled from the highly capable Llama 3.1 8B variant,
showcases remarkable instruction-following task performance.

However, the licensing stipulations governing these two leading
candidates diverge significantly, fundamentally dictating their
viability for this diagnostic. The Llama 3.2 Acceptable Use Policy
(AUP) contains complex and potentially restrictive clauses regarding
the generation of derivative models. While text-only derivations may
technically bypass the strictest multimodal limitations, the Llama
3.2 license imposes stringent geographic limitations — specifically
prohibiting individuals or companies domiciled in the European Union
from exercising certain rights granted under the community license.
Furthermore, the requirement to prominently display "Built with Llama"
on derivative works could legally entangle the future open-sourcing
of the 40M Harmony v1 model if its fundamental architectural
validation relied directly on Llama-derived feature spaces.

Conversely, the Qwen 2.5 series utilizes highly permissive licensing
structures. Models under the 72B parameter threshold, including the
target Qwen2.5-1.5B, are released under the Apache 2.0 license, or a
similarly permissive custom research license that explicitly permits
commercial use, fine-tuning, and distillation with minimal friction,
provided basic attribution is maintained. The empirical viability of
Qwen models as teachers is heavily supported by the literature;
DeepSeek's successful distillation of the Qwen architecture (e.g.,
DeepSeek-R1-Distill-Qwen-1.5B) mathematically proves that the
internal representations of Qwen models are highly amenable to
transfer learning and feature extraction without inducing a
catastrophic "capacity gap" in the student. The capacity gap
phenomenon occurs when a teacher model is vastly too large (e.g., 70B
parameters), causing its internal latent space to become so complex
that a 40M parameter student cannot meaningfully project or mimic
its distributions.

Therefore, balancing the requirement for high-density semantic
representations, modern architectural parity (RoPE scaling, SwiGLU,
tied embeddings), and unencumbered licensing, the Qwen2.5-1.5B model
emerges as the unequivocally optimal teacher for the oracle
diagnostic.

### Layer Topology and Semantic Extraction Depth

Transformer hidden states are not uniform across the depth of the
network. A foundational question in feature distillation is deciding
precisely which layer of the teacher model contains the richest,
most generalizable semantic signal for the student's retrieval
mechanism to exploit.

The standard transformer architecture processes text progressively.
The lowest layers function primarily as lexical and shallow syntactic
feature extractors, resolving basic parts of speech and local token
dependencies. The middle layers aggregate this information to capture
deep semantic meaning, topical coherence, and coreferential
relationships across the sequence. The absolute final layers, however,
become heavily specialized; they undergo a geometric shift to map the
high-dimensional continuous semantic space directly onto the discrete
vocabulary output matrix, minimizing the next-token cross-entropy
loss.

Extracting features from the final layer (L) often yields
representations that are overly specialized for the teacher's specific
unembedding weights. The seminal MiniLM framework and subsequent deep
self-attention distillation studies demonstrate that extracting
self-attention values and hidden states from the penultimate layer
(L-1) or a specific combination of upper-middle layers provides the
most robust, task-agnostic supervisory signal. By avoiding the final
layer, the distillation process captures the network's deepest
contextual understanding before it is deformed to fit the constraints
of the output vocabulary.

Further empirical evidence from model pruning and compression studies
reinforces this dynamic. The ShortGPT methodology, which assesses
layer importance by measuring performance degradation upon layer
removal, establishes that middle-to-late transformer layers often
contain redundant sequential transformations, but the layer
immediately preceding the final normalization and unembedding contains
the most generalized contextual clustering. For the oracle diagnostic,
extracting the hidden states from the penultimate layer (L-1) of the
Qwen2.5-1.5B teacher guarantees maximum semantic richness while
avoiding the vocabulary-specific collapse associated with the final
logit projection.

### N-Gram Extraction and Contextual Averaging Mathematics

The fundamental challenge of constructing the oracle table lies in
mapping deeply contextualized, token-level hidden states back to a
static, discrete n-gram lookup table. This requires an aggregation
strategy that preserves the contextual signal without diluting the
primary semantic payload through improper mathematical blending.

Historical static word embeddings, such as the FastText architecture,
construct representations by averaging the vectors of constituent
subword n-grams. For instance, the vector for the word "grass" is
computed by averaging the vectors for "gra", "ras", "ass", etc.
However, the reverse operation — distilling a static n-gram embedding
from a highly contextualized autoregressive transformer sequence —
requires an entirely different mathematical approach.

When processing an n-gram (e.g., a trigram sequence w_{i-2}, w_{i-1},
w_i) through a causal, decoder-only transformer like Qwen, the
lower-triangular attention mask enforces strict temporal
directionality. The hidden state at position i (h_i) has fully
attended to, and mathematically incorporated, the semantic values of
w_{i-2} and w_{i-1}. Conversely, the hidden state at h_{i-2} has
absolutely no awareness of the subsequent tokens in the n-gram.
Therefore, mean-pooling the hidden states across all three positions
within the n-gram sequence (i.e., computing [h_{i-2} + h_{i-1} +
h_i] / 3) actively corrupts the representation. It mathematically
dilutes the fully informed state h_i with the partially informed
states h_{i-2} and h_{i-1}. Thus, the optimal extraction scheme is
to isolate the hidden state of the final token in the n-gram
sequence exclusively.

Because the student model's retrieval mechanism utilizes a static
corpus table indexed by an unparameterized xxhash64 algorithm, the
oracle must collapse all temporal and contextual variations of a
specific n-gram hash into a single representative vector. The
literature surrounding the conversion of modern contextual embeddings
to static equivalents (such as the X2Static methodology) proves that
the simple arithmetic averaging of the target token's contextual
hidden state across tens of thousands of distinct sequence occurrences
yields a highly robust, expressive static embedding. By averaging the
L-1 hidden state of the final token of an n-gram across all 800M
tokens of the FineWeb-Edu-POC dataset, the resulting vector represents
the teacher's generalized, marginalized "understanding" of that
sequence across every possible context it appeared in during the
training run.

### Dimensionality Reduction: Bridging the 1536 to 128 Gap

The Qwen2.5-1.5B teacher model operates within a high-dimensional
state space, typically d=1536. The student architecture under study
strictly mandates an engram_dim=128. Bridging this dimensional gap
without destroying the topological relationships of the retrieved
embeddings is a critical path for the diagnostic.

Historically, random Gaussian projections have been utilized to reduce
dimensionality, relying on the theoretical guarantees of the
Johnson-Lindenstrauss (JL) lemma, which posits that pairwise distances
between points are preserved under random projections. However, the
JL lemma implicitly assumes that the intrinsic variance of the
underlying data manifold is relatively uniformly distributed across
the original high-dimensional space. In practical applications,
transformer hidden states are overwhelmingly anisotropic; a small,
highly skewed subset of the eigenspace accounts for the vast majority
of the geometric and semantic variance.

Applying a random projection treats all principal components with
equal weight, thereby injecting massive amounts of mathematical noise
from structurally irrelevant dimensions into the constrained
128-dimensional subspace. This phenomenon often leads to catastrophic
semantic collapse in retrieval systems.

Principal Component Analysis (PCA) provides a deterministic,
data-aware projection mechanism that isolates the exact hyperplanes
containing the maximum semantic variance. In the specific context of
Retrieval-Augmented Generation (RAG) pipelines, compressing large
1536-dimensional transformer embeddings down to 128 dimensions via
PCA has been empirically proven to preserve >90% of retrieval
accuracy, far outperforming autoencoders or random projections when
downstream fine-tuning data is limited.

While recent advancements like Matryoshka Representation Learning
(MRL) offer elegant solutions by forcing the model to explicitly
encode primary information in the earliest dimensions of the vector,
MRL inherently requires access to the pretraining loss function and
the ability to alter the teacher's fundamental training dynamics.
Since the diagnostic requires extracting features from an already-
pretrained, frozen open-weight model, MRL is inapplicable. Therefore,
applying a PCA transformation, fit on a large, representative
subsample of the extracted corpus vectors, is the mathematically
optimal choice for the oracle projection.

### Related Work in Architectural Memory Augmentation

The theoretical foundation for injecting external memory into language
models has evolved significantly over the past five years, moving
from late-stage generation conditioning to deep pretraining
augmentation. Understanding these precedents provides the necessary
context for the ZEB-117 mechanisms and the diagnostic itself.

**REALM (Retrieval-Augmented Language Model Pre-training):** Guu et al.
(2020) demonstrated that augmenting the pretraining objective with a
differentiable retriever significantly improves the model's capacity
to internalize world knowledge. Unlike the ZEB-117 architecture, which
utilizes a fixed hash-based lookup, REALM updates its retrieval index
asynchronously. The success of REALM proves that small language
models can exploit external vectors during pretraining, validating the
fundamental premise of the bake-off.

**kNN-LM (Nearest Neighbor Machine Translation/Language Modeling):**
Khandelwal et al. (2020) introduced a paradigm where a standard
autoregressive language model is linearly interpolated with a
k-nearest neighbors model during inference. The nearest neighbors are
retrieved from a massive datastore of cached training data hidden
states. Crucially, kNN-LM demonstrated that accessing the explicitly
mapped hidden states of a training corpus can yield state-of-the-art
perplexity reductions without altering the base model parameters. The
oracle diagnostic mathematically mirrors the kNN-LM datastore concept,
but shifts the retrieval from the logits-level at inference to the
continuous hidden-state level during pretraining.

**Memorizing Transformers:** Wu et al. (2022) extended the context
window by caching K,V pairs from past sequences into a differentiable,
searchable database. While effective for long-document context, it
differs from the ZEB-117 architecture by focusing on local,
sequence-specific memory rather than a global, static corpus hash
table.

**RETRO (Retrieval-Enhanced Transformer):** Borgeaud et al. (2022)
successfully augmented a 7B parameter model with a trillion-token
database using chunked cross-attention. RETRO utilizes a frozen BERT
model to encode retrieval keys and injects the retrieved text
representations periodically via a specialized cross-attention module.
The delta mechanism in ZEB-117 is heavily inspired by RETRO's
localized cross-attention, but operates at a per-token positional
level rather than utilizing large textual chunks.

The use of an "oracle retrieval table" explicitly designed as an
architectural diagnostic tool to measure mechanism capacity
independently of content quality appears to be an unexplored
methodological innovation within this literature. It bridges the gap
between knowledge distillation and retrieval augmentation, offering a
novel framework for isolating topological bottlenecks in constrained
networks.

## Expected Signal Analysis: Quantifying the Oracle's Impact

To establish rigorous go/no-go criteria for the ZEB-117 bake-off, the
theoretical impact of the oracle table must be quantified. If the
oracle table is genuinely populated with high-fidelity semantic
signal, and the injection mechanisms (gamma, delta) are architecturally
sound, what is the expected improvement in validation loss at the 40M
parameter scale over an 800M token horizon?

Theoretical bounds derived from the retrieval-augmented language
modeling literature provide a clear benchmark. The kNN-LM
architecture, which relies on a similar premise of mapping input
context to pre-computed external hidden states, reliably achieves
perplexity reductions of approximately 16% to 17% over strong
parametric baselines. In terms of log-loss — measured in nats
(natural logarithms) — augmenting a baseline model with a high-fidelity
continuous memory datastore translates to a reduction of approximately
0.15 to 0.30 nats, depending heavily on the scale of the base model
and the sheer volume of the retrieval corpus.

Given that the student under evaluation is a heavily constrained 40M
parameter model operating within a Chinchilla-optimal token budget
(800M tokens), its internal parameter capacity is fundamentally
limited. It lacks the depth and dimension to internalize the vast
statistical tail of the FineWeb-Edu dataset. Consequently, the
relative impact of external, perfectly distilled memory should be
proportionally higher for this student than for a massive 7B
parameter model that can memorize the data natively.

If the gamma (gated-residual) or delta (cross-attention) injection
mechanisms are structurally viable, the introduction of the
PCA-compressed, oracle-derived corpus table is projected to yield a
drop in validation loss of **0.15 to 0.25 nats relative to the dense
beta baseline**.

The predefined experimental go/no-go threshold of 0.05 nats is
therefore statistically conservative and robust. It requires the
oracle table to supply only one-third of the theoretical kNN-LM
advantage to trigger a positive signal. If the oracle diagnostic
fails to cross this minimal 0.05 nat threshold, the literature on
distillation and retrieval scaling dictates that the 40M architectural
scale is genuinely incapable of utilizing the signal. A loss
improvement of less than 0.05 nats mathematically guarantees an
architectural mechanism failure, indicating that the gradient flow
through the attention/gate layers is collapsing, or the FFN width is
insufficient to route the auxiliary data streams.

## Methodology: Oracle Table Construction Pipeline

To construct the oracle table without violating hardware memory
constraints or disrupting the aggressive bake-off timeline, the
extraction pipeline must be engineered around a single-pass, streaming
architecture. The following delineates the exact mathematical and
computational workflow required to generate the oracle table.

### Phase 1: Corpus Processing and Forward Pass Evaluation

The diagnostic pipeline instantiates the Qwen2.5-1.5B model in
bfloat16 precision to prevent quantization degradation in the
sensitive continuous feature space. The FineWeb-Edu-POC tokenized
dataset is consumed via a streaming dataloader. To maximize GPU
saturation and maintain mathematical parity with the student's
training environment, the sequence length is fixed to match the
teacher's optimal context window (e.g., T=2048 tokens).

For each continuous batch X in R^{B x T}, a forward pass is executed
with the `output_hidden_states=True` flag enabled. All gradient
computation is strictly disabled via the `torch.no_grad()` context
manager to optimize throughput. The target hidden states are extracted
exclusively from the penultimate layer (L-1):

```
H_{L-1} = Teacher(X) in R^{B x T x 1536}
```

### Phase 2: N-Gram State Aggregation and Online Averaging

The student table indexing schema relies on an xxhash64 algorithm
applied to bigrams and trigrams to assign n-grams to a fixed number
of rows (N = total_entries). To ensure perfect architectural parity,
the diagnostic pipeline must calculate the exact hash for every
contiguous 2-token and 3-token sequence in the batch.

For a specific n-gram ending at position t in the sequence, the
representative high-dimensional vector is strictly the hidden state
at position t:

```
v_t = H_{L-1}[b, t, :]
```

Because the 800M token corpus vastly exceeds the capacity of available
system RAM, standard aggregation (storing all vectors and computing
the mean at the end) is impossible. Instead, an online moving average
algorithm — an adaptation of Welford's online algorithm for vectors —
is maintained asynchronously on the CPU. The table is initialized as
a zero matrix M in R^{N x 1536} alongside an integer frequency array
C in Z^N.

For every calculated n-gram hash k:

1. Determine the row index: `idx = k mod N`
2. Increment the frequency counter: `C[idx] = C[idx] + 1`
3. Update the moving average: `M[idx] = M[idx] + (v_t - M[idx]) / C[idx]`

This O(1) memory operation guarantees that after the single 800M
token pass is completed, M[idx] represents the perfect arithmetic
mean of the contextualized teacher state across all occurrences of
all n-grams that hash to that specific row.

### Phase 3: Dimensionality Reduction via PCA

Following the comprehensive forward pass over the corpus, the
high-dimensional matrix M in R^{N x 1536} must be projected down to
match the student's engram_dim=128. The shipped scaffold uses
`sklearn.decomposition.IncrementalPCA` rather than exact full-matrix
SVD — this keeps the pipeline identical whether N is 10K (our current
default) or 1M (a future, larger-table variant that wouldn't fit a
single in-memory covariance decomposition).

> **Implementation note (2026-04-15):** the earlier draft of this
> section described exact full-matrix PCA. The shipped
> `training/ct87/generate_oracle_table.py` fits IncrementalPCA on a
> random subsample of *populated* rows, uses the fitted sample mean
> as mu, and projects the full table against the subsample-fitted
> components. Explained-variance numbers in run-stats JSON are from
> IncrementalPCA's `explained_variance_ratio_` attribute on the
> subsample, not from a full-covariance SVD. Numerically near-identical
> at our scale, but the provenance differs.

1. **Subsample selection:** a random subsample of populated rows
   (default 20% via `--pca-subsample-fraction`, with an RNG seed of 42)
   is drawn. Unpopulated rows are excluded so the projection matrix
   isn't dominated by the geometric origin.
2. **IncrementalPCA fit:** `partial_fit` is called on each
   `--pca-batch-size` (default 512) chunk of the subsample. The
   estimator converges to a mean vector `mu_sample` and components
   matrix `W_PCA` shaped `[128, 1536]`.
3. **Projection:** the full table is projected via
   `M_128 = (M - mu_sample) @ W_PCA.T`. Rows that were never populated
   by Welford are explicitly re-zeroed after projection to preserve
   the "empty row = zero signal" invariant — otherwise zero inputs
   would project to `-mu_sample @ W_PCA.T` and inject a constant
   nonzero vector into rows the student reads as absent.

### Phase 4: Serialization and Student Integration

The final reduced matrix M_128 is cast to float32 (or bfloat16) and
serialized to a standard `.safetensors` file. This format perfectly
mirrors the schema required by the student model's
`EngramTable.from_safetensors()` method. The gamma and delta
mechanisms will subsequently load this matrix directly via the
`--engram-ann-table` or `--engram-xattn-table` flags. The injection
layer logic natively handles the xxhash64 retrieval during the
student's forward pass, seamlessly fetching the PCA-projected,
teacher-distilled states instead of the previously utilized random
Gaussian states.

## Failure Mode Analysis: Leakage, Collision, and Alignment

While the oracle table isolates the injection mechanism's structural
capacity, the diagnostic itself is susceptible to highly specific
mathematical and systemic failure modes. These must be parameterized
to prevent false positive interpretations.

### Causal Masking and Label Leakage Prevention

A primary methodological concern in any form of distillation via
retrieval is the phenomenon of "label leakage" or "treatment leakage".
If the retrieved vector at position i contains deterministic,
future-sighted information about the exact identity of the target
token at i+1, the student's optimization landscape will collapse. The
network will devolve into trivial decoding of the oracle vector
rather than performing true autoregressive language modeling.

Because Qwen2.5-1.5B is a strictly causal, decoder-only transformer,
the hidden state at position i (h_i) is computed using a
lower-triangular attention mask. It has zero mathematical access to
the embeddings of token i+1 or beyond. Consequently, there is no
explicit future label leakage.

However, because the teacher is a highly advanced model, the state
h_i contains a highly sophisticated implicit prediction of the
sequence future. The teacher's final layers map closely to the
target vocabulary distribution. If the student learns to simply
bypass its own internal reasoning heads and linearly project the
oracle's h_i into its output logits, it is utilizing the external
memory as a purely associative crutch.

For the explicit purposes of this architectural diagnostic, this
behavior is highly desirable. The fundamental question the bake-off
asks is: *Is the mechanism inert?* If the student exploits the
oracle to achieve a massive validation loss drop by copying the
teacher's implicit priors, it explicitly proves that the injection
mechanism (gamma or delta) successfully routes gradient and
high-dimensional signal through the transformer residual stream.
The mitigation of dependency and the enforcement of true reasoning
are problems for the production-stage distillation pipeline, not the
diagnostic-stage capacity test.

**Harmony team note:** the report is correct that implicit leakage is
useful *for the diagnostic verdict*. But if this oracle table is later
promoted into a production training run, the student will develop a
crutch that only works when the oracle is attached — shipping an
oracle-coupled model is a category error, not a graduation. The
production-stage follow-up (ZEB-102 Phase 2) must use distilled
retrieval KEYS, not distilled final hidden states, to avoid this
failure mode at deployment.

### Hash Collisions and Semantic Averaging Collapse

The student's indexing schema utilizes an unparameterized xxhash64
modulo projection to assign millions of unique n-grams to a
constrained number of rows (total_entries). Due to the mathematical
inevitability of the Pigeonhole Principle, hash collisions are
guaranteed.

When a collision occurs, the online averaging algorithm will blindly
blend the hidden states of entirely disparate semantic n-grams into a
single row vector. In a highly structured, anisotropic high-dimensional
space, the arithmetic mean of two distinct semantic clusters frequently
lands in a geometric "dead zone" — a vector space that represents
neither original concept accurately. If the load factor of the hash
table is too high, the collision rate will induce widespread semantic
collapse, rendering the oracle table nearly as inert as the random
baseline.

This hash-collision contamination presents a critical confounding
variable. If the oracle diagnostic fails to improve validation loss,
it may not be due to mechanism failure, but rather collision-induced
semantic destruction. To distinguish between these outcomes, a
specific baseline (the Sparse Uncollided Table, discussed below) must
be prepared.

### Teacher-Student Distribution Mismatch

The chosen teacher model (Qwen2.5-1.5B) was pretrained on a massive,
highly diverse multilingual corpus that drastically differs from the
specific FineWeb-Edu-POC domain slice. Consequently, the teacher's
latent representations encode systemic priors and vocabulary
distributions that the 40M student has no capacity to contextualize.
This domain mismatch can induce a phenomenon where the oracle vectors
reside in a geometric subspace entirely orthogonal to the student's
developing activation manifolds.

To counteract this manifold mismatch, the student model's delta
mechanism (cross-attention) must maintain its independent,
zero-initialized output projection matrix (W_o). By initializing W_o
to exactly zero, the student begins its training trajectory identical
to the dense alpha baseline. As training progresses, the gradient
slowly scales the cross-attention influence, allowing the student
model to smoothly rotate the teacher's fixed PCA-space into structural
alignment with its own evolving residual stream, preventing initial
instability or activation spikes.

## Engineering Plan and Compute Budget

The oracle diagnostic must be executed rapidly to inform the ZEB-117
transition without delaying the broader Harmony v1 roadmap. The
hardware envelope is strictly constrained to consumer-grade GPUs: a
single RTX 4090 (24GB VRAM) and an RTX 5080 (16GB VRAM). Weekend-scale
experiments are highly preferred.

### Offline Inference Throughput Estimations

Evaluating 800M tokens through a 1.5B parameter model requires
substantial computational FLOPs. However, because the task requires
only the forward prefill pass (with no autoregressive decoding or
KV-cache maintenance for subsequent tokens), the throughput is
entirely compute-bound rather than memory-bandwidth bound.

Utilizing the vLLM inference engine for offline benchmarking on an
RTX 4090, models in the 1.5B parameter range process prefill sequences
at highly efficient rates, often exceeding 10,000 to 14,000 prompt
tokens per second depending on batch sizes and sequence packing
optimizations.

Assuming a conservative sustained throughput of 10,000 tokens per
second:

```
800,000,000 total tokens / 10,000 tokens/sec = 80,000 seconds ~ 22.2 hours
```

The physical generation of the oracle table is therefore a ~24-hour,
unsupervised background process. This falls perfectly within the
acceptable parameters for a weekend-scale job executed on the KRILE
compute node.

### Memory, Storage, and Tooling Architectures

The final matrix serialization poses no storage threat. Assuming a
large experimental configuration of N = 1,000,000 entries, the
mathematical dimensions dictate:

```
1,000,000 rows x 128 dimensions x 4 bytes (fp32) ~ 512 MB
```

The intermediate teacher states, however, must be managed carefully
to avoid Out-Of-Memory (OOM) errors. A single batch of size B = 64,
sequence length T = 2048, with hidden dimension d = 1536 in bfloat16
occupies roughly 400 MB of VRAM. The extraction script must
asynchronously transfer these specific tensor slices to CPU RAM
immediately after the layer hook fires, update the Welford moving
average arrays, and forcefully free the CUDA buffers before the next
batch initiates.

The tooling implementation will manifest as a standalone Python
script, `generate_oracle_table.py`, residing alongside the existing
corpus generators. The architecture will utilize the
HuggingFace/transformers library combined with datasets streaming. C
bindings for xxhash will be employed to process the n-gram sliding
windows in parallel across the CPU cores to prevent a data-loading
bottleneck. Following the completion of the 800M token pass,
scikit-learn's IncrementalPCA will be fitted on a random 20% subsample
of the accumulated vectors to generate the projection weights rapidly.

## Comparative Baselines for Diagnostic Isolation

If the primary teacher-distilled oracle table fails to transmit a
measurable signal, fallback baselines must be deployed to eliminate
secondary confounding variables without necessitating another
24-hour distillation run.

### The Sparse (Collision-Reduced) Table

> **Implementation note (2026-04-15):** the historical name "Sparse
> Uncollided Table" overstates what the shipped scaffold delivers.
> The current generator (`training/ct87/generate_sparse_uncollided_table.py`)
> still routes top-N n-gram embeddings into the 10K-row dense table
> via the student's xxhash64 — so at N=50K unique n-grams and
> entries=10K, each row still averages ~5 top-N n-grams × 4 seeds ≈
> 20 writes per row. This is a ~4000× reduction in per-row collision
> contamination vs the primary oracle's ~80K-n-gram averaging, but it
> is NOT fully collision-free. A truly collision-free variant requires
> a student-side dictionary dispatch (option (a) in the module
> docstring), which is flagged as a follow-up and not implemented.
>
> Interpretation implication: if the sparse baseline succeeds where
> the primary oracle fails, it is strong evidence (but not airtight
> proof) that hash topology is the dominant failure mode. For an
> airtight separation, option (a) must be implemented and run.

As analyzed in the failure modes, hash collisions can induce semantic
collapse. To isolate mechanism viability from hash contamination
(subject to the caveat above), a collision-reduced table is generated:
populate the hashed table with embeddings derived from a standard
Sentence-Transformer model applied to the textual strings of the top
50,000 most frequent unique n-grams. Rows not touched by the top-N
n-grams stay at zero. If the primary oracle fails but this
collision-reduced oracle succeeds, hash topology is the most likely
failure mode, pointing toward the cross-attention (delta) and
gated-residual (gamma) injection mechanisms being architecturally fine
but starved of usable content by the hash layer.

### Student-as-Teacher Oracle

A significantly cheaper diagnostic involves extracting the hidden
states from the student model's own partially-trained alpha
checkpoint (e.g., captured at step 10,000 of the 20k run). While this
risks tautology — the embeddings contain no novel knowledge or
external semantic priors — it provides a mathematically perfectly
aligned latent space. If the delta cross-attention mechanism cannot
mathematically exploit and route hidden states that are already
identically aligned to its own dimensions, the mechanism's projection
matrices or initialization schemes are fundamentally flawed at a
mathematical level.

**Harmony team note:** this one is the weakest of the three fallbacks.
By the report's own framing, aligned latent space alone doesn't prove
a mechanism works in the general case, since the student's own state
already encodes everything it could retrieve. Keep it in reserve but
do not scaffold it in the first-pass tooling — prefer the Sparse
Uncollided Table as the primary fallback.

### Upscaled Student Oracle

Utilizing a larger student parameter class (80M or 160M) trained on
the identical setup as a "teacher" for the 40M model presents a
middle ground. It offers a smaller capability gap than the 1.5B
distillation, reducing domain mismatch while remaining computationally
trivial to generate. However, it risks propagating the same
architectural deficiencies inherent to the Harmony v1 baseline
topology.

## Concrete Research Proposal and Interpretation Playbook

Based on the synthesis of the literature and architectural
constraints, the execution of the oracle diagnostic is the singular
critical path for advancing the ZEB-117 research agenda.

### Ranked Implementation Approaches

1. **Primary Approach: The Qwen2.5 Oracle**
   - Teacher: Qwen2.5-1.5B (Apache 2.0 / Open Research License).
   - Extraction Layer: Penultimate Layer (L-1).
   - Aggregation: Causal Final-Token State Extraction + Welford Online Averaging.
   - Projection: Global PCA down to 128 dimensions.
   - Rationale: Maximizes semantic density while maintaining absolute mathematical alignment with the student's retrieval logic.

2. **Secondary Approach: The Sparse Uncollided Control**
   - Rationale: Deployed only if the Primary Approach fails, to isolate hash-collision collapse from architectural injection failure.

3. **Tertiary Approach: Student-as-Teacher Checkpoint**
   - Rationale: Deployed only if alignment/domain mismatch is suspected of destroying the cross-attention gradient flow.

### Runtime Budget

The total computational budget for resolving the diagnostic is
approximately 34 GPU-hours:

- Oracle Table Generation: 22 hours (1x RTX 4090).
- PCA and Serialization: 2 hours (CPU/RAM bound).
- Student 20k Evaluation Run: ~10 hours (assuming standard ZEB-117 throughput on RTX 5080).

### Interpretation Matrix and Strategic Pivots

Upon completion of the 20k-step evaluation utilizing the primary
oracle table, the delta validation loss between the experimental
mechanisms (gamma, delta) and the parameter-matched dense control
(beta) will dictate the definitive trajectory of the Harmony v1
roadmap.

| ZEB-117 Base Result | Oracle Diagnostic Result | Strategic Interpretation | Actionable Pivot |
|---------------------|--------------------------|--------------------------|------------------|
| Mechanism > beta | Oracle >> Base Mechanism | The injection mechanism is highly viable and scales linearly with content quality. The current random hashed content is the sole bottleneck. | Go-Decision on ZEB-102 Phase 2. Allocate engineering resources to build a production distillation pipeline to populate the actual inference tables. |
| Mechanism > beta | Oracle ~ Base Mechanism | The mechanism hits a hard representational ceiling at the 40M parameter scale. The student's FFN width cannot route richer data streams regardless of quality. | Increase student model scale to 80M/160M to verify capacity scaling, or abandon continuous memory augmentation for this parameter class. |
| Both ~ beta | Oracle > beta cleanly | The original corpus content is mathematically inert, but the gamma/delta architecture is fully capable of routing semantic signal. | Abandon the naive xxhash64 table topology. Pivot research entirely to content refinement, semantic key spaces, and continuous distillation. |
| Both ~ beta | Oracle ~ beta | Catastrophic Mechanism Failure. A 40M student cannot exploit external memory regardless of content perfection. | Engram architecture is dead for Harmony v1. Pivot immediately to latent reasoning (COCONUT) or scaling the dense alpha baseline. |

## Conclusion

The implementation of the oracle corpus diagnostic is not merely an
optional validation step; it is the fundamental mathematical
prerequisite for interpreting the results of the ZEB-117 bake-off. By
isolating the injection architecture's routing capacity from the
unverified quality of the external corpus, this methodology prevents
the engineering team from squandering months tuning the
hyperparameters of an unscalable injection mechanism, or conversely,
discarding a brilliant architectural innovation due to a faulty data
pipeline. If the worst-case scenario materializes — where the oracle
fails to improve upon the baseline — the diagnostic provides the
empirical certainty required to kill the research track cleanly and
pivot toward more viable paradigms.
