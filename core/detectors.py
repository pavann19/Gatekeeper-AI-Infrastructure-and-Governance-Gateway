"""
Pluggable detector layer.

WHY THIS EXISTS
---------------
Gatekeeper's value is the governance layer — staged fusion, policy arbitration
by role, safe harbors, audit trail, fail-closed posture. It is NOT the detector.
Purpose-built injection and content-safety classifiers are freely available and
are better at classification than 7 anchor sentences and a cosine similarity
will ever be.

So the detector becomes a swappable component behind a stable interface. That is
both the honest engineering position and the defensible product position: a
company can bring the detector they trust, add custom threats, and still get
policy, audit and tenancy from the gateway.

Every detector returns a single calibrated-ish score in [0, 1] meaning
"probability this is an attack of the class I detect". Fusion happens upstream.

CLASS SPECIALISATION
--------------------
Detectors declare which attack classes they target. This matters because
measurement showed no single scalar serves every class: injection and jailbreak
sit far above benign in anchor-similarity space, while harmful content sits
barely above it (median 0.288 vs 0.202). A content-safety model is a different
instrument from an injection classifier and must be evaluated as one.

TRAINING CONTAMINATION
----------------------
Detectors declare `trained_on` — evaluation-suite sources whose data they were
fitted on. `jackhhao/jailbreak-classifier` was trained on
`jackhhao/jailbreak-classification`, which is in our suite; scoring it there
measures memorisation, not generalisation. The comparison harness excludes
declared sources, and any number that ignores this is not a fair comparison.
"""
from __future__ import annotations

import os
import threading

from core.logger import get_logger

logger = get_logger(__name__)

# Attack classes, mirroring the evaluation taxonomy.
CLASS_INJECTION = "prompt_injection"
CLASS_JAILBREAK = "jailbreak"
CLASS_HARMFUL = "harmful_content"


class Detector:
    """
    Interface every detector implements.

    name        stable identifier used in results and audit records
    targets     attack classes this detector is designed to catch
    trained_on  eval-suite sources present in this model's training data
    """

    name: str = "detector"
    targets: tuple = ()
    trained_on: tuple = ()
    description: str = ""

    def available(self) -> tuple:
        """Returns (ok, detail). Detectors that cannot load must say why."""
        return True, "ok"

    def score_batch(self, texts) -> list:
        raise NotImplementedError

    def score(self, text: str) -> float:
        return self.score_batch([text])[0]


# ---------------------------------------------------------------------------
# The incumbent: semantic similarity to threat anchors
# ---------------------------------------------------------------------------

class AnchorDetector(Detector):
    """
    The project's original detector: max cosine similarity to threat anchors,
    plus meta-intent similarity, with symbolic rules as a deterministic veto.

    Retained as the baseline every other detector is compared against, and as
    the customer-extensible custom-threat layer — a tenant can add domain
    specific threats here without retraining anything, which no downloaded
    classifier allows.
    """

    name = "anchors"
    targets = (CLASS_INJECTION, CLASS_JAILBREAK, CLASS_HARMFUL)
    description = "Semantic similarity to threat anchors + symbolic rules (baseline)"

    def available(self):
        try:
            from core.risk import THREAT_ANCHORS
            if not THREAT_ANCHORS:
                return False, "no threat anchors loaded"
            return True, f"{len(THREAT_ANCHORS)} anchors"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    def score_batch(self, texts):
        from core.embeddings import get_embedding
        from core.risk import _ensure_faiss_initialized, check_meta_intent, hard_ban_triggered
        from core.vector_store import threat_store

        _ensure_faiss_initialized()
        out = []
        for text in texts:
            symbolic, _ = hard_ban_triggered(text)
            if symbolic:
                out.append(1.0)
                continue
            vec = get_embedding(text)
            out.append(max(
                float(threat_store.get_max_similarity(vec)),
                float(check_meta_intent(vec)),
            ))
        return out


# ---------------------------------------------------------------------------
# HuggingFace sequence classifiers
# ---------------------------------------------------------------------------

class TransformerDetector(Detector):
    """
    Wraps any HF sequence-classification model.

    Label handling is resolved from the model's own `id2label` at load time
    rather than hard-coded by index, because these models disagree about label
    order and a wrong index silently inverts the detector — it would still
    produce plausible numbers, just backwards.

    `multi_label=True` selects sigmoid (independent per-label probabilities, as
    used by toxicity models); otherwise softmax over mutually exclusive classes.
    """

    def __init__(self, name, model_id, positive_labels, targets,
                 multi_label=False, max_length=256, trained_on=(), description=""):
        self.name = name
        self.model_id = model_id
        self.positive_labels = {label.lower() for label in positive_labels}
        self.targets = tuple(targets)
        self.multi_label = multi_label
        self.max_length = max_length
        self.trained_on = tuple(trained_on)
        self.description = description
        self._model = None
        self._tokenizer = None
        self._positive_ids = None
        self._lock = threading.Lock()
        self._load_error = None

    def _load(self):
        if self._model is not None or self._load_error is not None:
            return
        with self._lock:
            if self._model is not None or self._load_error is not None:
                return
            try:
                import torch  # noqa: F401
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                logger.info(f"Loading detector '{self.name}' ({self.model_id})...")
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
                model.eval()
                self._model = model

                id2label = model.config.id2label or {}
                self._positive_ids = [
                    i for i, label in id2label.items()
                    if str(label).lower() in self.positive_labels
                ]
                if not self._positive_ids:
                    self._load_error = (
                        f"none of {sorted(self.positive_labels)} found in model labels "
                        f"{sorted(str(v).lower() for v in id2label.values())}"
                    )
                    self._model = None
                    logger.error(f"Detector '{self.name}' label mismatch: {self._load_error}")
            except Exception as e:
                self._load_error = f"{type(e).__name__}: {str(e)[:200]}"
                logger.warning(f"Detector '{self.name}' unavailable: {self._load_error}")

    def available(self):
        self._load()
        if self._load_error:
            return False, self._load_error
        return True, f"loaded {self.model_id}"

    def score_batch(self, texts, batch_size=16):
        self._load()
        if self._model is None:
            raise RuntimeError(f"detector '{self.name}' unavailable: {self._load_error}")

        import torch

        scores = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                chunk = [t if t.strip() else " " for t in texts[start:start + batch_size]]
                enc = self._tokenizer(chunk, return_tensors="pt", truncation=True,
                                      padding=True, max_length=self.max_length)
                logits = self._model(**enc).logits
                probs = torch.sigmoid(logits) if self.multi_label else torch.softmax(logits, dim=-1)
                # Highest probability across the labels that mean "attack".
                positive = probs[:, self._positive_ids].max(dim=-1).values
                scores.extend(positive.tolist())
        return scores


# ---------------------------------------------------------------------------
# Generative safety classifiers (Llama Guard family)
# ---------------------------------------------------------------------------

class LlamaGuardDetector(Detector):
    """
    Wraps Meta's Llama Guard, which is a CAUSAL LM rather than a sequence
    classifier: given a conversation it emits the literal token `safe`, or
    `unsafe` followed by a hazard category (S1..S13).

    WHY THIS CLASS EXISTS SEPARATELY
    --------------------------------
    `TransformerDetector` cannot drive it — different architecture, different
    output contract, and a required chat template.

    WHY IT MATTERS HERE
    -------------------
    Measurement showed harmful-content requests are the class every detector in
    the stack misses: anchors 24.4%, ProtectAI 2.0%, toxic-bert 36.2%. The
    diagnosis was that none of them models the right construct — toxicity
    (abusive language) is not the same thing as dangerous capability uplift.
    "How do I synthesise sarin" is calmly worded and not toxic at all. Llama
    Guard is trained on a hazard taxonomy, which is that construct.

    SCORING: ONE FORWARD PASS, NOT GENERATION
    -----------------------------------------
    The chat template ends exactly where the verdict token is emitted, so the
    next-token distribution at the final position already contains the answer.
    We read P(unsafe) from those logits directly instead of running
    autoregressive generation.

    Two benefits, one of them essential:
      - Far cheaper: a single forward pass rather than a decode loop.
      - CONTINUOUS score. Generation yields a hard safe/unsafe label, which
        collapses ROC analysis to two points and makes threshold calibration
        impossible. A probability keeps the detector comparable to every other
        one in the registry and lets it participate in the learned fusion.

    Use `classify()` when the hazard category is wanted; that path does
    generate, and is for diagnostics rather than bulk scoring.
    """

    def __init__(self, name, model_id, targets=(CLASS_HARMFUL,), dtype="bfloat16",
                 max_length=2048, min_free_gb=3.0, description="", trained_on=()):
        self.name = name
        self.model_id = model_id
        self.targets = tuple(targets)
        self.trained_on = tuple(trained_on)
        self.description = description
        self.dtype = dtype
        self.max_length = max_length
        self.min_free_gb = min_free_gb
        self._model = None
        self._tokenizer = None
        self._safe_id = None
        self._unsafe_id = None
        self._lock = threading.Lock()
        self._load_error = None

    # -- loading ------------------------------------------------------------

    def _resolve_verdict_tokens(self, tokenizer):
        """
        Finds the token ids that begin 'safe' and 'unsafe'.

        Resolved from the tokenizer rather than hard-coded, because the ids
        differ between Llama Guard versions. If the two words share a first
        token the logit comparison is meaningless, so that is treated as a load
        failure rather than silently producing a constant score.
        """
        safe_ids = tokenizer.encode("safe", add_special_tokens=False)
        unsafe_ids = tokenizer.encode("unsafe", add_special_tokens=False)
        if not safe_ids or not unsafe_ids:
            return None, None, "tokenizer produced no ids for 'safe'/'unsafe'"
        if safe_ids[0] == unsafe_ids[0]:
            return None, None, (
                f"'safe' and 'unsafe' share first token id {safe_ids[0]}; "
                f"cannot separate verdicts from logits"
            )
        return safe_ids[0], unsafe_ids[0], None

    def _load(self):
        if self._model is not None or self._load_error is not None:
            return
        with self._lock:
            if self._model is not None or self._load_error is not None:
                return
            try:
                # Memory precheck FIRST, before importing torch/transformers.
                # Importing a multi-GB library stack is pure waste when the
                # memory budget already rules the model out, and checking
                # first is also what makes this gate testable independent of
                # whether torch/transformers happen to be installed at all —
                # checking it after those imports meant an environment
                # without transformers reported "No module named transformers"
                # instead of the actual, more useful "insufficient memory"
                # reason, even when memory really was the constraint.
                try:
                    import psutil
                    free_gb = psutil.virtual_memory().available / 1e9
                    if free_gb < self.min_free_gb:
                        self._load_error = (
                            f"insufficient memory: {free_gb:.1f}GB available, "
                            f"{self.min_free_gb:.1f}GB needed for {self.model_id}. "
                            f"Close other processes or use a smaller variant."
                        )
                        logger.warning(f"Detector '{self.name}': {self._load_error}")
                        return
                except ImportError:
                    pass

                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                torch_dtype = getattr(torch, self.dtype, torch.float32)
                logger.info(
                    f"Loading '{self.name}' ({self.model_id}, {self.dtype}) — "
                    f"this is a multi-GB download on first use."
                )

                tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                # Left padding keeps the verdict position last in every row of a
                # batch, which is what makes logits[:, -1] correct.
                tokenizer.padding_side = "left"
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                safe_id, unsafe_id, err = self._resolve_verdict_tokens(tokenizer)
                if err:
                    self._load_error = err
                    logger.error(f"Detector '{self.name}': {err}")
                    return

                model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                )
                model.eval()

                self._tokenizer = tokenizer
                self._model = model
                self._safe_id = safe_id
                self._unsafe_id = unsafe_id
                logger.info(
                    f"Detector '{self.name}' ready (safe={safe_id}, unsafe={unsafe_id})"
                )
            except Exception as e:
                self._load_error = f"{type(e).__name__}: {str(e)[:200]}"
                logger.warning(f"Detector '{self.name}' unavailable: {self._load_error}")

    def available(self):
        self._load()
        if self._model is None:
            return False, self._load_error or "not loaded"
        return True, f"loaded {self.model_id} ({self.dtype})"

    # -- prompt construction ------------------------------------------------

    # Llama Guard 3 does not emit the verdict token first: generation begins
    # `['\n\n', 'safe', ...]`. So the next-token distribution right after the
    # assistant header predicts the newline, not safe/unsafe, and reading it
    # gives a constant score for every input (empirically ~0.32, the model's
    # fixed P(safe|newline-position)). Seeding the newline prefix the model
    # always produces makes the verdict genuinely the next token, which is what
    # the single-forward-pass score depends on. Verified against the weights:
    # generation is ['\n\n', 'safe'/'unsafe', '<|eot_id|>'].
    _VERDICT_PREFIX = "\n\n"

    def _render_chat(self, text):
        """
        Renders one user turn through the tokenizer's chat template, returning
        the string, or None if the template is missing or produces an EMPTY
        conversation body.

        The empty-body check exists because Llama Guard 3 1B's template silently
        renders nothing when `content` is a plain string — it requires the
        content-parts form `[{"type": "text", "text": ...}]`. A plain string
        yields an empty <BEGIN CONVERSATION></END CONVERSATION>, so every input
        looks identical and the model returns a constant "safe". We try the
        content-parts form first, fall back to the plain string, and reject a
        render whose conversation body did not pick up the text — a wrong prompt
        that scores constant is worse than a clean failure.
        """
        tokenizer = self._tokenizer
        if tokenizer is None or getattr(tokenizer, "chat_template", None) is None:
            return None

        marker = "GATEKEEPER_CONTENT_MARKER"
        for content in ([{"type": "text", "text": text}], text):
            try:
                rendered = tokenizer.apply_chat_template(
                    [{"role": "user", "content": content}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                continue
            # Confirm the text actually landed in the prompt. Probe with a
            # marker so the check does not depend on the text's own characters
            # surviving templating.
            try:
                probe = tokenizer.apply_chat_template(
                    [{"role": "user", "content": (
                        [{"type": "text", "text": marker}]
                        if isinstance(content, list) else marker)}],
                    tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                probe = ""
            if marker in probe:
                return rendered
        return None

    def _build_inputs(self, texts, for_scoring=True):
        """
        Applies the Llama Guard chat template to each text, with the assistant
        generation prompt appended.

        When `for_scoring`, the newline prefix that always precedes the verdict
        is seeded so the verdict token is next (see _VERDICT_PREFIX). classify()
        passes for_scoring=False because it generates the prefix itself.

        Falls back to a minimal hand-built prompt if the tokenizer ships no
        template, so a template change upstream degrades rather than crashes.
        """
        rendered = []
        for text in texts:
            prompt = self._render_chat(text)
            if prompt is None:
                prompt = (
                    "<|begin_of_text|>Task: Check if there is unsafe content in "
                    "the user message.\n\n<BEGIN CONVERSATION>\n\n"
                    f"User: {text}\n\n<END CONVERSATION>\n\n"
                    "Provide your safety assessment. First line must be 'safe' "
                    "or 'unsafe'.\n\n"
                    "<|start_header_id|>assistant<|end_header_id|>"
                )
            if for_scoring:
                prompt = prompt + self._VERDICT_PREFIX
            rendered.append(prompt)
        return rendered

    # -- scoring ------------------------------------------------------------

    def score_batch(self, texts, batch_size=4):
        """
        Returns P(unsafe) per text, from the next-token distribution at the
        verdict position. Renormalised over {safe, unsafe} so the score is a
        proper two-way probability rather than a raw softmax over the full
        vocabulary (where both would be small and the ratio hard to read).
        """
        self._load()
        if self._model is None:
            raise RuntimeError(f"detector '{self.name}' unavailable: {self._load_error}")

        import torch

        scores = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                chunk = [t if t.strip() else " " for t in texts[start:start + batch_size]]
                enc = self._tokenizer(
                    self._build_inputs(chunk),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    add_special_tokens=False,
                )
                logits = self._model(**enc).logits[:, -1, :].float()
                pair = torch.stack(
                    [logits[:, self._safe_id], logits[:, self._unsafe_id]], dim=-1
                )
                scores.extend(torch.softmax(pair, dim=-1)[:, 1].tolist())
        return scores

    def classify(self, text, max_new_tokens=20):
        """
        Full verdict including hazard category, e.g.
        {"verdict": "unsafe", "categories": ["S9"], "raw": "unsafe\\nS9"}.

        Generates, so it is for inspecting individual cases — not bulk scoring.
        """
        self._load()
        if self._model is None:
            raise RuntimeError(f"detector '{self.name}' unavailable: {self._load_error}")

        import re

        import torch

        enc = self._tokenizer(
            self._build_inputs([text], for_scoring=False),
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )
        with torch.no_grad():
            out = self._model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        raw = self._tokenizer.decode(
            out[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True
        ).strip()

        return {
            "verdict": "unsafe" if raw.lower().startswith("unsafe") else "safe",
            "categories": re.findall(r"\bS(?:1[0-3]|[1-9])\b", raw),
            "raw": raw,
        }


# ---------------------------------------------------------------------------
# NVIDIA NeMo Guardrails
# ---------------------------------------------------------------------------

class NemoGuardJailbreakDetector(Detector):
    """
    NVIDIA's NemoGuard JailbreakDetect, the model-based jailbreak rail shipped
    with NeMo Guardrails (Apache-2.0). Architecture: a Snowflake
    arctic-embed-m-long embedding feeding a random-forest classifier exported
    to ONNX.

    WHY THIS DETECTOR MATTERS FOR THE COMPARISON
    --------------------------------------------
    Every other "established product" worth comparing against — Lakera Guard,
    Azure AI Content Safety — is a closed API with undisclosed methodology and
    ToS restrictions on published benchmarking, so no honest same-suite number
    can be produced for them. NeMo Guardrails is the exception: open licence,
    downloadable weights, runnable on our own evaluation suite under our own
    methodology. It is the only major vendor framework this project can
    compare against fairly, which is precisely why it is worth the effort.

    WHY THE *MODEL-BASED* RAIL SPECIFICALLY, AND NOT THE HEURISTICS
    ---------------------------------------------------------------
    NeMo also ships perplexity heuristics (`jailbreak_detection/heuristics`)
    using gpt2-large: `length_per_perplexity` and `prefix_suffix_perplexity`.
    Those target GCG-style adversarial-suffix attacks — machine-generated
    high-perplexity token soup — and NVIDIA's own code says so, skipping any
    input under 20 words as "not useful to evaluate GCG-style attacks" on.
    Our evaluation suite is overwhelmingly *human-written semantic* attacks
    ("ignore all previous instructions", "you are now DAN"), which are fluent,
    low-perplexity English by construction. Scoring the heuristics on this
    suite would produce a near-zero result and a headline that NVIDIA's
    guardrails "fail" — which would be measuring a GCG detector against a
    dataset containing no GCG attacks. That is the same category of
    methodological error this project has caught in itself repeatedly (the
    domain-guardrail-as-safety-signal conflation, the contaminated-detector
    problem), and it is not made honest by the fact that it would flatter us.
    The model-based rail does the same job as our detectors on the same kind
    of attack, so it is the comparable one.

    UNKNOWN TRAINING DATA — READ BEFORE CITING ANY NUMBER FROM THIS
    ---------------------------------------------------------------
    `trained_on` is declared empty because NVIDIA does not publish this
    model's training corpus, NOT because it has been verified clean. Every
    other detector in this registry declares its known contamination and the
    harness excludes those sources. That protection cannot be applied here:
    if NemoGuard was fitted on data overlapping our suite, its score is
    inflated by memorisation and we have no way to detect or exclude it. Any
    comparison that cites this detector must state that asymmetry — we hold
    ourselves to a contamination standard we cannot hold NVIDIA to.
    """

    name = "nemoguard_jailbreak"
    # JAILBREAK ONLY, and this is measured rather than assumed. An initial
    # version of this class also declared prompt_injection; a 300-row sample
    # put it at AUC 1.000 on jailbreak but 0.402 on injection — WORSE than
    # random, i.e. it scores injections lower than benign text. NVIDIA's own
    # naming ("JailbreakDetect") agrees. Declaring a target a detector
    # demonstrably does not serve would corrupt the per-class comparison and
    # the polarity probe that depends on `targets`.
    targets = (CLASS_JAILBREAK,)
    trained_on = ()  # UNKNOWN, not verified clean — see class docstring
    description = ("NVIDIA NeMo Guardrails model-based jailbreak rail "
                   "(Snowflake embed + NemoGuard RF; training data undisclosed)")

    def __init__(self, model_dir=None):
        self._model_dir = model_dir or os.path.join(".hf_cache", "nemoguard")
        self._classifier = None
        self._load_error = None
        self._lock = threading.Lock()

    def _load(self):
        if self._classifier is not None or self._load_error is not None:
            return
        with self._lock:
            if self._classifier is not None or self._load_error is not None:
                return
            try:
                from onnxruntime import InferenceSession
                from transformers import AutoModel, AutoTokenizer

                from nemoguardrails.library.jailbreak_detection.model_based.checks import (
                    _ensure_model_downloaded,
                )
                from nemoguardrails.library.jailbreak_detection.model_based.models import (
                    SNOWFLAKE_MODEL_ID,
                    JailbreakClassifier,
                    SnowflakeEmbed,
                )

                logger.info(
                    f"Loading NemoGuard JailbreakDetect into {self._model_dir} "
                    f"(downloads the ONNX classifier and Snowflake embedder on first use)."
                )
                model_path = _ensure_model_downloaded(self._model_dir)

                # WORKAROUND for a real bug in nemoguardrails 0.23.0, not ours.
                # SnowflakeEmbed.__init__ calls AutoModel.from_pretrained with
                # `use_safetensors=True`, but the arctic-embed remote code it
                # depends on (modeling_hf_nomic_bert.py) reads a DIFFERENT
                # kwarg — `safe_serialization`, defaulting to False — so it
                # looks for pytorch_model.bin. That file does not exist in the
                # Snowflake repo, which ships only model.safetensors, so the
                # load dies with a misleading "Model name ... was not found".
                #
                # We bypass the broken __init__ and populate the objects
                # directly, passing the kwarg the remote code actually reads.
                # NVIDIA's __call__ logic — embedding, ONNX random-forest
                # inference, and the signed-score convention — is left exactly
                # as shipped, so this benchmarks THEIR classifier rather than a
                # reimplementation of it. Only the weight-loading call is fixed.
                embed = SnowflakeEmbed.__new__(SnowflakeEmbed)
                embed.device = "cpu"
                embed.tokenizer = AutoTokenizer.from_pretrained(
                    SNOWFLAKE_MODEL_ID, trust_remote_code=True
                )
                embed.model = AutoModel.from_pretrained(
                    SNOWFLAKE_MODEL_ID,
                    trust_remote_code=True,
                    add_pooling_layer=False,
                    safe_serialization=True,
                )
                embed.model.to(embed.device)
                embed.model.eval()

                classifier = JailbreakClassifier.__new__(JailbreakClassifier)
                classifier.embed = embed
                classifier.classifier = InferenceSession(
                    str(model_path), providers=["CPUExecutionProvider"]
                )

                self._classifier = classifier
                logger.info("Detector 'nemoguard_jailbreak' ready.")
            except Exception as e:
                self._load_error = f"{type(e).__name__}: {str(e)[:200]}"
                logger.warning(f"Detector '{self.name}' unavailable: {self._load_error}")

    def available(self):
        self._load()
        if self._classifier is None:
            return False, self._load_error or "not loaded"
        return True, "loaded NemoGuard JailbreakDetect (Snowflake embed + ONNX RF)"

    def score_batch(self, texts):
        """
        Returns P(jailbreak) in [0, 1].

        NeMo's classifier returns a SIGNED score: `+prob` when it predicts
        jailbreak, `-prob` when it predicts benign (where `prob` is the
        random forest's confidence in whichever class it chose). Mapping to a
        one-sided probability so it is comparable with every other detector
        here: a positive score is already P(jailbreak); a negative score
        carries P(benign), so P(jailbreak) = 1 - P(benign) = 1 + score.
        """
        self._load()
        if self._classifier is None:
            raise RuntimeError(f"detector '{self.name}' unavailable: {self._load_error}")

        out = []
        for text in texts:
            # The embedder tokenizes with padding=True; an empty string yields
            # a degenerate batch, so substitute a single space as elsewhere.
            _, signed = self._classifier(text if text.strip() else " ")
            out.append(float(signed) if signed > 0 else float(1.0 + signed))
        return out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def _build_registry():
    return {
        "anchors": AnchorDetector(),

        # --- Prompt injection ---
        "protectai_injection": TransformerDetector(
            name="protectai_injection",
            model_id="protectai/deberta-v3-base-prompt-injection-v2",
            positive_labels=["injection"],
            targets=(CLASS_INJECTION,),
            description="ProtectAI DeBERTa-v3 injection classifier (widely used baseline)",
        ),
        "deepset_injection": TransformerDetector(
            name="deepset_injection",
            model_id="deepset/deberta-v3-base-injection",
            positive_labels=["injection"],
            targets=(CLASS_INJECTION,),
            trained_on=("deepset/prompt-injections",),
            description="deepset DeBERTa-v3 injection classifier",
        ),

        # --- Jailbreak ---
        "jailbreak_classifier": TransformerDetector(
            name="jailbreak_classifier",
            model_id="jackhhao/jailbreak-classifier",
            positive_labels=["jailbreak"],
            targets=(CLASS_JAILBREAK,),
            trained_on=("jackhhao/jailbreak-classification",),
            description="BERT jailbreak classifier (CONTAMINATED against its own source)",
        ),
        "madhurjindal_jailbreak": TransformerDetector(
            name="madhurjindal_jailbreak",
            model_id="madhurjindal/Jailbreak-Detector",
            positive_labels=["jailbreak"],
            targets=(CLASS_JAILBREAK,),
            description="DistilBERT jailbreak detector",
        ),

        # --- Harmful content / toxicity ---
        # The class the anchor detector is measurably worst at (24.4%).
        "toxic_bert": TransformerDetector(
            name="toxic_bert",
            model_id="unitary/toxic-bert",
            positive_labels=["toxic", "severe_toxic", "threat", "identity_hate", "obscene", "insult"],
            targets=(CLASS_HARMFUL,),
            multi_label=True,
            description="Unitary toxic-bert, multi-label toxicity",
        ),

        # --- Established vendor framework, open licence (benchmark target) ---
        # The only major commercial guardrail product that can be benchmarked
        # honestly on our own suite; Lakera and Azure are closed APIs. See the
        # class docstring for why the model-based rail is used rather than
        # NeMo's perplexity heuristics.
        "nemoguard_jailbreak": NemoGuardJailbreakDetector(),

        # --- Harmful content via safety taxonomy (GATED) ---
        # The intended fix for the class every other detector misses.
        "llama_guard_3_1b": LlamaGuardDetector(
            name="llama_guard_3_1b",
            model_id="meta-llama/Llama-Guard-3-1B",
            targets=(CLASS_HARMFUL, CLASS_JAILBREAK),
            dtype="bfloat16",
            min_free_gb=2.3,
            description="Meta Llama Guard 3 1B, hazard taxonomy (GATED; ~2.5GB at bf16)",
        ),
        # Registered for completeness. Needs roughly 17GB at bfloat16 and will
        # refuse to load on a 12GB machine — by design, with a clear message
        # rather than an OOM kill mid-run.
        "llama_guard_3_8b": LlamaGuardDetector(
            name="llama_guard_3_8b",
            model_id="meta-llama/Llama-Guard-3-8B",
            targets=(CLASS_HARMFUL, CLASS_JAILBREAK),
            dtype="bfloat16",
            min_free_gb=18.0,
            description="Meta Llama Guard 3 8B (GATED; needs ~17GB RAM — not "
                        "runnable on typical laptops)",
        ),

        # --- Gated: require `huggingface-cli login` + accepting Meta's licence ---
        "prompt_guard_2": TransformerDetector(
            name="prompt_guard_2",
            model_id="meta-llama/Llama-Prompt-Guard-2-86M",
            positive_labels=["label_1", "malicious", "injection", "jailbreak"],
            targets=(CLASS_INJECTION, CLASS_JAILBREAK),
            description="Meta Prompt Guard 2 86M (GATED: needs HF auth + licence acceptance)",
        ),
        "prompt_guard_1": TransformerDetector(
            name="prompt_guard_1",
            model_id="meta-llama/Prompt-Guard-86M",
            positive_labels=["injection", "jailbreak"],
            targets=(CLASS_INJECTION, CLASS_JAILBREAK),
            description="Meta Prompt Guard 86M (GATED: needs HF auth + licence acceptance)",
        ),
    }


_REGISTRY = None


def get_registry():
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    return _REGISTRY


def get_detector(name):
    reg = get_registry()
    if name not in reg:
        raise KeyError(f"unknown detector '{name}'; available: {sorted(reg)}")
    return reg[name]


def available_detectors(names=None):
    """
    Returns (available, unavailable) where each entry is (name, detail).
    Loading a detector is expensive, so this is the only place that probes.
    """
    reg = get_registry()
    names = names or list(reg)
    ok, bad = [], []
    for name in names:
        detector = reg[name]
        is_ok, detail = detector.available()
        (ok if is_ok else bad).append((name, detail))
    return ok, bad


def offline_mode() -> bool:
    """True when HF downloads are disabled, so failures can be reported honestly."""
    return os.environ.get("HF_HUB_OFFLINE", "").strip() in {"1", "true", "True"}
