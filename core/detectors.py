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
        from core.updates import check_dynamic_threats
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
                float(check_dynamic_threats(vec)),
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
        self.positive_labels = {l.lower() for l in positive_labels}
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
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                # Memory precheck. A 1B model at bfloat16 needs ~2.5GB resident;
                # float32 doubles that. Failing here with a clear message beats
                # an OOM kill halfway through a scoring run.
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

    def _build_inputs(self, texts):
        """
        Applies the Llama Guard chat template to each text.

        Falls back to a minimal hand-built prompt if the tokenizer ships no
        template, so a template change upstream degrades rather than crashes.
        """
        rendered = []
        for text in texts:
            chat = [{"role": "user", "content": text}]
            try:
                rendered.append(
                    self._tokenizer.apply_chat_template(chat, tokenize=False)
                )
            except Exception:
                rendered.append(
                    "<|begin_of_text|>Task: Check if there is unsafe content in "
                    "the user message.\n\n<BEGIN CONVERSATION>\n\n"
                    f"User: {text}\n\n<END CONVERSATION>\n\n"
                    "Provide your safety assessment. First line must be 'safe' "
                    "or 'unsafe'.\n"
                )
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
            self._build_inputs([text]),
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

        # --- Harmful content via safety taxonomy (GATED) ---
        # The intended fix for the class every other detector misses.
        "llama_guard_3_1b": LlamaGuardDetector(
            name="llama_guard_3_1b",
            model_id="meta-llama/Llama-Guard-3-1B",
            targets=(CLASS_HARMFUL, CLASS_JAILBREAK),
            dtype="bfloat16",
            min_free_gb=3.0,
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
