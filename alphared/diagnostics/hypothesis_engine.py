"""
HypothesisEngine -- Generates experiment hypotheses via Groq LLM API.

AutoResearch alignment:
  - Agent reads actual model code, proposes a concrete code change
  - LLM returns: title + description + actual unified diff (code patch)
  - Static fallback used when no API key (testing / offline)
"""

import json
import os
import requests
from dataclasses import dataclass, field
from typing import Optional
from .diagnostics_engine import ProblemType, DiagnosisResult


@dataclass
class Hypothesis:
    id:             str
    problem:        str
    title:          str
    description:    str
    code_change:    dict    # structured params for static mode
    code_patch:     str     # actual unified diff / code snippet (LLM mode)
    priority:       int
    estimated_gain: str
    raw_llm_output: str = ""

    def to_dict(self) -> dict:
        return {
            "id":             self.id,
            "problem":        self.problem,
            "title":          self.title,
            "description":    self.description,
            "code_change":    self.code_change,
            "code_patch":     self.code_patch,
            "priority":       self.priority,
            "estimated_gain": self.estimated_gain,
        }


# ---------------------------------------------------------------------------
# Static fallback library (offline / test mode)
# ---------------------------------------------------------------------------
_STATIC_LIBRARY = {
    ProblemType.NAN_LOSS: [
        {"title": "Reduce learning rate 10x",
         "description": "NaN loss is often caused by LR too high. 10x reduction stabilizes training.",
         "code_change": {"learning_rate": {"action": "multiply", "factor": 0.1}},
         "priority": 1, "estimated_gain": "Stabilizes training immediately"},
        {"title": "Add gradient clipping (max_norm=1.0)",
         "description": "Clips exploding gradients before they produce NaN.",
         "code_change": {"gradient_clipping": {"action": "add", "max_norm": 1.0}},
         "priority": 2, "estimated_gain": "Prevents gradient explosion"},
        {"title": "Normalize input data",
         "description": "Large input values can produce NaN. Normalize to zero mean unit variance.",
         "code_change": {"data_normalization": {"action": "add", "type": "standardize"}},
         "priority": 3, "estimated_gain": "Eliminates NaN from bad inputs"},
    ],
    ProblemType.OVERFITTING: [
        {"title": "Add Dropout(0.3) after hidden layers",
         "description": "Dropout randomly disables neurons -- prevents memorization.",
         "code_change": {"dropout": {"action": "add", "rate": 0.3, "position": "after_hidden"}},
         "priority": 1, "estimated_gain": "2-4% val_accuracy improvement"},
        {"title": "Add L2 regularization (weight_decay=0.01)",
         "description": "Penalizes large weights -- improves generalization.",
         "code_change": {"weight_decay": {"action": "set", "value": 0.01}},
         "priority": 2, "estimated_gain": "1-3% val_accuracy improvement"},
        {"title": "Increase Dropout to 0.5",
         "description": "Stronger regularization when 0.3 was insufficient.",
         "code_change": {"dropout": {"action": "update", "rate": 0.5}},
         "priority": 3, "estimated_gain": "Stronger regularization"},
        {"title": "Add data augmentation",
         "description": "Artificially increases training diversity.",
         "code_change": {"data_augmentation": {"action": "add", "transforms": ["random_flip", "random_crop"]}},
         "priority": 4, "estimated_gain": "3-7% val_accuracy improvement"},
        {"title": "Reduce model size (halve hidden units)",
         "description": "Smaller model has less capacity to overfit.",
         "code_change": {"hidden_size": {"action": "multiply", "factor": 0.5}},
         "priority": 5, "estimated_gain": "Reduces overfitting capacity"},
    ],
    ProblemType.UNDERFITTING: [
        {"title": "Increase hidden layer size 2x",
         "description": "More capacity to learn complex patterns.",
         "code_change": {"hidden_size": {"action": "multiply", "factor": 2.0}},
         "priority": 1, "estimated_gain": "3-6% accuracy improvement"},
        {"title": "Add one more hidden layer",
         "description": "Deeper network learns more abstract features.",
         "code_change": {"num_layers": {"action": "add", "count": 1}},
         "priority": 2, "estimated_gain": "4-8% accuracy improvement"},
        {"title": "Increase learning rate 3x",
         "description": "LR too low causes slow or stuck learning.",
         "code_change": {"learning_rate": {"action": "multiply", "factor": 3.0}},
         "priority": 3, "estimated_gain": "Faster convergence"},
        {"title": "Train for 2x more epochs",
         "description": "Model simply needs more training time.",
         "code_change": {"epochs": {"action": "multiply", "factor": 2.0}},
         "priority": 4, "estimated_gain": "More training time"},
        {"title": "Remove regularization (weight_decay=0)",
         "description": "Underfitting model should not be constrained by regularization.",
         "code_change": {"weight_decay": {"action": "set", "value": 0.0}},
         "priority": 5, "estimated_gain": "Removes unnecessary constraint"},
    ],
    ProblemType.PLATEAU: [
        {"title": "Add ReduceLROnPlateau scheduler",
         "description": "Automatically reduces LR when loss stops improving.",
         "code_change": {"lr_scheduler": {"action": "add", "type": "ReduceLROnPlateau", "factor": 0.5, "patience": 3}},
         "priority": 1, "estimated_gain": "Often breaks plateaus immediately"},
        {"title": "Switch optimizer to AdamW",
         "description": "AdamW uses adaptive learning rates -- better at escaping plateaus.",
         "code_change": {"optimizer": {"action": "change", "to": "AdamW", "lr": 0.001, "weight_decay": 0.01}},
         "priority": 2, "estimated_gain": "2-5% improvement after plateau break"},
        {"title": "Add CosineAnnealingLR scheduler",
         "description": "Cyclically varies LR -- helps escape local minima.",
         "code_change": {"lr_scheduler": {"action": "add", "type": "CosineAnnealingLR", "T_max": 10}},
         "priority": 3, "estimated_gain": "Helps escape local minima"},
        {"title": "Add BatchNormalization layers",
         "description": "Stabilizes training -- often breaks plateaus.",
         "code_change": {"batch_norm": {"action": "add", "position": "after_linear"}},
         "priority": 4, "estimated_gain": "1-3% improvement + faster convergence"},
    ],
    ProblemType.EXPLODING_GRADIENT: [
        {"title": "Add gradient clipping (max_norm=1.0)",
         "description": "Clips gradient norm to 1.0 -- prevents explosion.",
         "code_change": {"gradient_clipping": {"action": "add", "max_norm": 1.0}},
         "priority": 1, "estimated_gain": "Immediately stabilizes training"},
        {"title": "Reduce learning rate 5x",
         "description": "High LR causes exploding gradients.",
         "code_change": {"learning_rate": {"action": "multiply", "factor": 0.2}},
         "priority": 2, "estimated_gain": "Reduces gradient magnitude"},
        {"title": "Add gradient clipping (max_norm=0.5)",
         "description": "More aggressive clipping when 1.0 was insufficient.",
         "code_change": {"gradient_clipping": {"action": "update", "max_norm": 0.5}},
         "priority": 3, "estimated_gain": "Stronger gradient control"},
    ],
}

# OpenRouter -- routes to best available model
OPENROUTER_MODEL   = "meta-llama/llama-3.3-70b-instruct"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
openrouter_api_key="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"

class HypothesisEngine:
    """
    Two modes:

    LLM mode (openrouter_api_key set):
      - Reads actual model code
      - Sends to Groq LLM with diagnosis + history
      - LLM returns title + description + code_patch (actual code changes)
      - Same as AutoResearch: agent reads code, proposes concrete edits

    Static mode (no key):
      - Uses _STATIC_LIBRARY
      - For testing / offline use
    """

    def __init__(self, openrouter_api_key: str = ""):
        self._api_key  = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self._tried:   list[str] = []
        self._counter: int = 0

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        diagnosis:           DiagnosisResult,
        model_type:          str        = "deep_learning",
        top_n:               int        = 1,
        experiment_history:  list[dict] = None,
        model_code:          str        = "",
    ) -> list[Hypothesis]:
        if diagnosis.problem == ProblemType.HEALTHY:
            return []
        if self._api_key:
            return self._generate_llm(diagnosis, model_type, top_n,
                                      experiment_history or [], model_code)
        return self._generate_static(diagnosis, model_type, top_n)

    def mark_tried(self, hypothesis: Hypothesis, improved: bool):
        self._tried.append(hypothesis.title)

    def remaining_count(self, problem: ProblemType) -> int:
        if self._api_key:
            return 999  # LLM never exhausts
        candidates = _STATIC_LIBRARY.get(problem, [])
        tried      = set(self._tried)
        return sum(1 for c in candidates if c["title"] not in tried)

    def all_exhausted(self, problem: ProblemType) -> bool:
        if self._api_key:
            return False
        return self.remaining_count(problem) == 0

    # ------------------------------------------------------------------ #
    #  Groq LLM mode                                                       #
    # ------------------------------------------------------------------ #

    def _generate_llm(
        self,
        diagnosis:           DiagnosisResult,
        model_type:          str,
        top_n:               int,
        experiment_history:  list[dict],
        model_code:          str,
    ) -> list[Hypothesis]:
        history_str  = self._format_history(experiment_history)
        code_snippet = model_code[:4000] if model_code else "(model code not provided)"

        system_prompt = (
            "You are an autonomous ML research agent -- like AutoResearch by Karpathy. "
            "Your job is to read the model code, understand the current problem, "
            "and propose ONE concrete improvement that has not been tried yet. "
            "You think deeply about the code and suggest meaningful changes: "
            "architecture, optimizer, hyperparameters, regularization, batch size, etc. "
            "Respond ONLY with valid JSON -- no markdown fences, no explanation outside JSON."
        )

        user_prompt = f"""## Current training problem
{diagnosis.message}
Severity: {diagnosis.severity}  |  Confidence: {diagnosis.confidence:.0%}

## Model type
{model_type}

## Already tried experiments (do NOT repeat these)
{history_str if history_str else "None yet -- this is the first experiment."}

## Current model code
```python
{code_snippet}
```

## Simplicity criterion (from AutoResearch)
All else being equal, simpler is better. A small improvement that adds ugly complexity
is not worth it. An improvement from deleting code is a win. Weigh complexity cost
against improvement magnitude.

## Task
Suggest exactly ONE hypothesis that modifies the model code above.
Return ONLY this JSON structure:
{{
  "title": "short description max 60 chars",
  "description": "why this change should help (2-3 sentences)",
  "code_patch": "the actual Python code change -- show the modified lines or new function. Be specific with real values.",
  "estimated_gain": "expected improvement e.g. 2-4% accuracy or faster convergence"
}}"""

        try:
            response = requests.post(
                OPENROUTER_API_URL,
                headers={
                    "Authorization":  f"Bearer {self._api_key}",
                    "Content-Type":   "application/json",
                    "HTTP-Referer":   "https://alphared.ai",
                    "X-Title":        "AlphaRed Agent",
                },
                json={
                    "model":       OPENROUTER_MODEL,
                    "max_tokens":  600,
                    "temperature": 0.7,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                },
                timeout=30,
            )
            response.raise_for_status()
            raw_text = response.json()["choices"][0]["message"]["content"].strip()

            # Strip markdown fences if Groq wraps in ```json ... ```
            clean = raw_text
            if clean.startswith("```"):
                parts = clean.split("```")
                # parts[1] is content between first ``` pair
                clean = parts[1].lstrip("json").strip() if len(parts) > 1 else clean

            parsed = json.loads(clean)
            self._counter += 1

            hyp = Hypothesis(
                id             = f"hyp_{self._counter:03d}",
                problem        = diagnosis.problem.value,
                title          = parsed.get("title", "Groq suggestion"),
                description    = parsed.get("description", ""),
                code_change    = {"patch": "see code_patch field"},
                code_patch     = parsed.get("code_patch", ""),
                priority       = 1,
                estimated_gain = parsed.get("estimated_gain", "unknown"),
                raw_llm_output = raw_text,
            )
            return [hyp]

        except Exception as e:
            print(f"[HypothesisEngine] OpenRouter call failed ({e}), using static fallback")
            return self._generate_static(diagnosis, model_type, top_n)

    def _format_history(self, history: list[dict]) -> str:
        if not history:
            return ""
        lines = []
        for h in history[-10:]:
            status = "COMMITTED" if h.get("improved") else "REVERTED"
            lines.append(
                f"- [{status}] {h.get('title', '?')}  delta={h.get('improvement_delta', 0):+.4f}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Static fallback                                                     #
    # ------------------------------------------------------------------ #

    def _generate_static(
        self,
        diagnosis:  DiagnosisResult,
        model_type: str,
        top_n:      int,
    ) -> list[Hypothesis]:
        candidates = list(_STATIC_LIBRARY.get(diagnosis.problem, []))

        if model_type == "traditional_ml":
            candidates = [
                c for c in candidates
                if not any(kw in str(c["code_change"]).lower()
                           for kw in ["dropout", "batch_norm", "gradient_clipping", "hidden_size"])
            ]

        tried     = set(self._tried)
        results   = []

        for candidate in candidates:
            if candidate["title"] in tried:
                continue
            self._counter += 1
            hyp = Hypothesis(
                id             = f"hyp_{self._counter:03d}",
                problem        = diagnosis.problem.value,
                title          = candidate["title"],
                description    = candidate["description"],
                code_change    = candidate["code_change"],
                code_patch     = "",   # no patch in static mode
                priority       = candidate["priority"],
                estimated_gain = candidate["estimated_gain"],
            )
            results.append(hyp)
            if len(results) >= top_n:
                break

        return results