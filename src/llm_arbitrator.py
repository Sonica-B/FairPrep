"""
Phase 4: LLM as a Fairness Arbitrator (The AI Method)
======================================================
Instead of asking the LLM for a binary answer, use a Multi-Perspective
Prompt for flagged pairs identified by the Behavioral Gatekeeper.

Prompt Strategy:
  - Standard: "Are these tables unionable?"
  - Fairness-Aware: "A Non-Native English speaker found these tables
    ambiguous. Could the column headers be interpreted differently in
    a non-Western context? Validating the minority perspective."

This module provides the framework and prompt templates. Actual LLM
inference requires a model backend (Llama-3.3, Qwen, or API-based).

References:
  - CultureLLM (Li et al., NeurIPS 2024)
  - DEM-MoE (Xu et al., 2025)
  - Trust or Escalate (ICLR 2025)
"""

from dataclasses import dataclass


@dataclass
class ArbitrationRequest:
    """A request for LLM fairness arbitration on a flagged table pair."""
    question_id: str
    table_description: str
    minority_group: str
    behavioral_signal: str
    majority_decision: int
    minority_decision: int


PROMPT_TEMPLATES = {
    "standard": (
        "Given the following two tables, determine if they are unionable "
        "(i.e., they share compatible schemas and can be meaningfully combined).\n\n"
        "Table Pair Description:\n{table_description}\n\n"
        "Answer with: UNIONABLE or NOT_UNIONABLE, and explain your reasoning."
    ),
    "fairness_aware": (
        "A {minority_group} annotator found the following table pair ambiguous "
        "(behavioral signals: {behavioral_signal}). The majority group decided: "
        "{majority_decision_text}.\n\n"
        "Table Pair Description:\n{table_description}\n\n"
        "Consider whether the column headers or data values could be interpreted "
        "differently from a diverse cultural, linguistic, or domain perspective. "
        "Is the minority annotator's interpretation ({minority_decision_text}) "
        "a valid alternative rather than simply an error?\n\n"
        "Respond with:\n"
        "1. Your unionability assessment: UNIONABLE / NOT_UNIONABLE\n"
        "2. Is the minority perspective valid? YES / NO\n"
        "3. Confidence: HIGH / MEDIUM / LOW\n"
        "4. Brief reasoning."
    ),
    "multi_perspective": (
        "You are evaluating whether two tables can be unioned. Different annotators "
        "have disagreed on this pair.\n\n"
        "Table Pair Description:\n{table_description}\n\n"
        "Perspective A ({group_a_name}, {n_group_a} annotators): {group_a_decision_text}\n"
        "Perspective B ({group_b_name}, {n_group_b} annotators): {group_b_decision_text}\n\n"
        "Analyze both perspectives. Consider:\n"
        "- Could schema differences be interpreted differently by domain experts vs. non-experts?\n"
        "- Could language-dependent column names favor native English speakers?\n"
        "- Is there genuine semantic ambiguity that makes both interpretations valid?\n\n"
        "Respond with:\n"
        "1. Your assessment: UNIONABLE / NOT_UNIONABLE / AMBIGUOUS\n"
        "2. Which perspective(s) are valid: A / B / BOTH\n"
        "3. Confidence: HIGH / MEDIUM / LOW\n"
        "4. Reasoning."
    ),
}


@dataclass
class ArbitrationResult:
    """Result from LLM fairness arbitration."""
    question_id: str
    llm_decision: str            # UNIONABLE / NOT_UNIONABLE / AMBIGUOUS
    minority_valid: bool         # Is the minority perspective valid?
    confidence: str              # HIGH / MEDIUM / LOW
    reasoning: str
    prompt_type: str             # Which prompt template was used


class LLMFairnessArbitrator:
    """
    Framework for LLM-based fairness arbitration on flagged table pairs.
    This is a stub/framework — actual inference requires connecting to
    an LLM backend (e.g., Llama-3.3, Qwen, or an API).
    """

    def __init__(self, model_name: str = "placeholder"):
        self.model_name = model_name

    def build_prompt(
        self,
        request: ArbitrationRequest,
        prompt_type: str = "fairness_aware",
    ) -> str:
        """Build a prompt from a template and arbitration request."""
        template = PROMPT_TEMPLATES[prompt_type]

        decision_map = {1: "UNIONABLE", 0: "NOT_UNIONABLE"}

        return template.format(
            table_description=request.table_description,
            minority_group=request.minority_group,
            behavioral_signal=request.behavioral_signal,
            majority_decision_text=decision_map.get(request.majority_decision, "UNKNOWN"),
            minority_decision_text=decision_map.get(request.minority_decision, "UNKNOWN"),
        )

    def arbitrate(self, request: ArbitrationRequest, prompt_type: str = "fairness_aware") -> ArbitrationResult:
        """
        Stub: In production, this sends the prompt to an LLM and parses response.
        Currently returns a placeholder result for framework testing.
        """
        prompt = self.build_prompt(request, prompt_type)

        # Placeholder — replace with actual LLM inference
        return ArbitrationResult(
            question_id=request.question_id,
            llm_decision="AMBIGUOUS",
            minority_valid=True,
            confidence="MEDIUM",
            reasoning=f"[STUB] Prompt generated ({len(prompt)} chars). "
                      f"Connect to {self.model_name} for actual inference.",
            prompt_type=prompt_type,
        )

    def batch_arbitrate(
        self,
        requests: list,
        prompt_type: str = "fairness_aware",
    ) -> list:
        """Arbitrate a batch of flagged table pairs."""
        return [self.arbitrate(req, prompt_type) for req in requests]
