#!/usr/bin/env python3
"""Easy, reproducible experiment for the paper:
"Detecting Suboptimal Sub-Tasks in Agentic AI Loops via Multi-Level Hashing
and Semantic Analysis."

The script intentionally runs without an LLM or external service. It creates
controlled LangGraph-like trajectories, applies five detector levels, performs
baseline and ablation comparisons, and exports paper-ready statistics.

For the final paper, replace the TF-IDF semantic backend with a Sentence
Transformer + FAISS and replace the heuristic judge with a validated LLM judge.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, TypedDict

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

try:
    from langgraph.graph import END, START, StateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False


SEMANTIC_BACKEND = "tfidf"
_SENTENCE_MODEL = None

LABELS = [
    "repeated_action",
    "redundant_search_loop",
    "semantic_repeat",
    "repeated_error_response",
    "hallucination",
]


@dataclass
class Step:
    run_id: str
    scenario: str
    step_index: int
    agent_id: str
    question: str
    subgoal: str
    action: str
    tool_name: str
    tool_args: dict[str, Any]
    observation: str
    response: str
    evidence: str
    progress: float
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cost_usd: float
    ground_truth_labels: list[str] = field(default_factory=list)
    anomaly_onset_step: int | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class Detection:
    detected: bool
    label: str = ""
    level: int = 0
    score: float = 0.0
    threshold: float = 0.0
    matched_step: int | None = None
    detector_latency_ms: float = 0.0


@dataclass
class RunResult:
    run_id: str
    scenario: str
    truth_label: str
    truth_positive: bool
    predicted_label: str
    predicted_positive: bool
    detected_level: int
    detection_step: int | None
    detection_delay: int | None
    baseline_tokens: int
    protected_tokens: int
    baseline_tool_calls: int
    protected_tool_calls: int
    baseline_latency_ms: float
    protected_latency_ms: float
    baseline_cost_usd: float
    protected_cost_usd: float
    detector_overhead_ms: float
    detector_cost_usd: float
    baseline_success: int
    protected_success: int


class GraphState(TypedDict, total=False):
    steps: list[Step]
    variant: str
    result: RunResult


VOLATILE_KEYS = {"timestamp", "request_id", "run_id", "trace_id", "span_id"}


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"https?://\S+", "<url>", text)
    text = re.sub(r"\b[0-9a-f]{8,}\b", "<id>", text)
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_obj(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(k): normalize_obj(v)
            for k, v in sorted(value.items())
            if str(k).lower() not in VOLATILE_KEYS
        }
    if isinstance(value, list):
        return [normalize_obj(v) for v in value]
    if isinstance(value, str):
        return normalize_text(value)
    return value


def canonical_action_payload(step: Step) -> str:
    payload = {
        "agent_id": step.agent_id,
        "question": normalize_text(step.question),
        "subgoal": normalize_text(step.subgoal),
        "action": normalize_text(step.action),
        "tool_name": normalize_text(step.tool_name),
        "tool_args": normalize_obj(step.tool_args),
        "observation": normalize_text(step.observation),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def action_text(step: Step) -> str:
    return " | ".join(
        [
            normalize_text(step.subgoal),
            normalize_text(step.action),
            normalize_text(step.tool_name),
            json.dumps(normalize_obj(step.tool_args), sort_keys=True),
        ]
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize_text(text))


def simhash64(text: str) -> int:
    vector = [0] * 64
    counts = Counter(tokens(text))
    for token, weight in counts.items():
        digest = int(hashlib.md5(token.encode("utf-8")).hexdigest()[:16], 16)
        for bit in range(64):
            vector[bit] += weight if digest & (1 << bit) else -weight
    result = 0
    for bit, value in enumerate(vector):
        if value >= 0:
            result |= 1 << bit
    return result


def simhash_similarity(a: int, b: int) -> float:
    return 1.0 - ((a ^ b).bit_count() / 64.0)


def token_shingles(text: str, width: int = 2) -> set[str]:
    parts = tokens(text)
    if len(parts) < width:
        return set(parts)
    return {" ".join(parts[i : i + width]) for i in range(len(parts) - width + 1)}


def minhash_signature(text: str, permutations: int = 64) -> tuple[int, ...]:
    shingles = token_shingles(text) or {"<empty>"}
    signature = []
    for permutation in range(permutations):
        minimum = min(
            int(hashlib.sha1(f"{permutation}:{shingle}".encode("utf-8")).hexdigest()[:16], 16)
            for shingle in shingles
        )
        signature.append(minimum)
    return tuple(signature)


def minhash_similarity(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    return safe_div(sum(x == y for x, y in zip(a, b)), len(a))


def lsh_band_keys(signature: tuple[int, ...], bands: int = 8) -> set[str]:
    rows = max(1, len(signature) // bands)
    keys = set()
    for band in range(bands):
        start = band * rows
        chunk = signature[start : start + rows]
        keys.add(f"{band}:" + sha256_text(json.dumps(chunk)))
    return keys


def semantic_similarity(a: str, b: str) -> float:
    global _SENTENCE_MODEL
    if normalize_text(a) == normalize_text(b):
        return 1.0
    if SEMANTIC_BACKEND == "sentence-transformer":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Install sentence-transformers or use --semantic-backend tfidf"
            ) from exc
        if _SENTENCE_MODEL is None:
            _SENTENCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = _SENTENCE_MODEL.encode([a, b], normalize_embeddings=True)
        return float(np.dot(embeddings[0], embeddings[1]))

    # Laptop-friendly demonstration backend.
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    try:
        matrix = vectorizer.fit_transform([a, b])
    except ValueError:
        return 0.0
    return float(cosine_similarity(matrix[0], matrix[1])[0, 0])


def heuristic_hallucination(step: Step) -> float:
    """Return a demo unsupported-answer score in [0, 1].

    This is deliberately conservative and only handles the factual test pattern
    used by the benchmark. Replace it with a structured LLM judge for the paper.
    """
    evidence = normalize_text(step.evidence)
    response = normalize_text(step.response)
    if not evidence or not response:
        return 0.0
    factual_answers = {"paris", "berlin", "munich", "london", "rome"}
    response_answers = factual_answers.intersection(tokens(response))
    evidence_answers = factual_answers.intersection(tokens(evidence))
    if response_answers and not response_answers.issubset(evidence_answers):
        return 1.0
    return 0.0


class Detector:
    def __init__(
        self,
        enabled_levels: set[int],
        repeat_count: int = 2,
        simhash_threshold: float = 0.80,
        semantic_threshold: float = 0.24,
        response_repeat_count: int = 2,
        judge_threshold: float = 0.80,
        window: int = 4,
    ) -> None:
        self.enabled_levels = enabled_levels
        self.repeat_count = repeat_count
        self.simhash_threshold = simhash_threshold
        self.semantic_threshold = semantic_threshold
        self.response_repeat_count = response_repeat_count
        self.judge_threshold = judge_threshold
        self.window = window
        self.action_hashes: list[str] = []
        self.action_texts: list[str] = []
        self.simhashes: list[int] = []
        self.minhashes: list[tuple[int, ...]] = []
        self.lsh_keys: list[set[str]] = []
        self.response_hashes: list[str] = []
        self.progress_history: list[float] = []

    def check(self, step: Step) -> Detection:
        start = time.perf_counter_ns()
        action_payload = canonical_action_payload(step)
        action_hash = sha256_text(action_payload)
        current_text = action_text(step)
        current_simhash = simhash64(current_text)
        current_minhash = minhash_signature(current_text)
        current_lsh_keys = lsh_band_keys(current_minhash)
        response_hash = sha256_text(normalize_text(step.response))
        previous_progress = self.progress_history[-1] if self.progress_history else step.progress
        low_progress = (step.progress - previous_progress) <= 0.03

        detection = Detection(False)

        # Level 1: exact canonical repetition.
        if 1 in self.enabled_levels and low_progress:
            recent = self.action_hashes[-self.window :]
            occurrences = recent.count(action_hash)
            if occurrences >= self.repeat_count - 1:
                matched = max(i for i, value in enumerate(self.action_hashes) if value == action_hash)
                detection = Detection(True, "repeated_action", 1, 1.0, 1.0, matched)

        # Level 2: near-duplicate action/query using SimHash.
        if (
            not detection.detected
            and 2 in self.enabled_levels
            and low_progress
            and "search" in normalize_text(step.tool_name)
        ):
            recent_start = max(0, len(self.simhashes) - self.window)
            best_score = 0.0
            best_step = None
            for index in range(recent_start, len(self.simhashes)):
                # LSH is the candidate filter; SimHash/MinHash provide scores.
                lsh_candidate = bool(current_lsh_keys.intersection(self.lsh_keys[index]))
                sim_score = simhash_similarity(current_simhash, self.simhashes[index])
                min_score = minhash_similarity(current_minhash, self.minhashes[index])
                score = max(sim_score, min_score if lsh_candidate else 0.0)
                if score > best_score:
                    best_score, best_step = score, index
            if best_score >= self.simhash_threshold:
                detection = Detection(
                    True,
                    "redundant_search_loop",
                    2,
                    best_score,
                    self.simhash_threshold,
                    best_step,
                )

        # Level 3: paraphrased action/plan.
        if (
            not detection.detected
            and 3 in self.enabled_levels
            and low_progress
            and normalize_text(step.tool_name) == "planner"
        ):
            recent_start = max(0, len(self.action_texts) - self.window)
            best_score = 0.0
            best_step = None
            for index in range(recent_start, len(self.action_texts)):
                score = semantic_similarity(current_text, self.action_texts[index])
                if score > best_score:
                    best_score, best_step = score, index
            if best_score >= self.semantic_threshold:
                detection = Detection(
                    True,
                    "semantic_repeat",
                    3,
                    best_score,
                    self.semantic_threshold,
                    best_step,
                )

        # Level 4: repeated user-visible/tool error response.
        if not detection.detected and 4 in self.enabled_levels and low_progress:
            recent = self.response_hashes[-self.window :]
            occurrences = recent.count(response_hash)
            response_is_error = bool(
                re.search(r"\b(error|failed|unavailable|500|timeout)\b", normalize_text(step.response))
            )
            if occurrences >= self.response_repeat_count - 1 and response_is_error:
                matched = max(i for i, value in enumerate(self.response_hashes) if value == response_hash)
                detection = Detection(
                    True,
                    "repeated_error_response",
                    4,
                    1.0,
                    1.0,
                    matched,
                )

        # Level 5: demo judge.
        if not detection.detected and 5 in self.enabled_levels:
            score = heuristic_hallucination(step)
            if score >= self.judge_threshold:
                detection = Detection(
                    True,
                    "hallucination",
                    5,
                    score,
                    self.judge_threshold,
                    None,
                )

        self.action_hashes.append(action_hash)
        self.action_texts.append(current_text)
        self.simhashes.append(current_simhash)
        self.minhashes.append(current_minhash)
        self.lsh_keys.append(current_lsh_keys)
        self.response_hashes.append(response_hash)
        self.progress_history.append(step.progress)
        detection.detector_latency_ms = (time.perf_counter_ns() - start) / 1_000_000
        return detection


class RetryLimitDetector:
    """Simple baseline: stop after three uses of the same tool."""

    def __init__(self, limit: int = 3) -> None:
        self.limit = limit
        self.counts: Counter[str] = Counter()

    def check(self, step: Step) -> Detection:
        start = time.perf_counter_ns()
        self.counts[step.tool_name] += 1
        detected = self.counts[step.tool_name] >= self.limit
        return Detection(
            detected=detected,
            label="fixed_retry_limit" if detected else "",
            level=-1 if detected else 0,
            score=float(self.counts[step.tool_name]),
            threshold=float(self.limit),
            detector_latency_ms=(time.perf_counter_ns() - start) / 1_000_000,
        )


def variant_detector(variant: str):
    if variant == "retry_limit":
        return RetryLimitDetector(limit=3)
    enabled = {
        "exact_only": {1},
        "L1": {1},
        "L1_L2": {1, 2},
        "L1_L3": {1, 2, 3},
        "L1_L4": {1, 2, 3, 4},
        "L1_L5": {1, 2, 3, 4, 5},
        "full": {1, 2, 3, 4, 5},
    }[variant]
    return Detector(enabled_levels=enabled)


def baseline_success_for(scenario: str) -> int:
    return 1 if scenario.startswith("healthy") else 0


def process_run(steps: list[Step], variant: str) -> RunResult:
    detector = variant_detector(variant)
    detected: Detection | None = None
    detection_step: int | None = None
    overhead = 0.0

    for step in steps:
        current = detector.check(step)
        overhead += current.detector_latency_ms
        if current.detected:
            detected = current
            detection_step = step.step_index
            break

    truth_labels = set(label for step in steps for label in step.ground_truth_labels)
    truth_label = next(iter(truth_labels), "healthy")
    truth_positive = bool(truth_labels)
    predicted_positive = detected is not None
    predicted_label = detected.label if detected else "healthy"
    onset = next((step.anomaly_onset_step for step in steps if step.anomaly_onset_step is not None), None)
    delay = detection_step - onset if predicted_positive and onset is not None else None

    baseline_tokens = sum(s.total_tokens for s in steps)
    baseline_tools = sum(1 for s in steps if s.tool_name)
    baseline_latency = sum(s.latency_ms for s in steps)
    baseline_cost = sum(s.cost_usd for s in steps)

    if detection_step is None:
        consumed = steps
    else:
        consumed = [s for s in steps if s.step_index <= detection_step]

    # Small intervention cost: summarize, switch strategy, or escalate.
    intervention_tokens = 90 if detected else 0
    intervention_latency = 180.0 if detected else 0.0
    intervention_cost = 0.00018 if detected else 0.0
    protected_tokens = sum(s.total_tokens for s in consumed) + intervention_tokens
    protected_tools = sum(1 for s in consumed if s.tool_name)
    protected_latency = sum(s.latency_ms for s in consumed) + intervention_latency + overhead
    protected_cost = sum(s.cost_usd for s in consumed) + intervention_cost

    baseline_success = baseline_success_for(steps[0].scenario)
    if truth_positive and predicted_positive:
        protected_success = 1  # demo assumes a safe strategy switch succeeds
    elif not truth_positive and predicted_positive:
        protected_success = 0  # premature stop
    else:
        protected_success = baseline_success

    return RunResult(
        run_id=steps[0].run_id,
        scenario=steps[0].scenario,
        truth_label=truth_label,
        truth_positive=truth_positive,
        predicted_label=predicted_label,
        predicted_positive=predicted_positive,
        detected_level=detected.level if detected else 0,
        detection_step=detection_step,
        detection_delay=delay,
        baseline_tokens=baseline_tokens,
        protected_tokens=protected_tokens,
        baseline_tool_calls=baseline_tools,
        protected_tool_calls=protected_tools,
        baseline_latency_ms=baseline_latency,
        protected_latency_ms=protected_latency,
        baseline_cost_usd=baseline_cost,
        protected_cost_usd=protected_cost,
        detector_overhead_ms=overhead,
        detector_cost_usd=intervention_cost,
        baseline_success=baseline_success,
        protected_success=protected_success,
    )


def build_langgraph() -> Any:
    """Build the minimal LangGraph: observe -> evaluate -> diagnose/intervene."""
    if not LANGGRAPH_AVAILABLE:
        return None

    def observe(state: GraphState) -> GraphState:
        return state

    def evaluate(state: GraphState) -> GraphState:
        return {"result": process_run(state["steps"], state["variant"])}

    graph = StateGraph(GraphState)
    graph.add_node("observe", observe)
    graph.add_node("evaluate_and_intervene", evaluate)
    graph.add_edge(START, "observe")
    graph.add_edge("observe", "evaluate_and_intervene")
    graph.add_edge("evaluate_and_intervene", END)
    return graph.compile()


def execute_run(steps: list[Step], variant: str, graph: Any) -> RunResult:
    if graph is None:
        return process_run(steps, variant)
    output = graph.invoke({"steps": steps, "variant": variant})
    return output["result"]


def make_step(
    rng: random.Random,
    run_id: str,
    scenario: str,
    index: int,
    action: str,
    tool: str,
    args: dict[str, Any],
    observation: str,
    response: str,
    progress: float,
    truth: str = "",
    onset: int | None = None,
    evidence: str = "",
    subgoal: str = "Complete the assigned sub-task",
    question: str = "Complete the task efficiently and correctly",
) -> Step:
    prompt = rng.randint(260, 520)
    completion = rng.randint(55, 135)
    latency = rng.uniform(520, 1450)
    # Demonstration price, not tied to any current provider/model.
    cost = prompt * 0.0000015 + completion * 0.000006
    return Step(
        run_id=run_id,
        scenario=scenario,
        step_index=index,
        agent_id="research-agent",
        question=question,
        subgoal=subgoal,
        action=action,
        tool_name=tool,
        tool_args=args,
        observation=observation,
        response=response,
        evidence=evidence,
        progress=progress,
        prompt_tokens=prompt,
        completion_tokens=completion,
        latency_ms=latency,
        cost_usd=cost,
        ground_truth_labels=[truth] if truth else [],
        anomaly_onset_step=onset,
    )


def scenario_exact(rng: random.Random, run_id: str) -> list[Step]:
    steps = []
    for i in range(6):
        steps.append(
            make_step(
                rng,
                run_id,
                "exact_loop",
                i,
                "Search the document repository for the annual report",
                "search_docs",
                {"query": "ACME annual report 2025"},
                "Returned annual_report_2025.pdf",
                "I found the same annual report and will search again.",
                0.20,
                truth="repeated_action",
                onset=1,
                subgoal="Retrieve ACME's annual report",
            )
        )
    return steps


def scenario_near(rng: random.Random, run_id: str) -> list[Step]:
    queries = [
        "ACME Q4 earnings",
        "ACME fourth quarter earnings",
        "ACME Q4 financial results",
        "latest ACME fourth-quarter earnings",
        "ACME earnings report for Q4",
        "Q4 results released by ACME",
    ]
    return [
        make_step(
            rng,
            run_id,
            "near_duplicate_loop",
            i,
            f"Search news using query: {query}",
            "news_search",
            {"query": query},
            "The same two earnings articles were returned",
            "The query returned the same earnings coverage.",
            0.25,
            truth="redundant_search_loop",
            onset=1,
            subgoal="Find ACME Q4 earnings news",
        )
        for i, query in enumerate(queries)
    ]


def scenario_semantic(rng: random.Random, run_id: str) -> list[Step]:
    plans = [
        "Create a welcome email, guide the user through a product tour, then send a survey",
        "Start with an introductory message, provide an in-app walkthrough, and request feedback afterward",
        "Greet the customer, demonstrate the main product features, and collect opinions in a questionnaire",
        "Send an onboarding note, show a guided feature demonstration, and finish with a feedback form",
        "Introduce the service by email, lead a product walkthrough, and ask the new user for feedback",
        "Welcome the account owner, explain the product interactively, and follow up with a satisfaction survey",
    ]
    return [
        make_step(
            rng,
            run_id,
            "semantic_plan_loop",
            i,
            plan,
            "planner",
            {"plan_version": i},
            "Critic rejected the plan without actionable feedback",
            "I will generate another onboarding plan.",
            0.15,
            truth="semantic_repeat",
            onset=1,
            subgoal="Design a three-stage customer onboarding journey",
        )
        for i, plan in enumerate(plans)
    ]


def scenario_error(rng: random.Random, run_id: str) -> list[Step]:
    actions = [
        "Fetch the transcript directly",
        "Retry the transcript endpoint with a longer timeout",
        "Request transcript metadata first",
        "Use the archived transcript endpoint",
        "Ask the transcript proxy service",
        "Retry through the backup route",
    ]
    tools = ["fetch_transcript", "fetch_transcript", "metadata_api", "archive_api", "proxy_api", "backup_api"]
    return [
        make_step(
            rng,
            run_id,
            "repeated_error_loop",
            i,
            action,
            tools[i],
            {"video_id": "V-42", "attempt": i},
            "HTTP 500: transcript service unavailable",
            "Error: transcript service unavailable (HTTP 500).",
            0.10,
            truth="repeated_error_response",
            onset=1,
            subgoal="Retrieve a video transcript",
        )
        for i, action in enumerate(actions)
    ]


def scenario_hallucination(rng: random.Random, run_id: str) -> list[Step]:
    steps = [
        make_step(
            rng,
            run_id,
            "hallucination",
            0,
            "Retrieve evidence for the capital of France",
            "knowledge_lookup",
            {"country": "France"},
            "Evidence: The capital of France is Paris.",
            "I found a reference answer.",
            0.60,
            evidence="The capital of France is Paris.",
            question="What is the capital of France?",
            subgoal="Answer using retrieved evidence",
        ),
        make_step(
            rng,
            run_id,
            "hallucination",
            1,
            "Synthesize the factual answer",
            "answer_generator",
            {"use_evidence": True},
            "Evidence remains: The capital of France is Paris.",
            "The capital of France is Berlin.",
            0.60,
            truth="hallucination",
            onset=1,
            evidence="The capital of France is Paris.",
            question="What is the capital of France?",
            subgoal="Answer using retrieved evidence",
        ),
        make_step(
            rng,
            run_id,
            "hallucination",
            2,
            "Return the final answer",
            "finalizer",
            {},
            "No additional evidence was collected",
            "Berlin is the capital of France.",
            0.60,
            truth="hallucination",
            onset=1,
            evidence="The capital of France is Paris.",
            question="What is the capital of France?",
            subgoal="Answer using retrieved evidence",
        ),
    ]
    return steps


def scenario_healthy_code(rng: random.Random, run_id: str) -> list[Step]:
    failures = [8, 6, 4, 2, 1, 0]
    steps = []
    for i, remaining in enumerate(failures):
        progress = (8 - remaining) / 8
        steps.append(
            make_step(
                rng,
                run_id,
                "healthy_code_repair",
                i,
                f"Modify implementation based on test failures; {remaining} remain",
                "run_tests",
                {"suite": "unit", "iteration": i},
                f"Test run completed: {remaining} failures remain",
                f"Progress made; {remaining} tests still fail.",
                progress,
                subgoal="Repair code until all tests pass",
            )
        )
    return steps


def scenario_healthy_polling(rng: random.Random, run_id: str) -> list[Step]:
    statuses = ["queued", "running 20%", "running 55%", "running 90%", "completed"]
    return [
        make_step(
            rng,
            run_id,
            "healthy_async_polling",
            i,
            "Poll the asynchronous export job with backoff",
            "job_status",
            {"job_id": "J-100", "poll": i},
            f"Job status: {status}",
            f"The export job is {status}.",
            i / (len(statuses) - 1),
            subgoal="Wait for a legitimate asynchronous job to finish",
        )
        for i, status in enumerate(statuses)
    ]


SCENARIO_BUILDERS: list[Callable[[random.Random, str], list[Step]]] = [
    scenario_exact,
    scenario_near,
    scenario_semantic,
    scenario_error,
    scenario_hallucination,
    scenario_healthy_code,
    scenario_healthy_polling,
]


def generate_runs(repetitions: int, seed: int) -> list[list[Step]]:
    rng = random.Random(seed)
    runs: list[list[Step]] = []
    for repeat in range(repetitions):
        builders = list(SCENARIO_BUILDERS)
        rng.shuffle(builders)
        for builder in builders:
            scenario_name = builder.__name__.replace("scenario_", "")
            run_id = f"{scenario_name}-{repeat:03d}"
            runs.append(builder(rng, run_id))
    return runs


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def binary_metrics(results: list[RunResult]) -> dict[str, float]:
    y_true = np.array([int(r.truth_positive) for r in results])
    y_pred = np.array([int(r.predicted_positive) for r in results])
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    delays = [r.detection_delay for r in results if r.detection_delay is not None and r.truth_positive]
    healthy = max(1, sum(not r.truth_positive for r in results))
    return {
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_flags_per_100_healthy_runs": 100.0 * fp / healthy,
        "mean_detection_delay_steps": statistics.mean(delays) if delays else math.nan,
        "p95_detection_delay_steps": float(np.percentile(delays, 95)) if delays else math.nan,
        "mean_detector_overhead_ms_per_run": statistics.mean(r.detector_overhead_ms for r in results),
        "p95_detector_overhead_ms_per_run": float(np.percentile([r.detector_overhead_ms for r in results], 95)),
    }


def bootstrap_metric(
    results: list[RunResult],
    metric_fn: Callable[[list[RunResult]], float],
    seed: int,
    samples: int = 1000,
) -> tuple[float, float]:
    rng = random.Random(seed)
    values = []
    n = len(results)
    for _ in range(samples):
        sample = [results[rng.randrange(n)] for _ in range(n)]
        values.append(metric_fn(sample))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def f1_only(results: list[RunResult]) -> float:
    return binary_metrics(results)["f1"]


def efficiency_metrics(results: list[RunResult], seed: int) -> dict[str, float]:
    baseline_tokens = sum(r.baseline_tokens for r in results)
    protected_tokens = sum(r.protected_tokens for r in results)
    baseline_tools = sum(r.baseline_tool_calls for r in results)
    protected_tools = sum(r.protected_tool_calls for r in results)
    baseline_latency = sum(r.baseline_latency_ms for r in results)
    protected_latency = sum(r.protected_latency_ms for r in results)
    baseline_cost = sum(r.baseline_cost_usd for r in results)
    protected_cost = sum(r.protected_cost_usd for r in results)

    per_run_token_savings = [
        safe_div(r.baseline_tokens - r.protected_tokens, r.baseline_tokens) for r in results
    ]
    rng = random.Random(seed)
    boot_means = []
    for _ in range(1000):
        sample = [per_run_token_savings[rng.randrange(len(per_run_token_savings))] for _ in results]
        boot_means.append(statistics.mean(sample))

    token_pairs = [r.baseline_tokens - r.protected_tokens for r in results]
    try:
        wilcoxon_result = wilcoxon(token_pairs, alternative="greater", zero_method="wilcox")
        p_value = float(wilcoxon_result.pvalue)
    except ValueError:
        p_value = math.nan

    premature = sum((not r.truth_positive) and r.predicted_positive for r in results)
    interventions = sum(r.predicted_positive for r in results)
    correct_interventions = sum(r.truth_positive and r.predicted_positive for r in results)

    return {
        "token_savings_percent": 100.0 * safe_div(baseline_tokens - protected_tokens, baseline_tokens),
        "token_savings_95ci_low_percent": 100.0 * float(np.percentile(boot_means, 2.5)),
        "token_savings_95ci_high_percent": 100.0 * float(np.percentile(boot_means, 97.5)),
        "tool_call_savings_percent": 100.0 * safe_div(baseline_tools - protected_tools, baseline_tools),
        "latency_savings_percent": 100.0 * safe_div(baseline_latency - protected_latency, baseline_latency),
        "gross_cost_savings_percent": 100.0 * safe_div(baseline_cost - protected_cost, baseline_cost),
        "net_cost_saved_usd": baseline_cost - protected_cost,
        "baseline_task_success_rate": statistics.mean(r.baseline_success for r in results),
        "protected_task_success_rate": statistics.mean(r.protected_success for r in results),
        "task_success_delta_points": 100.0 * (
            statistics.mean(r.protected_success for r in results)
            - statistics.mean(r.baseline_success for r in results)
        ),
        "intervention_precision": safe_div(correct_interventions, interventions),
        "premature_stop_rate": safe_div(premature, sum(not r.truth_positive for r in results)),
        "paired_wilcoxon_token_savings_p": p_value,
    }


def per_class_metrics(results: list[RunResult]) -> pd.DataFrame:
    rows = []
    for label in LABELS:
        tp = sum(r.truth_label == label and r.predicted_label == label for r in results)
        fp = sum(r.truth_label != label and r.predicted_label == label for r in results)
        fn = sum(r.truth_label == label and r.predicted_label != label for r in results)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        delays = [
            r.detection_delay
            for r in results
            if r.truth_label == label and r.predicted_label == label and r.detection_delay is not None
        ]
        rows.append(
            {
                "class": label,
                "support": sum(r.truth_label == label for r in results),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mean_delay_steps": statistics.mean(delays) if delays else math.nan,
            }
        )
    return pd.DataFrame(rows)


def write_annotation_template(runs: list[list[Step]], output: Path) -> None:
    rows = []
    for steps in runs:
        for s in steps:
            rows.append(
                {
                    "run_id": s.run_id,
                    "scenario": s.scenario,
                    "step_index": s.step_index,
                    "agent_id": s.agent_id,
                    "subgoal": s.subgoal,
                    "action": s.action,
                    "tool_name": s.tool_name,
                    "tool_args": json.dumps(s.tool_args, sort_keys=True),
                    "observation": s.observation,
                    "response": s.response,
                    "progress_score_reviewer": "",
                    "labels_reviewer": "",
                    "alternative_available_yes_no": "",
                    "safe_to_intervene_yes_no": "",
                    "notes": "",
                }
            )
    pd.DataFrame(rows).to_csv(output, index=False)


def markdown_table(df: pd.DataFrame, digits: int = 3) -> str:
    rendered = df.copy()
    for column in rendered.select_dtypes(include=["float", "float64"]).columns:
        rendered[column] = rendered[column].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
    return rendered.to_markdown(index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=30, help="Runs per scenario")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument(
        "--semantic-backend",
        choices=["tfidf", "sentence-transformer"],
        default="tfidf",
    )
    args = parser.parse_args()
    global SEMANTIC_BACKEND
    SEMANTIC_BACKEND = args.semantic_backend
    args.output.mkdir(parents=True, exist_ok=True)

    runs = generate_runs(args.repetitions, args.seed)
    graph = build_langgraph()

    variants = ["retry_limit", "exact_only", "L1_L2", "L1_L3", "L1_L4", "full"]
    all_results: dict[str, list[RunResult]] = {}
    comparison_rows = []

    for variant in variants:
        results = [execute_run(steps, variant, graph) for steps in runs]
        all_results[variant] = results
        metrics = binary_metrics(results)
        ci_low, ci_high = bootstrap_metric(results, f1_only, seed=args.seed + len(variant))
        comparison_rows.append(
            {
                "method": variant,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "f1_ci_low": ci_low,
                "f1_ci_high": ci_high,
                "false_flags_per_100_healthy_runs": metrics["false_flags_per_100_healthy_runs"],
                "mean_detection_delay_steps": metrics["mean_detection_delay_steps"],
                "mean_overhead_ms_per_run": metrics["mean_detector_overhead_ms_per_run"],
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(args.output / "detector_comparison.csv", index=False)

    ablation_df = comparison_df[comparison_df["method"].isin(["exact_only", "L1_L2", "L1_L3", "L1_L4", "full"])].copy()
    ablation_df["configuration"] = ["L1", "L1-L2", "L1-L3", "L1-L4", "L1-L5"]
    ablation_df = ablation_df[[
        "configuration",
        "precision",
        "recall",
        "f1",
        "false_flags_per_100_healthy_runs",
        "mean_detection_delay_steps",
        "mean_overhead_ms_per_run",
    ]]
    ablation_df.to_csv(args.output / "ablation.csv", index=False)

    full_results = all_results["full"]
    full_df = pd.DataFrame([asdict(r) for r in full_results])
    full_df.to_csv(args.output / "run_summary.csv", index=False)
    per_class_df = per_class_metrics(full_results)
    per_class_df.to_csv(args.output / "per_class_metrics.csv", index=False)

    # Every raw step for auditing and later human annotation.
    raw_steps = [asdict(step) for steps in runs for step in steps]
    pd.DataFrame(raw_steps).to_csv(args.output / "trace_steps.csv", index=False)
    write_annotation_template(runs, args.output / "human_annotation_template.csv")

    efficiency = efficiency_metrics(full_results, args.seed)
    full_binary = binary_metrics(full_results)
    summary = {
        "status": "ILLUSTRATIVE_SYNTHETIC_RESULT_NOT_FOR_FINAL_PUBLICATION",
        "langgraph_used": LANGGRAPH_AVAILABLE,
        "semantic_backend": SEMANTIC_BACKEND,
        "seed": args.seed,
        "runs_per_scenario": args.repetitions,
        "total_runs": len(runs),
        "total_steps": len(raw_steps),
        "scenario_counts": dict(Counter(steps[0].scenario for steps in runs)),
        "full_detector": full_binary,
        "efficiency": efficiency,
    }
    (args.output / "example_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    paper_tables = f"""# Illustrative Paper Tables — Do Not Publish as Real-Agent Evidence

These results were produced from controlled synthetic traces. They validate the
experiment plumbing only. Replace them with held-out real LangGraph traces and
human labels before publication.

## Dataset

- Runs: **{len(runs)}**
- Steps: **{len(raw_steps)}**
- Runs per scenario: **{args.repetitions}**
- LangGraph package used: **{LANGGRAPH_AVAILABLE}**

## Baseline comparison

{markdown_table(comparison_df)}

## Cumulative ablation

{markdown_table(ablation_df)}

## Per-class full-detector results

{markdown_table(per_class_df)}

## Paired operational impact

| Metric | Illustrative value |
|---|---:|
| Token savings | {efficiency['token_savings_percent']:.2f}% |
| Token savings 95% bootstrap CI | [{efficiency['token_savings_95ci_low_percent']:.2f}%, {efficiency['token_savings_95ci_high_percent']:.2f}%] |
| Tool-call savings | {efficiency['tool_call_savings_percent']:.2f}% |
| Latency savings | {efficiency['latency_savings_percent']:.2f}% |
| Gross cost savings | {efficiency['gross_cost_savings_percent']:.2f}% |
| Intervention precision | {efficiency['intervention_precision']:.3f} |
| Premature-stop rate | {efficiency['premature_stop_rate']:.3f} |
| Paired Wilcoxon p-value for token savings | {efficiency['paired_wilcoxon_token_savings_p']:.6f} |

## Example wording

> In the controlled synthetic benchmark, the full cascade achieved precision
> {full_binary['precision']:.3f}, recall {full_binary['recall']:.3f}, and F1
> {full_binary['f1']:.3f}. It reduced total token use by
> {efficiency['token_savings_percent']:.2f}% in paired simulated runs. These
> values demonstrate implementation behavior only and are not used as evidence
> of real-world generalization.
"""
    (args.output / "paper_tables.md").write_text(paper_tables, encoding="utf-8")

    print("Experiment complete")
    print(f"LangGraph installed/used: {LANGGRAPH_AVAILABLE}")
    print(f"Runs: {len(runs)}, steps: {len(raw_steps)}")
    print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nFull detector efficiency:")
    for key, value in efficiency.items():
        print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")
    print(f"\nFiles written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
