
import os
from typing import List, Dict

from .schema import TraceEvent

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
except ImportError:  # pragma: no cover
    AutoTokenizer = None
    AutoModelForCausalLM = None
    torch = None


class JudgeResult:
    def __init__(self, label: str, details: str = ""):
        self.label = label  # e.g., "ok", "hallucination", "irrelevant"
        self.details = details


# Global model + tokenizer, loaded once per process
_model = None
_tokenizer = None
_device = "cpu"


def _init_llama():
    """Initialize Llama judge model once, if possible."""
    global _model, _tokenizer, _device

    if AutoTokenizer is None or AutoModelForCausalLM is None:
        return

    if _model is not None and _tokenizer is not None:
        return

    model_id = os.getenv("LLAMA_JUDGE_MODEL", "unsloth/llama-3-8b-Instruct-bnb-4bit")
    hf_token = os.getenv("HF_TOKEN", None)

    if torch is not None and torch.cuda.is_available():
        _device = "cuda"
    else:
        _device = "cpu"

    _tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    _model = AutoModelForCausalLM.from_pretrained(
        model_id,
        #token=hf_token,
        device_map=_device,
    )
    _model.eval()


def _build_messages(e: TraceEvent) -> List[Dict[str, str]]:
    """Construct chat messages for the judge LLM."""
    system = (
        "You are an evaluation model acting as an LLM-as-a-Judge. "
        "Given a user question, the agent's response, and the observed context, "
        "decide whether the response is (1) relevant to the question, "
        "(2) faithful to the context, or (3) hallucinated / off-topic. "
        "Respond with one word label from {ok, irrelevant, hallucination} "
        "and a short explanation."
    )

    context = (
        f"Question: {e.question}"
        f"Subtask: {e.subtask}"
        f"Action: {e.action}"
        f"Tool: {e.tool_name}"
        f"Tool args: {e.tool_args}"
        f"Observation: {e.observation}"
        f"Response: {e.response}"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": context},
    ]


def llama_judge(e: TraceEvent) -> JudgeResult:
    """Use a Llama 3-style Instruct model as judge, if available.

    The model and tokenizer are loaded once globally. Inference is wrapped
    in torch.no_grad() to avoid accumulating graphs and leaking memory
    across iterations.
    """
    _init_llama()

    if _model is None or _tokenizer is None:
        return JudgeResult("ok", "Llama judge not initialized; transformers/torch may be missing.")

    messages = _build_messages(e)
    inputs = _tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )

    if torch is not None:
        inputs = inputs.to(_model.device)

    with torch.no_grad():  # type: ignore[attr-defined]
        outputs = _model.generate(
            **inputs,
            max_new_tokens=128,
            eos_token_id=_tokenizer.eos_token_id,
        )

    gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    gen = _tokenizer.decode(gen_ids, skip_special_tokens=True)
    text = gen.strip().lower()

    if "hallucination" in text:
        label = "hallucination"
    elif "irrelevant" in text or "off-topic" in text:
        label = "irrelevant"
    else:
        label = "ok"

    return JudgeResult(label, gen.strip())


def simple_judge(e: TraceEvent) -> JudgeResult:
    """Fallback heuristic judge, used if Llama judge is unavailable or fails."""
    q = e.question.lower()
    r = e.response.lower()

    if not r.strip():
        return JudgeResult("irrelevant", "Empty response.")

    tokens = [t for t in q.split() if len(t) > 4]
    if tokens and not any(t in r for t in tokens):
        return JudgeResult("irrelevant", "Response may not address question keywords.")

    return JudgeResult("ok")


def judge(e: TraceEvent) -> JudgeResult:
    """Main entry: try Llama, fall back to heuristic on error or OOM."""
    try:
        return llama_judge(e)
    except Exception as exc:  # pragma: no cover
        # Optional: free CUDA cache after an OOM to avoid repeated failures
        try:
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        return simple_judge(e)
