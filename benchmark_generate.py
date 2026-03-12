#!/usr/bin/env python3
"""
Generation wrapper for LLaDA masked diffusion model.

Adapts the infill_denoise loop for open-ended code generation (append-and-denoise)
so that the model can be evaluated on standard coding benchmarks like HumanEval/MBPP.
"""

from typing import List, Optional

import torch

from inference import infill_denoise, load_model_and_tokenizer  # noqa: F401


# ============================================================
# Mask Token Resolution (matches data_preprocess.py logic)
# ============================================================
MASK_TOKEN = "<|mask|>"


def resolve_mask_token_id(tokenizer, model: Optional[torch.nn.Module] = None) -> int:
    """
    Resolve the mask token id, adding <|mask|> if needed.

    If a model is provided and the tokenizer vocab is extended, the model's
    embedding table is resized to match.
    """
    if tokenizer.mask_token_id is not None:
        return tokenizer.mask_token_id
    if MASK_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [MASK_TOKEN]}
        )
        if model is not None:
            model.resize_token_embeddings(len(tokenizer))
    tid = tokenizer.convert_tokens_to_ids(MASK_TOKEN)
    if tid is None:
        raise RuntimeError("Could not resolve mask token id from tokenizer.")
    return tid


# ============================================================
# Stop-Sequence Truncation
# ============================================================
DEFAULT_STOP_SEQUENCES = [
    "\ndef ",
    "\nclass ",
    "\n\n\n",
    "\nif __name__",
]


def truncate_at_stop(text: str, stop_sequences: List[str]) -> str:
    """Truncate generated text at the earliest stop sequence."""
    earliest = len(text)
    for seq in stop_sequences:
        idx = text.find(seq)
        if idx != -1 and idx < earliest:
            earliest = idx
    return text[:earliest]


# ============================================================
# Core Generation (Append-and-Denoise)
# ============================================================
def generate_completion(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    mask_token_id: Optional[int] = None,
    steps: int = 128,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    stop_sequences: Optional[List[str]] = None,
    num_samples: int = 1,
    device: str = "cuda",
) -> List[str]:
    """
    Generate code completions using LLaDA's masked diffusion process.

    Approach (Append-and-Denoise):
      1. Tokenize the prompt (length P).
      2. Append M mask tokens to create a sequence of length P+M.
      3. Run infill_denoise on the masked suffix.
      4. Decode and truncate at stop sequences.

    Args:
        model: LLaDA model (or LoRA-merged model).
        tokenizer: Corresponding tokenizer.
        prompt: Text prompt (e.g., function signature + docstring).
        max_new_tokens: Number of mask tokens to append (M).
        mask_token_id: Token id for masking. Resolved automatically if None.
        steps: Number of denoising iterations.
        temperature: Sampling temperature. 0=greedy, >0 for diverse samples.
        remasking: 'low_confidence' or 'random'.
        stop_sequences: List of strings to truncate at. Uses defaults if None.
        num_samples: Number of completions to generate (for pass@k).
        device: 'cuda' or 'cpu'.

    Returns:
        List of `num_samples` completion strings (prompt is NOT included).
    """
    if mask_token_id is None:
        mask_token_id = resolve_mask_token_id(tokenizer, model)

    if stop_sequences is None:
        stop_sequences = DEFAULT_STOP_SEQUENCES

    # Tokenize prompt, cap to avoid excessively long sequences
    MAX_PROMPT_LEN = 1536  # leave room for generation within 2048 context
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(prompt_ids) > MAX_PROMPT_LEN:
        print(f"[WARN] Prompt truncated from {len(prompt_ids)} to {MAX_PROMPT_LEN} tokens")
        prompt_ids = prompt_ids[:MAX_PROMPT_LEN]
    P = len(prompt_ids)
    M = max_new_tokens
    total_len = P + M

    completions = []
    for sample_idx in range(num_samples):
        # Build input: [prompt_tokens | mask_tokens]
        input_ids = torch.full(
            (1, total_len), mask_token_id, dtype=torch.long, device=device
        )
        input_ids[0, :P] = torch.tensor(prompt_ids, dtype=torch.long, device=device)

        # Mask positions: only the appended region
        mask_positions = torch.zeros(total_len, dtype=torch.bool, device=device)
        mask_positions[P:] = True

        # Set seed per sample for reproducible diversity
        if temperature > 0 and num_samples > 1:
            torch.manual_seed(42 + sample_idx)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42 + sample_idx)

        # Run denoising
        result = infill_denoise(
            model=model,
            input_ids=input_ids,
            mask_positions=mask_positions,
            mask_token_id=mask_token_id,
            steps=steps,
            temperature=temperature,
            remasking=remasking,
            return_logits=False,
        )

        # Decode only the generated region
        gen_ids = result["predicted_ids"][0, P:].tolist()
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # Truncate at stop sequences
        gen_text = truncate_at_stop(gen_text, stop_sequences)

        completions.append(gen_text)

    return completions


# ============================================================
# Instruct-Wrapped Generation
# ============================================================
INSTRUCT_SYSTEM = "You are a helpful coding assistant. Complete the following Python function."


def generate_completion_instruct(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    mask_token_id: Optional[int] = None,
    steps: int = 128,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    stop_sequences: Optional[List[str]] = None,
    num_samples: int = 1,
    device: str = "cuda",
) -> List[str]:
    """
    Generate completions using the instruct chat template.

    Wraps the prompt in a chat-style format before running append-and-denoise.
    Falls back to raw generation if the tokenizer has no chat template.
    """
    # Try to use the chat template
    try:
        messages = [
            {"role": "system", "content": INSTRUCT_SYSTEM},
            {"role": "user", "content": f"Complete this function:\n\n{prompt}"},
        ]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback: simple prefix
        formatted = f"{INSTRUCT_SYSTEM}\n\n{prompt}"

    completions = generate_completion(
        model=model,
        tokenizer=tokenizer,
        prompt=formatted,
        max_new_tokens=max_new_tokens,
        mask_token_id=mask_token_id,
        steps=steps,
        temperature=temperature,
        remasking=remasking,
        stop_sequences=stop_sequences,
        num_samples=num_samples,
        device=device,
    )

    # Post-process: extract just the function body from instruct output
    cleaned = []
    for comp in completions:
        # If the model repeats the prompt signature, strip it
        # Look for the function body starting after any repeated signature
        body = _extract_function_body(comp, prompt)
        cleaned.append(body)

    return cleaned


def _extract_function_body(completion: str, original_prompt: str) -> str:
    """
    Try to extract just the function body if the model repeated the signature.
    If the completion starts with recognizable code, return as-is.
    """
    lines = original_prompt.strip().split("\n")
    if not lines:
        return completion

    sig_line = lines[0].strip()
    if not sig_line:
        return completion

    idx = completion.find(sig_line)
    if idx != -1:
        # Find a non-empty last line of the prompt to use as landmark
        last_line = ""
        for line in reversed(lines):
            if line.strip():
                last_line = line.strip()
                break
        if last_line:
            prompt_end = completion.find(last_line, idx)
            if prompt_end != -1:
                after = prompt_end + len(last_line)
                return completion[after:]
    return completion
