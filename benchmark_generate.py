#!/usr/bin/env python3
"""
Generation wrapper for LLaDA masked diffusion model.

Adapts the infill_denoise loop for open-ended code generation (append-and-denoise)
so that the model can be evaluated on standard coding benchmarks like HumanEval/MBPP.
"""

from typing import List, Optional

import torch

from inference import (
    batch_infill_denoise,
    infill_denoise,
    load_model_and_tokenizer,  # noqa: F401
)


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
DEFAULT_BATCH_SIZE = 4  # fits 8B model in bf16 on A100 80GB with 2048 seq_len


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
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = "cuda",
) -> List[str]:
    """
    Generate code completions using LLaDA's masked diffusion process.

    Approach (Append-and-Denoise):
      1. Tokenize the prompt (length P).
      2. Append M mask tokens to create a sequence of length P+M.
      3. Run batched infill_denoise on the masked suffix.
      4. Decode and truncate at stop sequences.

    When num_samples > 1, samples are generated in batches of `batch_size`
    for improved throughput on GPU.

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
        batch_size: Max samples to generate in parallel. Set to 1 to disable batching.
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

    # Shared mask positions (same for all samples from the same prompt)
    mask_positions = torch.zeros(total_len, dtype=torch.bool, device=device)
    mask_positions[P:] = True

    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)

    completions = []
    remaining = num_samples
    sample_offset = 0

    while remaining > 0:
        chunk = min(remaining, batch_size)

        # Set seed for this chunk for reproducible diversity
        if temperature > 0 and num_samples > 1:
            torch.manual_seed(42 + sample_offset)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42 + sample_offset)

        # Build batched input: (chunk, total_len) = [prompt | masks]
        input_ids = torch.full(
            (chunk, total_len), mask_token_id, dtype=torch.long, device=device
        )
        input_ids[:, :P] = prompt_tensor.unsqueeze(0).expand(chunk, -1)

        if chunk == 1:
            # Single sample — use original infill_denoise (avoids overhead)
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
            predicted = result["predicted_ids"]  # (1, seq_len)
        else:
            # Batched denoising
            predicted = batch_infill_denoise(
                model=model,
                input_ids=input_ids,
                mask_positions=mask_positions,
                mask_token_id=mask_token_id,
                steps=steps,
                temperature=temperature,
                remasking=remasking,
            )  # (chunk, seq_len)

        # Decode each sample in the chunk
        for b in range(chunk):
            gen_ids = predicted[b, P:].tolist()
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            gen_text = truncate_at_stop(gen_text, stop_sequences)
            completions.append(gen_text)

        remaining -= chunk
        sample_offset += chunk

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
    batch_size: int = DEFAULT_BATCH_SIZE,
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
        batch_size=batch_size,
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
