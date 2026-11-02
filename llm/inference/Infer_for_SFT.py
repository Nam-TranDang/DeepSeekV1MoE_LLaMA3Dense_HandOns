#!/usr/bin/env python3
"""
INFERENCE SCRIPT FOR SFT / INSTRUCT-TUNED MODELS (ChatML)
==========================================================
Supports both DeepSeek-MoE and LLaMA-3 architectures.
Features:
  - Real-time token streaming to stdout
  - ChatML formatting (<|im_start|>, <|im_end|>)
  - Stop token handling (<|im_end|>, <|endoftext|>)
  - Single prompt mode (--prompt "...")
  - Interactive multi-turn chat mode (CLI chatbot)
  - Speed benchmark (tokens per second)

Usage:
  # 1. Single prompt test:
  python inference/Infer_for_SFT.py --prompt "Explain what is machine learning in simple terms."

  # 2. Interactive chat mode (multi-turn conversation):
  python inference/Infer_for_SFT.py --interactive

  # 3. Custom temperature and max tokens:
  python inference/Infer_for_SFT.py --prompt "Write a short poem about coding." --temp 0.8 --max_new 200
"""

import argparse
import os
import sys
import time

import torch
from tokenizers import Tokenizer

# Add project root and models to sys.path
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
for p_dir in [PROJECT_ROOT, MODELS_DIR]:
    if p_dir not in sys.path:
        sys.path.insert(0, p_dir)

DEFAULT_CKPT = os.path.join(
    PROJECT_ROOT, "checkpoints", "post_train", "DeepSeek-MoE", "DeepSeek-MoE_177M_sft_smoltalk.pt"
)
DEFAULT_TOK = os.path.join(
    PROJECT_ROOT, "data", "smoltalk_posttrain_raw", "tokenizer.json"
)
DEFAULT_SYSTEM = "You are a helpful, respectful, and honest assistant."


def parse_args():
    p = argparse.ArgumentParser(description="Inference for SFT / ChatML LLMs")
    p.add_argument("--ckpt", default=DEFAULT_CKPT, help="Path to SFT checkpoint .pt")
    p.add_argument("--tok", default=DEFAULT_TOK, help="Path to tokenizer.json")
    p.add_argument("--prompt", type=str, default=None, help="Single prompt mode")
    p.add_argument("--interactive", "-i", action="store_true", help="Launch interactive multi-turn chat session")
    p.add_argument("--system", type=str, default=DEFAULT_SYSTEM, help="System prompt")
    p.add_argument("--max_new", type=int, default=300, help="Max new tokens to generate")
    p.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    p.add_argument("--top_k", type=int, default=50, help="Top-K sampling filtering")
    p.add_argument("--top_p", type=float, default=0.9, help="Top-P (nucleus) filtering")
    p.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty (1.0 = disabled)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    return p.parse_args()


def load_model(ckpt_path, device):
    """Load model architecture and weights based on checkpoint metadata."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    print(f"-> Loading model from: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = ck.get("config", {})
    model_type = ck.get("model_type")

    if not model_type:
        if "intermediate_size" in cfg_dict:
            model_type = "llama"
        else:
            model_type = "deepseek"

    if model_type == "llama":
        from model_llama import LlamaConfig, LlamaForCausalLM
        cfg = LlamaConfig(**cfg_dict)
        model = LlamaForCausalLM(cfg).to(device)
    else:
        from model_deepseek import DeepSeekConfig, DeepSeekForCausalLM
        cfg = DeepSeekConfig(**cfg_dict)
        model = DeepSeekForCausalLM(cfg).to(device)

    model.load_state_dict(ck["model"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    arch_display = ck.get("arch_name", model_type.upper())
    print(f"   Architecture: {arch_display} ({total_params / 1e6:.1f}M parameters)")
    return model, cfg, model_type


def format_chatml_prompt(messages):
    """
    Format a list of message dicts [{'role': 'user', 'content': '...'}] into ChatML format.
    Appends '<|im_start|>assistant\\n' at the end to prompt the assistant generation.
    """
    formatted = []
    for m in messages:
        role = m["role"]
        content = m["content"].strip()
        formatted.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    # Prompt the assistant to respond
    formatted.append("<|im_start|>assistant\n")
    return "\n".join(formatted)


def sample_next_token(logits, temp=0.7, top_k=50, top_p=0.9, generated_ids=None, repetition_penalty=1.1):
    """Apply temperature, repetition penalty, top-k, and top-p sampling."""
    logits = logits.float()

    # Repetition penalty
    if repetition_penalty != 1.0 and generated_ids is not None and len(generated_ids) > 0:
        seen = set(generated_ids)
        for token_id in seen:
            if logits[token_id] > 0:
                logits[token_id] /= repetition_penalty
            else:
                logits[token_id] *= repetition_penalty

    # Temperature scaling
    if temp > 0:
        logits = logits / temp
    else:
        return torch.argmax(logits, dim=-1, keepdim=True)

    # Top-K filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float("Inf")

    # Top-P (nucleus) filtering
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float("Inf")

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate_stream(
    model,
    tok,
    prompt_text,
    block_size,
    device,
    max_new=300,
    temp=0.7,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.1,
    stop_token_ids=None,
):
    """
    Generate response tokens with real-time streaming to stdout.
    Stops generation when a stop token (<|im_end|>, <|endoftext|>) is encountered.
    """
    if stop_token_ids is None:
        stop_token_ids = []

    input_ids = tok.encode(prompt_text).ids
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    generated_ids = []
    t_start = time.time()

    for step in range(max_new):
        # Truncate context to model's maximum window size
        ctx = input_tensor[:, -block_size:]

        with torch.autocast(device, dtype=torch.bfloat16):
            outputs = model(ctx)
            next_token_logits = outputs["logits"][0, -1, :]

        next_token = sample_next_token(
            next_token_logits,
            temp=temp,
            top_k=top_k,
            top_p=top_p,
            generated_ids=generated_ids[-100:] if generated_ids else None,
            repetition_penalty=repetition_penalty,
        )

        token_id = next_token.item()

        # Stop condition
        if token_id in stop_token_ids:
            break

        generated_ids.append(token_id)
        input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)

        # Stream token text directly to console
        token_text = tok.decode([token_id])
        sys.stdout.write(token_text)
        sys.stdout.flush()

    duration = time.time() - t_start
    tok_per_sec = len(generated_ids) / max(0.001, duration)
    return tok.decode(generated_ids), len(generated_ids), tok_per_sec


def main():
    args = parse_args()

    # Load Tokenizer
    if not os.path.exists(args.tok):
        raise FileNotFoundError(f"Tokenizer not found at: {args.tok}")
    tok = Tokenizer.from_file(args.tok)

    # Resolve Stop Token IDs
    im_end_id = tok.token_to_id("<|im_end|>")
    eos_id = tok.token_to_id("<|endoftext|>")
    stop_ids = [tid for tid in [im_end_id, eos_id] if tid is not None]

    # Load Model
    model, cfg, model_type = load_model(args.ckpt, args.device)
    block_size = cfg.block_size

    print(f"-> Context Window: {block_size} tokens | Stop Tokens: {stop_ids}")
    print(f"-> Sampling Settings: temp={args.temp}, top_k={args.top_k}, top_p={args.top_p}, rep_penalty={args.repetition_penalty}")
    print("=" * 70)

    # 1. Single Prompt Mode
    if args.prompt and not args.interactive:
        messages = [
            {"role": "system", "content": args.system},
            {"role": "user", "content": args.prompt},
        ]
        prompt_text = format_chatml_prompt(messages)
        print(f"\n[USER]: {args.prompt}")
        print("\n[ASSISTANT]: ", end="", flush=True)

        _, n_tokens, speed = generate_stream(
            model=model,
            tok=tok,
            prompt_text=prompt_text,
            block_size=block_size,
            device=args.device,
            max_new=args.max_new,
            temp=args.temp,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            stop_token_ids=stop_ids,
        )
        print(f"\n\n[Stats: {n_tokens} tokens generated in {n_tokens / max(1, speed):.2f}s | {speed:.1f} tok/s]\n")
        return

    # 2. Interactive Multi-turn Chat Mode
    print("🤖 SFT Interactive Chatbot Launched!")
    print("   Type your question and press Enter.")
    print("   Commands: 'clear' to reset conversation | 'exit' or 'quit' to end session.")
    print("=" * 70)

    history = [{"role": "system", "content": args.system}]

    while True:
        try:
            user_input = input("\n[User]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting session. Bye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "q"]:
            print("Session ended.")
            break

        if user_input.lower() in ["clear", "reset"]:
            history = [{"role": "system", "content": args.system}]
            print("🧹 Conversation history cleared.")
            continue

        history.append({"role": "user", "content": user_input})
        prompt_text = format_chatml_prompt(history)

        print("\n[Assistant]: ", end="", flush=True)
        response_text, n_tokens, speed = generate_stream(
            model=model,
            tok=tok,
            prompt_text=prompt_text,
            block_size=block_size,
            device=args.device,
            max_new=args.max_new,
            temp=args.temp,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            stop_token_ids=stop_ids,
        )
        print(f"\n({speed:.1f} tok/s)")
        history.append({"role": "assistant", "content": response_text.strip()})


if __name__ == "__main__":
    main()
