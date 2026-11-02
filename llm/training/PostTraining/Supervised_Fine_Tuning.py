"""
 The exact name of this technique is SFT (Supervised Fine-Tuning), commonly referred to as Instruction Tuning.
  ──────
  ### Is it RLHF?

  No, this is NOT RLHF. (You likely mean RLHF — Reinforcement Learning from Human Feedback).

  Here is the difference:

   Feature      │ Your Current Code (SFT / Instruction Tuning)                                       │ RLHF (Reinforcement Learning from Human Feedback)
  ──────────────┼────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────
   How it works │ The model is given direct examples of ideal answers and learns via standard Cross- │ The model generates multiple answers, and a Reward Model scores them using
                │ Entropy Loss (predicting the next token).                                          │ Reinforcement Learning (PPO / GRPO).
   Data type    │ Single target dialogues (SmolTalk): user asks → assistant gives the correct        │ Paired comparisons: Prompt + Response A (Chosen) vs. Response B (Rejected).
                │ answer.                                                                            │
   Analogy      │ Reading a textbook with solutions: "Here is a question, and here is exactly how an │ Taking an exam with scoring: "Try answering this yourself, and I will give you a
                │ expert answers it."                                                                │ score from 1 to 10."
   Stage        │ Step 2 (Prerequisite)                                                              │ Step 3 (Optional refinement)
  ──────
  ### The Industry LLM Hierarchy

    1. Pre-Training        --> Learn language & world facts (Raw internet text)
    2. SFT / Instruct      --> [YOUR SCRIPT] Learn how to follow instructions & chat (SmolTalk)
    3. RLHF / DPO          --> Align tone, safety, and human preference (Reward scoring)

  │ Note
  │ RLHF cannot be run on a raw base model directly. SFT is the mandatory bridge: it first teaches the model the concept of being an assistant and answering questions in ChatML format.



TRAIN SCRIPT: POST-TRAINING (INSTRUCT / SFT FINE-TUNING)
Tải trọng số đã Pretrain -> Fine-tune với dữ liệu hội thoại (SmolTalk) -> Xuất checkpoint SFT.

Ví dụ:
  # Fine-tune LLaMA-3 từ checkpoint pretrain
  python training/PostTraining/finetune.py \
      --pretrained_ckpt checkpoints/pre_train/LLaMA-3/LLaMA-3_73M_pretrained_smollm.pt \
      --data_dir data/smoltalk_posttrain_bin \
      --dataset_name smoltalk \
      --max_steps 3000 --lr 1e-4

  # Fine-tune DeepSeek-MoE từ checkpoint pretrain
  python training/PostTraining/finetune.py \
      --pretrained_ckpt checkpoints/pre_train/DeepSeek-MoE/DeepSeek-MoE_177M_pretrained_smollm.pt \
      --data_dir data/smoltalk_posttrain_bin \
      --dataset_name smoltalk \
      --max_steps 3000 --lr 1e-4 --batch_size 8 --grad_accum 8



You should keep 1 unified file (finetune.py).
  Here is why this is the standard approach in both your codebase and modern industry frameworks:
  ──────
  ### 1. Why 1 Unified File is the Best Practice

  #### A. Consistency Across Your Codebase

  Your repository already follows this design pattern:

  • train.py trains both LLaMA-3 and DeepSeek-MoE via --model llama / --model deepseek.
  • generate.py runs inference on both architectures from their checkpoint metadata.
  • Keeping finetune.py unified maintains a clean 3-stage pipeline:

    train.py ⟶ finetune.py ⟶ generate.py

  #### B. 95% of the Code is Identical (DRY Principle)
  The core SFT training pipeline does not care about layer internals:
  • Data streaming (np.memmap + slicing + pinned memory)
  • Token-level cross-entropy loss computation
  • Optimizer schedule (AdamW cosine decay + warmup)
  • Gradient accumulation and gradient clipping
  • Validation loop, perplexity calculation, checkpoint payload serialization, and CSV logging

  If you duplicate the code into finetune_llama.py and finetune_moe.py, every improvement or bug fix you make (e.g., hyperparameter tuning, loss masking, sequence packing) will have to be
  copied and pasted into both files. Over time, the two files diverge and become hard to maintain.
  #### C. Industry Standard Approach
  In production LLM frameworks:

  • Hugging Face TRL / Trainer, Meta torchtune, Axolotl, and Megatron-LM do not write separate training scripts for different model architectures.
  • The model architecture is encapsulated inside its module (model_llama.py and model_deepseek.py).
  • The training script acts as a generic Trainer Engine that interacts with standard outputs (out["loss"], out["ce"]).
  ──────
  ### 2. How finetune.py Handles Both Automatically

  You don't even need a --model flag when fine-tuning. The script inspects the checkpoint itself:

  │ Diagram exceeds terminal width (392 > 189 cols)
  │ Displayed as code block. Widen terminal to view inline.

    flowchart TD
        CKPT["Pretrained Checkpoint (.pt)"] --> INSPECT{"Inspect Metadata & Config"}
        INSPECT -->|"model_type: 'deepseek'"| MOE["Load DeepSeekConfig & DeepSeekForCausalLM<br/>(12 layers, 8 routed + 1 shared experts)"]
        INSPECT -->|"model_type: 'llama'"| LLAMA["Load LlamaConfig & LlamaForCausalLM<br/>(12 layers dense SwiGLU)"]
        
        MOE --> LOOP["Unified SFT Training Loop<br/>(batching, AdamW, autocast bfloat16, clip)"]
        LLAMA --> LOOP
        
        LOOP --> LOG{"Telemetry Log"}
        LOG -->|DeepSeek| PRINT1["Logs CE + Aux + Z-Loss + Router Load Balance"]
        LOG -->|LLaMA| PRINT2["Logs CE + PPL + Throughput"]
        
        LOOP --> SAVE["Organized Checkpoint Output"]
        SAVE --> P1["checkpoints/post_train/DeepSeek-MoE/..."]
        SAVE --> P2["checkpoints/post_train/LLaMA-3/..."]

  1. Auto-Detection (lines 107–128):
      • If loading DeepSeek-MoE_177M_pretrained_smollm.pt, it automatically instantiates DeepSeekForCausalLM.
      • If loading LLaMA-3_73M_pretrained_smollm.pt, it automatically instantiates LlamaForCausalLM.
  2. MoE vs. Dense Telemetry (lines 268–315):
      • For MoE: Tracks auxiliary load balancing loss, router z-loss, and expert dispatch balance (load max, load min, load ent).
      • For LLaMA: Smoothly skips router metrics and logs standard Cross-Entropy loss and Perplexity.
  3. Structured Save Directories (lines 137–141):
      • LLaMA checkpoints are saved to: checkpoints/post_train/LLaMA-3/LLaMA-3_73M_sft_smoltalk.pt
      • MoE checkpoints are saved to: checkpoints/post_train/DeepSeek-MoE/DeepSeek-MoE_177M_sft_smoltalk.pt

  ──────
  ### Comparison Summary

   Criteria                                                    │ 1 Unified File (finetune.py)                                │ 2 Separate Files (finetune_llama.py & finetune_moe.py)
  ─────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────
   Code Duplication                                            │ Zero duplication                                            │ ~90% duplicate code across both files
   Maintainability                                             │ High (fix once, applies to both)                            │ Low (must fix and test both files)
   Ease of Use                                                 │ Just point --pretrained_ckpt to any .pt file                │ Must remember which script matches which checkpoint
   Consistency                                                 │ Matches train.py and generate.py                            │ Breaks project architectural symmetry
   Industry Standard                                           │ Follows Hugging Face / Torchtune / Megatron                 │ Considered an anti-pattern in production pipelines
  ──────
  ### How to Run Either Model Using the Same Script

  #### Fine-Tuning DeepSeek-MoE:

    /home/dangnam/venv/bin/python training/PostTraining/finetune.py \
        --pretrained_ckpt checkpoints/pre_train/DeepSeek-MoE/DeepSeek-MoE_177M_pretrained_smollm.pt \
        --data_dir data/smoltalk_posttrain_bin \
        --dataset_name smoltalk \
        --batch_size 8 --grad_accum 8 \
        --max_steps 3000 --lr 1e-4

  #### Fine-Tuning LLaMA-3:

    /home/dangnam/venv/bin/python training/PostTraining/finetune.py \
        --pretrained_ckpt checkpoints/pre_train/LLaMA-3/LLaMA-3_73M_pretrained_smollm.pt \
        --data_dir data/smoltalk_posttrain_bin \
        --dataset_name smoltalk \
        --batch_size 8 --grad_accum 8 \
        --max_steps 3000 --lr 1e-4

  Conclusion: Keep the single unified finetune.py. It is already fully equipped and cleanly structured for both.




>>> [SFT] step 3000/3000 (ep 0.20) | val_ce 1.6418 | val_ppl 5.2
======================================================================
🎉 SFT Post-Training Completed Successfully!
  Total Steps       : 3,000 steps
  Total Tokens      : 196,608,000 tokens (196.6M)
  Accumulated Epochs: 0.20 epochs
  Total Train Time  : 1h 34m 2s (1.57 hours)
  SFT Weights Saved : /home/dangnam/hand_ons/llm/checkpoints/post_train/DeepSeek-MoE/DeepSeek-MoE_177M_sft_smoltalk.pt

"""
import argparse
import csv
import math
import os
import sys
import time
from dataclasses import asdict

import numpy as np
import torch

# Đảm bảo import được model_llama và model_deepseek từ project root
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
for p_dir in [PROJECT_ROOT, MODELS_DIR]:
    if p_dir not in sys.path:
        sys.path.insert(0, p_dir)

DEFAULT_CHECKPOINT_ROOT = (
    os.path.join(PROJECT_ROOT, "checkpoints", "post_train")
    if os.path.exists(os.path.join(PROJECT_ROOT, "checkpoints", "post_train"))
    else os.path.join(PROJECT_ROOT, "checkpoint")
)


def format_time(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS."""
    seconds = int(max(0, seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def format_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration like '8h 15m 24s'."""
    seconds = int(max(0, seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    parts = []
    if h > 0:
        parts.append(f"{h}h")
    if m > 0 or h > 0:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)


p = argparse.ArgumentParser(description="Post-training (Instruct / SFT) for LLM")
p.add_argument("--pretrained_ckpt", required=True, help="Đường dẫn tới file weights đã pretrain (.pt)")
p.add_argument("--data_dir", default="data/smoltalk_posttrain_bin", help="Thư mục chứa train.bin & val.bin từ prepare_data_posttrain.py")
p.add_argument("--checkpoint_dir", default=DEFAULT_CHECKPOINT_ROOT, help="Thư mục gốc chứa weights")
p.add_argument("--out_dir", default=None, help="Tùy chọn ghi đè thư mục lưu checkpoint")
p.add_argument("--stage", default="sft", help="Tên stage (mặc định: 'sft')")
p.add_argument("--dataset_name", default="smoltalk", help="Tên dataset post-train để đặt tên weight")
p.add_argument("--batch_size", type=int, default=8, help="micro-batch size (khuyến nghị 8 để an toàn VRAM trên GPU 16GB)")
p.add_argument("--grad_accum", type=int, default=8, help="gradient accumulation steps (8 x 8 x 1024 = 65,536 tokens/step)")
p.add_argument("--lr", type=float, default=1e-4, help="Learning rate cho SFT (thường thấp hơn pretrain: 1e-4 đến 5e-5)")
p.add_argument("--min_lr_ratio", type=float, default=0.1)
p.add_argument("--warmup", type=int, default=100, help="Số steps warmup cho SFT")
p.add_argument("--max_steps", type=int, default=3000, help="Tổng steps fine-tune SFT")
p.add_argument("--weight_decay", type=float, default=0.1)
p.add_argument("--clip", type=float, default=1.0)
p.add_argument("--log_interval", type=int, default=10)
p.add_argument("--eval_interval", type=int, default=200)
p.add_argument("--eval_iters", type=int, default=30)
p.add_argument("--ckpt_interval", type=int, default=500)
p.add_argument("--resume", action="store_true", help="Tiếp tục từ checkpoint SFT dở dang nếu có")
args = p.parse_args()

# Chuẩn hóa đường dẫn tương đối theo PROJECT_ROOT nếu gọi từ thư mục con
for path_attr in ["pretrained_ckpt", "data_dir", "checkpoint_dir"]:
    val = getattr(args, path_attr)
    if val and not os.path.isabs(val) and not os.path.exists(val):
        candidate = os.path.join(PROJECT_ROOT, val)
        if os.path.exists(candidate) or os.path.exists(os.path.dirname(candidate)):
            setattr(args, path_attr, candidate)

torch.manual_seed(1337)
np.random.seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
dev = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- 1) Tải Pretrained Checkpoint
if not os.path.exists(args.pretrained_ckpt):
    raise FileNotFoundError(f"Không tìm thấy pretrained checkpoint tại: '{args.pretrained_ckpt}'")

print(f"-> Đang nạp pretrained checkpoint từ: {args.pretrained_ckpt} ...")
ck = torch.load(args.pretrained_ckpt, map_location=dev)

model_type = ck.get("model_type")
arch_name = ck.get("arch_name")
config_dict = ck.get("config", {})

if not model_type or not arch_name:
    # Dự đoán nếu checkpoint cũ không lưu model_type
    if "intermediate_size" in config_dict:
        model_type = "llama"
        arch_name = "LLaMA-3"
    else:
        model_type = "deepseek"
        arch_name = "DeepSeek-MoE"

if model_type == "llama":
    from model_llama import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(**config_dict)
    raw_model = LlamaForCausalLM(cfg).to(dev)
else:
    from model_deepseek import DeepSeekConfig, DeepSeekForCausalLM
    cfg = DeepSeekConfig(**config_dict)
    raw_model = DeepSeekForCausalLM(cfg).to(dev)

# Nạp trọng số mô hình từ pretrain
raw_model.load_state_dict(ck["model"])
print(f"   Đã nạp thành công weights của {arch_name}!")

total, active = raw_model.num_params()
param_str = f"{int(round(total / 1e6))}M"

# ---------------- 2) Thiết lập thư mục lưu checkpoint SFT
save_dir = args.out_dir if args.out_dir else os.path.join(args.checkpoint_dir, arch_name)
os.makedirs(save_dir, exist_ok=True)

main_weight_name = f"{arch_name}_{param_str}_{args.stage}_{args.dataset_name}.pt"
main_ckpt_path = os.path.join(save_dir, main_weight_name)


# ---------------- 3) Nạp dữ liệu Post-train (train.bin, val.bin)
def load(split):
    bin_path = os.path.join(args.data_dir, f"{split}.bin")
    if not os.path.exists(bin_path):
        raise FileNotFoundError(
            f"Không tìm thấy file {bin_path}.\n"
            f"Hãy chạy: python data_prep/prepare_data_posttrain.py --dataset ... --tokenizer ... trước!"
        )
    return np.memmap(bin_path, dtype=np.uint16, mode="r")


train_data, val_data = load("train"), load("val")
T = cfg.block_size
tok_per_step = args.batch_size * args.grad_accum * T
tokens_train = len(train_data)
total_train_tokens = args.max_steps * tok_per_step
total_epochs = total_train_tokens / tokens_train

print("=" * 70)
print(f"Stage              : POST-TRAINING (SFT / INSTRUCT)")
print(f"Model Architecture : {arch_name} ({model_type.upper()})")
print(f"Parameters         : Total {total/1e6:.1f}M | Active {active/1e6:.1f}M")
print(f"Context Window (T) : {T} tokens | Vocab Size: {cfg.vocab_size}")
print(f"Post-train Data    : {tokens_train:,} tokens ({tokens_train/1e6:.1f}M tokens) tại {args.data_dir}")
print(f"Tokens/Step        : {tok_per_step:,} (batch={args.batch_size}, accum={args.grad_accum}, seq={T})")
print(f"Total Steps        : {args.max_steps:,} steps -> {total_train_tokens/1e6:.1f}M tokens ({total_epochs:.1f} epochs)")
print(f"Target SFT Weights : {main_ckpt_path}")
print("=" * 70)


def get_batch(data):
    ix = torch.randint(len(data) - T - 1, (args.batch_size,)).tolist()
    x = torch.stack([torch.from_numpy(data[i:i + T].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + T].astype(np.int64)) for i in ix])
    if dev == "cuda":
        return x.pin_memory().to(dev, non_blocking=True), y.pin_memory().to(dev, non_blocking=True)
    return x.to(dev), y.to(dev)


# Tách riêng tham số >=2D (Linear, Embedding có weight decay) và <2D (RMSNorm, bias không decay)
decay = [p for p in raw_model.parameters() if p.dim() >= 2]
no_decay = [p for p in raw_model.parameters() if p.dim() < 2]
opt = torch.optim.AdamW(
    [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
    lr=args.lr,
    betas=(0.9, 0.95),
    fused=True if torch.cuda.is_available() else False,
)


def get_lr(step):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    decay_ratio = (step - args.warmup) / max(1, args.max_steps - args.warmup)
    decay_ratio = min(1.0, max(0.0, decay_ratio))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    min_lr = args.lr * args.min_lr_ratio
    return min_lr + coeff * (args.lr - min_lr)


@torch.no_grad()
def evaluate():
    raw_model.eval()
    ce_losses = []
    for _ in range(args.eval_iters):
        x, y = get_batch(val_data)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = raw_model(x, y)
            ce_losses.append(out["ce"].item())
    raw_model.train()
    return float(np.mean(ce_losses))


def save(step, is_periodic=False):
    payload = {
        "model": raw_model.state_dict(),
        "opt": opt.state_dict(),
        "step": step,
        "config": asdict(cfg),
        "model_type": model_type,
        "arch_name": arch_name,
        "param_str": param_str,
        "stage": args.stage,
        "dataset_name": args.dataset_name,
        "args": vars(args),
    }
    torch.save(payload, main_ckpt_path)
    if is_periodic:
        step_filename = f"{arch_name}_{param_str}_{args.stage}_{args.dataset_name}_step{step}.pt"
        torch.save(payload, os.path.join(save_dir, step_filename))


start = 0
if args.resume and os.path.exists(main_ckpt_path):
    sft_ck = torch.load(main_ckpt_path, map_location=dev)
    raw_model.load_state_dict(sft_ck["model"])
    opt.load_state_dict(sft_ck["opt"])
    start = sft_ck["step"]
    print(f"-> Resumed SFT checkpoint {main_ckpt_path} at step {start}")

# ---------------- Training Loop
log_path = os.path.join(save_dir, f"{arch_name}_{param_str}_sft_log.csv")
log_f = open(log_path, "a", newline="")
log = csv.writer(log_f)
if start == 0:
    log.writerow([
        "step", "tokens", "train_ce", "val_ce", "aux", "z", "lr", "gnorm",
        "tok_per_s", "max_load", "min_load", "load_entropy", "mem_gb"
    ])

tok_per_step = args.batch_size * args.grad_accum * T
acc = dict(ce=0.0, aux=0.0, z=0.0)
counts_acc = None
t0 = time.time()
train_start_time = time.time()
raw_model.train()

model = raw_model

for step in range(start, args.max_steps):
    for g in opt.param_groups:
        g["lr"] = get_lr(step)

    for _ in range(args.grad_accum):
        x, y = get_batch(train_data)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, y)
        (out["loss"] / args.grad_accum).backward()

        acc["ce"] += out["ce"].detach() / args.grad_accum
        if "aux" in out:
            acc["aux"] += out["aux"].detach() / args.grad_accum
            acc["z"] += out["z"].detach() / args.grad_accum
        if out.get("counts") is not None:
            counts_acc = out["counts"] if counts_acc is None else counts_acc + out["counts"]

    gnorm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.clip)
    opt.step()
    opt.zero_grad(set_to_none=True)

    s = step + 1
    if s % args.log_interval == 0:
        torch.cuda.synchronize()
        dt = time.time() - t0
        t0 = time.time()
        n = args.log_interval
        ce = acc["ce"].item() / n
        aux = acc["aux"].item() / n if isinstance(acc["aux"], torch.Tensor) else 0.0
        z = acc["z"].item() / n if isinstance(acc["z"], torch.Tensor) else 0.0
        acc = dict(ce=0.0, aux=0.0, z=0.0)

        mx = mn = ent = float("nan")
        if counts_acc is not None:
            frac = counts_acc.float() / counts_acc.sum(1, keepdim=True)
            mx = frac.max(1).values.mean().item()
            mn = frac.min(1).values.mean().item()
            ent = (-(frac * (frac + 1e-9).log()).sum(1) / math.log(frac.shape[1])).mean().item()
            counts_acc = None

        tps = n * tok_per_step / dt
        mem = torch.cuda.max_memory_allocated() / 1e9

        elapsed_sec = time.time() - train_start_time
        steps_done = s - start
        steps_left = args.max_steps - s
        avg_step_time = elapsed_sec / max(1, steps_done)
        eta_sec = steps_left * avg_step_time
        curr_epoch = (s * tok_per_step) / tokens_train
        time_str = f"{format_time(elapsed_sec)}<{format_time(eta_sec)}"

        if model_type == "deepseek":
            print(f"[SFT] step {s:6d}/{args.max_steps} | ep {curr_epoch:.2f}/{total_epochs:.1f} | ce {ce:.4f} | aux {aux:.3f} | lr {get_lr(step):.2e} | gn {gnorm:.2f} | "
                  f"{tps/1e3:.1f}k tok/s | load max {mx:.2f} min {mn:.3f} ent {ent:.3f} | {mem:.1f}GB | {time_str}")
        else:
            print(f"[SFT] step {s:6d}/{args.max_steps} | ep {curr_epoch:.2f}/{total_epochs:.1f} | ce {ce:.4f} | lr {get_lr(step):.2e} | gn {gnorm:.2f} | "
                  f"{tps/1e3:.1f}k tok/s | {mem:.1f}GB | {time_str}")

        log.writerow([
            s, s * tok_per_step, ce, "", aux, z, get_lr(step), float(gnorm),
            tps, mx, mn, ent, mem
        ])
        log_f.flush()

    if s % args.eval_interval == 0:
        curr_epoch = (s * tok_per_step) / tokens_train
        v = evaluate()
        print(f">>> [SFT] step {s}/{args.max_steps} (ep {curr_epoch:.2f}) | val_ce {v:.4f} | val_ppl {math.exp(v):.1f}")
        log.writerow([s, s * tok_per_step, "", v] + [""] * 9)
        log_f.flush()
        t0 = time.time()

    if s % args.ckpt_interval == 0:
        save(s, is_periodic=True)

save(args.max_steps, is_periodic=False)

total_train_time = time.time() - train_start_time
peak_bytes = torch.cuda.max_memory_allocated()
peak_gb = peak_bytes / 1e9
total_tokens_done = args.max_steps * tok_per_step
final_accum_epochs = total_tokens_done / tokens_train

print("=" * 70)
print("🎉 SFT Post-Training Completed Successfully!")
print(f"  Total Steps       : {args.max_steps:,} steps")
print(f"  Total Tokens      : {total_tokens_done:,} tokens ({total_tokens_done/1e6:.1f}M)")
print(f"  Accumulated Epochs: {final_accum_epochs:.2f} epochs")
print(f"  Total Train Time  : {format_duration(total_train_time)} ({total_train_time/3600:.2f} hours)")
print(f"  SFT Weights Saved : {main_ckpt_path}")
print("=" * 70)
