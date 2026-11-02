"""Ví dụ chạy sinh văn bản:
  python inference/generate.py --ckpt checkpoint/LLaMA-3/LLaMA-3_73M_pretrained_cosmopedia-100k.pt --tok data/smoltalk_posttrain_raw/tokenizer.json --prompt "Machine learning is"
  python inference/generate.py --ckpt checkpoints/pre_train/DeepSeek-MoE/DeepSeek-MoE_177M_pretrained_smollm.pt --tok data/smoltalk_posttrain_raw/tokenizer.json  --prompt "Photosynthesis is the process where" --max_new 150 --temp 0.8

Text Generation:
Autoregressive sampling with Temperature and Top-K filtering.
"""
import argparse
import os
import sys

import torch
from tokenizers import Tokenizer

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
for p_dir in [PROJECT_ROOT, MODELS_DIR]:
    if p_dir not in sys.path:
        sys.path.insert(0, p_dir)

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--tok", default=None)
p.add_argument("--prompt", default="The")
p.add_argument("--max_new", type=int, default=100)
p.add_argument("--temp", type=float, default=0.8)
p.add_argument("--top_k", type=int, default=50)
args = p.parse_args()

ck = torch.load(args.ckpt, map_location="cuda")
model_type = ck.get("model_type")

if model_type == "llama" or "intermediate_size" in ck.get("config", {}):
    from model_llama import LlamaConfig, LlamaForCausalLM
    model = LlamaForCausalLM(LlamaConfig(**ck["config"])).cuda().eval()
else:
    from model_deepseek import DeepSeekConfig, DeepSeekForCausalLM
    model = DeepSeekForCausalLM(DeepSeekConfig(**ck["config"])).cuda().eval()

model.load_state_dict(ck["model"])
tok = Tokenizer.from_file(args.tok)

ids = torch.tensor([tok.encode(args.prompt).ids], device="cuda")
for _ in range(args.max_new):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(ids[:, -model.c.block_size:])["logits"][:, -1].float() / args.temp
    v, _ = logits.topk(args.top_k)
    logits[logits < v[:, [-1]]] = -float("inf")
    ids = torch.cat([ids, torch.multinomial(logits.softmax(-1), 1)], dim=1)
print(tok.decode(ids[0].tolist()))
