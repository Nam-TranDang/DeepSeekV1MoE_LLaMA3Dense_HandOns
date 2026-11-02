# ./run_sft.sh --> CLi run SFT
#!/bin/bash
set -e

# Navigate to project directory
cd /home/dangnam/hand_ons/llm

# Use virtualenv python
PYTHON="/home/dangnam/venv/bin/python"

echo "======================================================================"
echo "🚀 Starting SFT Post-Training: DeepSeek-MoE on SmolTalk"
echo "======================================================================"

$PYTHON training/PostTraining/Supervised_Fine_Tuning.py \
    --pretrained_ckpt /home/dangnam/hand_ons/llm/checkpoints/pre_train/DeepSeek-MoE/DeepSeek-MoE_177M_pretrained_smollm.pt \
    --data_dir /home/dangnam/hand_ons/llm/data/smoltalk_posttrain_raw \
    --dataset_name smoltalk \
    --batch_size 8 \
    --grad_accum 8 \
    --max_steps 3000 \
    --lr 1e-4 \
    --ckpt_interval 500 \
    --eval_interval 200
