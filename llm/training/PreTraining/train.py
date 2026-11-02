"""Train LLaMA-3 (Dense) hoặc DeepSeek-MoE (Sparse).

Ví dụ:
  # LLaMA-3 Dense
  python train.py --preset tiny --model llama
  python train.py --preset small --model llama --batch_size 8 --grad_accum 8

  # DeepSeek-MoE
  python train.py --preset tiny --model deepseek
  python train.py --preset small --model deepseek --batch_size 8 --grad_accum 8
    >>> step 2000 | val_ce 2.9635 | val_ppl 19.4
step   2020 | ce 2.8405 | lr 5.96e-04 | gn 0.37 | 77.9k tok/s | 7.4GB
step   2040 | ce 2.8280 | lr 5.96e-04 | gn 0.38 | 91.4k tok/s | 7.4GB
step   2060 | ce 2.8328 | lr 5.96e-04 | gn 0.38 | 91.4k tok/s | 7.4GB
step   2080 | ce 2.8270 | lr 5.96e-04 | gn 0.37 | 91.3k tok/s | 7.4GB
step   2100 | ce 2.8177 | lr 5.96e-04 | gn 0.36 | 91.3k tok/s | 7.4GB
step   2120 | ce 2.8230 | lr 5.96e-04 | gn 0.36 | 91.3k tok/s | 7.4GB
step   2140 | ce 2.8155 | lr 5.96e-04 | gn 0.37 | 91.3k tok/s | 7.4GB
step   2160 | ce 2.7884 | lr 5.96e-04 | gn 0.37 | 91.2k tok/s | 7.4GB
step   2180 | ce 2.7888 | lr 5.96e-04 | gn 0.37 | 91.3k tok/s | 7.4GB
step   2200 | ce 2.7976 | lr 5.96e-04 | gn 0.36 | 91.3k tok/s | 7.4GB
step   2220 | ce 2.7852 | lr 5.95e-04 | gn 0.38 | 91.3k tok/s | 7.4GB
step   2240 | ce 2.7859 | lr 5.95e-04 | gn 0.38 | 91.3k tok/s | 7.4GB
step   2260 | ce 2.7774 | lr 5.95e-04 | gn 0.37 | 91.3k tok/s | 7.4GB
step   2280 | ce 2.7788 | lr 5.95e-04 | gn 0.36 | 91.3k tok/s | 7.4GB
step   2300 | ce 2.7630 | lr 5.95e-04 | gn 0.37 | 91.3k tok/s | 7.4GB
step   2320 | ce 2.7870 | lr 5.95e-04 | gn 0.37 | 91.3k tok/s | 7.4GB
step   2340 | ce 2.7720 | lr 5.95e-04 | gn 0.39 | 91.3k tok/s | 7.4GB
step   2360 | ce 2.7414 | lr 5.95e-04 | gn 0.37 | 91.3k tok/s | 7.4GB
step   2380 | ce 2.7470 | lr 5.95e-04 | gn 0.37 | 91.3k tok/s | 7.4GB
step   2400 | ce 2.7525 | lr 5.94e-04 | gn 0.37 | 91.3k tok/s | 7.4GB
step   2420 | ce 2.7535 | lr 5.94e-04 | gn 0.38 | 91.3k tok/s | 7.4GB
step   2440 | ce 2.7536 | lr 5.94e-04 | gn 0.36 | 91.3k tok/s | 7.4GB
step   2460 | ce 2.7355 | lr 5.94e-04 | gn 0.36 | 91.3k tok/s | 7.4GB
step   2480 | ce 2.7321 | lr 5.94e-04 | gn 0.36 | 91.3k tok/s | 7.4GB
step   2500 | ce 2.7325 | lr 5.94e-04 | gn 0.39 | 91.3k tok/s | 7.4GB
>>> step 2500 | val_ce 2.8983 | val_ppl 18.1
step   2520 | ce 2.7177 | lr 5.94e-04 | gn 0.38 | 91.3k tok/s | 7.4GB
step   2540 | ce 2.7229 | lr 5.94e-04 | gn 0.36 | 91.3k tok/s | 7.4GB
step   2560 | ce 2.7213 | lr 5.94e-04 | gn 0.36 | 91.3k tok/s | 7.4GB
step   2580 | ce 2.7239 | lr 5.93e-04 | gn 0.37 | 91.3k tok/s | 7.4GB
step   2600 | ce 2.7090 | lr 5.93e-04 | gn 0.35 | 91.3k tok/s | 7.4GB
step   2620 | ce 2.7154 | lr 5.93e-04 | gn 0.35 | 91.3k tok/s | 7.4GB
step   2640 | ce 2.6963 | lr 5.93e-04 | gn 0.35 | 91.3k tok/s | 7.4GB
step   2660 | ce 2.7064 | lr 5.93e-04 | gn 0.36 | 91.3k tok/s | 7.4GB
step   2680 | ce 2.6895 | lr 5.93e-04 | gn 0.37 | 91.3k tok/s | 7.4GB
step   2700 | ce 2.6860 | lr 5.93e-04 | gn 0.35 | 91.3k tok/s | 7.4GB
step   2720 | ce 2.6830 | lr 5.92e-04 | gn 0.36 | 91.3k tok/s | 7.4GB
step   2740 | ce 2.6922 | lr 5.92e-04 | gn 0.36 | 91.3k tok/s | 7.4GB
step   2760 | ce 2.6716 | lr 5.92e-04 | gn 0.35 | 91.3k tok/s | 7.4GB
step   2780 | ce 2.6680 | lr 5.92e-04 | gn 0.36 | 91.3k tok/s | 7.4GB
step   2800 | ce 2.6619 | lr 5.92e-04 | gn 0.36 | 91.3k tok/s | 7.4GB
step   2820 | ce 2.6725 | lr 5.92e-04 | gn 0.36 | 91.3k tok/s | 7.4GB
step   2840 | ce 2.6719 | lr 5.92e-04 | gn 0.37 | 91.3k tok/s | 7.4GB
step   2860 | ce 2.6662 | lr 5.92e-04 | gn 0.36 | 91.3k tok/s | 7.4GB
step   2880 | ce 2.6687 | lr 5.91e-04 | gn 0.46 | 91.3k tok/s | 7.4GB
step   2900 | ce 2.6718 | lr 5.91e-04 | gn 0.36 | 91.3k tok/s | 7.4GB
step   2920 | ce 2.6557 | lr 5.91e-04 | gn 0.36 | 91.3k tok/s | 7.4GB
step   2940 | ce 2.6483 | lr 5.91e-04 | gn 0.37 | 91.3k tok/s | 7.4GB
step   2960 | ce 2.6618 | lr 5.91e-04 | gn 0.36 | 91.3k tok/s | 7.4GB
step   2980 | ce 2.6358 | lr 5.91e-04 | gn 0.35 | 91.3k tok/s | 7.4GB
step   3000 | ce 2.6302 | lr 5.90e-04 | gn 0.36 | 91.3k tok/s | 7.4GB
>>> step 3000 | val_ce 2.8363 | val_ppl 17.1
step   3020 | ce 2.6489 | lr 5.90e-04 | gn 0.37 | 77.6k tok/s | 7.4GB
step   3040 | ce 2.6547 | lr 5.90e-04 | gn 0.35 | 91.4k tok/s | 7.4GB
step   3060 | ce 2.6288 | lr 5.90e-04 | gn 0.35 | 91.3k tok/s | 7.4GB
step   3080 | ce 2.6326 | lr 5.90e-04 | gn 0.37 | 91.3k tok/s | 7.4GB
step   3100 | ce 2.6308 | lr 5.90e-04 | gn 0.35 | 91.3k tok/s | 7.4GB
step   3120 | ce 2.6272 | lr 5.90e-04 | gn 0.36 | 91.3k tok/s | 7.4GB
step   3140 | ce 2.6227 | lr 5.89e-04 | gn 0.36 | 91.3k tok/s | 7.4GB
step   3160 | ce 2.6226 | lr 5.89e-04 | gn 0.36 | 91.2k tok/s | 7.4GB
step   3180 | ce 2.6181 | lr 5.89e-04 | gn 0.35 | 91.3k tok/s | 7.4GB
step   3200 | ce 2.6259 | lr 5.89e-04 | gn 0.36 | 91.3k tok/s | 7.4GB
step   3220 | ce 2.6168 | lr 5.89e-04 | gn 0.35 | 91.3k tok/s | 7.4GB
step   3240 | ce 2.6154 | lr 5.89e-04 | gn 0.35 | 91.3k tok/s | 7.4GB
step   3260 | ce 2.6136 | lr 5.88e-04 | gn 0.35 | 91.3k tok/s | 7.4GB
step   3280 | ce 2.6111 | lr 5.88e-04 | gn 0.36 | 91.3k tok/s | 7.4GB
step   3300 | ce 2.6150 | lr 5.88e-04 | gn 0.35 | 91.3k tok/s | 7.4GB
step   3320 | ce 2.5954 | lr 5.88e-04 | gn 0.34 | 91.3k tok/s | 7.4GB
step   3340 | ce 2.5991 | lr 5.88e-04 | gn 0.35 | 91.3k tok/s | 7.4GB
step   3360 | ce 2.5837 | lr 5.88e-04 | gn 0.35 | 91.3k tok/s | 7.4GB
step   3380 | ce 2.5888 | lr 5.87e-04 | gn 0.36 | 91.3k tok/s | 7.4GB
step   3400 | ce 2.5935 | lr 5.87e-04 | gn 0.34 | 91.3k tok/s | 7.4GB
step   3420 | ce 2.5787 | lr 5.87e-04 | gn 0.37 | 91.3k tok/s | 7.4GB
step   3440 | ce 2.5926 | lr 5.87e-04 | gn 0.35 | 91.3k tok/s | 7.4GB
step   3460 | ce 2.5911 | lr 5.87e-04 | gn 0.37 | 91.3k tok/s | 7.4GB
step   3480 | ce 2.5675 | lr 5.87e-04 | gn 0.36 | 91.3k tok/s | 7.4GB
step   3500 | ce 2.5682 | lr 5.86e-04 | gn 0.37 | 91.3k tok/s | 7.4GB
>>> step 3500 | val_ce 2.7892 | val_ppl 16.3
step   3520 | ce 2.5676 | lr 5.86e-04 | gn 0.36 | 91.3k tok/s | 7.4GB
step   3540 | ce 2.5610 | lr 5.86e-04 | gn 0.35 | 91.3k tok/s | 7.4GB
step   3560 | ce 2.5692 | lr 5.86e-04 | gn 0.36 | 91.3k tok/s | 7.4GB
step   3580 | ce 2.5616 | lr 5.86e-04 | gn 0.35 | 91.3k tok/s | 7.4GB
step   3600 | ce 2.5602 | lr 5.85e-04 | gn 0.35 | 91.3k tok/s | 7.4GB
step   3620 | ce 2.5642 | lr 5.85e-04 | gn 0.48 | 91.3k tok/s | 7.4GB
step   3640 | ce 2.5605 | lr 5.85e-04 | gn 0.35 | 91.3k tok/s | 7.4GB
step   3660 | ce 2.5624 | lr 5.85e-04 | gn 0.36 | 91.3k tok/s | 7.4GB
step   3680 | ce 2.5575 | lr 5.85e-04 | gn 0.34 | 91.3k tok/s | 7.4GB
step   3700 | ce 2.5505 | lr 5.84e-04 | gn 0.39 | 91.3k tok/s | 7.4GB
step   3720 | ce 2.5643 | lr 5.84e-04 | gn 0.35 | 91.3k tok/s | 7.4GB
step   3740 | ce 2.5635 | lr 5.84e-04 | gn 0.35 | 91.3k tok/s | 7.4GB
step   3760 | ce 2.5424 | lr 5.84e-04 | gn 0.37 | 91.3k tok/s | 7.4GB
step   3780 | ce 2.5487 | lr 5.84e-04 | gn 0.36 | 91.3k tok/s | 7.4GB
step   3800 | ce 2.5384 | lr 5.84e-04 | gn 0.36 | 91.3k tok/s | 7.4GB
step   3820 | ce 2.5275 | lr 5.83e-04 | gn 0.35 | 91.3k tok/s | 7.4GB
step   3840 | ce 2.5409 | lr 5.83e-04 | gn 0.35 | 91.3k tok/s | 7.4GB
step   3860 | ce 2.5372 | lr 5.83e-04 | gn 0.36 | 91.3k tok/s | 7.4GB

=================================
With MoE 
step  14280 | ce 1.2098 | aux 1.043 | lr 6.33e-05 | gn 0.65 | 34.0k tok/s | load max 0.16 min 0.097 ent 0.992 | 10.4GB
step  14300 | ce 1.2002 | aux 1.042 | lr 6.31e-05 | gn 0.65 | 34.4k tok/s | load max 0.16 min 0.096 ent 0.992 | 10.4GB
step  14320 | ce 1.2068 | aux 1.042 | lr 6.29e-05 | gn 0.64 | 34.3k tok/s | load max 0.16 min 0.096 ent 0.992 | 10.4GB
step  14340 | ce 1.2131 | aux 1.041 | lr 6.28e-05 | gn 0.64 | 33.8k tok/s | load max 0.16 min 0.097 ent 0.992 | 10.4GB
step  14360 | ce 1.2179 | aux 1.041 | lr 6.26e-05 | gn 0.64 | 33.8k tok/s | load max 0.16 min 0.097 ent 0.992 | 10.4GB
step  14380 | ce 1.2346 | aux 1.040 | lr 6.24e-05 | gn 0.68 | 34.5k tok/s | load max 0.16 min 0.098 ent 0.992 | 10.4GB
step  14400 | ce 1.2104 | aux 1.041 | lr 6.23e-05 | gn 0.65 | 34.3k tok/s | load max 0.16 min 0.098 ent 0.992 | 10.4GB
step  14420 | ce 1.2167 | aux 1.041 | lr 6.21e-05 | gn 0.65 | 33.9k tok/s | load max 0.16 min 0.097 ent 0.992 | 10.4GB
step  14440 | ce 1.2213 | aux 1.041 | lr 6.20e-05 | gn 0.67 | 34.1k tok/s | load max 0.16 min 0.097 ent 0.992 | 10.4GB
step  14460 | ce 1.1984 | aux 1.041 | lr 6.19e-05 | gn 0.65 | 34.2k tok/s | load max 0.16 min 0.097 ent 0.992 | 10.4GB
step  14480 | ce 1.2139 | aux 1.042 | lr 6.17e-05 | gn 0.64 | 34.1k tok/s | load max 0.16 min 0.097 ent 0.992 | 10.4GB
step  14500 | ce 1.1972 | aux 1.043 | lr 6.16e-05 | gn 0.64 | 33.8k tok/s | load max 0.16 min 0.097 ent 0.992 | 10.4GB
>>> step 14500 | val_ce 3.2328 | val_ppl 25.4
step  14520 | ce 1.2220 | aux 1.041 | lr 6.15e-05 | gn 0.66 | 33.6k tok/s | load max 0.16 min 0.097 ent 0.992 | 10.4GB
step  14540 | ce 1.2124 | aux 1.042 | lr 6.13e-05 | gn 0.68 | 33.7k tok/s | load max 0.16 min 0.097 ent 0.992 | 10.4GB
step  14560 | ce 1.2102 | aux 1.041 | lr 6.12e-05 | gn 0.65 | 34.2k tok/s | load max 0.16 min 0.097 ent 0.992 | 10.4GB
step  14580 | ce 1.1917 | aux 1.041 | lr 6.11e-05 | gn 0.64 | 34.0k tok/s | load max 0.16 min 0.097 ent 0.992 | 10.4GB
step  14600 | ce 1.2096 | aux 1.041 | lr 6.10e-05 | gn 0.69 | 34.3k tok/s | load max 0.16 min 0.097 ent 0.992 | 10.4GB
step  14620 | ce 1.1946 | aux 1.042 | lr 6.09e-05 | gn 0.67 | 34.4k tok/s | load max 0.16 min 0.097 ent 0.992 | 10.4GB
step  14640 | ce 1.2041 | aux 1.041 | lr 6.08e-05 | gn 0.66 | 34.5k tok/s | load max 0.16 min 0.097 ent 0.992 | 10.4GB
step  14660 | ce 1.1970 | aux 1.041 | lr 6.07e-05 | gn 0.66 | 34.4k tok/s | load max 0.16 min 0.098 ent 0.992 | 10.4GB
step  14680 | ce 1.1744 | aux 1.042 | lr 6.07e-05 | gn 0.65 | 34.3k tok/s | load max 0.16 min 0.096 ent 0.992 | 10.4GB
step  14700 | ce 1.1768 | aux 1.042 | lr 6.06e-05 | gn 0.67 | 34.3k tok/s | load max 0.16 min 0.097 ent 0.992 | 10.4GB
step  14720 | ce 1.1898 | aux 1.042 | lr 6.05e-05 | gn 0.64 | 34.5k tok/s | load max 0.16 min 0.097 ent 0.992 | 10.4GB
step  14740 | ce 1.1862 | aux 1.043 | lr 6.04e-05 | gn 0.67 | 34.4k tok/s | load max 0.16 min 0.096 ent 0.992 | 10.4GB
step  14760 | ce 1.1930 | aux 1.039 | lr 6.04e-05 | gn 0.64 | 34.1k tok/s | load max 0.16 min 0.097 ent 0.992 | 10.4GB
step  14780 | ce 1.1855 | aux 1.043 | lr 6.03e-05 | gn 0.65 | 34.2k tok/s | load max 0.16 min 0.096 ent 0.992 | 10.4GB
step  14800 | ce 1.1947 | aux 1.042 | lr 6.03e-05 | gn 0.65 | 34.5k tok/s | load max 0.16 min 0.098 ent 0.992 | 10.4GB
step  14820 | ce 1.1879 | aux 1.043 | lr 6.02e-05 | gn 0.63 | 33.9k tok/s | load max 0.16 min 0.097 ent 0.992 | 10.4GB
step  14840 | ce 1.1976 | aux 1.042 | lr 6.02e-05 | gn 0.68 | 34.1k tok/s | load max 0.16 min 0.096 ent 0.992 | 10.4GB
step  14860 | ce 1.1874 | aux 1.041 | lr 6.01e-05 | gn 0.69 | 34.2k tok/s | load max 0.16 min 0.097 ent 0.992 | 10.4GB
step  14880 | ce 1.1727 | aux 1.041 | lr 6.01e-05 | gn 0.69 | 34.0k tok/s | load max 0.16 min 0.097 ent 0.992 | 10.4GB
step  14900 | ce 1.1839 | aux 1.042 | lr 6.01e-05 | gn 0.66 | 33.9k tok/s | load max 0.16 min 0.096 ent 0.992 | 10.4GB
step  14920 | ce 1.1783 | aux 1.042 | lr 6.00e-05 | gn 0.68 | 34.1k tok/s | load max 0.16 min 0.096 ent 0.992 | 10.4GB
step  14940 | ce 1.1929 | aux 1.041 | lr 6.00e-05 | gn 0.68 | 34.1k tok/s | load max 0.16 min 0.097 ent 0.992 | 10.4GB
step  14960 | ce 1.1779 | aux 1.042 | lr 6.00e-05 | gn 0.65 | 34.0k tok/s | load max 0.16 min 0.097 ent 0.992 | 10.4GB
step  14980 | ce 1.1781 | aux 1.041 | lr 6.00e-05 | gn 0.66 | 34.4k tok/s | load max 0.16 min 0.097 ent 0.993 | 10.4GB
step  15000 | ce 1.1662 | aux 1.041 | lr 6.00e-05 | gn 0.65 | 34.1k tok/s | load max 0.16 min 0.097 ent 0.992 | 10.4GB
>>> step 15000 | val_ce 3.2661 | val_ppl 26.2




Cli for new MoE - 25/10/2026
python training/PreTraining/train.py --preset small --model deepseek \
        --data_dir data/smollm_pretrain_raw \
        --dataset_name smollm \
        --batch_size 8 --grad_accum 8 \
        --max_steps 17600 --eval_interval 500 --ckpt_interval 5000

/home/dangnam/venv/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
======================================================================
Architecture : DeepSeek-MoE
Parameters   : Total 176.7M | Active/token 72.9M
Hardware     : NVIDIA GeForce RTX 5070 Ti (15.5 GB VRAM)
Dataset Size : 1,153,772,909 tokens (1153.8M tokens)
Tokens/Step  : 65,536 (batch=8, accum=8, seq=1024)
Target Budget: 17,600 steps | 1153.4M tokens | Est. 1.00 Epochs
Save Folder  : /home/dangnam/hand_ons/llm_scratch/checkpoint/DeepSeek-MoE
Weight File  : DeepSeek-MoE_177M_pretrained_smollm.pt

New MoE
step  17480/17600 | ep 0.99/1.0 | ce 1.8845 | aux 1.024 | lr 6.01e-05 | gn 0.35 | 33.0k tok/s | load max 0.16 min 0.101 ent 0.995 | 11.1GB | 04:19:26<03:40
step  17500/17600 | ep 0.99/1.0 | ce 1.8882 | aux 1.023 | lr 6.00e-05 | gn 0.35 | 30.8k tok/s | load max 0.15 min 0.101 ent 0.995 | 11.1GB | 04:20:08<03:03
>>> step 17500/17600 (ep 0.99) | val_ce 2.0347 | val_ppl 7.6
step  17520/17600 | ep 1.00/1.0 | ce 1.8825 | aux 1.023 | lr 6.00e-05 | gn 0.36 | 35.3k tok/s | load max 0.15 min 0.101 ent 0.995 | 11.1GB | 04:20:49<02:26
step  17540/17600 | ep 1.00/1.0 | ce 1.8785 | aux 1.023 | lr 6.00e-05 | gn 0.35 | 37.1k tok/s | load max 0.15 min 0.101 ent 0.995 | 11.1GB | 04:21:24<01:50
step  17560/17600 | ep 1.00/1.0 | ce 1.8817 | aux 1.023 | lr 6.00e-05 | gn 0.39 | 34.6k tok/s | load max 0.15 min 0.101 ent 0.995 | 11.1GB | 04:22:02<01:13
step  17580/17600 | ep 1.00/1.0 | ce 1.8901 | aux 1.024 | lr 6.00e-05 | gn 0.36 | 34.8k tok/s | load max 0.16 min 0.101 ent 0.995 | 11.1GB | 04:22:40<00:36
step  17600/17600 | ep 1.00/1.0 | ce 1.8832 | aux 1.025 | lr 6.00e-05 | gn 0.35 | 37.1k tok/s | load max 0.16 min 0.102 ent 0.995 | 11.1GB | 04:23:15<00:00
======================================================================
🎉 Training Completed Successfully!
  Total Steps       : 17,600 steps
  Total Tokens      : 1,153,433,600 tokens (1153.4M)
  Accumulated Epochs: 1.00 epochs
  Total Train Time  : 4h 23m 19s (4.39 hours)
  Avg Throughput    : 35.7k tok/s
  Peak VRAM Usage   : 11.07 GB (10.31 GiB)
  Weights Saved At  : /home/dangnam/hand_ons/llm_scratch/checkpoint/DeepSeek-MoE/DeepSeek-MoE_177M_pretrained_smollm.pt
======================================================================
LLAMA Dense - 27 Oct 2026



python training/PreTraining/train.py \
        --preset small \
        --model llama \
        --data_dir data/smollm_pretrain_raw \
        --dataset_name smollm \
        --batch_size 8 \
        --grad_accum 8 \
        --max_steps 17600 \
        --eval_interval 500 \
        --ckpt_interval 3000
/home/dangnam/venv/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
======================================================================                                                      2:58 26-Oct-26
Architecture : LLaMA-3
Parameters   : Total 72.9M | Active/token 72.9M
Hardware     : NVIDIA GeForce RTX 5070 Ti (15.5 GB VRAM)
Dataset Size : 1,153,772,909 tokens (1153.8M tokens)                                                                       -
Tokens/Step  : 65,536 (batch=8, accum=8, seq=1024)
Target Budget: 17,600 steps | 1153.4M tokens | Est. 1.00 Epochs
Save Folder  : /home/dangnam/hand_ons/llm_scratch/checkpoint/LLaMA-3
Weight File  : LLaMA-3_73M_pretrained_smollm.pt
======================================================================
>>> step 17500/17600 (ep 0.99) | val_ce 2.0955 | val_ppl 8.1
step  17520/17600 | ep 1.00/1.0 | ce 2.0556 | lr 6.00e-05 | gn 0.39 | 91.2k tok/s | 7.4GB | 03:30:55<00:57
step  17540/17600 | ep 1.00/1.0 | ce 2.0806 | lr 6.00e-05 | gn 0.39 | 91.2k tok/s | 7.4GB | 03:31:09<00:43
step  17560/17600 | ep 1.00/1.0 | ce 2.0610 | lr 6.00e-05 | gn 0.39 | 91.2k tok/s | 7.4GB | 03:31:23<00:28
step  17580/17600 | ep 1.00/1.0 | ce 2.0691 | lr 6.00e-05 | gn 0.39 | 91.2k tok/s | 7.4GB | 03:31:38<00:14
step  17600/17600 | ep 1.00/1.0 | ce 2.0708 | lr 6.00e-05 | gn 0.39 | 91.2k tok/s | 7.4GB | 03:31:52<00:00
======================================================================
🎉 Training Completed Successfully!
  Total Steps       : 17,600 steps
  Total Tokens      : 1,153,433,600 tokens (1153.4M)
  Accumulated Epochs: 1.00 epochs
  Total Train Time  : 3h 31m 54s (3.53 hours)
  Avg Throughput    : 90.7k tok/s
  Peak VRAM Usage   : 7.37 GB (6.86 GiB)
  Weights Saved At  : /home/dangnam/hand_ons/llm_scratch/checkpoint/LLaMA-3/LLaMA-3_73M_pretrained_smollm.pt


"""
import argparse
import csv
import math
import os
import time
from dataclasses import asdict

import numpy as np
import torch


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


import sys

# Đảm bảo import được các module model (models/model_llama.py, models/model_deepseek.py) từ bất kỳ đâu
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
for p_dir in [PROJECT_ROOT, MODELS_DIR]:
    if p_dir not in sys.path:
        sys.path.insert(0, p_dir)

# Checkpoint lưu trữ mặc định tại checkpoints/pre_train hoặc checkpoints
DEFAULT_CHECKPOINT_ROOT = (
    os.path.join(PROJECT_ROOT, "checkpoints", "pre_train")
    if os.path.exists(os.path.join(PROJECT_ROOT, "checkpoints", "pre_train"))
    else os.path.join(PROJECT_ROOT, "checkpoint")
)

p = argparse.ArgumentParser()
p.add_argument("--preset", default="tiny", choices=["tiny", "small"])
p.add_argument("--model", default="deepseek", choices=["deepseek", "llama", "moe", "dense"])
p.add_argument("--data_dir", default="data/smollm_pretrain_raw")
p.add_argument("--checkpoint_dir", default=DEFAULT_CHECKPOINT_ROOT, help="Thư mục gốc chứa weights")
p.add_argument("--out_dir", default=None, help="Tùy chọn ghi đè thư mục lưu checkpoint")
p.add_argument("--stage", default="pretrained", help="Giai đoạn huấn luyện: pretrained, sft, cot")
p.add_argument("--dataset_name", default="cosmopedia-100k", help="Tên dataset để đặt tên file weight")
p.add_argument("--vocab_size", type=int, default=16384)
p.add_argument("--batch_size", type=int, default=8, help="micro-batch size")
p.add_argument("--grad_accum", type=int, default=4, help="gradient accumulation steps")
p.add_argument("--lr", type=float, default=6e-4)
p.add_argument("--min_lr_ratio", type=float, default=0.1)
p.add_argument("--warmup", type=int, default=500)
p.add_argument("--max_steps", type=int, default=15000, help="Số Step ~ cách tách nhỏ tính của epoch - default 30000")
p.add_argument("--weight_decay", type=float, default=0.1)
p.add_argument("--clip", type=float, default=1.0)
p.add_argument("--log_interval", type=int, default=20)
p.add_argument("--eval_interval", type=int, default=500)
p.add_argument("--eval_iters", type=int, default=50)
p.add_argument("--ckpt_interval", type=int, default=3000)
p.add_argument("--compile", action="store_true", help="torch.compile (hoạt động tốt với LLaMA dense)")
p.add_argument("--resume", action="store_true", help="Tự động tìm và tiếp tục từ checkpoint mặc định")
p.add_argument("--resume_ckpt", default=None, help="Đường dẫn file .pt cụ thể muốn nạp để tiếp tục train")

# DeepSeek MoE ablations
p.add_argument("--top_k", type=int, default=None, dest="num_experts_per_tok")
p.add_argument("--n_shared", type=int, default=None, dest="n_shared_experts")
p.add_argument("--n_expert", type=int, default=None, dest="n_routed_experts")
p.add_argument("--aux_coef", type=float, default=None, dest="aux_loss_coef")
args = p.parse_args()

# Chuẩn hóa đường dẫn tương đối theo PROJECT_ROOT nếu gọi từ thư mục khác
if not os.path.isabs(args.data_dir) and not os.path.exists(args.data_dir):
    candidate = os.path.join(PROJECT_ROOT, args.data_dir)
    if os.path.exists(candidate):
        args.data_dir = candidate

if not os.path.isabs(args.checkpoint_dir) and not os.path.exists(args.checkpoint_dir):
    candidate = os.path.join(PROJECT_ROOT, args.checkpoint_dir)
    if os.path.exists(candidate) or os.path.exists(os.path.dirname(candidate)):
        args.checkpoint_dir = candidate

torch.manual_seed(1337)
np.random.seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
dev = "cuda"

# ---------------- Model & Config selection
if args.model in ["llama", "dense"]:
    from model_llama import LlamaConfig, LlamaForCausalLM, PRESETS as LLAMA_PRESETS
    model_type = "llama"
    arch_name = "LLaMA-3"
    cfg = LlamaConfig(**LLAMA_PRESETS[args.preset])
    cfg.vocab_size = args.vocab_size
    raw_model = LlamaForCausalLM(cfg).to(dev)
else:
    from model_deepseek import DeepSeekConfig, DeepSeekForCausalLM, PRESETS as DEEPSEEK_PRESETS
    model_type = "deepseek"
    arch_name = "DeepSeek-MoE"
    cfg = DeepSeekConfig(**DEEPSEEK_PRESETS[args.preset])
    cfg.vocab_size = args.vocab_size
    for k in ["num_experts_per_tok", "n_shared_experts", "n_routed_experts", "aux_loss_coef"]:
        if getattr(args, k) is not None:
            setattr(cfg, k, getattr(args, k))
    raw_model = DeepSeekForCausalLM(cfg).to(dev)

total, active = raw_model.num_params()
param_str = f"{int(round(total / 1e6))}M"

# ---------------- Structured Output Folder & File Naming
# Cấu trúc: checkpoint/<arch_name>/<arch_name>_<param_str>_<stage>_<dataset_name>.pt
# Ví dụ: checkpoint/LLaMA-3/LLaMA-3_75M_pretrained_cosmopedia-100k.pt
#         checkpoint/DeepSeek-MoE/DeepSeek-MoE_190M_pretrained_cosmopedia-100k.pt
save_dir = args.out_dir if args.out_dir else os.path.join(args.checkpoint_dir, arch_name)
os.makedirs(save_dir, exist_ok=True)

main_weight_name = f"{arch_name}_{param_str}_{args.stage}_{args.dataset_name}.pt"
main_ckpt_path = os.path.join(save_dir, main_weight_name)

# ---------------- Data loading (np.memmap)
def load(split):
    bin_path = os.path.join(args.data_dir, f"{split}.bin")
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Không tìm thấy file {bin_path}. Hãy chạy prepare_data.py trước!")
    return np.memmap(bin_path, dtype=np.uint16, mode="r")

train_data, val_data = load("train"), load("val")
T = cfg.block_size
tok_per_step = args.batch_size * args.grad_accum * T
tokens_train = len(train_data)
total_tokens_target = args.max_steps * tok_per_step
total_epochs = total_tokens_target / tokens_train

gpu_name = torch.cuda.get_device_name(dev) if torch.cuda.is_available() else "CPU"
gpu_total_vram = (torch.cuda.get_device_properties(dev).total_memory / (1024**3)) if torch.cuda.is_available() else 0.0

print("=" * 70)
print(f"Architecture : {arch_name}")
print(f"Parameters   : Total {total/1e6:.1f}M | Active/token {active/1e6:.1f}M")
print(f"Hardware     : {gpu_name} ({gpu_total_vram:.1f} GB VRAM)")
print(f"Dataset Size : {tokens_train:,} tokens ({tokens_train/1e6:.1f}M tokens)")
print(f"Tokens/Step  : {tok_per_step:,} (batch={args.batch_size}, accum={args.grad_accum}, seq={T})")
print(f"Target Budget: {args.max_steps:,} steps | {total_tokens_target/1e6:.1f}M tokens | Est. {total_epochs:.2f} Epochs")
print(f"Save Folder  : {save_dir}")
print(f"Weight File  : {main_weight_name}")
print("=" * 70)

model = torch.compile(raw_model) if args.compile else raw_model


def get_batch(data):
    ix = torch.randint(len(data) - T - 1, (args.batch_size,)).tolist()
    x = torch.stack([torch.from_numpy(data[i:i + T].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + T].astype(np.int64)) for i in ix])
    return x.pin_memory().to(dev, non_blocking=True), y.pin_memory().to(dev, non_blocking=True)


# ---------------- Optimizer & LR Schedule
decay = [p for p in raw_model.parameters() if p.dim() >= 2]
no_decay = [p for p in raw_model.parameters() if p.dim() < 2]
opt = torch.optim.AdamW(
    [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
    lr=args.lr, betas=(0.9, 0.95), fused=True
)
min_lr = args.lr * args.min_lr_ratio


def get_lr(it):
    if it < args.warmup:
        return args.lr * (it + 1) / args.warmup
    if it >= args.max_steps:
        return min_lr
    r = (it - args.warmup) / (args.max_steps - args.warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * r)) * (args.lr - min_lr)


@torch.no_grad()
def evaluate():
    model.eval()
    s = 0.0
    for _ in range(args.eval_iters):
        x, y = get_batch(val_data)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s += model(x, y)["ce"].item()
    model.train()
    return s / args.eval_iters


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
    # Luôn ghi đè weight chính thống
    torch.save(payload, main_ckpt_path)

    # Nếu là checkpoint định kỳ, lưu kèm step
    if is_periodic:
        step_filename = f"{arch_name}_{param_str}_{args.stage}_{args.dataset_name}_step{step}.pt"
        torch.save(payload, os.path.join(save_dir, step_filename))


start = 0
ckpt_to_resume = None
if args.resume_ckpt:
    if not os.path.exists(args.resume_ckpt):
        raise FileNotFoundError(f"Không tìm thấy file checkpoint: {args.resume_ckpt}")
    ckpt_to_resume = args.resume_ckpt
elif args.resume and os.path.exists(main_ckpt_path):
    ckpt_to_resume = main_ckpt_path

if ckpt_to_resume:
    ck = torch.load(ckpt_to_resume, map_location=dev)
    raw_model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    start = ck["step"]
    print(f"Resumed from {ckpt_to_resume} at step {start}")

# ---------------- Training Loop
log_path = os.path.join(save_dir, f"{arch_name}_{param_str}_{args.dataset_name}_log.csv")
log_f = open(log_path, "a", newline="")
log = csv.writer(log_f)
if start == 0 and os.path.getsize(log_path) == 0:
    log.writerow([
        "step", "tokens", "train_ce", "val_ce", "aux", "z", "lr", "gnorm",
        "tok_per_s", "max_load", "min_load", "load_entropy", "mem_gb"
    ])

tok_per_step = args.batch_size * args.grad_accum * T
acc = dict(ce=0.0, aux=0.0, z=0.0)
counts_acc = None
t0 = time.time()
train_start_time = time.time()
model.train()

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
            print(f"step {s:6d}/{args.max_steps} | ep {curr_epoch:.2f}/{total_epochs:.1f} | ce {ce:.4f} | aux {aux:.3f} | lr {get_lr(step):.2e} | gn {gnorm:.2f} | "
                  f"{tps/1e3:.1f}k tok/s | load max {mx:.2f} min {mn:.3f} ent {ent:.3f} | {mem:.1f}GB | {time_str}")
        else:
            print(f"step {s:6d}/{args.max_steps} | ep {curr_epoch:.2f}/{total_epochs:.1f} | ce {ce:.4f} | lr {get_lr(step):.2e} | gn {gnorm:.2f} | "
                  f"{tps/1e3:.1f}k tok/s | {mem:.1f}GB | {time_str}")

        log.writerow([
            s, s * tok_per_step, ce, "", aux, z, get_lr(step), float(gnorm),
            tps, mx, mn, ent, mem
        ])
        log_f.flush()

    if s % args.eval_interval == 0:
        curr_epoch = (s * tok_per_step) / tokens_train
        v = evaluate()
        print(f">>> step {s}/{args.max_steps} (ep {curr_epoch:.2f}) | val_ce {v:.4f} | val_ppl {math.exp(v):.1f}")
        log.writerow([s, s * tok_per_step, "", v] + [""] * 9)
        log_f.flush()
        t0 = time.time()

    if s % args.ckpt_interval == 0:
        save(s, is_periodic=True)

save(args.max_steps, is_periodic=False)

total_train_time = time.time() - train_start_time
peak_bytes = torch.cuda.max_memory_allocated()
peak_gb = peak_bytes / 1e9
peak_gib = peak_bytes / (1024**3)
total_tokens_done = args.max_steps * tok_per_step
final_accum_epochs = total_tokens_done / tokens_train

print("=" * 70)
print("🎉 Training Completed Successfully!")
print(f"  Total Steps       : {args.max_steps:,} steps")
print(f"  Total Tokens      : {total_tokens_done:,} tokens ({total_tokens_done/1e6:.1f}M)")
print(f"  Accumulated Epochs: {final_accum_epochs:.2f} epochs")
print(f"  Total Train Time  : {format_duration(total_train_time)} ({total_train_time/3600:.2f} hours)")
if total_train_time > 0 and (args.max_steps - start) > 0:
    print(f"  Avg Throughput    : {((args.max_steps - start) * tok_per_step) / total_train_time / 1e3:.1f}k tok/s")
if torch.cuda.is_available():
    print(f"  Peak VRAM Usage   : {peak_gb:.2f} GB ({peak_gib:.2f} GiB)")
print(f"  Weights Saved At  : {main_ckpt_path}")
print("=" * 70)
