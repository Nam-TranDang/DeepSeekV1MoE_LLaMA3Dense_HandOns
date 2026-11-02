# Pretrain LLM From Scratch: LLaMA-3 (Dense) vs DeepSeek-MoE (Sparse)

Dự án tự xây dựng và huấn luyện mô hình ngôn ngữ lớn (LLM) từ đầu (from scratch) với 2 kiến trúc hiện đại nhất hiện nay:

1. **LLaMA-3 Dense**: Mạng thuần dense kế thừa kiến trúc LLaMA-3 (RMSNorm, RoPE, GQA, SwiGLU).
2. **DeepSeek-MoE**: Mạng Mixture-of-Experts hiện đại (1 Shared Expert + 8 Routed Experts, Top-2, Auxiliary Load-Balancing Loss, Router z-loss).

Dataset huấn luyện: **`HuggingFaceTB/cosmopedia-100k`** (~100k bài viết/sách giáo khoa sạch, ~75M tokens, < 1GB).
Tối ưu VRAM: Được cấu hình để chạy an toàn trên GPU **RTX 5060 Ti (VRAM $\le$ 8 GB)**, giữ mức sử dụng thực tế chỉ khoảng **3.5 – 5.5 GB**.

---

## 📁 Cấu Trúc Mã Nguồn

```
llm_scratch/
├── prepare_data.py      # Pipeline tải/stream dataset, train BPE tokenizer, đóng gói train.bin & val.bin
├── model_llama.py       # Kiến trúc LLaMA-3 Dense độc lập (100% self-contained)
├── model_deepseek.py    # Kiến trúc DeepSeek-MoE độc lập (100% self-contained)
├── train.py             # Script pretraining hỗ trợ cả LLaMA và DeepSeek, bfloat16 AMP, Cosine LR
├── generate.py          # Script sinh văn bản autoregressive sampling (Top-K, Temperature)
├── README.md            # Tài liệu hướng dẫn từng bước
└── checkpoint/          # Thư mục tự động lưu weights phân cấp theo kiến trúc
    ├── LLaMA-3/         # Chứa weights và log của LLaMA-3
    └── DeepSeek-MoE/    # Chứa weights và log của DeepSeek-MoE
```

---

## 🚀 Bước 0: Kiểm Tra Môi Trường & GPU

Kích hoạt môi trường ảo (virtualenv) của bạn và kiểm tra PyTorch cùng GPU:

```bash
cd /home/dangnam/hand_ons/llm_scratch

# Kiểm tra PyTorch và bfloat16 trên GPU
python -c "import torch; print('PyTorch:', torch.__version__, '| GPU:', torch.cuda.get_device_name(0), '| bf16 support:', torch.cuda.is_bf16_supported())"
```

*Yêu cầu thư viện:* `torch >= 2.5`, `datasets`, `tokenizers`, `numpy`, `tqdm`, `huggingface_hub`.

---

## 📦 Bước 1: Chuẩn Bị Dữ Liệu (Cosmopedia-100k)

Bạn có thể chọn **Cách A** (tải offline về máy trước) hoặc **Cách B** (stream trực tiếp từ Hugging Face).

### Cách A: Tải raw data về máy trước (Khuyên dùng)

```bash
# 1. Tải file Parquet về thư mục data/cosmopedia_raw (~600MB)
huggingface-cli download HuggingFaceTB/cosmopedia-100k --repo-type dataset --local-dir data/cosmopedia_raw

# 2. Train Tokenizer (16,384 vocab) & tokenize dữ liệu thành train.bin và val.bin
python prepare_data.py --dataset data/cosmopedia_raw --out data/cosmopedia_raw
python prepare_data.py --dataset data/smollm_pretrain_raw --out data/smollm_pretrain_raw
python prepare_data.py --dataset data/smoltalk_posttrain_raw --out data/smoltalk_posttrain_raw


```

### Cách B: Stream trực tiếp từ Hugging Face Hub

```bash
python prepare_data.py --dataset HuggingFaceTB/cosmopedia-100k --out data/cosmopedia
```

**Kết quả đầu ra tại `data/cosmopedia/`:**

* `tokenizer.json`: BPE Tokenizer với 16,384 vocab.
* `train.bin`: ~70M tokens dạng `uint16` (~140 MB trên ổ cứng).
* `val.bin`: 2M tokens dạng `uint16` (~4 MB trên ổ cứng).

---

## 🧪 Bước 2: Smoke Test (Preset `tiny`, VRAM ~1.2 GB, chạy vài phút)

Kiểm tra toàn bộ pipeline từ forward, backward, optimizer đến sinh văn bản:

### 1. Smoke test LLaMA-3 Tiny (~10M params)

```bash
python train.py --preset tiny --model llama --batch_size 16 --grad_accum 2 --max_steps 500 --eval_interval 100
```

### 2. Smoke test DeepSeek-MoE Tiny (~15M params)

```bash
python train.py --preset tiny --model deepseek --batch_size 16 --grad_accum 2 --max_steps 500 --eval_interval 100
```

**Dấu hiệu kiểm tra thành công:**

1. Loss ban đầu xấp xỉ $\ln(16384) \approx 9.7$, sau đó giảm dần đều theo từng step.
2. Với DeepSeek: thông số `aux` dao động quanh ~1.0 (các expert được định tuyến cân bằng).
3. File weights được tạo tự động tại:
   * `checkpoint/LLaMA-3/LLaMA-3_10M_pretrained_cosmopedia-100k.pt`
   * `checkpoint/DeepSeek-MoE/DeepSeek-MoE_15M_pretrained_cosmopedia-100k.pt`

### 3. Sinh thử văn bản sau smoke test

```bash
python generate.py --ckpt checkpoint/DeepSeek-MoE/DeepSeek-MoE_15M_pretrained_cosmopedia-100k.pt --prompt "The sun is"
```

---

## 🏋️ Bước 3: Huấn Luyện Chính Thức (Preset `small`, An Toàn Cho VRAM $\le$ 8GB)

Sử dụng cấu hình `--batch_size 8 --grad_accum 8` để giữ effective batch size là $65,536$ tokens/step, trong khi giới hạn peak VRAM ở mức **3.5 – 5.5 GB** (không đẩy GPU lên 100% giới hạn nhiệt/bộ nhớ):

### 1. Huấn luyện LLaMA-3 Dense (~73M params, Peak VRAM ~3.5 GB)

```bash
python train.py --preset small --model llama \
    --batch_size 8 --grad_accum 8 \
    --max_steps 15000 --eval_interval 500 --ckpt_interval 1000
```

* Weight file: `checkpoint/LLaMA-3/LLaMA-3_73M_pretrained_cosmopedia-100k.pt`
* Log metrics: `checkpoint/LLaMA-3/LLaMA-3_73M_log.csv`

### 2. Huấn luyện DeepSeek-MoE (~177M total / 73M active params, Peak VRAM ~5.2 GB)

```bash
python train.py --preset small --model deepseek \
    --batch_size 8 --grad_accum 8 \
    --max_steps 15000 --eval_interval 500 --ckpt_interval 1000
```

* Weight file: `checkpoint/DeepSeek-MoE/DeepSeek-MoE_177M_pretrained_cosmopedia-100k.pt`
* Log metrics: `checkpoint/DeepSeek-MoE/DeepSeek-MoE_177M_log.csv`

---

## 💬 Bước 4: Sinh Văn Bản (Inference)

Sau khi huấn luyện, kiểm tra khả năng sinh văn bản của mô hình:

```bash
# Kiểm tra model LLaMA-3
python generate.py \
    --ckpt checkpoint/LLaMA-3/LLaMA-3_73M_pretrained_cosmopedia-100k.pt \
    --prompt "Photosynthesis is the process where" \
    --max_new 150 --temp 0.8

# Kiểm tra model DeepSeek-MoE
python generate.py \
    --ckpt checkpoint/DeepSeek-MoE/DeepSeek-MoE_177M_pretrained_cosmopedia-100k.pt \
    --prompt "Photosynthesis is the process where" \
    --max_new 150 --temp 0.8

python generate.py \
    --ckpt checkpoint/DeepSeek-MoE/DeepSeek-MoE_177M_pretrained_smollm.pt \
    --prompt "Photosynthesis is the process where" \
    --max_new 150 --temp 0.8
```

---

## 🔬 Bước 5: So Sánh & Thí Nghiệm Ablation (DeepSeek-MoE)

Mỗi lần thay đổi một biến số để kiểm tra ảnh hưởng đến tốc độ học và cân bằng chuyên gia:

| Thí nghiệm                        | Cờ lệnh thêm     | Mục tiêu quan sát                                                                 |
| :---------------------------------- | :------------------ | :----------------------------------------------------------------------------------- |
| **Không có Shared Expert**  | `--n_shared 0`    | Xem khả năng giữ tri thức dùng chung bị suy giảm như thế nào               |
| **Top-1 routing**             | `--top_k 1`       | So sánh tốc độ và độ chính xác so với Top-2                                |
| **16 Experts**                | `--n_expert 16`   | Tăng dung lượng tham số và kiểm tra tải định tuyến                         |
| **Tắt Load Balancing Loss**  | `--aux_coef 0`    | Quan sát hiện tượng *Expert Collapse* (chỉ 1–2 expert nhận toàn bộ token) |
| **Tăng Load Balancing Loss** | `--aux_coef 0.05` | Ép định tuyến đồng đều hơn                                                  |

---

## ⚙️ Các Lưu Ý Quan Trọng Cho Phần Cứng & Code

1. **Khôi phục tiến trình khi bị ngắt (Resume):**
   Nếu bạn bấm `Ctrl+C` hoặc máy bị dừng, chỉ cần thêm cờ `--resume` vào lệnh cũ:

   ```bash
   python train.py --preset small --model deepseek --batch_size 8 --grad_accum 8 --resume
   ```
   Script sẽ tự động tìm checkpoint tương ứng trong thư mục `checkpoint/` và tiếp tục huấn luyện từ step đã lưu.
2. **Cờ `--compile` (PyTorch 2.0+):**

   * Hoạt động rất tốt với `--model llama` (tăng tốc độ forward/backward ~15-20%).
   * Không nên dùng với `--model deepseek` do vòng lặp dispatch Python có tính động (dynamic graph break).
3. **Giới hạn cứng VRAM cho PyTorch (Tùy chọn):**
   Nếu muốn đảm bảo PyTorch không bao giờ chiếm quá 75% VRAM (dành tài nguyên cho tác vụ hiển thị khác), có thể đặt biến môi trường:

   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   ```
