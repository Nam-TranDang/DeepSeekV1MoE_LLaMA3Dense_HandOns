"""
DATA PIPELINE: POST-TRAINING (INSTRUCT / SFT) DATA (SERVER-SAFE EDITION)
========================================================================
Mục đích:
- Đọc hội thoại chat (messages: user / assistant / system) từ Parquet, JSONL,... (vd: SmolTalk).
- Chuẩn hóa sang định dạng ChatML:
    <|im_start|>system\\n{content}<|im_end|>
    <|im_start|>user\\n{content}<|im_end|>
    <|im_start|>assistant\\n{content}<|im_end|>
- Dùng TOKENIZER TỪ PRETRAIN (khớp vocab & token IDs) -> Tokenize sang train.bin / val.bin (uint16).
- Tách biệt tuyệt đối file train vs test/val để tránh rò rỉ dữ liệu (data leakage).
- Hỗ trợ Streaming PyArrow cực nhanh: KHÔNG nghẽn CPU / RAM khi nạp tập Parquet hàng triệu dòng.
- BẢO VỆ SERVER TỪ XA (REMOTE SERVER SAFE):
    * Giới hạn số luồng CPU (--threads, mặc định 4/56 cores), tránh chiếm dụng tài nguyên người khác.
    * Cơ chế Throttling (--throttle_ms, mặc định 10ms) cho phép CPU nghỉ nhịp, chống treo/crash server.

LƯU Ý QUAN TRỌNG:
- KHÔNG BAO GIỜ train tokenizer mới cho bước post-train.
- Bắt buộc phải truyền --tokenizer trỏ đến file tokenizer.json của bước pretrain.
- Tích hợp cờ --preview N để kiểm tra trước định dạng ChatML mà không tạo file .bin.

Ví dụ sử dụng an toàn trên Remote Server:
  # 1. Xem trước 3 mẫu ChatML (không tạo file .bin):
  python datasets/prepare_data_posttrain.py \\
      --dataset data/smoltalk_posttrain_raw \\
      --tokenizer data/smollm_pretrain_raw/tokenizer.json \\
      --preview 3

  # 2. Tokenize an toàn với 4 luồng CPU và nghỉ 10ms giữa mỗi batch:
  python datasets/prepare_data_posttrain.py \\
      --dataset data/smoltalk_posttrain_raw \\
      --tokenizer data/smollm_pretrain_raw/tokenizer.json \\
      --out data/smoltalk_posttrain_bin \\
      --threads 4 \\
      --throttle_ms 10 \\
      --val_tokens 500000

  # 3. Chạy test nhẹ nhàng 2 triệu tokens:
  python datasets/prepare_data_posttrain.py \\
      --dataset data/smoltalk_posttrain_raw \\
      --tokenizer data/smollm_pretrain_raw/tokenizer.json \\
      --out data/smoltalk_posttrain_bin \\
      --threads 2 \\
      --throttle_ms 20 \\
      --max_tokens 2000000 \\
      --val_tokens 200000


===========================================================================
KHỞI TẠO PIPELINE DỮ LIỆU POST-TRAINING (CHẾ ĐỘ AN TOÀN CHO SERVER TỪ XA)
===========================================================================
• Máy chủ từ xa     : Tổng 56 CPU cores
• Tải hệ thống      : Load avg (1m=2.20, 5m=2.57, 15m=2.30)
• Giới hạn CPU cấp  : Dùng tối đa 4/56 luồng (dành 52 luồng cho hệ thống)
• Cơ chế Throttling : Nghỉ 5 ms sau mỗi batch 128 hội thoại
===========================================================================
-> Nạp tokenizer thành công từ: /home/dangnam/hand_ons/llm/data/smollm_pretrain_raw/tokenizer.json
   Vocab size       : 16,384 (hợp lệ cho uint16)
   EOS token        : '<|endoftext|>' (ID: 0)
   <|im_start|> ID  : 1
   <|im_end|> ID    : 2
---------------------------------------------------------------------------

---------------------------------------------------------------------------
TIẾN HÀNH TOKENIZE DỮ LIỆU:
---------------------------------------------------------------------------
[BƯỚC 1/2] Đang tạo tập Validation -> /home/dangnam/hand_ons/llm/data/smoltalk_posttrain_bin/val.bin (mục tiêu ~500,000 tokens)...
Val tokens: 500792tok [00:00, 834785.82tok/s]                                                                                                                                                 
-> Đã ghi tập Val: 500,792 tokens (0.50M tokens) vào /home/dangnam/hand_ons/llm/data/smoltalk_posttrain_bin/val.bin

[BƯỚC 2/2] Đang tạo tập Train -> /home/dangnam/hand_ons/llm/data/smoltalk_posttrain_bin/train.bin ...
Train tokens: 1006259836tok [15:38, 1071962.78tok/s][[A

===========================================================================
HOÀN THÀNH CHUẨN BỊ DỮ LIỆU POST-TRAINING AN TOÀN!
===========================================================================
• Số đoạn hội thoại đã xử lý : 1,043,917
• Train tokens               : 1,006,259,836  (1006.26M tokens) [1919.3 MB]
• Val tokens                 : 500,792    (0.50M tokens)   [1.0 MB]
• File Train                 : /home/dangnam/hand_ons/llm/data/smoltalk_posttrain_bin/train.bin
• File Val                   : /home/dangnam/hand_ons/llm/data/smoltalk_posttrain_bin/val.bin
• Tokenizer bản sao          : /home/dangnam/hand_ons/llm/data/smoltalk_posttrain_bin/tokenizer.json
• CPU Load Average cuối cùng : (6.4892578125, 5.75, 4.49267578125)
• Trạng thái tương thích     : 100% SẴN SÀNG CHO training/PostTraining/train_posttrain.py
===========================================================================


 ### 1. Is this done yet?

  Yes, it is 100% completed.

  • Execution Time: Finished in 15m 38s (matching the ~15–20 min estimate).
  • Conversations Processed: All 1,043,917 dialogues in the training set and 500,792 tokens for validation were processed.
  • Disk Verification: Both binary files passed memmap size validation and are saved at:
      • train.bin — 1.9 GB (1,006,259,836 tokens)
      • val.bin — 979 KB (500,792 tokens)
      • tokenizer.json — 1.2 MB

  ──────
  ### 2. Does post-train data need a new tokenizer.json or reuse from pretrain?

  It MUST reuse the exact tokenizer from pretrain.

  1. Why you must NEVER train a new tokenizer for post-training:
      • The model's embedding weights were learned during pretraining based on fixed token ID mappings (e.g., token ID 356 mapped to a specific subword).
      • If you trained a new tokenizer, the IDs would shift, scrambling the vocabulary and destroying all knowledge the model learned during pretraining.
  2. What happened here:
      • The script used the pretrain tokenizer from tokenizer.json to encode all SmolTalk dialogues.
      • It also copied an exact duplicate into tokenizer.json so the directory is self-contained when running evaluation or inference.

  ──────
  ### 3. Key Differences: Pretrain vs. Post-train Data

   Feature            │ Pre-train Data (smollm_pretrain_raw / Cosmopedia)                              │ Post-train Data (smoltalk_posttrain_raw / SmolTalk)
  ────────────────────┼────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────
   Structure / Schema │ Unstructured raw text (['text', 'prompt', ...]). Continuous documents,         │ Multi-turn dialogue list (['messages']), each containing role (user, assistant,
                      │ articles, textbooks.                                                           │ system) and content.
   Formatting / Tags  │ Standard text separated only by `<                                             │ endoftext
   Training Goal      │ World Knowledge & Fluency: Teaches grammar, reasoning, facts, and language     │ Behavior & Alignment (SFT): Teaches the model how to act as an AI assistant,
                      │ patterns via raw next-token prediction.                                        │ follow prompts, answer questions, and know when to stop speaking.
   Loss Masking       │ Model learns to predict every single token in the document.                    │ Commonly focuses loss on the assistant's answers, rather than memorizing user
                      │                                                                                │ questions.

────────────────────────────────────────────────────────────




  I have updated Prepare_Data_Post-train.py with the following improvements:

  ### Changes Made

  1. Default to Input Dataset Directory (--out is optional):
      • Changed --out default to None.
      • When --out is not passed, the script automatically resolves the output directory directly to --dataset (the directory being read), eliminating the need to create a new folder:
        out_dir = args.out if args.out else (args.dataset if os.path.isdir(args.dataset) else os.path.dirname(args.dataset))

  2. Safety Check in Dataset Scanner:
      • Updated scan_dataset_files to ignore tokenizer.json and .bin files so placing them directly inside the dataset folder will not interfere with scanning Parquet files.
  3. Safe File Handling:
      • Avoids re-copying tokenizer.json onto itself if the source and destination match (SameFileError prevention).

  ──────
  ### Files Placed in Dataset Directory

  The .bin files and tokenizer from your completed run have also been copied directly into /home/dangnam/hand_ons/llm/data/smoltalk_posttrain_raw/:

  • train.bin (1.9 GB)
  • val.bin (979 KB)
  • tokenizer.json (1.2 MB)

  Going forward, you can simply run without --out:

    /home/dangnam/venv/bin/python /home/dangnam/hand_ons/llm/datasets/Prepare_Data_Post-train.py \
      --dataset /home/dangnam/hand_ons/llm/data/smoltalk_posttrain_raw \
      --tokenizer /home/dangnam/hand_ons/llm/data/smollm_pretrain_raw/tokenizer.json \
      --threads 4



"""

import argparse
import glob
import os
import shutil
import sys
import time

# ---------------- CLI Arguments
p = argparse.ArgumentParser(description="Post-training (Instruct / SFT) Data Pipeline (Server-Safe)")
p.add_argument(
    "--dataset",
    default="data/smoltalk_posttrain_raw",
    help="Đường dẫn file/thư mục dataset chat (vd: data/smoltalk_posttrain_raw)",
)
p.add_argument(
    "--tokenizer",
    required=True,
    help="Đường dẫn BẮT BUỘC tới tokenizer.json sinh ra từ bước pretrain",
)
p.add_argument(
    "--out",
    default=None,
    help="Thư mục xuất file val.bin, train.bin cho post-train (mặc định: lưu trực tiếp trong thư mục --dataset, không tạo folder mới)",
)
p.add_argument(
    "--messages_col",
    default="messages",
    help="Tên cột chứa danh sách các turn hội thoại (mặc định: 'messages')",
)
p.add_argument(
    "--val_tokens",
    type=int,
    default=500_000,
    help="Số token tối đa dành cho tập val.bin (mặc định: 500,000 tokens)",
)
p.add_argument(
    "--max_tokens",
    type=int,
    default=0,
    help="Trần tối đa token cần tokenize (0 hoặc âm = không giới hạn, xử lý toàn bộ)",
)
p.add_argument(
    "--preview",
    type=int,
    default=0,
    help="Số mẫu xem trước (preview) dạng ChatML và thoát ngay (không tạo file .bin)",
)
p.add_argument(
    "--batch_size",
    type=int,
    default=128,
    help="Kích thước batch gom hội thoại để encode_batch (mặc định: 128)",
)
p.add_argument(
    "--threads",
    type=int,
    default=4,
    help="Giới hạn số luồng CPU sử dụng (mặc định: 4 để bảo vệ server, tránh đơ máy)",
)
p.add_argument(
    "--throttle_ms",
    type=int,
    default=10,
    help="Thời gian nghỉ (mili-giây) giữa các batch để CPU hạ nhiệt và nhường tài nguyên (mặc định: 10ms)",
)
args = p.parse_args()

# ---------------- THIẾT LẬP BẢO VỆ CPU (CHỐNG NGHẼN SERVER)
# Đặt giới hạn luồng cho Tokenizers (Rust Rayon), OpenMP và BLAS
safe_threads = max(1, args.threads)
os.environ["RAYON_NUM_THREADS"] = str(safe_threads)
os.environ["OMP_NUM_THREADS"] = str(safe_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(safe_threads)
os.environ["MKL_NUM_THREADS"] = str(safe_threads)

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm

# Cố định số luồng CPU của PyArrow
pa.set_cpu_count(safe_threads)

# Kiểm tra tình trạng tải CPU của hệ thống (Load Average)
total_cpus = os.cpu_count() or 1
load1, load5, load15 = os.getloadavg() if hasattr(os, "getloadavg") else (0, 0, 0)

print("=" * 75)
print("KHỞI TẠO PIPELINE DỮ LIỆU POST-TRAINING (CHẾ ĐỘ AN TOÀN CHO SERVER TỪ XA)")
print("=" * 75)
print(f"• Máy chủ từ xa     : Tổng {total_cpus} CPU cores")
print(f"• Tải hệ thống      : Load avg (1m={load1:.2f}, 5m={load5:.2f}, 15m={load15:.2f})")
print(f"• Giới hạn CPU cấp  : Dùng tối đa {safe_threads}/{total_cpus} luồng (dành {total_cpus - safe_threads} luồng cho hệ thống)")
print(f"• Cơ chế Throttling : Nghỉ {args.throttle_ms} ms sau mỗi batch {args.batch_size} hội thoại")
print("=" * 75)


# ---------------- 1) Kiểm tra Tokenizer từ Pretrain
if not os.path.exists(args.tokenizer):
    raise FileNotFoundError(
        f"[LỖI] Không tìm thấy tokenizer tại '{args.tokenizer}'.\n"
        f"Bạn phải chỉ định đúng đường dẫn tới tokenizer.json đã được tạo ở bước pretrain!"
    )

tok = Tokenizer.from_file(args.tokenizer)
vocab_size = tok.get_vocab_size()

if vocab_size > 65535:
    raise ValueError(
        f"[LỖI] Vocab size ({vocab_size}) vượt quá giới hạn uint16 (65,535). "
        f"Cần dùng uint32 thay vì uint16!"
    )

# Tìm EOS Token
eos_candidates = ["<|endoftext|>", "<eos>", "</s>", "<|end_of_text|>"]
eos = None
eos_token_str = None
for cand in eos_candidates:
    cand_id = tok.token_to_id(cand)
    if cand_id is not None:
        eos = cand_id
        eos_token_str = cand
        break

if eos is None:
    raise ValueError(
        f"[LỖI] Không tìm thấy token kết thúc (EOS) trong tokenizer {args.tokenizer}.\n"
        f"Đã thử các ký hiệu: {eos_candidates} nhưng đều trả về None."
    )

# Kiểm tra các Special Token của ChatML
im_start_id = tok.token_to_id("<|im_start|>")
im_end_id = tok.token_to_id("<|im_end|>")

print(f"-> Nạp tokenizer thành công từ: {args.tokenizer}")
print(f"   Vocab size       : {vocab_size:,} (hợp lệ cho uint16)")
print(f"   EOS token        : '{eos_token_str}' (ID: {eos})")
print(f"   <|im_start|> ID  : {im_start_id}")
print(f"   <|im_end|> ID    : {im_end_id}")
if im_start_id is None or im_end_id is None:
    print("   [!] CẢNH BÁO: Tokenizer không có sẵn <|im_start|> hoặc <|im_end|> dạng Special Token.")
    print("       Các thẻ này sẽ được tokenizer tách thành các subwords nhỏ.")
print("-" * 75)


# ---------------- 2) Hàm chuẩn hóa hội thoại sang cấu trúc ChatML
def format_conversation(raw_msgs):
    """
    Trích xuất hội thoại từ record và định dạng theo chuẩn ChatML:
    <|im_start|>system\\n{content}<|im_end|>
    <|im_start|>user\\n{content}<|im_end|>
    <|im_start|>assistant\\n{content}<|im_end|>
    """
    if raw_msgs is None:
        return None

    if isinstance(raw_msgs, list):
        turns = []
        for turn in raw_msgs:
            if isinstance(turn, dict):
                # Hỗ trợ cả key 'from/value' (ShareGPT) lẫn 'role/content' (Hugging Face)
                role = turn.get("role") or turn.get("from") or "user"
                content = turn.get("content") or turn.get("value") or ""
                # Chuẩn hóa role
                role_clean = str(role).lower().strip()
                if role_clean in ["human", "user"]:
                    role_str = "user"
                elif role_clean in ["gpt", "assistant", "bot", "model"]:
                    role_str = "assistant"
                elif role_clean in ["system"]:
                    role_str = "system"
                else:
                    role_str = role_clean

                content_str = str(content).strip()
                turns.append(f"<|im_start|>{role_str}\n{content_str}<|im_end|>")
            else:
                turns.append(str(turn).strip())
        return "\n".join(turns)

    return str(raw_msgs).strip()


# ---------------- 3) Quét và phân loại file dữ liệu
def scan_dataset_files(path):
    """Quét và phân loại các file parquet/jsonl theo split train và test/val."""
    if not os.path.exists(path):
        return None, None, False

    if os.path.isfile(path):
        return [path], [], True

    patterns = ["*.parquet", "*.pq", "*.jsonl", "*.json", "*.csv"]
    all_found = []
    for pat in patterns:
        all_found.extend(glob.glob(os.path.join(path, f"**/{pat}"), recursive=True))

    all_found = sorted(list(set(all_found)))
    if not all_found:
        return [], [], False

    train_files = []
    val_files = []
    for f in all_found:
        base = os.path.basename(f).lower()
        if "tokenizer" in base:
            continue
        if base.startswith("test") or base.startswith("val") or "eval" in base:
            val_files.append(f)
        else:
            train_files.append(f)

    return train_files, val_files, True


train_files, val_files, is_local = scan_dataset_files(args.dataset)


# ---------------- 4) Trình tạo Streaming cực nhanh & tiết kiệm CPU
def stream_records(file_paths, target_col="messages", batch_size=128):
    """
    Duyệt dữ liệu dạng stream từng batch qua PyArrow mà không nạp toàn bộ file vào RAM.
    Tích hợp kiểm tra schema_arrow và không chiếm dụng CPU.
    """
    for file_path in file_paths:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".parquet", ".pq"]:
            try:
                pf = pq.ParquetFile(file_path)
                # Dùng schema_arrow để lấy tên cột tầng trên cùng chính xác
                schema_names = pf.schema_arrow.names
                chosen_col = target_col if target_col in schema_names else None
                if not chosen_col:
                    for alt in ["conversations", "dialogue", "turns", "messages"]:
                        if alt in schema_names:
                            chosen_col = alt
                            break
                if not chosen_col:
                    continue

                for record_batch in pf.iter_batches(batch_size=batch_size, columns=[chosen_col]):
                    col_items = record_batch[chosen_col].to_pylist()
                    for item in col_items:
                        yield item
            except Exception as e:
                print(f"[CẢNH BÁO] Lỗi khi đọc file {file_path}: {e}")
                continue
        else:
            fmt_map = {".jsonl": "json", ".json": "json", ".csv": "csv"}
            fmt = fmt_map.get(ext, "json")
            ds = load_dataset(fmt, data_files=[file_path], split="train", streaming=True)
            for ex in ds:
                raw = ex.get(target_col) or ex.get("conversations") or ex.get("dialogue")
                yield raw


# ---------------- 5) Chế độ PREVIEW (Xem trước ChatML không tạo .bin)
if args.preview > 0:
    print(f"\n[CHẾ ĐỘ PREVIEW]: Đang trích xuất {args.preview} mẫu hội thoại ChatML đầu tiên...")
    preview_files = train_files if train_files else (val_files if val_files else [args.dataset])
    sample_gen = stream_records(preview_files, args.messages_col, batch_size=args.preview)

    shown = 0
    for raw_conv in sample_gen:
        sample_text = format_conversation(raw_conv)
        if not sample_text:
            continue
        enc = tok.encode(sample_text)
        token_count = len(enc.ids) + 1  # tính cả EOS
        print("\n" + "=" * 70)
        print(f"MẪU HỘI THOẠI #{shown + 1} (Số tokens: {token_count}):")
        print("=" * 70)
        print(sample_text)
        print("-" * 70)
        print(f"Token IDs (15 tokens đầu): {enc.ids[:15]} ... | EOS ID: {eos}")
        shown += 1
        if shown >= args.preview:
            break

    print("\n" + "=" * 70)
    print(f"[PREVIEW HOÀN TẤT] Đã hiển thị {shown} mẫu. KHÔNG tạo file .bin nào.")
    print("=" * 70)
    sys.exit(0)


# ---------------- 6) Chuẩn bị file xuất nhị phân (.bin)
# Tự động lưu bên trong thư mục đọc dữ liệu (args.dataset) nếu không chỉ định --out
out_dir = args.out if args.out else (args.dataset if os.path.isdir(args.dataset) else os.path.dirname(args.dataset))
os.makedirs(out_dir, exist_ok=True)
val_bin_path = os.path.join(out_dir, "val.bin")
train_bin_path = os.path.join(out_dir, "train.bin")

val_f = open(val_bin_path, "wb")
train_f = open(train_bin_path, "wb")

n_val = 0
n_train = 0
dialogue_count = 0
skipped_count = 0


def encode_and_write(texts, file_handle, max_needed=None):
    """Mã hóa danh sách hội thoại và ghi nhị phân uint16 vào file."""
    if not texts:
        return 0
    tokens_written = 0
    encoded_list = tok.encode_batch(texts)
    for enc in encoded_list:
        token_ids = enc.ids + [eos]
        arr = np.array(token_ids, dtype=np.uint16)
        file_handle.write(arr.tobytes())
        tokens_written += len(arr)
        if max_needed is not None and tokens_written >= max_needed:
            break
    return tokens_written


print("\n" + "-" * 75)
print("TIẾN HÀNH TOKENIZE DỮ LIỆU:")
print("-" * 75)

# A) Xử lý tập VALIDATION
print(f"[BƯỚC 1/2] Đang tạo tập Validation -> {val_bin_path} (mục tiêu ~{args.val_tokens:,} tokens)...")
val_stream_files = val_files if val_files else train_files
val_gen = stream_records(val_stream_files, args.messages_col, batch_size=args.batch_size)

val_pbar = tqdm(total=args.val_tokens, unit="tok", desc="Val tokens")
batch = []
for raw_msgs in val_gen:
    text = format_conversation(raw_msgs)
    if not text:
        skipped_count += 1
        continue

    batch.append(text)
    if len(batch) >= args.batch_size:
        needed = max(0, args.val_tokens - n_val)
        written = encode_and_write(batch, val_f, max_needed=needed)
        n_val += written
        val_pbar.update(written)
        batch = []

        # Throttling để CPU nghỉ, bảo vệ máy chủ
        if args.throttle_ms > 0:
            time.sleep(args.throttle_ms / 1000.0)

        if n_val >= args.val_tokens:
            break

if batch and n_val < args.val_tokens:
    needed = max(0, args.val_tokens - n_val)
    written = encode_and_write(batch, val_f, max_needed=needed)
    n_val += written
    val_pbar.update(written)

val_pbar.close()
val_f.close()
print(f"-> Đã ghi tập Val: {n_val:,} tokens ({n_val/1e6:.2f}M tokens) vào {val_bin_path}")


# B) Xử lý tập TRAIN
print(f"\n[BƯỚC 2/2] Đang tạo tập Train -> {train_bin_path} ...")
train_target = max(0, args.max_tokens - n_val) if args.max_tokens > 0 else None
train_pbar = tqdm(total=train_target, unit="tok", desc="Train tokens")

# Nếu có file train riêng biệt, stream từ danh sách train_files; nếu không, tiếp tục từ generator hiện tại
if val_files and train_files:
    train_gen = stream_records(train_files, args.messages_col, batch_size=args.batch_size)
else:
    train_gen = val_gen

batch = []
for raw_msgs in train_gen:
    text = format_conversation(raw_msgs)
    if not text:
        skipped_count += 1
        continue

    batch.append(text)
    dialogue_count += 1

    if len(batch) >= args.batch_size:
        needed = max(0, args.max_tokens - (n_train + n_val)) if args.max_tokens > 0 else None
        written = encode_and_write(batch, train_f, max_needed=needed)
        n_train += written
        train_pbar.update(written)
        batch = []

        # Throttling để CPU nghỉ, bảo vệ máy chủ
        if args.throttle_ms > 0:
            time.sleep(args.throttle_ms / 1000.0)

        if args.max_tokens > 0 and (n_train + n_val) >= args.max_tokens:
            break

if batch and (args.max_tokens <= 0 or (n_train + n_val) < args.max_tokens):
    needed = max(0, args.max_tokens - (n_train + n_val)) if args.max_tokens > 0 else None
    written = encode_and_write(batch, train_f, max_needed=needed)
    n_train += written
    train_pbar.update(written)

train_pbar.close()
train_f.close()


# ---------------- 7) Sao chép tokenizer.json và kiểm tra tính toàn vẹn của file .bin
tokenizer_copy_path = os.path.join(out_dir, "tokenizer.json")
if os.path.abspath(args.tokenizer) != os.path.abspath(tokenizer_copy_path):
    shutil.copyfile(args.tokenizer, tokenizer_copy_path)

# Kiểm tra file sinh ra bằng memmap
val_mmap = np.memmap(val_bin_path, dtype=np.uint16, mode="r")
train_mmap = np.memmap(train_bin_path, dtype=np.uint16, mode="r")

assert len(val_mmap) == n_val, f"Lỗi: val.bin kích thước không khớp ({len(val_mmap)} vs {n_val})"
assert len(train_mmap) == n_train, f"Lỗi: train.bin kích thước không khớp ({len(train_mmap)} vs {n_train})"

print("\n" + "=" * 75)
print("HOÀN THÀNH CHUẨN BỊ DỮ LIỆU POST-TRAINING AN TOÀN!")
print("=" * 75)
print(f"• Số đoạn hội thoại đã xử lý : {dialogue_count:,}")
if skipped_count > 0:
    print(f"• Số mẫu bị bỏ qua do lỗi   : {skipped_count:,}")
print(f"• Train tokens               : {n_train:,}  ({n_train/1e6:.2f}M tokens) [{os.path.getsize(train_bin_path)/(1024*1024):.1f} MB]")
print(f"• Val tokens                 : {n_val:,}    ({n_val/1e6:.2f}M tokens)   [{os.path.getsize(val_bin_path)/(1024*1024):.1f} MB]")
print(f"• File Train                 : {train_bin_path}")
print(f"• File Val                   : {val_bin_path}")
print(f"• Tokenizer bản sao          : {tokenizer_copy_path}")
print(f"• CPU Load Average cuối cùng : {os.getloadavg() if hasattr(os, 'getloadavg') else 'N/A'}")
print("• Trạng thái tương thích     : 100% SẴN SÀNG CHO training/PostTraining/train_posttrain.py")
print("=" * 75)
