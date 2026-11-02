"""
INDEPENDENT POST-TRAINING DATA READER & INSPECTOR
==================================================
Mục đích:
- Đọc và khảo sát cấu trúc file Parquet của bộ dữ liệu Post-Training (SFT/Instruct) như SmolTalk.
- KHÔNG sinh ra file .bin (no-bin inspection), không tốn dung lượng ổ đĩa.
- Hiển thị cấu trúc Schema, phân bố nguồn dữ liệu (sources/subsets), định dạng ChatML và thống kê token.

Cách dùng:
  # 1. Đọc tổng quan và in 3 mẫu hội thoại:
  python datasets/read_posttrain_data.py

  # 2. Đọc kèm Tokenizer để xem số token từng turn:
  python datasets/read_posttrain_data.py --tokenizer data/smollm_pretrain_raw/tokenizer.json

  # 3. Xuất kết quả preview ra file text:
  python datasets/read_posttrain_data.py --export_preview data/preview_smoltalk_posttrain.txt
"""

import argparse
import glob
import os
import sys
from collections import Counter

import pyarrow.parquet as pq


def parse_args():
    p = argparse.ArgumentParser(description="Khảo sát dữ liệu Post-Training Parquet (Không tạo file .bin)")
    p.add_argument(
        "--dataset",
        default="data/smoltalk_posttrain_raw",
        help="Đường dẫn thư mục hoặc file parquet post-training (mặc định: data/smoltalk_posttrain_raw)",
    )
    p.add_argument(
        "--tokenizer",
        default="data/smollm_pretrain_raw/tokenizer.json",
        help="Đường dẫn tokenizer.json (tùy chọn, để xem phân tích token)",
    )
    p.add_argument(
        "--split",
        choices=["all", "train", "test"],
        default="all",
        help="Chọn split để khảo sát chi tiết (mặc định: all)",
    )
    p.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Số lượng mẫu hội thoại hiển thị chi tiết (mặc định: 3)",
    )
    p.add_argument(
        "--stats",
        action="store_true",
        default=True,
        help="Thống kê phân bố nguồn dữ liệu (source/subset)",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=2,
        help="Giới hạn số luồng CPU (mặc định: 2 để an toàn cho remote server)",
    )
    p.add_argument(
        "--export_preview",
        type=str,
        default=None,
        help="Đường dẫn file .txt để xuất bản preview (nếu muốn lưu lại)",
    )
    return p.parse_args()


def find_parquet_files(path):
    """Tìm và phân loại các file parquet thành train và test/val."""
    if os.path.isfile(path):
        return [path], []

    all_files = sorted(glob.glob(os.path.join(path, "**/*.parquet"), recursive=True))
    if not all_files:
        all_files = sorted(glob.glob(os.path.join(path, "**/*.pq"), recursive=True))

    train_files = []
    test_files = []

    for f in all_files:
        base = os.path.basename(f).lower()
        if base.startswith("test") or base.startswith("val") or "eval" in base:
            test_files.append(f)
        else:
            train_files.append(f)

    return train_files, test_files


def format_chatml(messages):
    """Chuyển danh sách messages (role, content) sang định dạng ChatML tiêu chuẩn."""
    turns = []
    for turn in messages:
        if isinstance(turn, dict):
            role = turn.get("role") or turn.get("from") or "user"
            content = turn.get("content") or turn.get("value") or ""
            # Chuẩn hóa role
            role_lower = str(role).lower().strip()
            if role_lower in ["human", "user"]:
                role_str = "user"
            elif role_lower in ["gpt", "assistant", "bot", "model"]:
                role_str = "assistant"
            elif role_lower in ["system"]:
                role_str = "system"
            else:
                role_str = role_lower
            turns.append(f"<|im_start|>{role_str}\n{content}<|im_end|>")
        else:
            turns.append(str(turn))
    return "\n".join(turns)


def main():
    args = parse_args()
    
    # Giới hạn luồng CPU an toàn cho remote server
    safe_threads = max(1, args.threads)
    os.environ["RAYON_NUM_THREADS"] = str(safe_threads)
    os.environ["OMP_NUM_THREADS"] = str(safe_threads)
    import pyarrow as pa
    pa.set_cpu_count(safe_threads)

    output_lines = []

    def log(msg=""):
        print(msg)
        output_lines.append(msg)

    log("=" * 80)
    log("KHẢO SÁT DỮ LIỆU POST-TRAINING (SFT / INSTRUCT) - NO BINARY EXPORT")
    log(f"Đường dẫn dataset: {args.dataset}")
    log("=" * 80)

    if not os.path.exists(args.dataset):
        log(f"[LỖI] Không tìm thấy đường dẫn dataset tại: '{args.dataset}'")
        sys.exit(1)

    train_files, test_files = find_parquet_files(args.dataset)
    total_files = len(train_files) + len(test_files)

    if total_files == 0:
        log(f"[LỖI] Không tìm thấy file parquet nào trong '{args.dataset}'!")
        sys.exit(1)

    log(f"-> Tìm thấy tổng cộng {total_files} file Parquet:")
    log(f"   + Tập TRAIN : {len(train_files)} files")
    for f in train_files[:3]:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        log(f"     * {os.path.basename(f)} ({size_mb:.1f} MB)")
    if len(train_files) > 3:
        log(f"     * ... và {len(train_files) - 3} file train khác")

    log(f"   + Tập TEST/VAL: {len(test_files)} files")
    for f in test_files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        log(f"     * {os.path.basename(f)} ({size_mb:.1f} MB)")

    # 1. Đọc Schema từ file đầu tiên
    first_file = train_files[0] if train_files else test_files[0]
    schema = pq.read_schema(first_file)
    log("\n" + "-" * 80)
    log("1. CẤU TRÚC PARQUET SCHEMA (CÁC CỘT & KIỂU DỮ LIỆU):")
    log("-" * 80)
    for col_name, field in zip(schema.names, schema):
        log(f"  • Cột '{col_name}': {field.type}")

    # 2. Đếm số dòng tổng cộng qua PyArrow Metadata (cực nhanh, không cần nạp dữ liệu vào RAM)
    log("\n" + "-" * 80)
    log("2. THỐNG KÊ SỐ LƯỢNG MẪU HỘI THOẠI (ROWS):")
    log("-" * 80)
    train_rows = sum(pq.read_metadata(f).num_rows for f in train_files)
    test_rows = sum(pq.read_metadata(f).num_rows for f in test_files)
    total_rows = train_rows + test_rows

    log(f"  • Tổng số mẫu (All)   : {total_rows:,} hội thoại")
    log(f"  • Số mẫu Train        : {train_rows:,} ({train_rows/max(1, total_rows)*100:.1f}%)")
    log(f"  • Số mẫu Test/Val     : {test_rows:,} ({test_rows/max(1, total_rows)*100:.1f}%)")

    # 3. Phân bố nguồn dữ liệu (Source/Subset Distribution)
    if args.stats:
        log("\n" + "-" * 80)
        log("3. PHÂN BỐ NGUỒN DỮ LIỆU (SOURCE / SUBSET DISTRIBUTION):")
        log("-" * 80)
        source_counter = Counter()

        # Đọc cột 'source' từ test_file và shard train đầu tiên để lấy mẫu phân bố
        inspect_files = (test_files + train_files[:1]) if test_files else train_files[:2]
        for f in inspect_files:
            try:
                table = pq.read_table(f, columns=["source"])
                source_counter.update(table["source"].to_pylist())
            except Exception:
                pass

        total_sampled = sum(source_counter.values())
        if source_counter:
            log(f"  (Lấy mẫu thống kê trên {total_sampled:,} dòng từ {len(inspect_files)} files):")
            for src, count in source_counter.most_common(15):
                pct = (count / total_sampled) * 100
                log(f"    - {src:<32}: {count:>8,} dòng ({pct:>5.1f}%)")
            if len(source_counter) > 15:
                log(f"    - ... và {len(source_counter) - 15} nguồn khác")
        else:
            log("  (Dataset không có cột 'source' riêng biệt)")

    # 4. Kiểm tra Tokenizer nếu có
    tok = None
    if args.tokenizer and os.path.exists(args.tokenizer):
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(args.tokenizer)
        log("\n" + "-" * 80)
        log("4. THÔNG TIN TOKENIZER PRETRAIN ĐƯỢC CHỈ ĐỊNH:")
        log("-" * 80)
        log(f"  • File tokenizer   : {args.tokenizer}")
        log(f"  • Vocab size       : {tok.get_vocab_size():,}")
        log(f"  • ID <|endoftext|> : {tok.token_to_id('<|endoftext|>')}")
        log(f"  • ID <|im_start|>  : {tok.token_to_id('<|im_start|>')}")
        log(f"  • ID <|im_end|>    : {tok.token_to_id('<|im_end|>')}")
        if tok.token_to_id("<|im_start|>") is None or tok.token_to_id("<|im_end|>") is None:
            log("  [!] LƯU Ý: Tokenizer này không chứa token đặc biệt <|im_start|>/<|im_end|>!")
            log("             Các thẻ này sẽ bị tách thành các subwords nhỏ.")

    # 5. Đọc chi tiết các mẫu hội thoại (Samples)
    log("\n" + "-" * 80)
    log(f"5. CHI TIẾT CÁC MẪU HỘI THOẠI (HIỂN THỊ {args.num_samples} MẪU ĐẦU TIÊN):")
    log("-" * 80)

    # Chọn file để đọc mẫu
    target_files = train_files if args.split == "train" else (test_files if args.split == "test" else (train_files or test_files))
    sample_file = target_files[0]
    log(f"  Đang đọc mẫu từ: {sample_file}")

    table_sample = pq.read_table(sample_file)
    has_source = "source" in table_sample.column_names
    messages_col = "messages" if "messages" in table_sample.column_names else None

    if not messages_col:
        for c in ["conversations", "dialogue", "turns"]:
            if c in table_sample.column_names:
                messages_col = c
                break

    if not messages_col:
        log(f"[LỖI] Không tìm thấy cột hội thoại trong các cột: {table_sample.column_names}")
        return

    num_to_show = min(args.num_samples, table_sample.num_rows)
    for idx in range(num_to_show):
        msgs = table_sample[messages_col][idx].as_py()
        src = table_sample["source"][idx].as_py() if has_source else "N/A"

        log("\n" + "=" * 80)
        log(f"MẪU #{idx + 1} | Nguồn: {src} | Số lượt (messages): {len(msgs)}")
        log("=" * 80)

        # Hiển thị từng lượt thô
        log("[CÁC LƯỢT HỘI THOẠI GỐC (RAW TURNS)]:")
        for t_idx, turn in enumerate(msgs):
            role = turn.get("role") or turn.get("from") or "unknown"
            content = turn.get("content") or turn.get("value") or ""
            preview_content = content.strip()
            if len(preview_content) > 300:
                preview_content = preview_content[:300] + "... [còn tiếp]"
            log(f"  Turn {t_idx + 1} [{role.upper()}]:\n    {preview_content}\n")

        # Hiển thị định dạng ChatML
        chatml_text = format_chatml(msgs)
        log("-" * 40)
        log("[ĐỊNH DẠNG CHATML SẼ DÙNG ĐỂ TRAIN SFT]:")
        log("-" * 40)
        preview_chatml = chatml_text
        if len(preview_chatml) > 600:
            preview_chatml = preview_chatml[:600] + "\n... [còn tiếp]"
        log(preview_chatml)

        # Tokenize nếu có tokenizer
        if tok is not None:
            enc = tok.encode(chatml_text)
            eos_id = tok.token_to_id("<|endoftext|>")
            full_ids = enc.ids + ([eos_id] if eos_id is not None else [])
            log("-" * 40)
            log(f"[PHÂN TÍCH TOKEN (Độ dài: {len(full_ids)} tokens)]:")
            log(f"  Token IDs đầu tiên: {full_ids[:15]} ...")
            log(f"  Token IDs kết thúc : ... {full_ids[-10:]}")

    log("\n" + "=" * 80)
    log("TỔNG KẾT:")
    log(f"  • Post-training dataset đã sẵn sàng tại: {args.dataset}")
    log(f"  • Định dạng lưu trữ: Parquet ({len(train_files)} train shards, {len(test_files)} test shards)")
    log(f"  • Cột hội thoại: '{messages_col}' chứa cấu trúc chuẩn list[dict(role, content)]")
    log("=" * 80)

    # Xuất file preview nếu được yêu cầu
    if args.export_preview:
        out_dir = os.path.dirname(args.export_preview)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.export_preview, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines) + "\n")
        log(f"\n-> Đã xuất báo cáo khảo sát thành công ra file: {args.export_preview}")


if __name__ == "__main__":
    main()
