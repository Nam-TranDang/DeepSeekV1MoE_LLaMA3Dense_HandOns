"""
Goal: Read raw dataset (Parquet, JSONL, TXT) -> Train BPE Tokenizer -> Tokenize train.bin / val.bin (uint16) + tokenizer.json aka Vocab

For Dataset Pretrain - HuggingFace (smollm-corpus, cosmopedia-100k,...).

Input: data SHOULD down locally from Hugging Face

Code 2 main jobs:

- Train a Byte-Pair Encoding (BPE) Tokenizer (saves tokenizer.json).

- Convert text into numbers (token IDs) and write them as raw binary files (val.bin and train.bin) for training an LLM. --> Need not "Tokenize" for Training.



  #### Lines 53–61: Text Column Extractor (extract_text) (L54-L61)

    def extract_text(ex, target_col):

  • Different datasets use different column names (text, content, document, raw_text).
  • This function first checks if target_col exists in the record. If not, it falls back to known alternates ("text", "content", "document", "body", "raw_text"), raising a KeyError if
  none match.
  ──────
  #### Lines 63–91: Phase 1 — Train Byte-Level BPE Tokenizer (L63-L91)

  • **L63**: Loads the raw data: ds = load_any_dataset(args.dataset).
  • **L68-L87**: If tokenizer.json does not already exist:
      • **L69-L73**: Defines text_iter() generator to yield texts up to args.tok_docs (30,000 documents) so it doesn't exhaust memory.
      • **L75-L77**: Sets up HuggingFace's Rust-based ByteLevel BPE tokenizer:
          • models.BPE(): Base BPE algorithm.
          • pre_tokenizers.ByteLevel: Splits text at the byte level (allowing arbitrary Unicode characters to be tokenized without out-of-vocabulary <unk> tokens).
          • decoders.ByteLevel(): Decodes byte sequences back to UTF-8 strings.
      • **L78-L82**: Configures BpeTrainer with vocab_size=16384 and adds the <|endoftext|> special token.
      • **L85-L86**: Trains the tokenizer using tok.train_from_iterator(...) and saves the result to tokenizer.json.
  • **L89-L91**: Loads the tokenizer from disk and retrieves the integer ID for <|endoftext|> (used to separate documents).
  ──────
  #### Lines 93–110: Output Files & Buffer Flushing (flush) (L94-L110)

  • **L94-L95**: Opens binary files val.bin and train.bin in raw write binary ("wb") mode.
  • **L96-L97**: Initializes counters n_val = 0, n_train = 0, and an in-memory document batch = [].
  • **L100-L109**: flush(batch):
      • Uses tok.encode_batch(batch) for fast multi-threaded tokenization in Rust.
      • For each encoded document, appends [eos] (<|endoftext|>), then converts it to np.uint16 (ids = np.array(enc.ids + [eos], dtype=np.uint16)).
      • Writes raw bytes (ids.tobytes()) to val.bin until n_val >= args.val_tokens (2,000,000 tokens).
      • Once the validation set quota is filled, all subsequent tokens are written to train.bin.

  ──────
  #### Lines 112–130: Processing Loop & Finalization (L112-L130)

  • **L112-L113**: Initializes a tqdm progress bar with total token target (if --max_tokens is configured).
  • **L115-L123**: Iterates through the dataset:
      • Gathers 1,000 documents at a time into batch.
      • Calls flush(batch) to tokenize and write to disk in bulk.
      • Updates the progress bar.
      • Breaks early if args.max_tokens is exceeded.
  • **L125-L126**: Flushes any trailing documents left in batch.
  • **L128-L130**: Closes val_f and train_f, and prints total tokens written.
  ──────
  ### Output Summary

  Running this script produces 3 files in your target output directory (--out):

  1. tokenizer.json: The trained Byte-Level BPE tokenizer config and vocabulary.
  2. val.bin: Flat 1D binary array of uint16 token IDs for evaluation (first 2M tokens).
  3. train.bin: Flat 1D binary array of uint16 token IDs for pre-training.



"""
import argparse
import glob
import os
import sys

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tqdm import tqdm

# ---------------- CLI Arguments
p = argparse.ArgumentParser(description="Pretraining Data Pipeline: train BPE & tokenize sang .bin")
p.add_argument("--dataset", default="data/smollm_pretrain_raw", help="Đường dẫn file/thư mục dataset thô (parquet, jsonl, csv, txt)")
p.add_argument("--out", default="data/smollm_pretrain_bin", help="Thư mục xuất file tokenizer.json, val.bin, train.bin")

p.add_argument("--vocab_size", type=int, default=16384, help="Kích thước từ vựng (mặc định: 16,384 tokens)")
p.add_argument("--tok_docs", type=int, default=30_000, help="Số doc dùng để học vocab (mặc định: 30,000 docs; <= 0 để dùng toàn bộ)")
p.add_argument("--max_tokens", type=int, default=0, help="Trần tối đa token cần tokenize (0 hoặc âm = không giới hạn)")

p.add_argument("--val_tokens", type=int, default=2_000_000, help="Số token dành cho tập val.bin (mặc định: 2,000,000)")
p.add_argument("--text_col", default="text", help="Tên cột chứa nội dung văn bản (mặc định: 'text')")
args = p.parse_args()

os.makedirs(args.out, exist_ok=True) # Create output folder - if not exist


# ---------------- Load Raw Data (Parquet, JSONL, CSV, TXT, HF Hub)
def load_any_dataset(path):

    # If data local path "NOT EXIST" --> Stream from Hugging Face
    if not os.path.exists(path):    
        return load_dataset(path, split="train", streaming=True)

    # ONLY RUN when Path = a Single File path (e.g: Train.parquet)
    if os.path.isfile(path):        

        # Get the file extension and convert the characters to lowercase.
        ext = os.path.splitext(path)[1].lower()

        # Map each file extension to its file format type
        fmt_map = {".parquet": "parquet", ".pq": "parquet", ".jsonl": "json", ".json": "json", ".csv": "csv", ".txt": "text"}
        
        # Only run if extension supported from map 
        if ext in fmt_map:
            return load_dataset(fmt_map[ext], data_files=[path], split="train")
        raise ValueError(f"Định dạng file không được hỗ trợ: {ext}")

    # Default read - Path of "Folder" (like data/smollm_pretrain_raw), it uses recursive glob to find all data files
    if os.path.isdir(path):
        patterns = [("*.parquet", "parquet"), ("*.jsonl", "json"), ("*.json", "json"), ("*.csv", "csv"), ("*.txt", "text")]
        
        # Iterate the "pattern" -> Try find file based on file format. (VD folder chứa parquet - csc,... read từng cái)
        for pattern, fmt in patterns:

            # Find glob.glob (E.G: folder có 5 files parquet --> found sẽ chứa 5 path file đó). Vòng loop cover "iterate" on file extension
            found = sorted(glob.glob(os.path.join(path, f"**/{pattern}"), recursive=True))
            
            # Nếu found --> đã chứa các file parquet (return luôn - không đọc các file extension còn lại)
            if found:
                print(f"-> Tìm thấy {len(found)} file [{pattern}] tại: {path}")
                return load_dataset(fmt, data_files=found, split="train")

        # Fall back - nếu path dữ liệu không có - Hugging face re download.
        return load_dataset(path, split="train")


# ---------------- Extract text - extract text from a data record .parquet
'''
Different datasets name their text columns differently (E.G: smollm != cosmopedia)

'''
# c. The Extractor Function
def extract_text(ex, target_col):
    
    # Check the primary column name. Convert the value to a string and return it.
    if target_col in ex and ex[target_col] is not None:
        return str(ex[target_col])

    # Check alternative column names in sequence. Return the first valid text string
    for alt in ["text", "content", "document", "body", "raw_text"]:
        if alt in ex and ex[alt] is not None:
            return str(ex[alt])
    
    raise KeyError(f"Không tìm thấy cột văn bản '{target_col}'. Các cột hiện có: {list(ex.keys())}")

# Step 1: Load the Data
# Code start here ds is now a streamable list of dataset records (rows).
ds = load_any_dataset(args.dataset)

# Step 2: Phase 1 — Train the BPE Tokenizer Vocabulary & Train Byte-Level BPE Tokenizer
tok_path = os.path.join(args.out, "tokenizer.json")

# If not have Vocab aka tokenizer.json --> Start train 
'''
tok.train_from_iterator(...)
         │  (requests text samples)
         ▼
    text_iter()
         │  (needs text from each row)
         ▼
 extract_text(ex) ──► extracts string from column "text"
'''


if not os.path.exists(tok_path):
    
    # b. The Helper Generator 

    # 30k nay la max doc hay gi? 
    # Loops through the first 30,000  rows and calls extract_text() for each row
    def text_iter():

        for i, ex in enumerate(ds):
            if args.tok_docs > 0 and i >= args.tok_docs:
                break
            yield extract_text(ex, args.text_col)

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    doc_info = "TOÀN BỘ dataset" if args.tok_docs <= 0 else f"{args.tok_docs:,} docs"
    print(f"Đang train tokenizer BPE (vocab_size={args.vocab_size}, mẫu: {doc_info}) ...")
    
    # a. Actual Call To Train Tokenizer    
    tok.train_from_iterator(text_iter(), trainer)
    tok.save(tok_path)
    print(f"-> Đã lưu tokenizer tại: {tok_path}")

tok = Tokenizer.from_file(tok_path)
eos = tok.token_to_id("<|endoftext|>")
print(f"Vocab size: {tok.get_vocab_size()} | EOS ID: {eos}")

# ---- 2) Tokenize sang val.bin và train.bin   #### Lines 93–110: Output Files & Buffer Flushing (flush) (L94-L110)
val_f = open(os.path.join(args.out, "val.bin"), "wb")
train_f = open(os.path.join(args.out, "train.bin"), "wb")
n_val = n_train = 0
batch = []

'''
Converts words to numbers (uint16).
Appends [eos] (<|endoftext|>) at the end of each document.
Fills val.bin first (2M tokens).
Once val.bin is full, dumps everything else into train.bin.

'''

def flush(batch):
    global n_val, n_train
    for enc in tok.encode_batch(batch):
        ids = np.array(enc.ids + [eos], dtype=np.uint16)
        if n_val < args.val_tokens:
            val_f.write(ids.tobytes())
            n_val += len(ids)
        else:
            train_f.write(ids.tobytes())
            n_train += len(ids)


total_tokens_target = args.max_tokens if args.max_tokens > 0 else None
pbar = tqdm(total=total_tokens_target, unit="tok")


# Step 3: Phase 2 — Encode Text into Binary Files (train.bin & val.bin)
'''
Main Loop (for ex in ds)
    │
    ├─► 1. Calls extract_text(ex) ──► Appends text into batch list
    │
    └─► 2. Every 1,000 texts ─────► Calls flush(batch)
It takes rows from ds.
It calls extract_text() to get the text.
It collects 1,000 texts into memory.
Once 1,000 texts are ready, it calls flush(batch).

'''

for ex in ds:
    batch.append(extract_text(ex, args.text_col))
    if len(batch) == 1000:
        before = n_train + n_val
        flush(batch)
        batch = []
        pbar.update(n_train + n_val - before)
        if args.max_tokens > 0 and (n_train + n_val) >= args.max_tokens:
            break

if len(batch) > 0 and (args.max_tokens <= 0 or (n_train + n_val) < args.max_tokens):
    flush(batch)

val_f.close()
train_f.close()
print(f"Done! train tokens: {n_train:,} | val tokens: {n_val:,}")
