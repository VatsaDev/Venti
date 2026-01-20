import os
import torch
from transformers import AutoTokenizer
import glob
import itertools
from pathlib import Path
import numpy as np

DATA_DIR = "data/qwen_text" 
TOKENIZER_ID = "Qwen/Qwen3-1.7B" 
BATCH_SIZE = 1
BLOCK_SIZE = 512

def _load_data_shard(file: Path):
    with file.open("rb") as f:
        # Skip the 1024-byte header
        f.seek(1024)
        # Read as uint32
        data = np.fromfile(f, dtype=np.int16)
    return torch.from_numpy(data).to(torch.int64)

def create_data_generator(filename_pattern, batch_size, block_size):
    files = sorted(glob.glob(filename_pattern))
    if not files:
        raise ValueError(f"No files found for pattern: {filename_pattern}")
    
    file_iter = itertools.cycle([Path(f) for f in files])
    current_tokens = None
    current_pos = 0
    
    while True:
        if current_tokens is None or current_pos + (batch_size * block_size) + 1 > len(current_tokens):
            shard_path = next(file_iter)
            print(f"\n--- Loading Shard: {shard_path.name} ---")
            current_tokens = _load_data_shard(shard_path)
            current_pos = 0
        
        chunk = current_tokens[current_pos : current_pos + (batch_size * block_size) + 1]
        x = chunk[:-1].view(batch_size, block_size)
        yield x
        current_pos += batch_size * block_size

def check_data():

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    pattern = os.path.join(DATA_DIR, "*.bin")
    
    gen = create_data_generator(pattern, BATCH_SIZE, BLOCK_SIZE)

    print("\n" + "="*50)
    print("STARTING DATA VERIFICATION")
    print("="*50)

    for i in range(BATCH_SIZE):
        print(f"\n>>> RANDOM BATCH {i+1} <<<")
        xb = next(gen)
        
        # Stats Check
        max_id = xb.max().item()
        min_id = xb.min().item()
        print(f"Max Token ID: {max_id}")
        print(f"Min Token ID: {min_id}")

        # 16-bit Overflow Check
        if max_id < 65535:
            print("WARNING: All tokens are < 65535. Your data might still be uint16 or truncated!")
        elif max_id > 151643:
            print(f"INFO: High-range tokens detected (ID > 151643). This is good for Qwen.")

        decoded = tokenizer.decode(xb[0].tolist(), errors='replace')
        
        print("-" * 20)
        print("DECODED TEXT:")
        print(decoded[:1000])
        print("-" * 20)
        print("="*50)

if __name__ == "__main__":
    check_data()
