import numpy as np
from tokenizers import Tokenizer
import glob
import os

MODEL_ID = "Qwen/Qwen3-1.7B"
BIN_DIR = "vision_binned"

def inspect():
    # 1. Load Tokenizer
    tokenizer = Tokenizer.from_pretrained(MODEL_ID)
    
    # 2. Get first bin file
    files = sorted(glob.glob(os.path.join(BIN_DIR, "*.bin")))
    if not files:
        print("No .bin files found!")
        return
    
    target_file = files[0]
    print(f"Inspecting: {target_file}")

    # 3. Read the data
    with open(target_file, "rb") as f:
        # Skip the 256-byte (1024 int32) header
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        print(f"Header Report: Magic={header[0]}, Version={header[1]}, Tokens={header[2]}")
        
        # Read the first 1000 tokens (uint32)
        tokens = np.frombuffer(f.read(1000 * 4), dtype=np.uint32)

    # 4. Decode and Print
    decoded_text = tokenizer.decode(tokens.tolist())
    print("-" * 50)
    print(decoded_text)
    print("-" * 50)

if __name__ == "__main__":
    inspect()
