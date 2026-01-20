# Mini inference code here, add sampling settings and inference improvements

# Greedy vs Multinomial option
# stop tokens option 
# Max_new_tokens, is an option, also should be forced to be below ctx len 
# add TopK and TopP
# add a streaming option 
# Get KV cache working
# prefix suffix options 
# add repetition penalty options
# expand this to more tokenizers

# Tail free sampling, - Tail free sampling (TFS) is a text generation technique that aims to reduce the impact of less likely tokens, which may be less relevant, less coherent, or nonsensical, on the output. Similar to Top-P it tries to determine the bulk of the most likely tokens dynamically. But TFS filters out logits based on the second derivative of their probabilities. Adding tokens is stopped after the sum of the second derivatives reaches the parameter z. In short: TFS looks how quickly the probabilities of the tokens decrease and cuts off the tail of unlikely tokens using the parameter z. Typical values for z are in the range of 0.9 to 0.95. A value of 1.0 would include all tokens, and thus disables the effect of TFS.

# Locally Typical Sampling - Locally typical sampling promotes the generation of contextually coherent and diverse text by sampling tokens that are typical or expected based on the surrounding context. By setting the parameter p between 0 and 1, you can control the balance between producing text that is locally coherent and diverse. A value closer to 1 will promote more contextually coherent tokens, while a value closer to 0 will promote more diverse tokens. A value equal to 1 disables locally typical sampling.

# Smooth Sampling / Quadratic Sampling
#    - This sampling method differs from the truncation samplers (Top K, Top P, Min P) in that it is doing something that is fundamentally different to the raw token scores.
#    - We are tweaking the logits using a quadratic transformation, based on each token score's distance from the top token (the transformation centers on the top logit.) The coefficient is decided by the "smoothing factor" value.
#    - This is hard to explain without looking at the visualization, but the idea is that we make the topmost tokens more evenly probable while reducing the probability of extremely unlikely tokens.
#    - Higher values will be more deterministic, but it doesn't work quite like lower temperature would, as the scores of extremely closely competing top tokens will barely change. So if the original probabilities were 50/50 on the top two tokens, they will likely remain that way with higher smoothing factor values.
#    - The idea is that this can be used as an "all in one" sampler by itself, or in tandem with other methods if desired.

# The muse https://github.com/the-crypt-keeper/the-muse
# add beam search 
# Drugs https://github.com/EGjoni/DRUGS 
# minimum bayes risk decoding [https://github.com/ZurichNLP/mbr](https://github.com/ZurichNLP/mbr?scrlybrkr=4c9c022b)

# grammars
# - https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md
# - https://github.com/ggerganov/llama.cpp#constrained-output-with-grammars

# Mirostat
# - https://arxiv.org/abs/2007.14966

# EAGLE
# - https://arxiv.org/abs/2401.15077
# - https://github.com/SafeAILab/EAGLE

# Dynamic Temp
# - https://github.com/ggerganov/llama.cpp/issues/3483

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import tomllib
import argparse
import os
import math
import time

import model
from model import Transformer

# --- Setup Argument Parser ---
parser = argparse.ArgumentParser(description="Inference with KV Cache")
parser.add_argument("--conf", type=str, required=True, help="Path to the toml config file")
parser.add_argument("--ckpt", type=str, required=True, help="Path to the model checkpoint (.pt)")
parser.add_argument("--prompt", type=str, default="The secret to intelligence is", help="Starting text")
parser.add_argument("--new_tokens", type=int, default=50, help="Number of tokens to generate")
args = parser.parse_args()

# 1. Setup Tokenizer first to get the correct vocab_size
with open(args.conf, "rb") as f:
    data = tomllib.load(f)

tokenizer = AutoTokenizer.from_pretrained(data["tokenizer"])
vocab_size = 151680

device = "cuda"
ptdtype = torch.bfloat16

# --- Model Init ---
model.config.update({
    "n_embd": data["n_embd"], "n_layer": data["n_layer"], "n_head": data["n_head"],
    "ctx_len": data["block_size"], "vocab_size": vocab_size, "dropout": data["dropout"],
    "swa_ratio": data["swa_ratio"]
})

# 4. Initialize Model with the config dictionary
# Make sure your Transformer class looks at this config!
m_i = Transformer()

def load_model(checkpoint_path, model_instance):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    # Remove torch.compile prefixes if they exist
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    # CRITICAL: strict=False ignores the missing k_cache/v_cache keys
    msg = model_instance.load_state_dict(state_dict, strict=False)
    
    print(f"Missing keys (expected): {msg.missing_keys}")
    print(f"Unexpected keys: {msg.unexpected_keys}")

    model_instance.to(device=device, dtype=ptdtype)
    model_instance.eval()
    return model_instance

m = load_model(args.ckpt, m_i)
m = torch.compile(m, mode="reduce-overhead")

def get_kv_cache_size(model):
    total_bytes = 0
    for name, buffer in model.named_buffers():
        if "k_cache" in name or "v_cache" in name:
            total_bytes += buffer.nelement() * buffer.element_size()
    return total_bytes / (1024 * 1024)

# 3. Prepare Generation
m.reset_cache() # Clear any garbage from memory
inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
idx = inputs['input_ids']

print(f"\nPrompt: {args.prompt}")
print(f"Generating {args.new_tokens} tokens...\n")

# 4. Run Generation

t0 = time.perf_counter()

with torch.inference_mode():
    with torch.amp.autocast(device_type="cuda", dtype=ptdtype):
        generated_indices = m.generate(
            idx, 
            max_new_tokens=args.new_tokens, 
            temperature=0.01, 
            top_k=5
        )

t1 = time.perf_counter()

dt = t1 - t0

tokens_generated = generated_indices.size(1) - idx.size(1)
tokens_per_sec = tokens_generated / dt

# 5. Output results
output_text = tokenizer.decode(generated_indices[0], skip_special_tokens=True)
print("-" * 30)
print(output_text)
print("-" * 30)

# 6. Memory Stats
size_mb = get_kv_cache_size(m)
print(f"\nKV Cache Memory: {size_mb:.2f} MB")
print(f"Precision: {ptdtype}")
print(f"Generation Time: {dt:.2f}s")
print(f"Throughput: {tokens_per_sec:.2f} tok/s")
