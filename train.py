import os
import math
import time
import glob
import torch
import string
import random
import duckdb
import pickle
import tomllib
import argparse
import itertools
import subprocess
import numpy as np
from pathlib import Path
from torch.cuda.amp import GradScaler
from transformers import AutoTokenizer

import matplotlib
from matplotlib import pyplot as plt
 
import model
from model import Transformer

from datasets import load_dataset

import torch._inductor.config as config

# like reduce-overhead without the baggage

config.coordinate_descent_tuning = True
config.triton.unique_kernel_names = True
config.fx_graph_cache = True

# TOML config 

parser = argparse.ArgumentParser(description="conf")
parser.add_argument("--conf", type=str, default="", help="toml conf path")
args = parser.parse_args()

conf_path = args.conf

with open(conf_path, "rb") as f:
    data = tomllib.load(f)

batch_size = data["batch_size"]
block_size = data["block_size"]
eval_interval = data["eval_interval"]
grad_accum_steps = data["grad_accum_steps"]

lr = data["lr"]
min_lr = data["min_lr"]

max_iters = data["max_iters"]
eval_iters = data["eval_iters"]
warmup_iters = data["warmup_iters"]

beta1 = 0.9
beta2 = 0.95
weight_decay = data["weight_decay"]

max_grad_norm = 1.0 

ckpt_iter = 500

resume = False
resume_checkpoint = "/content/floppyLLM/checkpoints/sVtcrs_10000.pt" 

data_dir = data["data_dir"]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ptdtype = torch.float16

print(f"Using device: {device}")

# Initialize Scaler, needed for FP16 training on T4 to prevent underflow?
scaler = GradScaler(enabled=True)
ctx = torch.amp.autocast(device_type=device, dtype=ptdtype)

# Run Name
characters = string.ascii_letters + string.digits 
run_name = ''.join(random.choice(characters) for i in range(6))

# check plots/checkpoints
if not os.path.exists('checkpoints'): os.makedirs('checkpoints')
if not os.path.exists('plots'): os.makedirs('plots')

# make the duckdb database, new table for current run

table_format = f"""
CREATE TABLE _{run_name} (
   step INT,
   lr FLOAT,
   train_loss FLOAT,
   val_loss FLOAT, 
   train_pplx FLOAT,
   val_pplx FLOAT,
   val_bpb FLOAT,
   throughput FLOAT
);
"""

logging = duckdb.connect("log.db")
logging.sql(table_format)
logging.close()

def safe_log_to_duckdb(run_name, row_data, db_path="log.db", retries=5):
    for i in range(retries):
        try:
            with duckdb.connect(db_path) as con:
                con.execute(f'INSERT INTO "_{run_name}" VALUES {row_data}')
            return # Success!
        except duckdb.duckdb.IOException:
            # Database is locked by the plotter, wait 0.5s and retry
            time.sleep(0.5)
    print("WARNING: Could not write to log.db after 5 retries. Skipping log.")

# encoding 
tok = AutoTokenizer.from_pretrained("tokenizers/venti_4k")
encode = lambda s: tok.encode(s, add_special_tokens=True)
decode = lambda l: tok.decode(l)
vocab_size = tok.vocab_size
print(f"Using AutoTokenizer, vocab_size = {vocab_size}")

model.config["n_embd"] = data["n_embd"]
model.config["n_layer"] = data["n_layer"]
model.config["n_head"] = data["n_head"]
model.config["ctx_len"] = block_size
model.config["vocab_size"] = vocab_size
model.config["dropout"] = data["dropout"]
model.config["swa_ratio"] = data["swa_ratio"]

# hellaswag

print("Loading HellaSwag for evaluation...")
hswag = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=True)
hellaswag_bpb_history = []

# data Loading
def _load_data_shard(file: Path):
    header = torch.from_file(str(file), shared=False, size=256, dtype=torch.int32)
    with file.open("rb") as f:
        header_bytes = f.read(256 * 4)
        header = torch.frombuffer(header_bytes, dtype=torch.int32)
        assert header[0].item() == 20240520, f"magic number mismatch in {file}"
        assert header[1].item() == 1, f"unsupported version in {file}"
        num_tokens = int(header[2].item()) 
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=(device=='cuda'))
        tokens_np = tokens.numpy() 
        nbytes_read = f.readinto(tokens_np.data) 
        expected_bytes = 2 * num_tokens
        assert nbytes_read == expected_bytes, f"read mismatch in {file}"
    return tokens

def create_data_generator(filename_pattern: str, batch_size: int, block_size: int, rank: int = 0, world_size: int = 1):
    files = sorted(glob.glob(filename_pattern))
    if not files: 
        raise FileNotFoundError(f"No data files found: {filename_pattern}")
    
    print(f"Found {len(files)} data shards for pattern {filename_pattern}")
    file_iter = itertools.cycle([Path(file) for file in files])
    
    # Pre-calculate token counts
    local_batch_size = batch_size // world_size
    tokens_per_rank = local_batch_size * block_size
    global_tokens_per_step = batch_size * block_size
    
    current_tokens = None
    current_pos = 0

    while True:
        # Check if current shard has enough tokens for the WHOLE global batch
        if current_tokens is None or current_pos + global_tokens_per_step + 1 > len(current_tokens):
            next_file = next(file_iter)
            current_tokens = _load_data_shard(next_file)
            current_tokens = current_tokens.to(torch.int64) # Keep on CPU for faster slicing
            current_pos = 0
            
            # Ensure the shard is actually big enough to hold one batch
            if len(current_tokens) <= global_tokens_per_step + 1:
                current_tokens = None 
                continue 

        # --- THE FAST PART: CONTIGUOUS SLICING ---
        # 1. Calculate the window for THIS specific rank
        start_idx = current_pos + (rank * tokens_per_rank)
        end_idx = start_idx + tokens_per_rank + 1 # +1 for the target shift
        
        # 2. Slice once. This is a single C++ call.
        # This gives us (local_batch_size * block_size + 1) tokens
        chunk = current_tokens[start_idx : end_idx]
        
        # 3. Create X and Y using .view(). 
        # This is a "metadata-only" operation (zero memory copy).
        # x: [token_0, token_1, ..., token_N-1]
        # y: [token_1, token_2, ..., token_N]
        x = chunk[:-1].view(local_batch_size, block_size)
        y = chunk[1:].view(local_batch_size, block_size)

        # 4. Move to GPU in bulk.
        # Pin memory and non_blocking=True allow the GPU to copy while the CPU works.
        yield x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        # Advance the pointer by the GLOBAL batch tokens consumed
        current_pos += global_tokens_per_step

train_data_pattern = os.path.join("data", data_dir, f"{data_dir}_train_*.bin")
val_data_pattern = os.path.join("data", data_dir, f"{data_dir}_val_*.bin")

def get_batch_from_shards(split, data_gens):
    if split == 'train': X, Y = next(data_gens['train'])
    else: X, Y = next(data_gens['val'])
    return X, Y

train_data_gen = create_data_generator(train_data_pattern, batch_size, block_size, rank=0, world_size=1)
val_data_gen = create_data_generator(val_data_pattern, batch_size, block_size, rank=0, world_size=1)
data_gens = {'train': train_data_gen, 'val': val_data_gen}

# --- Model Init ---
model.config["vocab_size"] = vocab_size
model.config["block_size"] = block_size 

if resume:
    print(f"Resuming from checkpoint: {resume_checkpoint}")
    checkpoint = torch.load(resume_checkpoint, map_location=device)
    
    model_instance = Transformer()
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model_instance.load_state_dict(state_dict)
    m = model_instance.to(device)

    opt_muon, opt_adam = m.configure_optimizers(weight_decay, lr, (beta1, beta2), device_type=device)

    if 'opt_adam' in checkpoint and 'opt_muon' in checkpoint:
        opt_adam.load_state_dict(checkpoint['opt_adam'])
        opt_muon.load_state_dict(checkpoint['opt_muon'])

    if 'scaler' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler'])

    start_iter = checkpoint['iter'] + 1 
    run_name = checkpoint['run_name'] 

else:

    model_instance = Transformer()
    m = model_instance.to(device)
    
    # Init optimizers for fresh run
    opt_muon, opt_adam = m.configure_optimizers(weight_decay, lr, (beta1, beta2), device_type=device)
    start_iter = 0 
    print(f"Starting new run {run_name} from scratch")

p = sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f"{p/1e6:.2f} M parameters")

# tag layers for attn 
m._tag_layers(m)

# --- Compile ---
print("compilation step")
if device == "cuda":
    # trying reduce-overhead to see if cuda graphs fix perf (didnt work)
    compiled_model = torch.compile(m)
    print("compiled")
else:
    compiled_model = m
    print("skipped compilation")

# Loss Estimation
@torch.no_grad()
def estimate_loss(model_to_eval, data_gens):
    out = {}
    model_to_eval.eval() 
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch_from_shards(split, data_gens)
            with ctx:
                logits, loss = model_to_eval(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    return out

# text gen
@torch.no_grad()
def generate_text(model_to_gen, enc, max_new_tokens=1000, temperature=0.8, top_k=10):
    model_to_gen.eval() 
    start = "\n"
    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    print("Generating text...")
    with ctx: 
        y = model_to_gen.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
    generated_ids = y[0].tolist() 
    print(decode(generated_ids))
    model_to_gen.train()

# hellaswag_bpb (move to evals.py later)

@torch.no_grad()
def evaluate_hellaswag_bpb(model_to_eval, dataset, num_samples=100):
    model_to_eval.eval()
    total_nll = 0.0
    total_bytes = 0
    
    # We sample a subset to keep training moving fast
    indices = random.sample(range(len(dataset)), num_samples)
    
    for idx in indices:
        example = dataset[idx]
        ctx_str = example['ctx']
        label = int(example['label'])
        correct_answer_str = example['endings'][label]
        
        ctx_enc = encode(ctx_str)
        ans_enc = encode(correct_answer_str)
        full_enc = ctx_enc + ans_enc
        
        if len(full_enc) > block_size:
            full_enc = full_enc[-(block_size + 1):]
        
        x = torch.tensor(full_enc[:-1], dtype=torch.long, device=device)[None, ...]
        y = torch.tensor(full_enc[1:], dtype=torch.long, device=device)[None, ...]
        
        with ctx:
            # FIX: We pass Y as targets even if we don't use the model's internal loss.
            # This forces the model to return the FULL sequence of logits.
            logits, _ = model_to_eval(x, y) 
            
            ans_len = len(ans_enc)
            
            # Isolate the answer portion
            ans_logits = logits[0, -ans_len:, :].contiguous()
            ans_targets = y[0, -ans_len:].contiguous()
            
            # Safety check: if shapes still don't match, skip this sample
            if ans_logits.size(0) != ans_targets.size(0):
                continue
                
            nll = torch.nn.functional.cross_entropy(ans_logits, ans_targets, reduction='sum')
            
        total_nll += nll.item()
        total_bytes += len(correct_answer_str.encode('utf-8'))

    hswag_bpb = total_nll / (total_bytes * math.log(2)) if total_bytes > 0 else 0
    model_to_eval.train()
    return hswag_bpb

# Training Loop
time_s = time.time()
prev_time = time_s 

# Ensure optimizer state is on correct device
for opt in [opt_adam, opt_muon]:
    for state in opt.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor): state[k] = v.to(device)

opt_adam.zero_grad(set_to_none=True) 
opt_muon.zero_grad(set_to_none=True)

# bpb stuff

sample_xb, _ = get_batch_from_shards('val', data_gens)

# Decode them back to text to see how many UTF-8 bytes they actually represent
sample_text = tok.decode(sample_xb[0].tolist(), skip_special_tokens=True)
total_bytes = len(sample_text.encode('utf-8'))
total_tokens = len(sample_xb[0])

# Tokens per Byte ratio
tokens_per_byte = total_tokens / total_bytes

for iter_num in range(start_iter, max_iters + 1):

    # cos lr schedule 

    lr_iter = min_lr 
    
    if iter_num < warmup_iters:
        lr_iter = lr * iter_num / warmup_iters
    elif iter_num <= max_iters:
        decay_ratio = (iter_num - warmup_iters) / (max_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
        lr_iter = min_lr + coeff * (lr - min_lr)

    # lr assignment

    for opt in [opt_adam, opt_muon]:
        for param_group in opt.param_groups:
            param_group['lr'] = lr_iter

    # training steps

    if iter_num % eval_interval == 0 or iter_num == max_iters:
        
        time_n = time.time()
        elapsed = time_n - time_s
        dt = time_n - prev_time 
        prev_time = time_n

        throughput = (block_size * batch_size * grad_accum_steps * eval_interval) / dt
        mfu, tflops = m.estimate_mfu(block_size * batch_size * grad_accum_steps * eval_interval, dt) if hasattr(m, 'estimate_mfu') else 0.0

        losses = estimate_loss(m, data_gens)
        
        # loss
        train_loss = losses['train']
        val_loss = losses['val']
        
        # perplexity
        train_ppl = math.exp(train_loss)
        val_ppl = math.exp(val_loss)

        # bits/byte val loss (tokenizer invariant)
        val_bpb = (val_loss/math.log(2)) * tokens_per_byte
                
        print(f"step: {iter_num}, train loss: {losses['train']:.4f}, val loss: {val_loss:.4f}, train pplx: {train_ppl:.4f}, val pplx: {val_ppl:.4f}, val bpb: {val_bpb:.4f}, lr: {lr_iter:.6f}, elapsed: {elapsed/60:.2f} min, MFU: {mfu*100:.2f}%, throughput: {throughput:.2f} tok/s, tflops: {tflops:.2f}")

        # everything to duckdb 
        row_add = f"""({iter_num},{lr_iter},{train_loss:.4f},{val_loss:.4f},{train_ppl:.4f},{val_ppl:.4f},{val_bpb:.4f},{throughput:.4f})"""

        safe_log_to_duckdb(run_name, row_add)

        subprocess.Popen(["python", "plotgen.py", f"_{run_name}"]) 
         
    if iter_num == max_iters: break 

    m.train()
    
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):

        torch.compiler.cudagraph_mark_step_begin()

        xb, yb = get_batch_from_shards('train', data_gens)

        # FP16 Context (T4 safe)
        with ctx:
            logits, loss = compiled_model(xb.clone(), yb.clone())
            loss = loss / grad_accum_steps 

        # Scaled Backward Pass (Prevents underflow in FP16)
        scaler.scale(loss).backward()
        loss_accum += loss.item() * grad_accum_steps 

    # Unscale before clipping
    scaler.unscale_(opt_adam)
    scaler.unscale_(opt_muon)
    
    torch.nn.utils.clip_grad_norm_(m.parameters(), max_grad_norm)

    # Step with scaler
    scaler.step(opt_adam)
    scaler.step(opt_muon)
    
    scaler.update()

    opt_adam.zero_grad(set_to_none=True)
    opt_muon.zero_grad(set_to_none=True)

    # Checkpoint
    if iter_num % ckpt_iter == 0:
        ckpt_path = f'checkpoints/{run_name}_{iter_num}.pt'
        print(f"Saving checkpoint to {ckpt_path}")
        
        # Save standard checkpoint (Weights + Optimizer State)
        torch.save({
            'model': m.state_dict(),
            'opt_adam': opt_adam.state_dict(),
            'opt_muon': opt_muon.state_dict(),
            'scaler': scaler.state_dict(),
            'iter': iter_num,
            'run_name': run_name,
            'config': model.config
        }, ckpt_path)

        # FP16 Save (T4 Safe)
        print("Saving lightweight FP16 inference model...")
        fp16_inference_path = f'checkpoints/{run_name}_inference_fp16.pt'

        # fp16
        fp16_state_dict = {k: v.half() for k, v in m.state_dict().items()}

        # Weight Tying fix
        if 'transformer.wte.weight' in fp16_state_dict and 'lm_head.weight' in fp16_state_dict:
            fp16_state_dict['lm_head.weight'] = fp16_state_dict['transformer.wte.weight']

        # remove Causal Mask Buffer, mem save
        keys_to_remove = [k for k in fp16_state_dict.keys() if k.endswith('.attn.bias')]
        for k in keys_to_remove:
            del fp16_state_dict[k]

        torch.save(fp16_state_dict, fp16_inference_path)
        print(f"Saved optimized inference model to {fp16_inference_path}")

    # checking bench results at the same time as checkpoints makes sense
    if iter_num % ckpt_iter == 0:
        
        hswag_bpb = evaluate_hellaswag_bpb(m, hswag, num_samples=100)
        
        print(f"--- Benchmark Results at Step {iter_num} ---")
        print(f"HellaSwag BPB: {hswag_bpb:.4f}")

        if hasattr(m, 'generate'):
            generate_text(m, tok, max_new_tokens=200)

print('Training finished.')
