import os
import math
import time
import glob
import torch
import wandb
import string
import random
import duckdb
import tomllib
import argparse
import itertools
import subprocess
import numpy as np
from pathlib import Path
from torch.cuda.amp import GradScaler
from transformers import AutoTokenizer

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import model
from model import Transformer
from datasets import load_dataset
import torch._inductor.config as config

# torch compil optims
config.coordinate_descent_tuning = True
config.triton.unique_kernel_names = True
config.fx_graph_cache = True
torch.set_float32_matmul_precision('high')

def setup_ddp():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl')
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{local_rank}'
        torch.cuda.set_device(device)
    else:
        rank, local_rank, world_size = 0, 0, 1
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return rank, local_rank, world_size, device

ddp_rank, ddp_local_rank, ddp_world_size, device = setup_ddp()
master_process = ddp_rank == 0

# toml

parser = argparse.ArgumentParser(description="conf")
parser.add_argument("--conf", type=str, default="", help="toml conf path")
args = parser.parse_args()

with open(args.conf, "rb") as f:
    data = tomllib.load(f)

batch_size = data["batch_size"] # Global batch size
block_size = data["block_size"]
eval_interval = data["eval_interval"]
grad_accum_steps = data["grad_accum_steps"]
lr = data["lr"]
min_lr = data["min_lr"]
max_iters = data["max_iters"]
eval_iters = data["eval_iters"]
warmup_iters = data["warmup_iters"]
weight_decay = data["weight_decay"]
beta1, beta2 = 0.9, 0.95
max_grad_norm = 1.0
ckpt_iter = 500
resume = True
resume_checkpoint = "checkpoints/qGi1Gr_19500.pt"
data_dir = data["data_dir"]
ptdtype = torch.bfloat16

# wandb and duckdb

run_name = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(6))
if master_process:
    if not os.path.exists('checkpoints'): os.makedirs('checkpoints')
    if not os.path.exists('plots'): os.makedirs('plots')
    logging = duckdb.connect("log.db")
    logging.sql(f"CREATE TABLE IF NOT EXISTS _{run_name} (step INT, lr FLOAT, train_loss FLOAT, val_loss FLOAT, train_pplx FLOAT, val_pplx FLOAT, val_bpb FLOAT, throughput FLOAT);")
    logging.close()

    run = wandb.init(
        entity="vatsadev",
        project=f"reducto-1-{run_name}",
    )

    table = wandb.Table(columns=["step", "gen_text"])

def safe_log_to_duckdb(run_name, row_data):
    if not master_process: return
    try:
        with duckdb.connect("log.db") as con:
            con.execute(f'INSERT INTO "_{run_name}" VALUES {row_data}')
    except Exception as e:
        print(f"DuckDB Error: {e}")

def _load_data_shard(file: Path):
    with file.open("rb") as f:
        header_bytes = f.read(256 * 4)
        header = torch.frombuffer(header_bytes, dtype=torch.int32)
        num_tokens = int(header[2].item())
        tokens = torch.empty(num_tokens, dtype=torch.int32, pin_memory=True)
        f.readinto(tokens.numpy().data)
    return tokens

def create_data_generator(filename_pattern, batch_size, block_size, rank, world_size):
    files = sorted(glob.glob(filename_pattern))
    file_iter = itertools.cycle([Path(f) for f in files])
    local_batch_size = batch_size // world_size
    tokens_per_rank = local_batch_size * block_size
    global_tokens_per_step = batch_size * block_size
    current_tokens, current_pos = None, 0
    while True:
        if current_tokens is None or current_pos + global_tokens_per_step + 1 > len(current_tokens):
            current_tokens = _load_data_shard(next(file_iter)).to(torch.int64)
            current_pos = 0
            if len(current_tokens) <= global_tokens_per_step + 1: continue
        start_idx = current_pos + (rank * tokens_per_rank)
        end_idx = start_idx + tokens_per_rank + 1
        chunk = current_tokens[start_idx : end_idx]
        x = chunk[:-1].view(local_batch_size, block_size)
        y = chunk[1:].view(local_batch_size, block_size)
        yield x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        current_pos += global_tokens_per_step

train_gen = create_data_generator(os.path.join("data", data_dir, f"synth_text_train_*.bin"), batch_size, block_size, ddp_rank, ddp_world_size)
val_gen = create_data_generator(os.path.join("data", data_dir, f"synth_text_val_*.bin"), batch_size, block_size, ddp_rank, ddp_world_size)

# --- Tokenizer BPB Ratio ---
tok = AutoTokenizer.from_pretrained(data["tokenizer"]) #"tokenizers/venti_4k"
true_vocab_size = len(tok) 
vocab_size = ((true_vocab_size + 127) // 128) * 128

if master_process:
    # Estimate tokens_per_byte for BPB metrics
    sample_text = "The quick brown fox jumps over the lazy dog. Just some text to estimate byte density."
    tokens_per_byte = len(tok.encode(sample_text)) / len(sample_text.encode('utf-8'))
else:
    tokens_per_byte = 0

# --- Model Init ---
model.config.update({
    "n_embd": data["n_embd"], "n_layer": data["n_layer"], "n_head": data["n_head"],
    "ctx_len": block_size, "vocab_size": vocab_size, "dropout": data["dropout"],
    "swa_ratio": data["swa_ratio"], "block_size": block_size
})

model_instance = Transformer().to(device)
if resume:
    checkpoint = torch.load(resume_checkpoint, map_location=device)
    model_instance.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in checkpoint['model'].items()}, strict=False)
    start_iter = checkpoint['iter'] + 1
else:
    start_iter = 0

m = DDP(model_instance, device_ids=[ddp_local_rank], find_unused_parameters=True)
raw_model = m.module
opt_muon, opt_adam = raw_model.configure_optimizers(weight_decay, lr, (beta1, beta2), device_type='cuda')
compiled_model = torch.compile(m)

# --- Scaler & Context ---
scaler = GradScaler(enabled=True)
ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)

@torch.no_grad()
def estimate_loss():
    out = {}
    model_instance.eval()
    for split, gen in [('train', train_gen), ('val', val_gen)]:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            x, y = next(gen)
            with ctx:
                _, loss = compiled_model(x, y)
            losses[k] = loss
        avg_loss = losses.mean()
        dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
        out[split] = avg_loss.item()
    model_instance.train()
    return out

# --- Training Loop ---
time_s = time.time()
prev_time = time_s

if master_process:
    hswag = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=True)
    print(f"H100 Node Initialized. World Size: {ddp_world_size}")
    print(f"Params: {sum(p.numel() for p in m.parameters() if p.requires_grad)/1e6:.2f}M")

    total_params = sum(p.numel() for p in m.parameters())
    trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)

    print(f"Total Params: {total_params/1e6:.2f}M")
    print(f"Trainable Params: {trainable_params/1e6:.2f}M") # This will be the ~LM size

for iter_num in range(start_iter, max_iters + 1):
    
    # 1. Cosine LR
    if iter_num < warmup_iters:
        lr_iter = lr * iter_num / warmup_iters
    else:
        decay_ratio = (iter_num - warmup_iters) / (max_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        lr_iter = min_lr + coeff * (lr - min_lr)
    
    for opt in [opt_adam, opt_muon]:
        for pg in opt.param_groups: pg['lr'] = lr_iter

    # 2. Heavy Logging & Eval
    if iter_num % eval_interval == 0 or iter_num == max_iters:
        losses = estimate_loss()
        
        if master_process:

            time_n = time.time()
            elapsed = time_n - time_s
            dt = time_n - prev_time
            prev_time = time_n

            # Global Tokens processed in this interval
            tokens_in_interval = block_size * batch_size * grad_accum_steps * eval_interval
            throughput = tokens_in_interval / dt
            
            # MFU & TFLOPS calculation (calls method on raw_model)
            mfu, tflops = 0.0, 0.0
            if hasattr(raw_model, 'estimate_mfu'):
                mfu, tflops = raw_model.estimate_mfu(tokens_in_interval, dt)

            # Metrics
            train_loss, val_loss = losses['train'], losses['val']
            train_ppl, val_ppl = math.exp(train_loss), math.exp(val_loss)
            val_bpb = (val_loss / math.log(2)) * tokens_per_byte

            print(f"step: {iter_num} | train: {train_loss:.4f} | val: {val_loss:.4f} | bpb: {val_bpb:.4f} | "
                  f"pplx: {val_ppl:.2f} | lr: {lr_iter:.6f} | dt: {dt:.2f}s | "
                  f"tok/s: {throughput:.0f} | TFLOPS: {tflops:.2f} | MFU: {mfu*100:.1f}%")

            # wandb log step

            run.log({"train_loss": train_loss, "val_loss": val_loss, "train_ppl": train_ppl, "val_ppl": val_ppl})

            # DuckDB
            row = f"({iter_num},{lr_iter},{train_loss:.4f},{val_loss:.4f},{train_ppl:.4f},{val_ppl:.4f},{val_bpb:.4f},{throughput:.2f})"
            safe_log_to_duckdb(run_name, row)
            subprocess.Popen(["python", "plotgen.py", f"_{run_name}"])

    if iter_num % 100 == 0:

        if master_process:

            # output prints
            model_instance.eval()
            context_tokens = torch.tensor([[198]], dtype=torch.long, device=device) # sources say 198 is newline, verify later
            raw_model.reset_cache()

            with torch.no_grad():
                with ctx:
                    y = raw_model.generate(context_tokens, max_new_tokens=500, temperature=0.8, top_k=50)
            
            decoded_text = tok.decode(y[0].tolist())
            print(f"\n{'='*20} output at step {iter_num} {'='*20}\n{decoded_text}\n{'='*20}\n")
           
            sample_table = wandb.Table(columns=["step", "generation"])
            sample_table.add_data(iter_num, decoded_text)
            run.log({"samples": sample_table})

            raw_model.reset_cache()
            model_instance.train() 

    if iter_num == max_iters: break

    # 3. Training Step
    opt_adam.zero_grad(set_to_none=True)
    opt_muon.zero_grad(set_to_none=True)

    for micro_step in range(grad_accum_steps):
        # Sync only on the last step of accumulation
        m.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        xb, yb = next(train_gen)
        with ctx:
            logits, loss = compiled_model(xb, yb)
            loss = loss / grad_accum_steps
        scaler.scale(loss).backward()

    scaler.unscale_(opt_adam)
    scaler.unscale_(opt_muon)
    torch.nn.utils.clip_grad_norm_(m.parameters(), max_grad_norm)
    scaler.step(opt_adam)
    scaler.step(opt_muon)
    scaler.update()

    # 4. Checkpoint & HellaSwag (Rank 0)
    if iter_num > 0 and iter_num % ckpt_iter == 0 and master_process:
        ckpt_path = f'checkpoints/{run_name}_{iter_num}.pt'
        torch.save({
            'model': raw_model.state_dict(),
            'opt_adam': opt_adam.state_dict(),
            'opt_muon': opt_muon.state_dict(),
            'iter': iter_num,
            'run_name': run_name,
        }, ckpt_path)
        print(f"--> Saved checkpoint {ckpt_path}")

if master_process:
    print(f"Training Complete. Total time: {time.time()-time_s:.2f}s")
dist.destroy_process_group()
