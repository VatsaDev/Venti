# config sweep gen

import os
import subprocess
import glob

lr = [1e-3, 8e-4, 5e-4, 3e-4]
dropout = [0.0, 0.05, 0.1, 0.2]
weight_decay = [0.1, 0.01, 0.02, 0.05, 0.1]

sweeps_folder = "sweeps"
os.makedirs(sweeps_folder, exist_ok=True)

c = 0

for l in lr:
    for d in dropout:
        for w in weight_decay:
            toml_base = f"""
                n_embd = 64
                n_layer = 4
                n_head = 2

                batch_size = 4
                block_size = 4096 # ctx_len
                eval_interval = 20
                grad_accum_steps = 2 # basically microbatch

                lr = {l}
                min_lr = {l/10}

                max_iters = 4000
                eval_iters = 20
                warmup_iters = 200 

                dropout = {d}
                weight_decay = {w}

                data_dir = "synth"
                """
            
            path = sweeps_folder+f"/config_{c}.toml"
            
            with open(path, "w") as f:
                f.write(toml_base)

            f.close()

            c += 1

print("configs made")

sweep_configs = sorted(glob.glob("sweeps/*.toml"))

print(f"Found {len(sweep_configs)} configs. Starting sweep...")

for config in sweep_configs:
    print(f"LAUNCHING: {config}")
    
    # calls train.py and waits for it, has err checks
    try:
        subprocess.run(["python", "train.py", "--conf", config], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Run failed for {config}: {e}")
        # Append failure to markdown if you want
        with open("sweep_results.md", "a") as f:
            f.write(f"| {os.path.basename(config)} | ERROR | N/A | N/A | N/A |\n")

print("\nALL RUNS COMPLETE. Check sweep_results.md for the table.")
