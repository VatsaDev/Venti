# config sweep gen

import os
import subprocess
import glob

lr = [1e-3, 8e-4, 5e-4, 3e-4]
dropout = [0.0, 0.05, 0.1, 0.2]
weight_decay = [0.1, 0.01, 0.02, 0.05, 0.1]

sweeps_folder = "sweeps"
results_file = "sweep_results.md"
os.makedirs(sweeps_folder, exist_ok=True)

def get_finished_configs():
    finished = set()
    if not os.path.exists(results_file):
        return finished
    
    with open(results_file, "r") as f:
        for line in f:
            # lines starting with | and containing .toml
            if line.startswith("|") and ".toml" in line:
                parts = [p.strip() for p in line.split("|")]
                config_name = parts[1]
                # checks if the third column (Run ID) is NOT "ERROR"
                # ensures re-run crashed
                if len(parts) > 2 and parts[2] != "ERROR":
                    finished.add(config_name)
    return finished

c = 0
for l in lr:
    for d in dropout:
        for w in weight_decay:
            toml_base = f"""
                n_embd = 64
                n_layer = 4
                n_head = 2

                batch_size = 4
                block_size = 4096 
                eval_interval = 20
                grad_accum_steps = 2 

                lr = {l}
                min_lr = {l/10}

                max_iters = 4000
                eval_iters = 20
                warmup_iters = 200 

                dropout = {d}
                weight_decay = {w}

                data_dir = "synth"
                """
            
            path = os.path.join(sweeps_folder, f"config_{c}.toml")
            with open(path, "w") as f:
                f.write(toml_base)
            c += 1

print(f"Total configs generated/checked: {c}")

sweep_configs = sorted(glob.glob(f"{sweeps_folder}/*.toml"))
finished_list = get_finished_configs()

print(f"Found {len(sweep_configs)} total configs.")
print(f"Skipping {len(finished_list)} already completed runs.\n")

for config_path in sweep_configs:
    config_file = os.path.basename(config_path)
    
    if config_file in finished_list:
        # print(f"SKIPPING: {config_file}") # Optional: uncomment if you want to see skips
        continue

    print(f"LAUNCHING: {config_file}")
    
    try:
        # Pass the config path to your training script
        subprocess.run(["python", "train.py", "--conf", config_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Run failed for {config_file}: {e}")
        # Log error so we know it failed, but get_finished_configs won't skip it next time
        with open(results_file, "a") as f:
            f.write(f"| {config_file} | ERROR | N/A | N/A | N/A |\n")

print("\nALL RUNS COMPLETE.")
