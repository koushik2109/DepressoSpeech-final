import os
import subprocess
import yaml

config_path = "configs/training.yaml"

def run_training():
    print("Running training...")
    result = subprocess.run(["python3", "scripts/train_v2.py", "--config", config_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    best_ccc = -1.0
    for line in result.stdout.split('\n'):
        if "--> New best model saved with CCC:" in line:
            try:
                ccc = float(line.split(":")[-1].strip())
                if ccc > best_ccc:
                    best_ccc = ccc
            except:
                pass
    print(f"Training finished. Best CCC: {best_ccc}")
    if best_ccc == -1.0:
        print("Training failed or did not report CCC. Output:")
        print(result.stdout)
    return best_ccc

def update_config(lr, dropout, ccc_weight):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    config["model"]["dropout"] = dropout
    config["training"]["learning_rate"] = lr
    config["loss"]["ccc_weight"] = ccc_weight
    
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)

lrs = [5e-4, 1e-4, 8e-4]
dropouts = [0.1, 0.2, 0.3]
ccc_weights = [1.5, 2.0, 3.0]

for lr in lrs:
    for drop in dropouts:
        for cw in ccc_weights:
            print(f"Testing LR: {lr}, Dropout: {drop}, CCC Weight: {cw}")
            update_config(lr, drop, cw)
            ccc = run_training()
            if ccc >= 0.55:
                print(f"Success! Reached {ccc} CCC with LR={lr}, Drop={drop}, CCC Weight={cw}")
                exit(0)

print("Failed to reach 0.55 CCC after all trials.")
