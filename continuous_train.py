import os
import subprocess
import glob
import time

def run_training():
    config_dir = "training_configs"
    if not os.path.exists(config_dir):
        print(f"Error: {config_dir} directory not found.")
        return

    configs = sorted(glob.glob(os.path.join(config_dir, "*.yaml")))
    
    if not configs:
        print(f"No .yaml files found in {config_dir}.")
        return

    print(f"Found {len(configs)} configurations to train.")

    for config_path in configs:
        print(f"\n" + "="*50)
        print(f"Starting training with config: {config_path}")
        print("="*50 + "\n")

        try:
            result = subprocess.run(
                ["python", "train.py", "--config", config_path],
                check=False
            )

            if result.returncode == 0:
                print(f"\nSuccessfully finished training with {config_path}")
            else:
                print(f"\nTraining failed for {config_path} with return code {result.returncode}")
                print("Proceeding to the next configuration...")

        except Exception as e:
            print(f"\nAn error occurred while running {config_path}: {e}")
            print("Proceeding to the next configuration...")
        
        # Optional: small delay between runs to ensure system/GPU resources are fully released
        time.sleep(10)

    print("\nAll training configurations processed.")

if __name__ == "__main__":
    run_training()
