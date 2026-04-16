import torch
import time
import os
import sys
import argparse
from thop import profile
from pathlib import Path

# Add the project root (parent directory of utils/) to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import models
from model.APRBiCA.APRBiCA import APRBiCA
from model.pointLoc.PointLoc import PointLoc
from model.PosePN.PosePN import PosePN
from model.PosePN.PosePNPP import PosePNPP
from model.STCLoc.STCLoc import STCLoc
from model.PoseSOE.PoseSOE import PoseSOE
from model.HypLiLoc.HypLiLoc import HypLiLoc

def get_model_size_mb(model_path):
    if os.path.exists(model_path):
        size = os.path.getsize(model_path)
        return size / (1024 * 1024)
    return 0

def check_stats():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    models = {
        "PointLoc": PointLoc(),
        "PosePN": PosePN(hidden_units=512),
        "PosePNPP": PosePNPP(hidden_units=512),
        "PoseSOE": PoseSOE(hidden_units=512),
        "STCLoc": STCLoc(steps=1),
        "HypLiLoc": HypLiLoc(hidden_units=512),
        "APRBiCA": APRBiCA(hidden_units=512),
    }

    dummy_input = torch.randn(1, 4096, 3).to(device)

    print("| **Model** | **Params (M)** | **FLOPs (G)** | **Time (ms)** | **Weight (MB)** |")
    print("|:----------|:--------------:|:-------------:|:-------------:|:---------------:|")

    model_names_ordered = ["PointLoc", "PosePN", "PosePNPP", "PoseSOE", "STCLoc", "HypLiLoc", "APRBiCA"]

    temp_dir = Path("temp_weights")
    temp_dir.mkdir(exist_ok=True)

    for name in model_names_ordered:
        model = models[name]
        model = model.to(device)
        model.eval()

        try:
            with torch.no_grad():
                flops, params = profile(model, inputs=(dummy_input.clone(),), verbose=False)

                for _ in range(10):
                    _ = model(dummy_input.clone())

                start_time = time.time()
                for _ in range(50):
                    _ = model(dummy_input.clone())
                avg_time = (time.time() - start_time) / 100 * 1000 # ms

                weight_file = temp_dir / f"{name}.pth"
                torch.save(model.state_dict(), weight_file)
                weight_size = get_model_size_mb(weight_file)

                print(f"| {name:<9} | {params/1e6:^14.3f} | {flops/1e9:^13.3f} | {avg_time:^13.3f} | {weight_size:^15.3f} |")
        except Exception as e:
            print(f"| {name:<9} | {'Error':^14} | {'Error':^14} | {'Error':^14} | {'Error':^14} |")
            print(f"# Debug: {str(e)}")

if __name__ == "__main__":
    check_stats()
