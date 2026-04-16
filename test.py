import argparse
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.nclt import NCLT
from data.robotcar import RobotCar
from data.vreloc import VReLoc
from model.APRBiCA import APRBiCA
from model.HypLiLoc import HypLiLoc
from model.PosePN import PosePN
from model.PosePN import PosePNPP
from model.PoseSOE import PoseSOE
from model.STCLoc import STCLoc
from model.pointLoc.PointLoc import PointLoc
from utils.train_utils import load_config_as_namespace, quaternion_angular_error, qexp


def test():
    parser = argparse.ArgumentParser(description='Inference script for LiDAR-based APR')
    parser.add_argument('--model', type=str, required=True, help='Model name (pointloc, posepnpp, posepn, poseminkloc, stcloc, posesoe)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint (.pth.tar)')
    parser.add_argument('--dataset', type=str, default='robotcar', choices=['robotcar', 'nclt', 'vreloc'], help='Dataset name')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--save_fig', action='store_true', help='Save trajectory figure')
    
    # Model specific args
    parser.add_argument('--hidden_units', type=int, default=512, help='Hidden units for MARegressor')
    parser.add_argument('--stcloc_steps', type=int, default=3, help='Steps for STCLoc')
    parser.add_argument('--stcloc_skip', type=int, default=2, help='Skip for STCLoc')
    parser.add_argument('--num_class_loc', type=int, default=10, help='STCLoc num_class_loc')
    parser.add_argument('--num_class_ori', type=int, default=10, help='STCLoc num_class_ori')
    parser.add_argument('--grid_size', type=float, default=0.01, help='Grid size for PoseMinkLoc')
    parser.add_argument('--sparse_engine', type=str, default='spconv', choices=['spconv', 'minkowski'], help='Sparse engine for PoseMinkLoc')

    args = parser.parse_args()
    conf = load_config_as_namespace('conf.yaml')

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Initialize model
    if args.model.lower() == "pointloc":
        model = PointLoc()
    elif args.model.lower() == "posepnpp":
        model = PosePNPP(hidden_units=args.hidden_units)
    elif args.model.lower() == "posepn":
        model = PosePN(hidden_units=args.hidden_units)
    elif args.model.lower() == "poseminkloc":
        try:
            from model.PoseMinkLoc import PoseMinkLoc
        except ImportError:
            raise ImportError("MinkowskiEngine is required for PoseMinkLoc.")
        model = PoseMinkLoc(hidden_units=args.hidden_units, grid_size=args.grid_size, sparse_engine=args.sparse_engine)
    elif args.model.lower() == "stcloc":
        num_loc = args.num_class_loc
        model = STCLoc(
            steps=args.stcloc_steps,
            num_class_loc=num_loc * num_loc,
            num_class_ori=args.num_class_ori
        )
    elif args.model.lower() == "posesoe":
        model = PoseSOE(hidden_units=args.hidden_units)
    elif args.model.lower() == "hypliloc":
        model = HypLiLoc(hidden_units=args.hidden_units)
    elif args.model.lower() == "aprbica":
        model = APRBiCA(hidden_units=args.hidden_units)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model = model.to(device)

    # Load weights
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Check if checkpoint has 'model_state_dict'
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Handle potential DDP prefix 'module.'
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model.eval()

    # Load dataset
    if args.dataset == 'robotcar':
        test_set = RobotCar(data_dir=conf.robot_car_data_dir, training=False, num_class_loc=args.num_class_loc, num_class_ori=args.num_class_ori)
    elif args.dataset == 'nclt':
        test_set = NCLT(data_dir=conf.nclt_data_dir, training=False, num_class_loc=args.num_class_loc, num_class_ori=args.num_class_ori)
    elif args.dataset == 'vreloc':
        test_set = VReLoc(data_dir=conf.vReLoc_data_dir, training=False, num_class_loc=args.num_class_loc, num_class_ori=args.num_class_ori)
    
    if args.model.lower() == "stcloc" and args.stcloc_steps > 1:
        from data.composition import SequenceDataset
        test_set = SequenceDataset(test_set, steps=args.stcloc_steps, skip=args.stcloc_skip)

    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=conf.nThreads)

    pred_poses_list, targ_poses_list = [], []

    print("Running inference...")
    with torch.no_grad():
        for feed_dict in tqdm(test_loader):
            lidar = feed_dict['lidar_float32'].to(device)
            target_pose = feed_dict['pose_float32'].to(device)

            if lidar.dim() == 4: # [B, T, N, 3]
                B, T, N, C = lidar.shape
                lidar = lidar.view(B * T, N, C)
                target_pose = target_pose.view(B * T, -1)

            output = model(lidar)
            if isinstance(output, tuple):
                output = output[0]

            # Process output and target
            output_np = output.cpu().numpy().reshape((-1, 6))
            target_np = target_pose.cpu().numpy().reshape((-1, 6))

            # Convert log quaternion to standard quaternion
            q_out = [qexp(p[3:]) for p in output_np]
            output_np = np.hstack((output_np[:, :3], np.asarray(q_out)))
            
            q_targ = [qexp(p[3:]) for p in target_np]
            target_np = np.hstack((target_np[:, :3], np.asarray(q_targ)))

            base_test_set = test_set.dataset if hasattr(test_set, 'dataset') else test_set
            pose_m = base_test_set.mean_t
            pose_s = base_test_set.std_t
            output_np[:, :3] = (output_np[:, :3] * pose_s) + pose_m
            target_np[:, :3] = (target_np[:, :3] * pose_s) + pose_m

            for p in output_np: pred_poses_list.append(p)
            for t in target_np: targ_poses_list.append(t)

    # Calculate errors
    pred_poses = np.vstack(pred_poses_list)
    targ_poses = np.vstack(targ_poses_list)

    t_err = np.linalg.norm(pred_poses[:, :3] - targ_poses[:, :3], axis=1)
    q_err = np.array([quaternion_angular_error(p[3:], t[3:]) for p, t in zip(pred_poses, targ_poses)])

    print("\nResults:")
    print(f"Translation Error: Mean={np.mean(t_err):.3f}m, Median={np.median(t_err):.3f}m")
    print(f"Rotation Error:    Mean={np.mean(q_err):.3f}°, Median={np.median(q_err):.3f}°")

    if args.save_fig:
        plt.figure(figsize=(10, 8))
        plt.plot(targ_poses[:, 0], targ_poses[:, 1], 'y-', label='Ground Truth', linewidth=2)
        plt.scatter(pred_poses[:, 0], pred_poses[:, 1], c=t_err, cmap='jet', s=5, label='Predictions')
        plt.colorbar(label='Translation Error [m]')
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.title(f'Trajectory Error - {args.model}')
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.savefig(f'results_{args.model}.png')
        print(f"Figure saved to results_{args.model}.png")

if __name__ == '__main__':
    test()
