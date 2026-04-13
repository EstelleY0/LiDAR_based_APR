import sys
import os

# Prioritize local directory in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import re
import time
import torch
import torch.multiprocessing as mp
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.nclt import NCLT
from data.robotcar import RobotCar
from model.PosePN import PosePNPP
from model.STCLoc import STCLoc
from model.pointLoc.PointLoc import PointLoc
from utils.loss import AtLocCriterion
from utils.train_utils import setup, cleanup, set_seed, mkdirs, load_state_dict, load_config_as_namespace, \
    quaternion_angular_error, qexp


def main_worker(rank, world_size, conf, visible_gpus, args):
    try:
        local_gpu_id = visible_gpus[rank]

        setup(rank, world_size, local_gpu_id)
        set_seed(7 + rank)

        device = torch.device(f"cuda:{local_gpu_id}")
        if args.model.lower() == "pointloc":
            model = PointLoc().to(device)
        elif args.model.lower() == "posepnpp":
            model = PosePNPP(
                hidden_units=getattr(args, 'hidden_units', 512),
                freeze_backbone=getattr(args, 'freeze_backbone', False)
            ).to(device)
        elif args.model.lower() == "poseminkloc":
            try:
                from model.PoseMinkLoc import PoseMinkLoc
            except ImportError:
                raise ImportError("MinkowskiEngine is required for PoseMinkLoc but could not be imported.")
            model = PoseMinkLoc(
                hidden_units=getattr(args, 'hidden_units', 512),
                freeze_backbone=getattr(args, 'freeze_backbone', False),
                grid_size=getattr(args, 'grid_size', 0.01)
            ).to(device)
        elif args.model.lower() == "stcloc":
            model = STCLoc(
                steps=getattr(args, 'stcloc_steps', 1),
                freeze_backbone=getattr(args, 'freeze_backbone', False)
            ).to(device)
        else:
            raise ValueError("Not proper model input")

        model = DDP(model, device_ids=[local_gpu_id], find_unused_parameters=True)

        train_criterion = AtLocCriterion(saq=conf.beta, learn_beta=True)
        train_criterion.to(device)

        param_list = [{'params': model.parameters()}]

        optimizer = torch.optim.AdamW(param_list, lr=conf.lr, weight_decay=conf.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

        cur_file_path = os.path.realpath(__file__)
        ws_path = Path(cur_file_path).parent

        if conf.data_set == 'robotcar':
            data_dir = conf.robot_car_data_dir
            train_set = RobotCar(data_dir=data_dir, training=True)
            test_set = RobotCar(data_dir=data_dir, training=False)
        elif conf.data_set == 'nclt':
            data_dir = conf.nclt_data_dir
            train_set = NCLT(data_dir=data_dir, training=True)
            test_set = NCLT(data_dir=data_dir, training=False)
        else:
            raise ValueError("Not proper data set input")

        weights_folder = os.path.join(ws_path, f'{args.folder}', 'models')

        sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank)
        train_loader = DataLoader(train_set, batch_size=args.batchsize, sampler=sampler,
                                  num_workers=conf.nThreads, pin_memory=True, prefetch_factor=2, persistent_workers=True)

        sampler = DistributedSampler(test_set, num_replicas=world_size, rank=rank, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=args.batchsize, sampler=sampler,
                                 num_workers=conf.nThreads, pin_memory=True, prefetch_factor=2, persistent_workers=True)

        global_step = 0
        tq_mean_error_best = [10000., 10000., 0, 10000.]
        tq_median_error_best = [10000., 10000., 0, 10000.]

        resume_epoch = args.resume_epoch
        epochs = args.epochs
        if resume_epoch >= 0:
            if args.from_last:
                weight_files = glob.glob(os.path.join(weights_folder, 'epoch_*.pth.tar'))
                epoch_numbers = [int(re.search(r'epoch_(\d+)', f).group(1)) for f in weight_files if re.search(r'epoch_(\d+)', f)]
                if epoch_numbers:
                    max_epoch = max(epoch_numbers)
                    weights_filename = os.path.join(weights_folder, f'epoch_{max_epoch}.pth.tar')
            else:
                weights_filename = os.path.join(weights_folder, f'epoch_{args.resume_epoch}.pth.tar')

            if osp.isfile(weights_filename):
                checkpoint = torch.load(weights_filename, map_location=device, weights_only=False)
                load_state_dict(model.module, checkpoint['model_state_dict'])
                if 'global_step' in checkpoint:
                    global_step = checkpoint['global_step']

                if 'optim_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optim_state_dict'])
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(device)
                if 'lr_scheduler_state_dict' in checkpoint:
                    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
                if 'criterion_state_dict' in checkpoint:
                    if hasattr(train_criterion, 'load_state_dict'):
                        train_criterion.load_state_dict(checkpoint['criterion_state_dict'])
                if 'args' in checkpoint:
                    args = checkpoint['args']
                    args.resume_epoch = resume_epoch
                    args.epochs = epochs
                if 'tq_mean_error_best' in checkpoint:
                    tq_mean_error_best = checkpoint['tq_mean_error_best']
                if 'tq_median_error_best' in checkpoint:
                    tq_median_error_best = checkpoint['tq_median_error_best']
                print(f'Rank {rank}: Resumed weights from {weights_filename}')
            else:
                print(f'Rank {rank}: Could not load weights from {weights_filename}')
                sys.exit(-1)

        experiment_name = conf.exp_name
        writer = None

        t0 = time.time()

        for epoch in range(args.resume_epoch + 1, args.epochs):
            model.train()
            epoch_loss_sum = 0.0
            pose_loss_sum = 0.0
            epoch_loss_count = 0

            if rank == 0:
                writer = SummaryWriter(log_dir=osp.join(f"{args.folder}", 'runs', f"{epoch:d}"))

            sampler.set_epoch(epoch)
            if rank == 0:
                pbar = tqdm(enumerate(train_loader), total=len(train_loader))
            else:
                pbar = enumerate(train_loader)

            for batch_idx, feed_dict in pbar:
                lidar = feed_dict['lidar_float32'].to(device)
                pose = feed_dict['pose_float32'].to(device).detach()
                target = pose.clone().detach()

                with torch.set_grad_enabled(True):
                    output = model(lidar)
                    pose_loss = sum(train_criterion(output, target))

                    final_loss = pose_loss

                final_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                now_lr = optimizer.param_groups[0]['lr']
                torch.cuda.empty_cache()

                epoch_loss_sum += final_loss.item()
                pose_loss_sum += pose_loss.item()
                epoch_loss_count += 1

                if rank == 0:
                    pbar.set_description(desc='loss:{:.4f}'.format(final_loss.item()))
                    writer.add_scalar('Loss/iteration', final_loss.item(), global_step)
                    writer.add_scalar('Loss/pose', pose_loss.item(), global_step)
                    writer.add_scalar('train_criterion/sax', train_criterion.sax, global_step)
                    writer.add_scalar('train_criterion/saq', train_criterion.saq, global_step)
                    global_step += 1

            lr_scheduler.step()
            torch.cuda.empty_cache()

            if rank == 0:
                epoch_avg_loss = epoch_loss_sum / epoch_loss_count
                epoch_pose_avg = pose_loss_sum / epoch_loss_count
                writer.add_scalar('Epoch/loss_avg', epoch_avg_loss, epoch)
                writer.add_scalar('Epoch/loss_pose_avg', epoch_pose_avg, epoch)
                print("epoch {:03d} | epoch loss {:.4f}".format(epoch, epoch_avg_loss))

                filename = osp.join(weights_folder, f'epoch_{epoch}.pth.tar')
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.module.state_dict(),
                    'optim_state_dict': optimizer.state_dict(),
                    'criterion_state_dict': train_criterion.state_dict(),
                    'tq_mean_error_best': tq_mean_error_best,
                    'tq_median_error_best': tq_median_error_best,
                    'args': args,
                }, filename)

            if epoch % args.save_freq == 0:
                model.eval()
                pred_poses_list, targ_poses_list = [], []

                with torch.no_grad():
                    if rank == 0:
                        test_pbar = tqdm(test_loader, desc=f"[Epoch {epoch} Eval]")
                    else:
                        test_pbar = test_loader

                    for batch_idx, feed_dict in enumerate(test_pbar):
                        lidar = feed_dict['lidar_float32'].to(device)
                        pose = feed_dict['pose_float32'].to(device)
                        target = pose.clone().detach()

                        with torch.set_grad_enabled(False):
                            output = model(lidar)

                        s = output.size() # [b,6]
                        output = output.cpu().detach().numpy().reshape((-1, s[-1]))
                        target = target.cpu().detach().numpy().reshape((-1, s[-1]))

                        q = [qexp(p[3:]) for p in output]
                        output = np.hstack((output[:, :3], np.asarray(q)))
                        q = [qexp(p[3:]) for p in target]
                        target = np.hstack((target[:, :3], np.asarray(q)))

                        pose_m = test_set.mean_t
                        pose_s = test_set.std_t

                        output[:, :3] = (output[:, :3] * pose_s) + pose_m
                        target[:, :3] = (target[:, :3] * pose_s) + pose_m

                        for each_output in output:
                            pred_poses_list.append(each_output)

                        for each_target in target:
                            targ_poses_list.append(each_target)
                    def compute_error(pred_poses_list, targ_poses_list, fig_name, epoch):
                        pred_poses = np.vstack(pred_poses_list)
                        targ_poses = np.vstack(targ_poses_list)

                        # calculate errors
                        t_criterion = lambda t_pred, t_gt: np.linalg.norm(t_pred - t_gt)
                        q_criterion = quaternion_angular_error
                        t_loss = np.asarray([t_criterion(p, t) for p, t in zip(pred_poses[:, :3], targ_poses[:, :3])])
                        q_loss = np.asarray([q_criterion(p, t) for p, t in zip(pred_poses[:, 3:], targ_poses[:, 3:])])
                        t_median_error = np.median(t_loss)
                        q_median_error = np.median(q_loss)
                        t_mean_error = np.mean(t_loss)
                        q_mean_error = np.mean(q_loss)

                        print(f"[Epoch {epoch}]")
                        print(f"  Translation Error (median): {t_median_error:.3f} m")
                        print(f"  Rotation Error (median): {q_median_error:.3f} deg")
                        print(f"  Translation Error (mean): {t_mean_error:.3f} m")
                        print(f"  Rotation Error (mean): {q_mean_error:.3f} deg")

                        if conf.save_fig:
                            xyz_gt = targ_poses[:, :3]

                            xyz_pred = pred_poses[:, :3]

                            plt.figure(figsize=(10, 8))

                            for i in range(len(xyz_pred)):
                                gt_point = xyz_gt[i]
                                pred_point = xyz_pred[i]
                                dist = np.linalg.norm(gt_point[:2] - pred_point[:2])

                                if dist < 2.0:
                                    line_color = 'green'
                                    alp = 1.0
                                elif dist < 10:
                                    line_color = 'yellow'
                                    alp = 0.5
                                else:
                                    line_color = 'red'
                                    alp = 0.2

                                plt.plot([gt_point[0], pred_point[0]], [gt_point[1], pred_point[1]], color=line_color, linewidth=0.8, alpha=alp)

                            plt.plot(xyz_gt[:, 0], xyz_gt[:, 1], color='yellow', linewidth=1.5, label='GT')

                            plt.xlabel('X [m]')
                            plt.ylabel('Y [m]')
                            plt.title(f'Predicted vs GT Trajectory with Orientation [Epoch {epoch}]')
                            plt.legend()
                            plt.axis('equal')
                            plt.grid(True)

                            result_dir = os.path.join(ws_path, args.folder, 'figures')
                            os.makedirs(result_dir, exist_ok=True)
                            image_filename = osp.join(osp.expanduser(result_dir), f'{str(epoch)}_{fig_name}.png')
                            plt.savefig(image_filename)
                            print(f"Figure saved to {image_filename}")
                            plt.close()

                        return t_mean_error, q_mean_error, t_median_error, q_median_error

                    t_mean, q_mean, t_median, q_median = compute_error(pred_poses_list, targ_poses_list, '', epoch)

                    def update_best(txt):
                        txt.append(f'mean:   t {t_mean:.2f} m  q {q_mean:.2f}')
                        txt.append(f'median: t {t_median:.2f} m  q {q_median:.2f}')
                        if t_mean + q_mean < tq_mean_error_best[-1]:
                            tq_mean_error_best[:] = [t_mean, q_mean, epoch, t_mean + q_mean]
                        if t_median + q_median < tq_median_error_best[-1]:
                            tq_median_error_best[:] = [t_median, q_median, epoch, t_median + q_median]
                        txt.append(f'tq mean   best {tq_mean_error_best[0]:.2f}/{tq_mean_error_best[1]:.2f}\t Epoch {tq_mean_error_best[2]}')
                        txt.append(f'tq median best {tq_median_error_best[0]:.2f}/{tq_median_error_best[1]:.2f}\t Epoch {tq_median_error_best[2]}')
                        return txt

                    txt = update_best([
                        f'Train/test {experiment_name}\t Epoch {epoch}\t Lr {now_lr:.6f}\t Time {time.time() - t0:.2f}',
                        '-----------------------'
                    ])
                    with open(f'{args.folder}/results/results_{conf.bev_type}_{conf.bev_resize_size}.txt', 'a') as f:
                        f.write('\n'.join(txt) + '\n')

                    filename = osp.join(weights_folder, f'epoch_{epoch}.pth.tar')
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.module.state_dict(),
                        'optim_state_dict': optimizer.state_dict(),
                        'criterion_state_dict': train_criterion.state_dict(),
                        'tq_mean_error_best': tq_mean_error_best,
                        'tq_median_error_best': tq_median_error_best,
                        'args': args,
                    }, filename)

                if writer:
                    writer.flush()
                t0 = time.time()

        if writer:
            writer.close()
        cleanup()

    except KeyboardInterrupt:
        if rank == 0:
            filename = osp.join(weights_folder, f'epoch_{epoch}.pth.tar')
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.module.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
                'criterion_state_dict': train_criterion.state_dict(),
                'tq_mean_error_best': tq_mean_error_best,
                'tq_median_error_best': tq_median_error_best,
                'args': args,
            }, filename)

        if writer:
            writer.close()
        cleanup()


def run_ddp(args):
    conf = load_config_as_namespace('conf.yaml')

    visible_devices = conf.gpu_idx

    visible_gpus = [int(d.strip()) for d in visible_devices.split(",") if d.strip().isdigit()]
    world_size = len(visible_gpus)

    print(f"[INFO] CUDA_VISIBLE_DEVICES = '{visible_devices}' → world_size = {world_size}")
    mp.spawn(main_worker, args=(world_size, conf, visible_gpus, args), nprocs=world_size, join=True)


if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    args = load_config_as_namespace("conf.yaml")

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="robotcar", help='Dataset name')
    parser.add_argument('--model', type=str, default="pointloc", help='Model name (pointloc, posepnpp, poseminkloc, stcloc)')
    parser.add_argument('--grid_size', type=float, default=0.01, help='Voxel grid size for PoseMinkLoc (meters)')
    parser.add_argument('--stcloc_steps', type=int, default=1, help='Number of sequence steps for STCLoc temporal attention')
    parser.add_argument('--hidden_units', type=int, default=512, help='Hidden units for MARegressor (PosePN++)')
    parser.add_argument('--freeze_backbone', action='store_true', default=False, help='Freeze backbone parameters')
    parser.add_argument('--resume_epoch', type=int, default=-1, help='Resume epoch number')
    parser.add_argument('--epoch_test', type=int, default=0, help='Epoch number')
    parser.add_argument('--epochs', type=int, default=150, help='Epoch number')
    parser.add_argument('--save_freq', type=int, default=1000, help='Eval save frequency')
    parser.add_argument('--batchsize', type=int, default=32, help='Batch size')
    parser.add_argument('--batchsize_test', type=int, default=1, help='Test batch size')
    parser.add_argument('--folder', type=str, default='gpu1', help='Folder name for results or config')
    parser.add_argument('--from_last', type=bool, default=False, help='Load weight from last epoch')

    cli_args = parser.parse_args()
    for key, value in vars(cli_args).items():
        if value is not None:
            setattr(args, key, value)

    required_paths = [os.path.join(args.folder, 'runs'),
                      os.path.join(args.folder, 'models'),
                      os.path.join(args.folder, 'results')]
    mkdirs(required_paths)

    run_ddp(args)
