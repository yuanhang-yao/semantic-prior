import argparse
import os
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--model", type=str, default="MLCL", choices=("MLCL", "ALCL"))
    parser.add_argument("--save_tag", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--eval_start_epoch", type=int, default=100)
    parser.add_argument("--lr_inner", type=float, default=1e-3)
    parser.add_argument("--lr_outer", type=float, default=1e-3)
    parser.add_argument("--lr_alpha", type=float, default=1e-3)
    parser.add_argument("--dataset_root", type=str, default=None)
    args = parser.parse_args()
    print("=" * 48)
    print(f"Time : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PID  : {os.getpid()}")
    print("-" * 48)
    print("Arguments:")
    print("-" * 48)
    for key, value in vars(args).items():
        print(f"{key:<22}: {value}")
    print("=" * 48)
    return args
