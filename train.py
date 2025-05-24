###
### python train.py --dataset data/GREEDY10/GREEDY10_TRAIN1M.json --num_cities 10
###


import argparse
from pathlib import Path

import torch
from dataset import TrajectoryDataset
from model import GPT
from trainer import Trainer
import matplotlib.pyplot as plt
import numpy as np


# PARSE ARGS
# -----------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="scripts/data/MIXED10/MIXED10_TEST1K.json")
    p.add_argument("--num_cities", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--n_layer", type=int, default=1)
    p.add_argument("--n_head", type=int, default=1)
    p.add_argument("--n_embd", type=int, default=64)
    p.add_argument("--model_type", choices=["reward_conditioned", "naive"], default="reward_conditioned")
    p.add_argument("--max_iters", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=1000)
    p.add_argument("--save_every", type=int, default=50000)
    p.add_argument("--checkpoint_dir", type=str, default="scripts/checkpoints")
    return p.parse_args()


# MODEL CONFIG
# -----------------------------------------------------------------------------------------
def build_model(args):
    model_cfg = GPT.get_default_config()
    model_cfg.model_type = args.model_type
    model_cfg.num_cities = args.num_cities
    model_cfg.n_layer = args.n_layer
    model_cfg.n_head = args.n_head
    model_cfg.n_embd = args.n_embd
    return GPT(model_cfg)


# MAIN
# -----------------------------------------------------------------------------------------
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = TrajectoryDataset(args.dataset, args.num_cities)
    model = build_model(args).to(device)

    trainer_cfg = Trainer.get_default_config()
    trainer_cfg.device = device
    trainer_cfg.batch_size = args.batch_size
    trainer_cfg.learning_rate = args.learning_rate
    trainer_cfg.weight_decay = args.weight_decay
    trainer_cfg.grad_norm_clip = 1.0
    trainer_cfg.max_iters = args.max_iters
    trainer_cfg.num_workers = 0

    trainer = Trainer(trainer_cfg, model, ds)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def logger(tr):
        if tr.iter_num % args.log_every == 0:
            print(f"ITER {tr.iter_num:8d} | " f"LOSS {tr.last_loss:.4f} | " f"{tr.iter_dt*1000:6.1f} ms/iter")

    def checkpoint(tr):
        if (tr.iter_num % args.save_every == 0) or (tr.iter_num == trainer_cfg.max_iters):
            path = ckpt_dir / f"ckpt_iter_{tr.iter_num}.pt"
            torch.save(
                {
                    "model_state_dict": tr.model.state_dict(),
                    "iter": tr.iter_num,
                    "args": vars(args),
                },
                path,
            )
            print(f"\n✩ SAVED CHECKPOINT {path}\n")

    trainer.add_callback("on_batch_end", logger)
    trainer.add_callback("on_batch_end", checkpoint)

    losses = trainer.run()

    fig, ax = plt.subplots()
    ax.plot(losses)
    fig.savefig("scripts/plots/loss_over_timesteps")


if __name__ == "__main__":
    main()
