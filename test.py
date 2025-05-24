###
### pdm run python test.py --dataset data/GREEDY10/GREEDY10_TEST1K.json --num_cities 10  --ckpt checkpoints/ckpt_iter_300000.pt --save_dir plots
###


import argparse
import random

import numpy as np
import torch
from model import GPT
from utils import build_state, load_samples, plot_and_save, tour_length


# DECODE
# -----------------------------------------------------------------------------------------
def beam_search_decode(model, coords, init_rtg, device, beam_width, temperature=1.0):
    model.eval()
    k = coords.size(0)
    alpha = temperature

    # init beams (beam_width = 1 is greedy deconding)
    beams = [
        {
            "visited": torch.zeros(k, dtype=torch.bool, device=device).scatter_(
                0, torch.tensor([0], device=device), True
            ),
            "states": [],
            "timesteps": [],
            "masks": [],
            "tok_seq": [],
            "rtgs": [torch.tensor(init_rtg, device=device, dtype=torch.float32)],
            "last_idx": 0,
            "score": 0.0,
        }
    ]

    with torch.no_grad():
        for t in range(k - 1):
            candidates = []
            for b in beams:

                # new sequences of states, timesteps and masks
                states2 = b["states"] + [build_state(coords, b["visited"])]
                ts2 = b["timesteps"] + [torch.tensor([t + 1], device=device)]
                mask_vec = (~b["visited"][1:]).to(device)
                ms2 = b["masks"] + [mask_vec]

                st = torch.stack(states2).unsqueeze(0)  # (1, t+1, dim_state)
                ts = torch.stack(ts2).unsqueeze(0)  # (1, t+1, 1)
                ms = torch.stack(ms2).unsqueeze(0)  # (1, t+1, k-1)

                # actions until here
                if b["tok_seq"]:
                    ac = (
                        torch.tensor(b["tok_seq"], dtype=torch.long, device=device).unsqueeze(0).unsqueeze(-1)
                    )  # (1, t, 1)
                else:
                    ac = None

                # RTGs until here
                rtg_seq = torch.stack(b["rtgs"]).unsqueeze(0).unsqueeze(-1)  # (1, t+1)

                # forward
                logits, _ = model(
                    states=st,
                    actions=ac,
                    rtgs=rtg_seq,
                    timesteps=ts,
                    targets=None,
                    masks=ms,
                )
                last_logits = logits[0, -1]  # (k-1,)
                logp = torch.log_softmax(last_logits / alpha, dim=-1)
                valid_inds = mask_vec.nonzero(as_tuple=False).view(-1)

                for idx in valid_inds.tolist():
                    prev = b["last_idx"]
                    nxt = idx + 1

                    # compute the model step-wise reward
                    dist = torch.norm(coords[prev] - coords[nxt], p=2).item()
                    r_eff = -dist
                    new_rtg = b["rtgs"][-1] - r_eff

                    new_visited = b["visited"].clone()
                    new_visited[nxt] = True

                    candidates.append(
                        {
                            "visited": new_visited,
                            "states": states2,
                            "timesteps": ts2,
                            "masks": ms2,
                            "tok_seq": b["tok_seq"] + [idx],
                            "rtgs": b["rtgs"] + [torch.tensor(new_rtg, device=device)],
                            "last_idx": nxt,
                            "score": b["score"] + logp[idx].item(),
                        }
                    )

            # select best beams
            beams = sorted(candidates, key=lambda x: x["score"], reverse=True)[:beam_width]

    best_beam = max(beams, key=lambda x: x["score"])
    
    # return the city sequence (1-indexed)
    return [i + 1 for i in best_beam["tok_seq"]]


# MAIN
# -----------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="scripts/data/MIXED10/MIXED10_TEST1K.json")
    p.add_argument("--num_cities", type=int, default=10)
    p.add_argument("--ckpt", default="scripts/checkpoints/ckpt_iter_500000.pt")
    p.add_argument("--save_dir", default="scripts/plots")
    p.add_argument("--num_beams", type=int, default=1)
    p.add_argument("--num_plots", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = args.device

    # load checkpoint and rebuild naive model
    ckpt = torch.load(args.ckpt, map_location=device)
    saved = ckpt["args"]
    cfg = GPT.get_default_config()
    cfg.model_type = "reward_conditioned"
    cfg.num_cities = saved["num_cities"]
    cfg.n_layer = saved["n_layer"]
    cfg.n_head = saved["n_head"]
    cfg.n_embd = saved["n_embd"]

    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    # evaluate
    samples = load_samples(args.dataset)
    indices = random.sample(range(len(samples)), min(args.num_plots, len(samples)))
    teacher_lens, model_lens = [], []
    initializations = []

    for coords, teacher_actions, rtgs in samples:

        # TARGET RTG 𝓡₁=R₁+𝞭
        init_rtgs = rtgs[0] # + 𝞭
        initializations.append(init_rtgs)
        
        teacher_len = tour_length(coords, teacher_actions)
        teacher_lens.append(teacher_len)
        
        model_tours = beam_search_decode(model, coords.to(device), init_rtgs, device, args.num_beams)
        model_lens.append(tour_length(coords, model_tours))

    mean_initializations = np.mean(initializations)

    tmean = np.mean(teacher_lens)
    mmean = np.mean(model_lens)
    gap = (mmean - tmean) / tmean * 100.0

    print()
    print(f"MEAN   TARGET   RTG: {mean_initializations}")
    print(f"TEACHER MEAN LENGTH: {tmean:.4f}")
    print(f"MODEL  MEAN  LENGTH: {mmean:.4f}")
    print(f"RELATIVE    GAP    : {gap:+.2f}%")

    #for i, idx in enumerate(indices, 1):
    #    coords, teacher_actions, rtgs = samples[idx]
    #    init_rtgs = rtgs[0]
    #    coords = coords.to(device)
    #    model_tour = beam_search_decode(model, coords, init_rtgs, device, args.num_beams)
    #    plot_and_save(
    #        i,
    #        coords.cpu(),
    #        teacher_actions,
    #        model_tour,
    #        tour_length(coords, teacher_actions),
    #        tour_length(coords, model_tour),
    #        args.save_dir,
    #    )


if __name__ == "__main__":
    main()
