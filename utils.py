###
###
###


import json
import os
import random
import sys
from ast import literal_eval
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate


# MISCELLANEOUS
# -----------------------------------------------------------------------------------------
def set_seed(seed=54):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_samples(path):
    raw = json.loads(Path(path).read_text())
    samples = []
    for traj in raw:
        coords = torch.tensor(traj["x"], dtype=torch.float32)
        acts = traj["actions"]
        rtgs = traj["rtgs"]
        samples.append((coords, acts, rtgs))
    return samples


def pad_collate(batch):
    # not being used atm, we train on trajevtories of same size
    return default_collate(batch)


def plot_and_save(idx, coords, tour, length, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    # prepare full orders including return to 0
    tour_ = [0] + tour + [0]
    points = coords[tour_].numpy()

    # model plot
    fig, ax = plt.subplots()
    ax.scatter(coords[:, 0], coords[:, 1], s=20)
    ax.plot(points[:, 0], points[:, 1], "-x", label=f"Model Tour")
    ax.set_title(f"Sample {idx}: Model, len = {length}")
    ax.axis("equal")
    ax.legend()
    fig.savefig(os.path.join(save_dir, f"sample_{idx}_model.png"))
    plt.close(fig)


# TSP
# -----------------------------------------------------------------------------------------
def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)


def build_state(coords, visited):
    flag = visited.float().unsqueeze(-1)
    state_mat = torch.cat([coords, flag], dim=-1)
    return state_mat.flatten()


def tour_length(coords, tour):
    order = [0] + tour + [0]
    pts = coords[order]
    return (pts[1:] - pts[:-1]).norm(dim=-1).sum().item()


def tour_rewards(instance: np.ndarray, tour: List[int]) -> List[float]:
    n = len(tour)
    return [-euclidean_distance(instance[tour[i]], instance[tour[(i + 1) % n]]) for i in range(n)]


def compute_rtgs(rewards: List[float]) -> List[float]:
    rtg, cum = [], 0.0
    for r in reversed(rewards):
        cum += r
        rtg.insert(0, cum)
    return rtg


# CLASSES
# -----------------------------------------------------------------------------------------
class CfgNode:
    """a lightweight configuration class inspired by yacs"""

    # TODO: convert to subclass from a dict like in yacs?
    # TODO: implement freezing to prevent shooting of own foot
    # TODO: additional existence/override checks when reading/writing params?

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return self._str_helper(0)

    def _str_helper(self, indent):
        """need to have a helper to support nested indentation for pretty printing"""
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                parts.append("%s:\n" % k)
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append("%s: %s\n" % (k, v))
        parts = [" " * (indent * 4) + p for p in parts]
        return "".join(parts)

    def to_dict(self):
        """return a dict representation of the config"""
        return {k: v.to_dict() if isinstance(v, CfgNode) else v for k, v in self.__dict__.items()}

    def merge_from_dict(self, d):
        self.__dict__.update(d)

    def merge_from_args(self, args):
        """
        update the configuration from a list of strings that is expected
        to come from the command line, i.e. sys.argv[1:].

        The arguments are expected to be in the form of `--arg=value`, and
        the arg can use . to denote nested sub-attributes. Example:

        --model.n_layer=10 --trainer.batch_size=32
        """
        for arg in args:
            keyval = arg.split("=")
            assert len(keyval) == 2, "expecting each override arg to be of form --arg=value, got %s" % arg
            key, val = keyval  # unpack

            # first translate val into a python object
            try:
                val = literal_eval(val)
                """
                need some explanation here.
                - if val is simply a string, literal_eval will throw a ValueError
                - if val represents a thing (like an 3, 3.14, [1,2,3], False, None, etc.) it will get created
                """
            except ValueError:
                pass

            # find the appropriate object to insert the attribute into
            assert key[:2] == "--"
            key = key[2:]  # strip the '--'
            keys = key.split(".")
            obj = self
            for k in keys[:-1]:
                obj = getattr(obj, k)
            leaf_key = keys[-1]

            # ensure that this attribute exists
            assert hasattr(obj, leaf_key), f"{key} is not an attribute that exists in the config"

            # overwrite the attribute
            print("command line overwriting config attribute %s with %s" % (key, val))
            setattr(obj, leaf_key, val)
