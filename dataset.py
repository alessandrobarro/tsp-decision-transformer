###
###
###


import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from utils import pad_collate


class TrajectoryDataset(Dataset):

    def __init__(self, path: str | Path, num_cities: int):
        self.path        = Path(path)
        self.num_cities  = num_cities
        self.T           = num_cities - 1
        text             = self.path.read_text()
        self.data        = json.loads(text)

    def __len__(self):
        return len(self.data)

    def _build_state(self, coords, visited):
        return torch.tensor([c + [v] for c, v in zip(coords, visited)], dtype=torch.float32).flatten()

    def __getitem__(self, idx):
        sample   = self.data[idx]
        coords   = sample["x"]                               # list[list[2]]
        actions  = sample["actions"]                         # length T ints
        rtgs     = sample["rtgs"]
        masks    = sample["masks"]                           # T × (n-1)

        # build per-step visited flags
        visited  = torch.zeros(self.num_cities, dtype=torch.float32)
        visited[0] = 1.0                                    # start city always visited
        states   = []
        for a in actions:
            states.append(self._build_state(coords, visited))
            visited[a] = 1.0                                # mark the city we just visited

        actions_  = [a - 1 for a in actions]

        states     = torch.stack(states)                    # (T, n*3)
        actions_t  = torch.tensor(actions_, dtype=torch.long).unsqueeze(-1)  # (T,1)
        targets    = torch.tensor(actions_, dtype=torch.long)                # (T,)
        rtgs_t     = torch.tensor(rtgs,    dtype=torch.float32).unsqueeze(-1)
        timesteps  = torch.arange(1, self.T+1).unsqueeze(-1)                # (T,1)
        masks_t    = torch.tensor(masks, dtype=torch.bool)                  # (T,n-1)

        return dict(states=states,
                    actions=actions_t,
                    targets=targets,
                    rtgs=rtgs_t,
                    timesteps=timesteps,
                    masks=masks_t)







