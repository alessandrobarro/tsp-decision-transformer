###
###
###

import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import CfgNode as CN


# DECISION TRANSFORMER
# ----------------------------------------------------------------------------------------------------
class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.block_size = 3 * (config.num_cities - 1)
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(self.block_size, self.block_size)).view(1, 1, self.block_size, self.block_size),
        )
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(config.n_embd, 4 * config.n_embd),
                c_proj=nn.Linear(4 * config.n_embd, config.n_embd),
                act=NewGELU(),
                dropout=nn.Dropout(config.resid_pdrop),
            )
        )
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))  # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT Language Model"""

    @staticmethod
    def get_default_config():
        C = CN()
        C.n_layer = 8
        C.n_head = 8
        C.n_embd = 512
        C.num_cities = 10
        C.model_type = "reward_conditioned"
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        return C

    def __init__(self, config):
        super().__init__()
        self.config = config

        # useful dimensions
        self.num_cities = (
            self.config.num_cities
        )  # In a TSP5, num_cities = 5 (implicit start action 0 included! Although not part of the vocabulary...)
        self.max_timestep = self.num_cities - 1  # Since 0 is implicit, steps range from 1 to 4
        self.vocab_size = self.max_timestep
        self.block_size = 3 * self.max_timestep  # e.g. in a TSP5 [r1 s1 a1...r4 a4 a4]
        self.state_size = (
            self.num_cities * 3
        )  # e.g. in a TSP5 coordinates are **instead 5 pairs**, mask has also **length 5**, beware of this when considering the mask for the logits
        self.n_embd = config.n_embd
        self.model_type = config.model_type

        # transformer
        self.transformer = nn.ModuleDict(
            dict(
                drop=nn.Dropout(config.embd_pdrop),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(self.n_embd),
            )
        )
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size, bias=False)

        # DT embeddings
        self.state_encoder = nn.Linear(self.state_size, self.n_embd)
        self.ret_emb = nn.Linear(1, self.n_embd)
        # self.action_emb = nn.Embedding(self.vocab_size, self.n_embd, padding_idx=0)
        self.action_emb = nn.Embedding(self.vocab_size, self.n_embd)
        nn.init.normal_(self.action_emb.weight, mean=0.0, std=0.02)

        # position embeddings
        self.pos_emb = nn.Parameter(torch.zeros(1, self.block_size, self.n_embd))
        self.global_pos_emb = nn.Parameter(
            torch.zeros(1, self.max_timestep + 1, self.n_embd)
        )  # +1 because we allow timestep == num_cities ‑ 1 (max)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params / 1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()

        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        no_decay.add("pos_emb")
        no_decay.add("global_pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(param_dict.keys() - union_params) == 0, (
            "parameters %s were not separated into either decay/no_decay set!"
            % (str(param_dict.keys() - union_params),)
        )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(
        self,
        states,  # (B, T, state_size)
        actions=None,  # (B, T, 1)
        rtgs=None,  # (B, T, 1)
        timesteps=None,  # (B, T, 1)
        masks=None,  # (B, T, num_cities)
        targets=None,  # (B, T) actions to predict
    ):
        device = states.device
        B, T, _ = states.shape

        # state embeddings
        state_tok = self.state_encoder(states)  # (B, T, n_embd)

        # token layout depends on model type
        if actions is not None and self.model_type == "reward_conditioned":
            rtg_tok = self.ret_emb(rtgs)  # (B, T, n_embd)
            act_tok = self.action_emb(actions.squeeze(-1))  # (B, T, n_embd), es T=4 TSP5 (64, 4, 256)
            tok = torch.zeros((B, T * 3 - int(targets is None), self.n_embd), dtype=torch.float32, device=device)
            tok[:, ::3, :] = rtg_tok  # [r1 s1 a1, ..., r4 s4 a4], block size is 4 ! (?)
            tok[:, 1::3, :] = state_tok
            tok[:, 2::3, :] = act_tok[:, -T + int(targets is None) :, :]
        elif (
            actions is None and self.model_type == "reward_conditioned"
        ):  # only happens at very first timestep of evaluation
            rtg_tok = self.ret_emb(rtgs)  # (B, T, n_embd)
            tok = torch.zeros((B, T * 2, self.n_embd), dtype=torch.float32, device=device)
            tok[:, ::2, :] = rtg_tok  # really just [:, 0, :]
            tok[:, 1::2, :] = state_tok  # really just [:, 1, :]
        elif actions is not None and self.model_type == "naive":
            tok = torch.zeros((B, T * 2 - int(targets is None), self.n_embd), dtype=torch.float32, device=device)
            act_tok = self.action_emb(actions.squeeze(-1))  # (B, T, n_embd), es T=4 TSP5 (64, 4, 256)
            tok[:, ::2, :] = state_tok
            tok[:, 1::2, :] = act_tok[:, -T + int(targets is None) :, :]
        elif actions is None and self.model_type == "naive":  # only happens at very first timestep of evaluation
            tok = state_tok
        else:
            raise NotImplementedError()

        # positional embeddings
        seq_len = tok.shape[1]

        # make repeats per token: 2 tokens/time‐step for naive, 3 for reward‐conditioned
        if self.model_type == "naive":
            rep = 2
        else:  # reward_conditioned
            rep = 3

        # build a long enough step_ids, then crop to seq_len
        full = timesteps.squeeze(-1).repeat_interleave(rep, dim=1)  # (B, rep * T_eff)
        step_ids = full[:, :seq_len]
        # (B, seq_len)
        # now grab exactly seq_len position embeddings
        pos_emb = self.pos_emb[:, :seq_len, :]  # (1, seq_len, d)
        all_glob = self.global_pos_emb.repeat(B, 1, 1)  # (B, max_t+1, d)
        glob_emb = torch.gather(all_glob, 1, step_ids.unsqueeze(-1).expand(-1, -1, self.n_embd))  # (B, seq_len, d)

        # make input
        x = self.transformer.drop(tok + pos_emb + glob_emb)

        # forward the GPT model itself
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # we only care about the logits predicting the next city (i.e. positioned at the state tokens)
        if actions is not None and self.model_type == "reward_conditioned":
            logits = logits[:, 1::3, :]  # only keep predictions from state_embeddings
        elif actions is None and self.model_type == "reward_conditioned":
            logits = logits[:, 1:, :]
        elif actions is not None and self.model_type == "naive":
            logits = logits[:, ::2, :]  # only keep predictions from state_embeddings
        elif actions is None and self.model_type == "naive":
            logits = logits  # for completeness
        else:
            raise NotImplementedError()

        # mask invalid actions
        if masks is not None:  # keep only the columns for the “real” cities (1 … num_cities-1)
            invalid = masks == 0  # 1 → valid, 0 → invalid
            logits = logits.masked_fill(invalid, -1e9)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        return logits, loss
