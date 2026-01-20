import math
import torch
import inspect
import torch.nn as nn
from dataclasses import dataclass
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.functional import scaled_dot_product_attention as flash_attn
from torch.nn.attention.flex_attention import flex_attention as flex_attn
from torch.nn.attention.flex_attention import create_block_mask

from sig import ViT

config = {
    "n_embd": 128,
    "n_head": 2,  # keep a minimum of two heads, but always aim for head_size = 128 at scale
    "n_layer": 5,
    "dropout": 0.05,
    "vocab_size": 50257,
    "ctx_len": 1024,
    "swa_ratio": 4, # 3/4 layers swa, every 4th layer is global
    "bias": False,
    "gradient_checkpointing": True,
    "image_token_id": 200000
}

# adding vision conf

@dataclass
class SiglipConfig:
    vit_model_type: str = "google/siglip2-base-patch32-256"
    vit_img_size: int = 256       # this is technically the fixed version, adding qwen dynamic rn
    vit_patch_size: int = 32
    vit_hidden_dim: int = 768
    vit_inter_dim: int = 3072     # 4x hidden_dim
    vit_n_heads: int = 12
    vit_cls_flag: bool = False

SIGLIP_LAYERS = [1, 3, 5, 10] # 4 intermediate layers

# LANGUAGE MODEL PARTS

class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x is FP16 (from autocast). We force weight to match x
        # This hopefully triggers the 'Fused' kernel on the T4
        return torch.nn.functional.rms_norm(x, (x.shape[-1],), self.weight.to(x.dtype), self.eps)

# actually RoPE is completely replaced by M_Rope in qwen-3, its a 3d superset of rope
# basically rotate not with pos, but t (time), w (width) and h (height)
# videos need all 3, images keep t constant, and for text t=w=h
# this is the interleave variant, not MHRoPE

class MRoPE(nn.Module):
    def __init__(self, d_head):
        super().__init__()

        self.d_head = d_head
        self.ctx = config["ctx_len"]

        inv_freq = 1.0 / (1000000.0 ** (torch.arange(0, d_head, 2).float() / d_head )) # frequency tensor (1, 0.8, 0.6 ...)
        self.register_buffer('inv_freq', inv_freq)                                     # size (d_head//2, basically every other element along T)

        axis_id = torch.arange(d_head // 2) % 3 # axii tensor (0, 1, 2, 0 ...)
        self.register_buffer('axis_id', axis_id)

    def rotate_half(self, x):

        # x shape [bs, n_head, T, head_size/d_head] -> x1/x2 split
        # x1/x2 shape [bs n_head, T, d_head//2], ... ellipsis and before/after last dim

        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]

        # returns to [bs, n_h, T, h_s], but hs is [[-hs//2] : [hs//2]]

        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x, pos_ids): # pos_ids is just [[t, h, w] ...], ex. [[0, 0, 0], [0,0,1], [1,0,0]] for two patches from the same image and a text tok

        # from the looks of it mrope doesnt need the offset either, since we literally pass pos_ids

        # x shape [bs, n_head, T, hs]

        T = x.shape[2]

        # get the axii per position from the pos_ids
        # makes n_head//2 pairs, every mask in pos_ids gets stretched to axis size

        # shape (T, hs//2)

        positions = pos_ids[:, self.axis_id] # if pos_id was [0, 3, 2] (first image, third row, second patch), with axis_id its [0, 3, 2, 0, 3, 2 ...] to 128 (d_head)

        # (T, hs//2) * (hs//2) -> (T, hs//2), but now freq mult

        freqs = positions * self.inv_freq

        # concat 2 (T, hs//2) -> (T, hs)

        emb = torch.cat((freqs, freqs), dim=-1)

        cos = emb.cos().view(1, 1, T, self.d_head) # now with angles
        sin = emb.sin().view(1, 1, T, self.d_head) # (1, 1, T, hs)

        # (bs, n_h, T, h_s) * (1, 1, T, hs) -> (bs, n_h, T, hs)

        return (x * cos) + (self.rotate_half(x) * sin)

class RoPE(nn.Module):
    def __init__(self, d_head):
        super().__init__()
        self.d_head = d_head
        self.ctx = config['ctx_len']

        # Precompute cos and sin
        inv_freq = 1.0 / (1000000.0 ** (torch.arange(0, d_head, 2).float() / d_head))
        t = torch.arange(self.ctx, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer('cos', emb.cos().view(1, 1, self.ctx, d_head))
        self.register_buffer('sin', emb.sin().view(1, 1, self.ctx, d_head))

    def rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x, offset=0):
        # x shape: (B, nh, T, hs)
        T = x.size(2)

        # --- OFFSET LOGIC ---
        # Instead of 0:T, we take offset : offset + T
        # If offset is 50 and T is 1, we get the embedding for the 51st position.
        cos = self.cos[:, :, offset : offset + T, :]
        sin = self.sin[:, :, offset : offset + T, :]

        return (x * cos) + (self.rotate_half(x) * sin)

# mask mods and score mods for flex_attn, causal and SWA

def causal_mask_mod(score, b, h, q, kv):
    return torch.where(q >= kv, score, -float('inf'))

def swa_mask_mod(score, b, h, q, kv):
    window_size = 128 # Matching your screenshot
    return torch.where((q >= kv) & (q - kv <= window_size), score, -float('inf'))

def causal_mask_fn(b, h, q, kv):
    return q >= kv

def swa_mask_fn(b, h, q, kv):
    return (q >= kv) & (q - kv <= 128)

class MQA(nn.Module):
    def __init__(self, layer_n):
        super().__init__()

        assert config['n_embd'] % config['n_head'] == 0

        self.n_embd = config['n_embd']
        self.n_head = config['n_head']
        self.block_size = config['ctx_len']
        self.head_dim = self.n_embd // self.n_head
        self.window_size = 128

        self.c_attn = nn.Linear(config['n_embd'], config['n_embd'] + 2 * self.head_dim, bias=config.get('bias', False))
        self.c_proj = nn.Linear(config['n_embd'], config['n_embd'], bias=config.get('bias', False))
        self.attn_dropout = nn.Dropout(config['dropout'])
        self.resid_dropout = nn.Dropout(config['dropout'])

        self.is_global = (layer_n % config["swa_ratio"] == 0)
        self.cache_len = self.block_size if self.is_global else self.window_size

        mask_fn = causal_mask_fn if self.is_global else swa_mask_fn
        self.score_mod = causal_mask_mod if self.is_global else swa_mask_mod

        # Pre-compute block mask
        self.full_block_mask = create_block_mask(mask_fn, 1, 1, self.block_size, self.block_size, device="cuda")

        self.flex = torch.compile(flex_attn) # flex_attn needs a different compile for some reason

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.rope = MRoPE(self.head_dim)

        self.register_buffer("k_cache", torch.zeros(1, 1, self.cache_len, self.head_dim))
        self.register_buffer("v_cache", torch.zeros(1, 1, self.cache_len, self.head_dim))

    def forward(self, x, pos_ids, start_pos=0, use_cache=False):
        B, T, C = x.size()

        # Projections
        q, k, v = self.c_attn(x).split([self.n_embd, self.head_dim, self.head_dim], dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, 1, self.head_dim).transpose(1, 2)
        v = v.view(B, T, 1, self.head_dim).transpose(1, 2)

        # Norms and RoPE
        q, k = self.q_norm(q), self.k_norm(k)
        q = self.rope(q, pos_ids)
        k = self.rope(k, pos_ids)

        if use_cache:
            write_pos = start_pos % self.cache_len
            self.k_cache[:B, :, write_pos : write_pos + T, :] = k
            self.v_cache[:B, :, write_pos : write_pos + T, :] = v

            if start_pos + T > self.cache_len:
                # SWA Rolling: Unroll so indices are chronological
                full_k = torch.roll(self.k_cache[:B], shifts=-(write_pos + T), dims=2)
                full_v = torch.roll(self.v_cache[:B], shifts=-(write_pos + T), dims=2)
                kv_len = self.cache_len
            else:
                kv_len = start_pos + T
                full_k = self.k_cache[:B, :, :kv_len, :]
                full_v = self.v_cache[:B, :, :kv_len, :]
        else:
            full_k, full_v, kv_len = k, v, T

        full_k, full_v = full_k.to(q.dtype), full_v.to(q.dtype)

        # If sequence fits in one block, use dense score_mod path
        # If it's larger, use the block_mask sparsity path
        if T < 128:
            y = self.flex(q, full_k, full_v, score_mod=self.score_mod, enable_gqa=True)
        else:
            # Note: start_pos and T must be passed carefully to avoid re-compiles
            current_mask = self.full_block_mask._adjust(T, kv_len)
            y = self.flex(q, full_k, full_v, block_mask=current_mask, enable_gqa=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

# Swiglu replaces MLP

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        n = int((8/3) * config['n_embd'])
        appr = (n + 63) & ~(63) # make it a multiple of 64

        # combine gate and value
        self.gate_value_proj = nn.Linear(config['n_embd'], 2 * appr, bias=False) # Llama uses no bias
        self.linear_out = nn.Linear(appr, config['n_embd'], bias=False)
        self.silu = nn.SiLU()

        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x):

        # project input to 2 * appr, split the tensor in half, gate and val
        gate_value = self.gate_value_proj(x)
        gate, value = torch.chunk(gate_value, 2, dim=-1)

        x = self.silu(gate) * value
        x = self.linear_out(x)
        x = self.dropout(x)

        return x

class Merge2x2(nn.Module): # 2x2 patch merge from 32x32 to 64x64
    def __init__(self, d_in, d_out):
        super().__init__()

        self.merger = nn.Sequential(
            nn.Linear(4 * d_in, d_out),
            nn.GELU(),
            nn.Linear(d_out, d_out),
        )

    def forward(self, x):

        B, H, W, D = x.shape

        x = x.view(B, H//2, 2, W//2, 2, D)
        x = x.permute(0, 1, 3, 2, 4, 5) # shape B, H/2, W/2, 2, 2, D
        x = x.view(B, H//2, W//2, 4*D) # 4x4 concat

        return self.merger(x)

class Block(nn.Module):

    def __init__(self, layer_n):
        super().__init__()
        self.rm_1 = RMSNorm(config['n_embd'])
        self.attn = MQA(layer_n)
        self.rm_2 = RMSNorm(config['n_embd'])
        self.mlp = MLP()

        self.branch_scale = 1.0 / math.sqrt(config['n_layer']) # MuP residual rule a/root(L), a = 1.0 here

    def _c_forward(self, x, pos_ids, start_pos=0, use_cache=False): # this is the actual functionality that gets wrapped into checkpoint grad?
        x = x + self.branch_scale * self.attn(self.rm_1(x), pos_ids, start_pos=start_pos, use_cache=use_cache)
        x = x + self.branch_scale * self.mlp(self.rm_2(x))
        return x

    def forward(self, x, pos_ids, start_pos=0, use_cache=False):
        if self.training:
            return checkpoint(self._c_forward, x, pos_ids, start_pos, use_cache, use_reentrant=False)  # need for torch.compile

        return self._c_forward(x, pos_ids, start_pos, use_cache)

class Transformer(nn.Module):

    def __init__(self):
        super().__init__()

        self.block_size = config['ctx_len']

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config['vocab_size'], config['n_embd']), # trying to match the liquid vocab param numbers with qwen size tokenizer
            drop = nn.Dropout(config['dropout']),
            h = nn.ModuleList([Block(layer_n=i) for i in range(config["n_layer"])])
        ))

        # weight-tying was removed for performance purposes (literally useless at this scale from the looks of it)

        self.lm_head = nn.Linear(config['n_embd'], config['vocab_size'], bias=False)
        self.h_scale = 1/config['n_embd'] # muP

        sig_cfg = SiglipConfig()
        self.vit = ViT.from_pretrained(sig_cfg)

        self.patch_merge = Merge2x2(768, 768)

        self.deepstack_proj = nn.ModuleList([
            nn.Linear(768, config["n_embd"], bias=False)
            for _ in SIGLIP_LAYERS
        ])

        for p in self.vit.parameters():
            p.requires_grad = False

        # init all weights
        self.apply(self._init_weights)

        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config['n_layer']))

        print(f"Model initialized with {sum(p.numel() for p in self.parameters())/1e6:.2f}M parameters")


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # MuP init: std = 1 / sqrt(fan_in)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode_image(self, images):
        # images: (B, 3, H, W)
        feats = self.vit(images, return_hidden_states=True).hidden_states

        merged = []
        for l, proj in zip(SIGLIP_LAYERS, self.deepstack_proj):
            f = feats[l]                          # (B, HW, 768)
            B, N, D = f.shape
            H = W = int(math.sqrt(N))

            f = f.view(B, H, W, D)
            f = self.patch_merge(f)              # (B, H/2, W/2, 768)
            f = proj(f)                          # → n_embd
            merged.append(f)

        vis = torch.stack(merged).sum(0)         # deepstack sum
        return vis                               # (B, H/2, W/2, n_embd)

    def build_pos_ids(self, idx, H2, W2, start_pos=0):
        T = idx.size(1)
        pos = torch.zeros(T, 3, device=idx.device)

        # text: t=w=h=token index
        text_pos = torch.arange(T, device=idx.device) + start_pos
        pos[:, 0] = text_pos
        pos[:, 1] = text_pos
        pos[:, 2] = text_pos

        # image tokens
        img_mask = (idx[0] == config["image_token_id"])
        if img_mask.any():
            img_idx = torch.where(img_mask)[0]
            for k, i in enumerate(img_idx):
                pos[i, 0] = 0
                pos[i, 1] = k // W2   # row
                pos[i, 2] = k %  W2   # col

        return pos

    def forward(self, idx, targets=None, vision_features=None, start_pos=0, use_cache=False):

        device = idx.device
        b, t = idx.size()

        vis_mask = (idx == config['image_token_id'])

        # clone idx to replace image tokens with 0 so Embedding doesn't crash
        idx_tmp = idx.clone()
        idx_tmp[vis_mask] = 0
        tok_emb = self.transformer.wte(idx_tmp) # (bs, t, n_embd)

        H2 = W2 = 0
        if vis_mask.any():
            if vision_features is None:
                # text-only part but the data has image tokens err
                raise ValueError("Found image tokens in input but no vision_features provided.")

            vis = self.encode_image(vision_features)   # (B, H2, W2, C)
            B, H2, W2, C = vis.shape
            vis = vis.view(B, H2 * W2, C)

            projected_vis = vis.to(tok_emb.dtype)
            tok_emb[vis_mask] = projected_vis.view(-1, C)

        pos_ids = self.build_pos_ids(idx, H2, W2, start_pos)
        x = self.transformer.drop(tok_emb)

        for block in self.transformer.h:
            x = block(x, pos_ids, start_pos=start_pos, use_cache=use_cache)

        if targets is not None:
            # Full sequence forward (Training/prefll)
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        logits = logits * self.h_scale            # MuP
        logits = 30.0 * torch.tanh(logits / 30.0) # gemma cap

        return logits, loss


    def reset_cache(self):
        for block in self.transformer.h:
            block.attn.k_cache.zero_()
            block.attn.v_cache.zero_()

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, vision_features=None, temperature=1.0, top_k=None):
        b, t = idx.size()

        # PREFILL PHASE, Process the entire prompt at once to populate the KV cache
        logits, _ = self(idx, vision_features=vision_features, start_pos=0, use_cache=True)
        logits = logits[:, -1, :] # We only care about the last logit for the first prediction

        # Simple sampling logic
        logits = (logits * self.h_scale).tanh() * 30.0 / temperature # gemma softcap
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

        # DECODING PHASE only pass the NEWEST token (idx_next) and its position
        start_pos = t

        for _ in range(max_new_tokens - 1):
            # only pass the single last token [:, -1:]
            logits, _ = self(idx[:, -1:], start_pos=start_pos, use_cache=True)

            logits = logits[:, -1, :]
            logits = (logits * self.h_scale).tanh() * 30.0 / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)
            start_pos += 1

        return idx

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):

        # MuP, kept constant for Muon, didnt show much diff, maybe useful if you are extremely narrow
        # Also used as 1/(layer_mult) in AdamW, but the rule is only for RMSnorm in my code, so its basically useless
        # could try later, mostly not worth it

        depth_scale = 1.0 # EDIT later, micro model is 16 deep, l_mult = depth_big/depth_small

        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        muon_groups = []
        adam_groups = []

        for name, p in param_dict.items():
            if p.dim() < 2:

                # Norms/Biases
                is_global = any(k in name for k in ["ln_f", "embed", "head"])
                p_lr = learning_rate if is_global else (learning_rate * depth_scale)

                adam_groups.append({
                    'params': [p],
                    'lr': p_lr,
                    'weight_decay': 0.0
                })

            elif any(k in name for k in ["embed", "token", "wte", "head", "output"]):

                #  MuP rule constant LR
                adam_groups.append({
                    'params': [p],
                    'lr': learning_rate,
                    'weight_decay': weight_decay
                })

            else:

                # MuP and Muon Spectral Scaling
                dout, din = p.shape[0], p.shape[1]
                spectral_scaling = math.sqrt(max(1, dout / din))

                # Combine Muon base LR (5*LR) * spectral * depth
                p_lr = (5 * learning_rate) * spectral_scaling * depth_scale

                muon_groups.append({
                    'params': [p],
                    'lr': p_lr,
                    'weight_decay': weight_decay
                })

        # Prepare extra args (fused AdamW, etc)
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type.startswith('cuda')
        extra_args = dict(fused=True) if use_fused else dict()

        optimizer_adam = torch.optim.AdamW(adam_groups, betas=betas, **extra_args)
        optimizer_muon = torch.optim.Muon(muon_groups)

        return [optimizer_muon, optimizer_adam]

    def estimate_mfu(self, total_tokens_per_iter, dt):
        """
        Estimate Model FLOPs Utilization (MFU) using the 6N approximation.
        total_tokens_per_iter: batch_size * block_size * grad_accum_steps * world_size
        dt: time for one full training iteration (seconds)
        """

        # Simplified: sum(p.numel() for p in self.parameters())
        N = sum(p.numel() for p in self.parameters())

        # 6N comes from: 2N (forward) + 4N (backward)
        # This is the "standard" estimator for Transformer FLOPs
        flops_per_token = 6 * N
        flops_per_iter = flops_per_token * total_tokens_per_iter

        flops_promised = 990e12 # bf16 h100 number, without 2:4 sparsity

        flops_achieved = flops_per_iter / (dt * 4) # 4 gpu DDP
        mfu = flops_achieved / flops_promised
        tflops = flops_achieved / 1e12
        return mfu, tflops
