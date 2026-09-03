"""
nanoGPT model.py
教材用内部可視化機能追加版

通常学習:
    model.enable_visualization(False)

内部可視化:
    model.enable_visualization(True)
    logits, loss = model(idx)
    vd = model.visual_data

ckpt.ptとのパラメータ互換性を維持
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# ============================================================
# LayerNorm
# ============================================================
class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(
            input,
            self.weight.shape,
            self.weight,
            self.bias,
            1e-5
        )

# ============================================================
# Causal Self Attention
# ============================================================
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        # Q, K, Vをまとめて生成
        self.c_attn = nn.Linear(
            config.n_embd,
            3 * config.n_embd,
            bias=config.bias
        )

        # Attention出力projection
        self.c_proj = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=config.bias
        )

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # 通常学習時にはFlash Attentionを使用可能
        self.flash = hasattr(
            torch.nn.functional,
            "scaled_dot_product_attention"
        )

        if not self.flash:
            print(
                "WARNING: using slow attention. "
                "Flash Attention requires PyTorch >= 2.0"
            )

        # 可視化では必ず明示的なmaskが必要なので常に保持
        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones(
                    config.block_size,
                    config.block_size
                )
            ).view(
                1,
                1,
                config.block_size,
                config.block_size
            ),
            persistent=False
        )

        # 可視化フラグ
        self.visualization = False

        # 可視化データ
        self.visual_data = {}

    # --------------------------------------------------------
    # visualization ON/OFF
    # --------------------------------------------------------
    def enable_visualization(self, enabled=True):
        self.visualization = enabled

        if not enabled:
            self.visual_data = {}

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------
    def forward(self, x):

        B, T, C = x.size()

        # ====================================================
        # Q K V
        # ====================================================
        q_raw, k_raw, v_raw = self.c_attn(x).split(
            self.n_embd,
            dim=2
        )

        k = k_raw.view(
            B,
            T,
            self.n_head,
            C // self.n_head
        ).transpose(1, 2)

        q = q_raw.view(
            B,
            T,
            self.n_head,
            C // self.n_head
        ).transpose(1, 2)

        v = v_raw.view(
            B,
            T,
            self.n_head,
            C // self.n_head
        ).transpose(1, 2)

        # ====================================================
        # 可視化モード
        # ====================================================
        if self.visualization:

            # ------------------------------------------------
            # WQ WK WV
            #
            # PyTorch Linear:
            # y = x @ W.T + b
            #
            # c_attn.weight shape:
            # (3*n_embd, n_embd)
            # ------------------------------------------------
            W = self.c_attn.weight

            WQ = W[0:C, :]
            WK = W[C:2*C, :]
            WV = W[2*C:3*C, :]

            # ------------------------------------------------
            # Attention Score
            # ------------------------------------------------
            attention_score = (
                q @ k.transpose(-2, -1)
            ) * (
                1.0 / math.sqrt(k.size(-1))
            )

            # mask前のscoreを保存
            raw_score = attention_score.clone()

            # ------------------------------------------------
            # Causal Mask
            # ------------------------------------------------
            masked_score = attention_score.masked_fill(
                self.bias[:, :, :T, :T] == 0,
                float("-inf")
            )

            # ------------------------------------------------
            # Softmax
            # ------------------------------------------------
            att = F.softmax(
                masked_score,
                dim=-1
            )

            att_after_dropout = self.attn_dropout(att)

            # ------------------------------------------------
            # Context Vector
            # ------------------------------------------------
            context = att_after_dropout @ v

            # headを戻す
            context_merged = (
                context
                .transpose(1, 2)
                .contiguous()
                .view(B, T, C)
            )

            # ------------------------------------------------
            # WO
            # ------------------------------------------------
            y = self.c_proj(context_merged)
            y = self.resid_dropout(y)

            # ------------------------------------------------
            # 保存
            # ------------------------------------------------
            self.visual_data = {
                "WQ": WQ.detach().cpu(),
                "WK": WK.detach().cpu(),
                "WV": WV.detach().cpu(),

                "q": q.detach().cpu(),
                "k": k.detach().cpu(),
                "v": v.detach().cpu(),

                "attention_score":
                    raw_score.detach().cpu(),

                "masked_score":
                    masked_score.detach().cpu(),

                "attention":
                    att.detach().cpu(),

                "context":
                    context.detach().cpu(),

                "context_merged":
                    context_merged.detach().cpu(),

                "attn_output":
                    y.detach().cpu(),

                "WO":
                    self.c_proj.weight.detach().cpu()
            }

            return y

        # ====================================================
        # 通常モード
        # ====================================================
        if self.flash:

            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )

        else:

            att = (
                q @ k.transpose(-2, -1)
            ) * (
                1.0 / math.sqrt(k.size(-1))
            )

            att = att.masked_fill(
                self.bias[:, :, :T, :T] == 0,
                float("-inf")
            )

            att = F.softmax(
                att,
                dim=-1
            )

            att = self.attn_dropout(att)

            y = att @ v

        y = (
            y
            .transpose(1, 2)
            .contiguous()
            .view(B, T, C)
        )

        y = self.resid_dropout(
            self.c_proj(y)
        )

        return y

# ============================================================
# MLP
# ============================================================
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.c_fc = nn.Linear(
            config.n_embd,
            4 * config.n_embd,
            bias=config.bias
        )

        self.gelu = nn.GELU()

        self.c_proj = nn.Linear(
            4 * config.n_embd,
            config.n_embd,
            bias=config.bias
        )

        self.dropout = nn.Dropout(
            config.dropout
        )

        self.visualization = False
        self.visual_data = {}

    # --------------------------------------------------------
    # visualization ON/OFF
    # --------------------------------------------------------
    def enable_visualization(self, enabled=True):

        self.visualization = enabled

        if not enabled:
            self.visual_data = {}

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------
    def forward(self, x):

        # Linear
        fc = self.c_fc(x)

        # GELU
        gelu = self.gelu(fc)

        # Projection
        proj = self.c_proj(gelu)

        # Dropout
        out = self.dropout(proj)

        if self.visualization:

            self.visual_data = {
                "ffn_fc":
                    fc.detach().cpu(),

                "ffn_gelu":
                    gelu.detach().cpu(),

                "ffn_proj":
                    out.detach().cpu()
            }

        return out

# ============================================================
# Transformer Block
# ============================================================
class Block(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.ln_1 = LayerNorm(
            config.n_embd,
            bias=config.bias
        )

        self.attn = CausalSelfAttention(
            config
        )

        self.ln_2 = LayerNorm(
            config.n_embd,
            bias=config.bias
        )

        self.mlp = MLP(
            config
        )

        self.visualization = False
        self.visual_data = {}

    # --------------------------------------------------------
    # visualization ON/OFF
    # --------------------------------------------------------
    def enable_visualization(self, enabled=True):

        self.visualization = enabled

        self.attn.enable_visualization(
            enabled
        )

        self.mlp.enable_visualization(
            enabled
        )

        if not enabled:
            self.visual_data = {}

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------
    def forward(self, x):

        # ====================================================
        # LN1
        # ====================================================
        ln1 = self.ln_1(x)

        # ====================================================
        # Attention
        # ====================================================
        attn_output = self.attn(
            ln1
        )

        # ====================================================
        # Residual 1
        # ====================================================
        residual1 = x + attn_output

        # ====================================================
        # LN2
        # ====================================================
        ln2 = self.ln_2(
            residual1
        )

        # ====================================================
        # MLP
        # ====================================================
        mlp_output = self.mlp(
            ln2
        )

        # ====================================================
        # Residual 2
        # ====================================================
        residual2 = (
            residual1
            + mlp_output
        )

        # ====================================================
        # 保存
        # ====================================================
        if self.visualization:

            self.visual_data = {
                "ln1":
                    ln1.detach().cpu(),

                "residual1":
                    residual1.detach().cpu(),

                "ln2":
                    ln2.detach().cpu(),

                "residual2":
                    residual2.detach().cpu()
            }

            self.visual_data.update(
                self.attn.visual_data
            )

            self.visual_data.update(
                self.mlp.visual_data
            )

        return residual2

# ============================================================
# GPT Config
# ============================================================
@dataclass
class GPTConfig:

    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

# ============================================================
# GPT
# ============================================================
class GPT(nn.Module):

    def __init__(self, config):

        super().__init__()

        assert config.vocab_size is not None
        assert config.block_size is not None

        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(
                    config.vocab_size,
                    config.n_embd
                ),

                wpe=nn.Embedding(
                    config.block_size,
                    config.n_embd
                ),

                drop=nn.Dropout(
                    config.dropout
                ),

                h=nn.ModuleList(
                    [
                        Block(config)
                        for _ in range(config.n_layer)
                    ]
                ),

                ln_f=LayerNorm(
                    config.n_embd,
                    bias=config.bias
                )
            )
        )

        self.lm_head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False
        )

        # weight tying
        self.transformer.wte.weight = (
            self.lm_head.weight
        )

        # ====================================================
        # Visualization
        # ====================================================
        self.visualization = False
        self.visual_data = {}

        # ====================================================
        # Initialize weights
        # ====================================================
        self.apply(
            self._init_weights
        )

        for pn, p in self.named_parameters():

            if pn.endswith(
                "c_proj.weight"
            ):

                torch.nn.init.normal_(
                    p,
                    mean=0.0,
                    std=0.02 /
                    math.sqrt(
                        2 * config.n_layer
                    )
                )

        print(
            "number of parameters: %.2fM"
            % (
                self.get_num_params()
                / 1e6,
            )
        )

    # ========================================================
    # Enable visualization
    # ========================================================
    def enable_visualization(
        self,
        enabled=True
    ):

        self.visualization = enabled

        self.visual_data = {}

        for block in self.transformer.h:

            block.enable_visualization(
                enabled
            )

    # ========================================================
    # Number of parameters
    # ========================================================
    def get_num_params(
        self,
        non_embedding=True
    ):

        n_params = sum(
            p.numel()
            for p in self.parameters()
        )

        if non_embedding:

            n_params -= (
                self.transformer
                .wpe
                .weight
                .numel()
            )

        return n_params

    # ========================================================
    # Initialize weights
    # ========================================================
    def _init_weights(
        self,
        module
    ):

        if isinstance(
            module,
            nn.Linear
        ):

            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:

                torch.nn.init.zeros_(
                    module.bias
                )

        elif isinstance(
            module,
            nn.Embedding
        ):

            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    # ========================================================
    # Forward
    # ========================================================
    def forward(
        self,
        idx,
        targets=None
    ):

        device = idx.device

        b, t = idx.size()

        assert (
            t <= self.config.block_size
        ), (
            f"Cannot forward sequence "
            f"of length {t}, "
            f"block size is only "
            f"{self.config.block_size}"
        )

        pos = torch.arange(
            0,
            t,
            dtype=torch.long,
            device=device
        )

        # ====================================================
        # Token Embedding
        # ====================================================
        tok_emb = (
            self.transformer
            .wte(idx)
        )

        # ====================================================
        # Position Embedding
        # ====================================================
        pos_emb = (
            self.transformer
            .wpe(pos)
        )

        # ====================================================
        # Token + Position
        # ====================================================
        embedding = (
            tok_emb
            + pos_emb
        )

        x = self.transformer.drop(
            embedding
        )

        # ====================================================
        # Transformer Blocks
        # ====================================================
        for block in self.transformer.h:

            x = block(x)

        # ====================================================
        # Final LayerNorm
        # ====================================================
        final_embedding = (
            self.transformer
            .ln_f(x)
        )

        x = final_embedding

        # ====================================================
        # Logits / Loss
        # ====================================================
        if targets is not None:

            logits = self.lm_head(
                x
            )

            loss = F.cross_entropy(
                logits.view(
                    -1,
                    logits.size(-1)
                ),
                targets.view(-1),
                ignore_index=-1
            )

        else:

            # 通常nanoGPTと同じ
            logits = self.lm_head(
                x[:, [-1], :]
            )

            loss = None

        # ====================================================
        # Visualization data
        # ====================================================
        if self.visualization:

            self.visual_data = {
                "token_embedding":
                    tok_emb.detach().cpu(),

                "position_embedding":
                    pos_emb.detach().cpu(),

                "embedding":
                    embedding.detach().cpu(),

                "final_embedding":
                    final_embedding.detach().cpu(),

                "logits":
                    logits.detach().cpu(),

                "probability":
                    F.softmax(
                        logits,
                        dim=-1
                    ).detach().cpu()
            }

            # ------------------------------------------------
            # 1層モデル教材用
            #
            # 複数層の場合は各blockを
            # block_0, block_1 ...
            # としても保存
            # ------------------------------------------------
            for i, block in enumerate(
                self.transformer.h
            ):

                self.visual_data[
                    f"block_{i}"
                ] = block.visual_data

            # ------------------------------------------------
            # 現在のvisualize_forward.pyとの互換性
            #
            # 1層目のデータをトップレベルにもコピー
            # ------------------------------------------------
            if len(
                self.transformer.h
            ) > 0:

                self.visual_data.update(
                    self.transformer
                    .h[0]
                    .visual_data
                )

        return logits, loss

    # ========================================================
    # Crop block size
    # ========================================================
    def crop_block_size(
        self,
        block_size
    ):

        assert (
            block_size
            <= self.config.block_size
        )

        self.config.block_size = (
            block_size
        )

        self.transformer.wpe.weight = (
            nn.Parameter(
                self.transformer
                .wpe
                .weight[
                    :block_size
                ]
            )
        )

        for block in self.transformer.h:

            if hasattr(
                block.attn,
                "bias"
            ):

                block.attn.bias = (
                    block.attn.bias[
                        :,
                        :,
                        :block_size,
                        :block_size
                    ]
                )

    # ========================================================
    # From pretrained GPT-2
    # ========================================================
    @classmethod
    def from_pretrained(
        cls,
        model_type,
        override_args=None
    ):

        assert model_type in {
            "gpt2",
            "gpt2-medium",
            "gpt2-large",
            "gpt2-xl"
        }

        override_args = (
            override_args or {}
        )

        assert all(
            k == "dropout"
            for k in override_args
        )

        from transformers import (
            GPT2LMHeadModel
        )

        print(
            "loading weights from pretrained gpt: %s"
            % model_type
        )

        config_args = {

            "gpt2":
                dict(
                    n_layer=12,
                    n_head=12,
                    n_embd=768
                ),

            "gpt2-medium":
                dict(
                    n_layer=24,
                    n_head=16,
                    n_embd=1024
                ),

            "gpt2-large":
                dict(
                    n_layer=36,
                    n_head=20,
                    n_embd=1280
                ),

            "gpt2-xl":
                dict(
                    n_layer=48,
                    n_head=25,
                    n_embd=1600
                )

        }[model_type]

        print(
            "forcing vocab_size=50257, "
            "block_size=1024, bias=True"
        )

        config_args[
            "vocab_size"
        ] = 50257

        config_args[
            "block_size"
        ] = 1024

        config_args[
            "bias"
        ] = True

        if "dropout" in override_args:

            config_args[
                "dropout"
            ] = override_args[
                "dropout"
            ]

        config = GPTConfig(
            **config_args
        )

        model = GPT(
            config
        )

        sd = model.state_dict()

        sd_keys = sd.keys()

        sd_keys = [
            k
            for k in sd_keys
            if not k.endswith(
                ".attn.bias"
            )
        ]

        model_hf = (
            GPT2LMHeadModel
            .from_pretrained(
                model_type
            )
        )

        sd_hf = (
            model_hf.state_dict()
        )

        sd_keys_hf = (
            sd_hf.keys()
        )

        sd_keys_hf = [
            k
            for k in sd_keys_hf
            if not k.endswith(
                ".attn.masked_bias"
            )
        ]

        sd_keys_hf = [
            k
            for k in sd_keys_hf
            if not k.endswith(
                ".attn.bias"
            )
        ]

        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight"
        ]

        assert (
            len(sd_keys_hf)
            == len(sd_keys)
        )

        for k in sd_keys_hf:

            if any(
                k.endswith(w)
                for w in transposed
            ):

                assert (
                    sd_hf[k]
                    .shape[::-1]
                    == sd[k].shape
                )

                with torch.no_grad():

                    sd[k].copy_(
                        sd_hf[k].t()
                    )

            else:

                assert (
                    sd_hf[k].shape
                    == sd[k].shape
                )

                with torch.no_grad():

                    sd[k].copy_(
                        sd_hf[k]
                    )

        return model

    # ========================================================
    # Configure optimizer
    # ========================================================
    def configure_optimizers(
        self,
        weight_decay,
        learning_rate,
        betas,
        device_type
    ):

        param_dict = {
            pn: p
            for pn, p
            in self.named_parameters()
        }

        param_dict = {
            pn: p
            for pn, p
            in param_dict.items()
            if p.requires_grad
        }

        decay_params = [
            p
            for n, p
            in param_dict.items()
            if p.dim() >= 2
        ]

        nodecay_params = [
            p
            for n, p
            in param_dict.items()
            if p.dim() < 2
        ]

        optim_groups = [

            {
                "params":
                    decay_params,

                "weight_decay":
                    weight_decay
            },

            {
                "params":
                    nodecay_params,

                "weight_decay":
                    0.0
            }
        ]

        num_decay_params = sum(
            p.numel()
            for p in decay_params
        )

        num_nodecay_params = sum(
            p.numel()
            for p in nodecay_params
        )

        print(
            f"num decayed parameter tensors: "
            f"{len(decay_params)}, "
            f"with {num_decay_params:,} parameters"
        )

        print(
            f"num non-decayed parameter tensors: "
            f"{len(nodecay_params)}, "
            f"with {num_nodecay_params:,} parameters"
        )

        fused_available = (
            "fused"
            in inspect.signature(
                torch.optim.AdamW
            ).parameters
        )

        use_fused = (
            fused_available
            and device_type == "cuda"
        )

        extra_args = (
            dict(fused=True)
            if use_fused
            else dict()
        )

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            **extra_args
        )

        print(
            f"using fused AdamW: "
            f"{use_fused}"
        )

        return optimizer

    # ========================================================
    # Estimate MFU
    # ========================================================
    def estimate_mfu(
        self,
        fwdbwd_per_iter,
        dt
    ):

        N = self.get_num_params()

        cfg = self.config

        L = cfg.n_layer
        H = cfg.n_head
        Q = cfg.n_embd // cfg.n_head
        T = cfg.block_size

        flops_per_token = (
            6 * N
            + 12 * L * H * Q * T
        )

        flops_per_fwdbwd = (
            flops_per_token * T
        )

        flops_per_iter = (
            flops_per_fwdbwd
            * fwdbwd_per_iter
        )

        flops_achieved = (
            flops_per_iter
            * (1.0 / dt)
        )

        flops_promised = 312e12

        mfu = (
            flops_achieved
            / flops_promised
        )

        return mfu

    # ========================================================
    # Generate
    # ========================================================
    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=None
    ):

        for _ in range(
            max_new_tokens
        ):

            idx_cond = (
                idx
                if idx.size(1)
                <= self.config.block_size
                else idx[
                    :,
                    -self.config.block_size:
                ]
            )

            logits, _ = self(
                idx_cond
            )

            logits = (
                logits[:, -1, :]
                / temperature
            )

            if top_k is not None:

                v, _ = torch.topk(
                    logits,
                    min(
                        top_k,
                        logits.size(-1)
                    )
                )

                logits[
                    logits
                    < v[:, [-1]]
                ] = -float("Inf")

            probs = F.softmax(
                logits,
                dim=-1
            )

            idx_next = (
                torch.multinomial(
                    probs,
                    num_samples=1
                )
            )

            idx = torch.cat(
                (
                    idx,
                    idx_next
                ),
                dim=1
            )

        return idx