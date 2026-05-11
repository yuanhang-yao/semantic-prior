import torch
import torch.nn as nn
from transformers import AutoModel


class FrozenDINOBackbone(nn.Module):
    def __init__(self, model_name: str, device: str = "cuda"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.device = torch.device(device)
        self.feat_dim = getattr(self.model.config, "hidden_size")

    def _tokens_to_map(self, hs: torch.Tensor, pixel_values: torch.Tensor, num_reg: int):
        hs = hs[:, 1 + num_reg:, :]
        batch_size, num_tokens, channels = hs.shape

        patch_size = self.model.config.patch_size
        img_h, img_w = pixel_values.shape[-2:]
        h_patches = img_h // patch_size
        w_patches = img_w // patch_size

        return hs.transpose(1, 2).contiguous().view(batch_size, channels, h_patches, w_patches)

    @torch.no_grad()
    def forward(self, dino_inputs, return_last_k: int = 1):
        pixel_values = dino_inputs["pixel_values"].to(self.device)
        outputs = self.model(pixel_values=pixel_values, output_hidden_states=True)
        hidden_states = list(outputs.hidden_states) if outputs.hidden_states is not None else []
        if not hidden_states:
            hidden_states = [outputs.last_hidden_state]
        else:
            last_hidden = getattr(outputs, "last_hidden_state", None)
            if isinstance(last_hidden, torch.Tensor) and last_hidden.shape == hidden_states[-1].shape:
                hidden_states[-1] = last_hidden
        num_reg = getattr(self.model.config, "num_register_tokens", 0)

        k = int(return_last_k) if return_last_k is not None else 1
        if k <= 1:
            return self._tokens_to_map(hidden_states[-1], pixel_values, num_reg)
        k = max(1, min(k, len(hidden_states)))
        return [self._tokens_to_map(hidden_states[-k_idx], pixel_values, num_reg) for k_idx in range(k, 0, -1)]


class Modulator(nn.Module):
    def __init__(
        self,
        dino_backbone: FrozenDINOBackbone,
        stage_channels,
        hidden_dim: int = 256,
        residual_gamma: bool = True,
        dino_last_k: int = 13,
        attn_temperature: float = 1.0,
    ):
        super().__init__()
        self.dino = dino_backbone
        self.stage_channels = list(stage_channels)
        self.residual_gamma = residual_gamma
        self.dino_last_k = int(dino_last_k)
        self.attn_temperature = float(attn_temperature)

        self.layer_fuse_logits = nn.Parameter(torch.zeros(self.dino_last_k)) if self.dino_last_k > 1 else None
        self.attn_score = nn.Conv2d(self.dino.feat_dim, 1, kernel_size=1, bias=True)

        total_channels = sum(2 * c for c in self.stage_channels)
        self.mlp = nn.Sequential(
            nn.Linear(self.dino.feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, total_channels),
        )

        total_gamma_channels = sum(self.stage_channels)
        self.gamma_gate = nn.Parameter(torch.ones(total_gamma_channels))

    def forward(self, img_dino):
        dino_out = self.dino({"pixel_values": img_dino}, return_last_k=self.dino_last_k)

        if isinstance(dino_out, (list, tuple)):
            feats = list(dino_out)
            logits = self.layer_fuse_logits[: len(feats)]
            weights = torch.softmax(logits, dim=0)
            dino_feat = 0.0
            for i, feat in enumerate(feats):
                dino_feat = dino_feat + weights[i] * feat
        else:
            dino_feat = dino_out

        score = self.attn_score(dino_feat)
        attn = torch.softmax(score.flatten(2) / self.attn_temperature, dim=-1)
        z = (dino_feat.flatten(2) * attn).sum(dim=-1)
        mod_vec = self.mlp(z)

        gammas = []
        betas = []
        offset = 0
        gate_offset = 0
        batch_size = mod_vec.size(0)

        for channels in self.stage_channels:
            gamma = mod_vec[:, offset : offset + channels]
            offset += channels
            beta = mod_vec[:, offset : offset + channels]
            offset += channels

            gate_slice = self.gamma_gate[gate_offset : gate_offset + channels]
            gate_offset += channels
            gamma = gamma * gate_slice.view(1, channels)

            gamma = gamma.view(batch_size, channels, 1, 1)
            beta = beta.view(batch_size, channels, 1, 1)

            if self.residual_gamma:
                gamma = 1.0 + torch.tanh(gamma)

            gammas.append(gamma)
            betas.append(beta)

        return gammas, betas


class DINOCNNTeacher(nn.Module):
    def __init__(
        self,
        dino_backbone: FrozenDINOBackbone,
        base_model: nn.Module,
        stage_channels,
        hidden_dim: int = 256,
        residual_gamma: bool = True,
    ):
        super().__init__()
        self.base = base_model
        self.modulator = Modulator(
            dino_backbone=dino_backbone,
            stage_channels=stage_channels,
            hidden_dim=hidden_dim,
            residual_gamma=residual_gamma,
        )

    def forward(self, img_cnn, img_dino):
        feats = self.base.forward_backbone(img_cnn)

        modulation_scale = self.base.modulation_scale
        if modulation_scale <= 1e-6:
            return self.base.forward_head(feats)

        gammas, betas = self.modulator(img_dino)

        if isinstance(feats, dict):
            mod_feats = {}
            feat_values = list(feats.values())
            feat_keys = list(feats.keys())
            for i, (gamma, beta) in enumerate(zip(gammas, betas)):
                feat = feat_values[i]
                key = feat_keys[i]
                mod_feats[key] = (1.0 + modulation_scale * (gamma - 1.0)) * feat + modulation_scale * beta
            for i in range(len(gammas), len(feat_keys)):
                mod_feats[feat_keys[i]] = feat_values[i]
        else:
            mod_feats = []
            for feat, gamma, beta in zip(feats, gammas, betas):
                mod_feats.append((1.0 + modulation_scale * (gamma - 1.0)) * feat + modulation_scale * beta)

        return self.base.forward_head(mod_feats)
