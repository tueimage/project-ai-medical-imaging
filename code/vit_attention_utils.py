# vit_attention_utils.py

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ============================================================
# Helper functions
# ============================================================
def ptc_heat(attn_bhNN, head="mean", out_hw=(224, 224)):
    """
    Convert attention [B,H,N,N] into a patch→CLS heatmap [B,H,W].
    attn_bhNN : torch.Tensor
        Attention tensor of shape [B, H, N, N]
    head : str or int
        "mean" → average over all heads
        int    → select a specific head
    """
    # 1) Select average or an individual head
    if head == "mean":
        attn = attn_bhNN.mean(dim=1)    # [B, N, N]
    else:
        attn = attn_bhNN[:, head]       # [B, N, N]

    # 2) Extract patch → CLS
    # CLS index = 0, patches = 1..N-1
    ptc = attn[:, 1:, 0]                # [B, P]

    # 3) Reshape patches to grid
    B, P = ptc.shape
    grid = int(P ** 0.5)
    ptc = ptc.view(B, 1, grid, grid)

    # 4) Upsample to image resolution
    heat = F.interpolate(
        ptc,
        size=out_hw,
        mode="bilinear",
        align_corners=False
    )

    return heat[:, 0]                   # [B, H, W]

def denorm(x_chw, mean, std):
    """
    Undo normalization for visualization.
    """
    mean = torch.tensor(mean, device=x_chw.device).view(3, 1, 1)
    std = torch.tensor(std, device=x_chw.device).view(3, 1, 1)
    return (x_chw * std + mean).clamp(0, 1).permute(1, 2, 0)


# ============================================================
# timm-attn capture (medical CLIP)
# implementation taken from: https://github.com/huggingface/pytorch-image-models/discussions/1232
# ============================================================
def patch_timm_attention(attn_obj):
    def fwd(x, **kwargs):
        B, N, C = x.shape
        qkv = attn_obj.qkv(x).reshape(B, N, 3, attn_obj.num_heads, C // attn_obj.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * attn_obj.scale
        attn = attn.softmax(dim=-1)
        attn = attn_obj.attn_drop(attn)
        attn_obj.attn_map = attn
        attn_obj.cls_attn_map = attn[:, :, 0, 2:]

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = attn_obj.proj(x)
        x = attn_obj.proj_drop(x)
        return x

    attn_obj.forward = fwd


def ensure_mha_outputs_attn(mha):
    """
    Force MultiheadAttention to return per-head attention weights.
    """
    orig = mha.forward

    def forward(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False
        return orig(*args, **kwargs)

    mha.forward = forward


def hook_mha(mha):
    """
    Forward hook to store attention as mha.attn_map [B,H,N,N]
    """
    def hook_fn(module, inputs, output):
        if not (isinstance(output, tuple) and len(output) == 2):
            return
        _, w = output
        if w is None:
            return

        if w.dim() == 4:
            module.attn_map = w.detach()
        elif w.dim() == 3:
            H = module.num_heads
            BH, N, _ = w.shape
            B = BH // H
            module.attn_map = w.view(B, H, N, N).detach()

    return mha.register_forward_hook(hook_fn)


# ============================================================
# Main class
# ============================================================
class ViTAttentionViewer:
    """
    Compare patch→CLS attention between a medical ViT (timm wrapper)
    and a general CLIP ViT (MultiheadAttention block).
    """

    def __init__(
        self,
        model_medical,
        model_general,
        preprocess_fn,
        device,
        mean,
        std,
        image_size=224,
    ):
        self.model_medical = model_medical.eval()
        self.model_general = model_general.eval()
        self.preprocess = preprocess_fn
        self.device = device
        self.mean = mean
        self.std = std
        self.image_size = image_size

        # Patch medical ViT
        for blk in self.model_medical.visual.trunk.blocks:
            patch_timm_attention(blk.attn)

        # Patch + hook general ViT
        self._gen_hooks = []
        for blk in self.model_general.visual.transformer.resblocks:
            mha = blk.attn
            ensure_mha_outputs_attn(mha)
            self._gen_hooks.append(hook_mha(mha))

    # -------------------------
    # Extraction
    # -------------------------
    def run_medical(self, pil_img, layer, head="mean"):
        x = self.preprocess(pil_img).to(self.device)
        x = x.to(dtype=next(self.model_medical.parameters()).dtype)

        with torch.no_grad():
            _ = self.model_medical(x.unsqueeze(0))

        A = self.model_medical.visual.trunk.blocks[layer].attn.attn_map
        heat = ptc_heat(A.float(), head=head, out_hw=(self.image_size, self.image_size))[0]
        heat = heat.detach().cpu().numpy()
        img = denorm(x.float().cpu(), self.mean, self.std).cpu().numpy()
        return img, heat

    def run_general(self, pil_img, layer, head="mean"):
        x = self.preprocess(pil_img).to(self.device)
        x = x.to(dtype=next(self.model_general.parameters()).dtype)

        with torch.no_grad():
            _ = self.model_general.encode_image(x.unsqueeze(0))

        A = self.model_general.visual.transformer.resblocks[layer].attn.attn_map
        heat = ptc_heat(A.float(), head=head, out_hw=(self.image_size, self.image_size))[0]
        heat = heat.detach().cpu().numpy()
        img = denorm(x.float().cpu(), self.mean, self.std).numpy()
        return img, heat

    # -------------------------
    # Visualization
    # -------------------------
    def show_side_by_side(self, ds, idxs, layer, head="mean", alpha=0.5, cmap="jet"):
        n = len(idxs)
        fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
        if n == 1:
            axes = np.expand_dims(axes, 0)

        for r, idx in enumerate(idxs):
            pil = ds[idx]["image"]

            axes[r, 0].imshow(pil)
            axes[r, 0].axis("off")
            axes[r, 0].set_title(f"original | idx {idx}")

            im_m, h_m = self.run_medical(pil, layer, head)
            axes[r, 1].imshow(im_m)
            axes[r, 1].imshow(h_m, alpha=alpha, cmap=cmap)
            axes[r, 1].axis("off")
            axes[r, 1].set_title(f"medical | head={head}, layer={layer}")

            im_g, h_g = self.run_general(pil, layer, head)
            axes[r, 2].imshow(im_g)
            axes[r, 2].imshow(h_g, alpha=alpha, cmap=cmap)
            axes[r, 2].axis("off")
            axes[r, 2].set_title(f"general | head={head}, layer={layer}")

        plt.tight_layout()
        plt.show()