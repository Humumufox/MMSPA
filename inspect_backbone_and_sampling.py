import os
import math
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from models.backbone.Swin_Transformer import swin_backbone
from models.mps import MultiPartsSampling


def load_image(img_path, img_size=384):
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(img_path).convert("RGB")
    return tfm(img).unsqueeze(0)


def token_to_map(x, h, w):
    # x: [B, N, C] -> [B, C, H, W]
    b, n, c = x.shape
    assert n == h * w, f"N={n}, h*w={h*w}"
    return x.transpose(1, 2).reshape(b, c, h, w)


def save_heatmap(arr, save_path, title=None, cmap="jet"):
    plt.figure(figsize=(5, 5))
    plt.imshow(arr, cmap=cmap)
    plt.colorbar()
    if title:
        plt.title(title)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()


def normalize_2d(t):
    t = t.detach().float().cpu()
    t = t - t.min()
    denom = t.max().clamp(min=1e-8)
    return (t / denom).numpy()


def save_feature_maps(feats, out_dir):
    for i, f in enumerate(feats, 1):
        b, n, c = f.shape
        hw = int(math.sqrt(n))
        feat_map = token_to_map(f, hw, hw)  # [B, C, H, W]
        torch.save(feat_map.cpu(), os.path.join(out_dir, f"F{i}.pt"))

        mean_map = feat_map.mean(dim=1)[0]
        save_heatmap(normalize_2d(mean_map), os.path.join(out_dir, f"F{i}_mean.png"), title=f"F{i} mean map")

        # save a few channels
        for ch in [0, min(1, c - 1), min(5, c - 1), min(10, c - 1)]:
            ch_map = feat_map[0, ch]
            save_heatmap(normalize_2d(ch_map), os.path.join(out_dir, f"F{i}_ch{ch}.png"), title=f"F{i} channel {ch}")


def save_sample_maps(sample_maps, out_dir):
    # sample_maps can be Tensor [B, P, N] or list/tuple of tensors
    if isinstance(sample_maps, (list, tuple)):
        for stage_idx, sm in enumerate(sample_maps, 1):
            _save_single_sample_map(sm, os.path.join(out_dir, f"sampling_stage_{stage_idx}"), f"stage_{stage_idx}")
    else:
        _save_single_sample_map(sample_maps, os.path.join(out_dir, "sampling"), "sampling")


def _save_single_sample_map(sm, stage_dir, stage_name):
    os.makedirs(stage_dir, exist_ok=True)
    sm = sm.detach().cpu()  # [B, P, N]
    b, p, n = sm.shape
    hw = int(math.sqrt(n))

    torch.save(sm, os.path.join(stage_dir, f"{stage_name}.pt"))

    # save all parts for first sample if not too many; otherwise only first few
    max_parts = min(p, 8)
    for part_idx in range(max_parts):
        part_map = sm[0, part_idx].reshape(hw, hw)
        save_heatmap(normalize_2d(part_map), os.path.join(stage_dir, f"{stage_name}_part{part_idx}.png"),
                     title=f"{stage_name} part {part_idx}")

    mean_part_map = sm[0].mean(dim=0).reshape(hw, hw)
    save_heatmap(normalize_2d(mean_part_map), os.path.join(stage_dir, f"{stage_name}_mean.png"),
                 title=f"{stage_name} mean sampling map")


def inspect_all(img_path, out_dir="visualize_feature_and_sampling", img_size=384, device="cuda"):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Save original image
    raw_img = Image.open(img_path).convert("RGB").resize((img_size, img_size))
    raw_img.save(os.path.join(out_dir, "input_image.png"))

    x = load_image(img_path, img_size=img_size).to(device)

    # Backbone only for feature inspection
    backbone = swin_backbone(window_size=img_size // 32, img_size=img_size, num_classes=200, cross_layer=True).to(device)
    backbone.eval()

    with torch.no_grad():
        feats = backbone.forward_features(x)

    print("Backbone outputs:")
    for i, f in enumerate(feats, 1):
        print(f"F{i}.shape = {tuple(f.shape)}")
        print(f"F{i}.min = {f.min().item():.6f}, max = {f.max().item():.6f}, mean = {f.mean().item():.6f}, std = {f.std().item():.6f}")

    save_feature_maps(feats, out_dir)

    # Full MPSA model for sampling maps
    # feature_weights_pooling=False keeps the model simpler for visualization
    mps = MultiPartsSampling(
        dim=1024,
        input_size=img_size,
        backbone=backbone,
        parts_drop=0.2,
        parts_base=0.0,
        cross_layer=True,
        feature_weights_pooling=False,
    ).to(device)
    mps.eval()

    # collect sampling maps from each PartSampling module
    collected = []
    hooks = []

    def make_hook():
        def hook(module, inp, out):
            # PartSampling can return either x or (x, sample_map) depending on your code version
            if isinstance(out, tuple) and len(out) >= 2:
                collected.append(out[1])
            elif hasattr(module, "last_sample_map"):
                collected.append(module.last_sample_map)
        return hook

    # Prefer direct instrumentation if the model returns sample_map
    # If your current code returns only x, temporarily modify PartSampling.forward to `return x, sample_map`
    # for visualization, or use the hook below after adding `module.last_sample_map = sample_map`.
    for mod in mps.modules():
        if mod.__class__.__name__ == "PartSampling":
            hooks.append(mod.register_forward_hook(make_hook()))

    with torch.no_grad():
        _ = mps(x)

    for h in hooks:
        h.remove()

    if collected:
        print("Collected sampling maps:")
        for i, sm in enumerate(collected, 1):
            print(f"S{i}.shape = {tuple(sm.shape)}")
            print(f"S{i}.min = {sm.min().item():.6f}, max = {sm.max().item():.6f}, mean = {sm.mean().item():.6f}, std = {sm.std().item():.6f}")
        save_sample_maps(collected, out_dir)
    else:
        print("No sampling maps collected.")
        print("If your current PartSampling.forward only returns x, change it temporarily to return (x, sample_map) for visualization.")

    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    # Change to your HAM10000 image path
    img_path = r"sample_img\isic_2018\AKIEC_ISIC_0030844.jpg"
    inspect_all(img_path)
