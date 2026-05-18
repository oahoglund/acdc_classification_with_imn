import torch


def volumes_from_mask_torch_batched(
    mask: torch.Tensor,          # (B, N, H, W) int labels 0..K-1
    pixel_spacing: torch.Tensor, # (B, 3) (dx, dy, dz) in mm
    num_classes: int = 4,        # includes background=0
) -> torch.Tensor:
    if mask.dtype != torch.long:
        mask = mask.long()

    device = mask.device
    dtype = torch.float32
    pixel_spacing = pixel_spacing.to(device=device, dtype=dtype)

    dx = pixel_spacing[:, 0]  # (B,)
    dy = pixel_spacing[:, 1]  # (B,)
    dz = pixel_spacing[:, 2]  # (B,)
    pixel_area = dx * dy      # (B,)

    B, N, H, W = mask.shape
    m = mask.reshape(B * N, H * W)  # (B*N, HW)

    counts = torch.zeros((B * N, num_classes), device=device, dtype=dtype)
    ones = torch.ones_like(m, dtype=dtype)
    counts.scatter_add_(dim=1, index=m.clamp_(0, num_classes - 1), src=ones)

    counts = counts.view(B, N, num_classes)          # (B, N, K)
    areas = counts[:, :, 1:4].permute(0, 2, 1)       # (B, 3, N)
    areas = areas * pixel_area[:, None, None]        # (B, 3, N) mm^2

    S0 = areas[:, :, :-1]
    S1 = areas[:, :,  1:]

    vol_mm3 = ((S0 + S1 + torch.sqrt(S0 * S1)) * (dz[:, None, None] / 3.0)).sum(dim=2)  # (B, 3)
    return vol_mm3 / 1000.0  # mL

def clinical_metrics_from_volumes_torch_batched(
    ed_volumes: torch.Tensor,  # (B, 3)
    es_volumes: torch.Tensor,  # (B, 3)
    height: torch.Tensor,      # (B,)  (see note below)
    weight: torch.Tensor,      # (B,)
    myocardial_density: float = 1.05,  # g/mL
    BSA_formula: str = "mosteller",
    eps: float = 1e-8,
) -> torch.Tensor:
    device = ed_volumes.device
    dtype = ed_volumes.dtype

    h = height.to(device=device, dtype=dtype)
    w = weight.to(device=device, dtype=dtype)

    if BSA_formula.lower() == "mosteller":
        BSA = torch.sqrt(h * w / 3600.0)
    elif BSA_formula.lower() == "dubois":
        BSA = torch.tensor(0.007184, device=device, dtype=dtype) * (h ** 0.725) * (w ** 0.425)
    else:
        BSA = torch.sqrt(h * w / 3600.0)

    V_LV = ed_volumes[:, 0]
    V_RV = ed_volumes[:, 1]
    V_MYO = ed_volumes[:, 2]

    myocardial_mass = V_MYO * torch.tensor(myocardial_density, device=device, dtype=dtype)

    LVEDV_indexed = V_LV / (BSA + eps)
    RVEDV_indexed = V_RV / (BSA + eps)
    MYmass_indexed = myocardial_mass / (BSA + eps)

    LVEF = (V_LV - es_volumes[:, 0]) / (V_LV + eps)
    RVEF = (V_RV - es_volumes[:, 1]) / (V_RV + eps)

    # (B, 5)
    return torch.stack([LVEF, RVEF, LVEDV_indexed, RVEDV_indexed, MYmass_indexed], dim=1)

@torch.no_grad()
def compute_feature_stats(dataloader, device="cpu", BSA_formula = "dubois"):
    feats = []

    for batch in dataloader:
        ed_img, ed_gt, es_img, es_gt, info = batch

        height = info["height"].to(device)
        weight = info["weight"].to(device)
        spacing = info["spacing"].to(device)
        ed_gt = ed_gt.to(device)
        es_gt = es_gt.to(device)

        volumes_ed = volumes_from_mask_torch_batched(ed_gt, spacing)
        volumes_es = volumes_from_mask_torch_batched(es_gt, spacing)

        metrics = clinical_metrics_from_volumes_torch_batched(
            volumes_ed, volumes_es, height, weight,
            myocardial_density=1.05, BSA_formula= BSA_formula
        )

        feats.append(metrics.cpu())

    feats = torch.cat(feats, dim=0)
    mean = feats.mean(dim=0)
    std = feats.std(dim=0, unbiased=False)
    return mean, std