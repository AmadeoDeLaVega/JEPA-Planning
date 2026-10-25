"""Load a frozen, released DINO-WM from the vendored JEPA-WMs source."""

from __future__ import annotations

import importlib

import torch
import yaml

from jepa_planning import ROOT, UPSTREAM

# Upstream eval configs that describe each released DINO-WM architecture.
_CONFIGS = UPSTREAM / "configs/evals/simu_env_planning"
CONFIGS = {
    "pusht": _CONFIGS / "pt/dino-wm/pt_L2_cem_sourcedset_H6_nas6_ctxt2_r224_alpha0.1_ep96_decode.yaml",
    "metaworld": _CONFIGS / "mw/dino-wm/reach_L2_cem_sourcexp_H6_nas3_ctxt2_r224_alpha0.1_ep48_decode.yaml",
}


def load_world_model(env: str, device: str = "cpu"):
    """Load checkpoints/dino_wm_<env>.pth.tar as upstream's wrapper (`encode`, `unroll`).

    The decoder heads are left out: they only turn features back into images
    for visualisation.
    """
    from app.plan_common.datasets import get_data_stats
    from app.plan_common.datasets.preprocessor import Preprocessor
    from app.plan_common.datasets.transforms import make_inverse_transforms, make_transforms

    checkpoint = ROOT / "checkpoints" / f"dino_wm_{env}.pth.tar"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    with CONFIGS[env].open() as handle:
        cfg = yaml.safe_load(handle)

    model_cfg = cfg["model_kwargs"]
    data_cfg = model_cfg["data"]
    aug_cfg = model_cfg["data_aug"]
    pretrain_cfg = model_cfg["pretrain_kwargs"]
    pretrain_cfg["heads_cfg"] = {"architectures": {}}

    stats = get_data_stats(env)
    img_size = data_cfg["img_size"]
    transform = make_transforms(
        img_size=img_size,
        normalize=aug_cfg["normalize"],
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
    )
    preprocessor = Preprocessor(
        action_mean=torch.tensor(stats["action_mean"]),
        action_std=torch.tensor(stats["action_std"]),
        state_mean=torch.tensor(stats["state_mean"]),
        state_std=torch.tensor(stats["state_std"]),
        proprio_mean=torch.tensor(stats["proprio_mean"]),
        proprio_std=torch.tensor(stats["proprio_std"]),
        transform=transform,
        inverse_transform=make_inverse_transforms(img_size=img_size, **aug_cfg),
    )
    # What upstream's evals.simu_env_planning.eval.init_module does, without
    # importing the rest of its planning-eval code.
    model = importlib.import_module(model_cfg["module_name"]).init_module(
        folder=str(checkpoint.parent),
        checkpoint=checkpoint.name,
        model_kwargs=pretrain_cfg,
        wrapper_kwargs=model_cfg["wrapper_kwargs"],
        cfgs_data=data_cfg,
        device=torch.device(device),
        action_dim=stats["action_dim"],
        proprio_dim=stats["proprio_dim"],
        preprocessor=preprocessor,
    )
    model.eval()
    return model
