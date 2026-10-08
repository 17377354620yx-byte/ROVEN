"""Model entry points for the P2I-LReg rigid experiments."""

from geotransformer.modules.liver.registration_model import GeoTransformer


def create_model(cfg):
    expected_modules = {
        "geotransformer": (False, False),
        "rtor": (True, False),
        "rtor_a3": (True, True),
    }
    architecture = cfg.ablation.architecture
    if architecture not in expected_modules:
        raise ValueError(f"unsupported P2I-LReg architecture: {cfg.ablation.architecture}")
    expected_rtor, expected_a3 = expected_modules[architecture]
    if (
        bool(cfg.ablation.rtor_enabled) != expected_rtor
        or bool(cfg.ablation.a3_enabled) != expected_a3
    ):
        raise ValueError("architecture and enabled modules disagree")
    return GeoTransformer(cfg)


def forward_without_ground_truth(model, data_dict):
    """Inference-only forward that prevents pose supervision from reaching the model."""
    if model.training:
        raise ValueError("ground-truth-free forward is only valid in eval mode")
    inference_dict = dict(data_dict)
    inference_dict.pop("transform", None)
    return model(inference_dict)


__all__ = ["GeoTransformer", "create_model", "forward_without_ground_truth"]
