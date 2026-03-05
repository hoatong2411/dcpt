from copy import deepcopy

from basicsr.utils.registry import METRIC_REGISTRY

from .niqe import calculate_niqe
from .psnr_ssim import calculate_psnr, calculate_ssim
from .final_score import calculate_final_score

# Try to import lpips metrics, skip if not available
try:
    from .lpips_metric import calculate_lpips, calculate_lpips_pt
    __all__ = ["calculate_psnr", "calculate_ssim", "calculate_niqe", "calculate_lpips", "calculate_lpips_pt", "calculate_final_score"]
except ImportError:
    __all__ = ["calculate_psnr", "calculate_ssim", "calculate_niqe", "calculate_final_score"]


def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop("type")
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric
