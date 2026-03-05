"""Final Score metric - combines PSNR, SSIM, and LPIPS into a single score."""

import numpy as np
from basicsr.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_final_score(
    img,
    img2,
    crop_border=0,
    input_order="BCHW",
    test_y_channel=False,
    image_range=255.0,
    psnr_value=None,
    ssim_value=None,
    lpips_value=None,
    **kwargs,
):
    """Calculate Final Score from pre-computed metrics.

    Formula:
        Final_Score = PSNR(Y) + 10 * SSIM(Y) - 5 * LPIPS

    This metric is calculated from other metrics (PSNR, SSIM, LPIPS) 
    that must be calculated first in the same validation pass.

    Args:
        img, img2: Unused (required for metric interface)
        crop_border: Unused
        input_order: Unused
        test_y_channel: Unused
        image_range: Unused
        psnr_value: Pre-computed PSNR value
        ssim_value: Pre-computed SSIM value
        lpips_value: Pre-computed LPIPS value
        **kwargs: Additional arguments

    Returns:
        float: Final Score value
    """
    # Return a placeholder - actual calculation happens in sr_model
    return 0.0
