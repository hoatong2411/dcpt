"""LPIPS metric wrapper for basicsr."""

import numpy as np
import torch
import lpips as lpips_lib

from basicsr.metrics.metric_util import reorder_image
from basicsr.utils.registry import METRIC_REGISTRY


# Global LPIPS model instances
_lpips_models = {}


def get_lpips_model(net="vgg", use_gpu=True, verbose=False):
    """Get or create LPIPS model instance.
    
    Args:
        net (str): Network backbone ('alex', 'vgg', 'squeeze'). Default: 'vgg'.
        use_gpu (bool): Whether to use GPU. Default: True.
        verbose (bool): Print verbose info. Default: False.
    
    Returns:
        lpips.LPIPS: LPIPS model instance.
    """
    global _lpips_models
    
    key = (net, use_gpu)
    
    if key not in _lpips_models:
        device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
        model = lpips_lib.LPIPS(
            pretrained=True,
            net=net,
            version="0.1",
            lpips=True,
            spatial=False,
            pnet_rand=False,
            pnet_tune=False,
            use_dropout=True,
            eval_mode=True,
            verbose=verbose,
        )
        model.to(device)
        model.eval()
        _lpips_models[key] = model
        
        if not verbose:
            print(f"[LPIPS] Loaded {net} model on {device}")
    
    return _lpips_models[key]


@METRIC_REGISTRY.register()
def calculate_lpips(
    img,
    img2,
    crop_border=0,
    input_order="BCHW",
    net="vgg",
    test_y_channel=False,
    image_range=255.0,
    **kwargs,
):
    """Calculate LPIPS (Learned Perceptual Image Patch Similarity).

    Reference:
        The Unreasonable Effectiveness of Deep Features as a Perceptual Metric
        https://arxiv.org/abs/1801.03924

    Args:
        img (ndarray): Images with range [0, 255]. Shape: (B, C, H, W) or (B, H, W, C).
        img2 (ndarray): Images with range [0, 255]. Shape: (B, C, H, W) or (B, H, W, C).
        crop_border (int): Cropped pixels in each edge. Default: 0.
        input_order (str): Order of input ('BHWC' or 'BCHW'). Default: 'BCHW'.
        net (str): Network backbone ('alex', 'vgg', 'squeeze'). Default: 'vgg'.
        test_y_channel (bool): Test on Y channel only (ignored for LPIPS). Default: False.
        image_range (float): Image range (255 or 1). Default: 255.

    Returns:
        float: LPIPS value.
    """
    assert img.shape == img2.shape, f"Image shapes are different: {img.shape}, {img2.shape}."
    if input_order not in ["BHWC", "BCHW"]:
        raise ValueError(f"Wrong input_order {input_order}. Supported: 'BHWC' or 'BCHW'")

    # Reorder images to BCHW
    imgs = reorder_image(img, input_order=input_order)
    imgs2 = reorder_image(img2, input_order=input_order)

    # Normalize to [0, 1]
    if image_range == 255:
        imgs = imgs.astype(np.float32) / 255.0
        imgs2 = imgs2.astype(np.float32) / 255.0
    else:
        imgs = imgs.astype(np.float32)
        imgs2 = imgs2.astype(np.float32)

    # Crop borders if needed
    if crop_border != 0:
        imgs = imgs[:, :, crop_border:-crop_border, crop_border:-crop_border]
        imgs2 = imgs2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    # Convert to torch tensors (reorder_image returns BHWC, need to convert to BCHW)
    imgs_t = torch.from_numpy(imgs).permute(0, 3, 1, 2).float()  # BHWC -> BCHW
    imgs2_t = torch.from_numpy(imgs2).permute(0, 3, 1, 2).float()  # BHWC -> BCHW

    # Move to GPU if available
    if torch.cuda.is_available():
        imgs_t = imgs_t.cuda()
        imgs2_t = imgs2_t.cuda()

    # Get LPIPS model
    lpips_model = get_lpips_model(net=net, use_gpu=torch.cuda.is_available(), verbose=False)

    # Normalize to [-1, 1] as required by LPIPS
    imgs_t = 2 * imgs_t - 1
    imgs2_t = 2 * imgs2_t - 1

    # Calculate LPIPS
    with torch.no_grad():
        lpips_value = lpips_model(imgs_t, imgs2_t, normalize=False)

    return lpips_value.mean().item()


@METRIC_REGISTRY.register()
def calculate_lpips_pt(
    img,
    img2,
    crop_border=0,
    test_y_channel=False,
    net="vgg",
    **kwargs,
):
    """Calculate LPIPS from PyTorch tensors.

    Args:
        img (Tensor): Images with range [0, 1]. Shape: (B, C, H, W).
        img2 (Tensor): Images with range [0, 1]. Shape: (B, C, H, W).
        crop_border (int): Cropped pixels in each edge. Default: 0.
        test_y_channel (bool): Test on Y channel only. Default: False.
        net (str): Network backbone. Default: 'vgg'.

    Returns:
        float: LPIPS value.
    """
    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    # Move to GPU if available
    if torch.cuda.is_available():
        img = img.cuda()
        img2 = img2.cuda()

    # Get LPIPS model
    lpips_model = get_lpips_model(net=net, use_gpu=torch.cuda.is_available(), verbose=False)

    # Normalize to [-1, 1]
    img = 2 * img - 1
    img2 = 2 * img2 - 1

    # Calculate LPIPS
    with torch.no_grad():
        lpips_value = lpips_model(img, img2, normalize=False)

    return lpips_value.mean().item()
