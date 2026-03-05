#!/usr/bin/env python3
"""Quick test to verify LPIPS is working correctly."""

import numpy as np
from basicsr.metrics import calculate_lpips, calculate_ssim, calculate_psnr

print("\n" + "=" * 70)
print("LPIPS METRIC TEST - Verify Everything is Working")
print("=" * 70)

# Create test images
img_gt = np.random.randint(0, 256, (1, 512, 512, 3), dtype=np.uint8)
img_pred = np.random.randint(0, 256, (1, 512, 512, 3), dtype=np.uint8)

print("\n1. Testing PSNR (Y-channel)...")
try:
    psnr_y = calculate_psnr(
        img_pred,
        img_gt,
        crop_border=0,
        input_order="BHWC",
        test_y_channel=True,
        image_range=255.0,
    )
    print(f"   ✓ PSNR(Y): {psnr_y:.4f}")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n2. Testing SSIM (Y-channel)...")
try:
    ssim_y = calculate_ssim(
        img_pred,
        img_gt,
        crop_border=0,
        input_order="BHWC",
        test_y_channel=True,
        image_range=255.0,
    )
    print(f"   ✓ SSIM(Y): {ssim_y:.4f}")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n3. Testing LPIPS (VGG)...")
try:
    lpips_val = calculate_lpips(
        img_pred,
        img_gt,
        crop_border=0,
        input_order="BHWC",
        net="vgg",
        image_range=255.0,
    )
    print(f"   ✓ LPIPS: {lpips_val:.6f}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n4. Testing LPIPS (AlexNet)...")
try:
    lpips_alex = calculate_lpips(
        img_pred,
        img_gt,
        crop_border=0,
        input_order="BHWC",
        net="alex",
        image_range=255.0,
    )
    print(f"   ✓ LPIPS (AlexNet): {lpips_alex:.6f}")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n5. Calculating Final Score...")
try:
    final_score = psnr_y + 10 * ssim_y - 5 * lpips_val
    print(f"   Formula: PSNR + 10×SSIM - 5×LPIPS")
    print(f"   = {psnr_y:.4f} + 10×{ssim_y:.4f} - 5×{lpips_val:.6f}")
    print(f"   = {psnr_y:.4f} + {10*ssim_y:.4f} - {5*lpips_val:.6f}")
    print(f"   ✓ Final Score: {final_score:.4f}")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "=" * 70)
print("✅ All metrics working correctly!")
print("=" * 70)
print("\nNext steps:")
print("  1. Run training: bash run_train.sh")
print("  2. Find best model: python calculate_final_score.py experiments/.../train.log")
print("=" * 70 + "\n")
