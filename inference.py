"""
Inference script for DCPT (Degradation Classification Pre-Training)
Loads generator (net_g) and degradation classifier (net_dc) models
with proper routing based on detected degradation type.
"""

import torch
import torch.nn.functional as F
from basicsr.archs.nafnet_arch import NAFNetBaseline
from basicsr.archs.degrad_classify_arch import PromptIR_NoImg_DC
from PIL import Image
import torchvision.transforms as transforms
import os
import time

# ===== Configuration =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model paths
MODEL_PATH_G = "./experiments/NAFNet_AIO_5d_custom/models/net_g_60000.pth"
MODEL_PATH_DC = "./experiments/NAFNet_AIO_5d_custom/models/net_dc_60000.pth"

# Degradation types mapping (must match training order)
DEGRADATION_TYPES = {
    0: "Blur",
    1: "Haze",
    2: "Lowlight",
    3: "Rain",
    4: "Snow"
}

# ===== Load Generator Model =====
print("\n[Loading Generator Model (net_g)]")
net_g = NAFNetBaseline(
    img_channel=3,
    width=64,
    enc_blk_nums=[1, 1, 1, 28],
    middle_blk_num=1,
    dec_blk_nums=[1, 1, 1, 1],
    window_size=8
)

# Load checkpoint
ckpt_g = torch.load(MODEL_PATH_G, map_location=device)
if "params_ema" in ckpt_g:
    net_g.load_state_dict(ckpt_g["params_ema"])
    print(f"✓ Loaded net_g with params_ema")
elif "params" in ckpt_g:
    net_g.load_state_dict(ckpt_g["params"])
    print(f"✓ Loaded net_g with params")
else:
    net_g.load_state_dict(ckpt_g)
    print(f"✓ Loaded net_g directly")

net_g = net_g.to(device)
net_g.eval()

# ===== Load Degradation Classifier Model =====
print("[Loading Degradation Classifier Model (net_dc)]")
net_dc = PromptIR_NoImg_DC(
    feature_dims=[64, 128, 256, 512],
    num_res_blocks=2,
    num_classes=5,
    downsample=False
)

# Load checkpoint
ckpt_dc = torch.load(MODEL_PATH_DC, map_location=device)
if "params_ema" in ckpt_dc:
    net_dc.load_state_dict(ckpt_dc["params_ema"])
    print(f"✓ Loaded net_dc with params_ema")
elif "params" in ckpt_dc:
    net_dc.load_state_dict(ckpt_dc["params"])
    print(f"✓ Loaded net_dc with params")
else:
    net_dc.load_state_dict(ckpt_dc)
    print(f"✓ Loaded net_dc directly")

net_dc = net_dc.to(device)
net_dc.eval()

# ===== Hook for Feature Extraction =====
hook_outputs = []

def hook_forward_fn(module, input, output):
    """Hook function to capture intermediate feature maps"""
    if isinstance(output, tuple):
        output = output[-1]
    hook_outputs.append(output)

# Register hooks to decoder blocks
hooks = []
for name, module in net_g.named_modules():
    if "decoder" in name and name.count(".") == 1:
        hook = module.register_forward_hook(hook_forward_fn)
        hooks.append(hook)
        print(f"  Hook registered: {name}")

# ===== Preprocessing =====
to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

# ===== Inference Settings =====
input_folder = "./data/dataset_test_val/Blur/LQ/"  # Change this to your input folder
output_folder = os.path.join(os.path.dirname(input_folder), "Restored/")
os.makedirs(output_folder, exist_ok=True)

save_kwargs = dict(format="JPEG", quality=96, subsampling=0, optimize=False)

print(f"\n[Inference Configuration]")
print(f"Input folder:  {input_folder}")
print(f"Output folder: {output_folder}")

# ===== Inference =====
print(f"\n[Starting Inference]")
print("=" * 70)

processed_count = 0
degradation_counts = {name: 0 for name in DEGRADATION_TYPES.values()}
total_time = 0

with torch.no_grad():
    for file in sorted(os.listdir(input_folder)):
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            start_time = time.time()
            print(f"\n[{processed_count + 1}] Processing: {file}")
            
            try:
                # Load and prepare image
                img = Image.open(os.path.join(input_folder, file)).convert("RGB")
                img_tensor = to_tensor(img).unsqueeze(0).to(device)
                
                # Get original dimensions
                h, w = img_tensor.shape[2:]
                
                # Padding if needed (divisible by window size 8)
                pad_h = (8 - h % 8) % 8
                pad_w = (8 - w % 8) % 8
                img_tensor_padded = F.pad(
                    img_tensor, (0, pad_w, 0, pad_h), mode='reflect'
                )
                
                # Step 1: Forward pass with hooks to extract features
                hook_outputs = []
                _ = net_g(img_tensor_padded, hook=True)
                
                # Step 2: Classify degradation type using extracted features
                cls_output = net_dc(img_tensor_padded, hook_outputs[::-1])
                routing_weight = F.softmax(cls_output, dim=1)
                
                pred_class = cls_output.argmax(dim=1).item()
                degradation_name = DEGRADATION_TYPES[pred_class]
                confidence = routing_weight[0, pred_class].item()
                
                print(f"  Degradation: {degradation_name} (confidence: {confidence:.2%})")
                
                # Print confidence for all degradations
                print(f"  Class scores: ", end="")
                for i, name in DEGRADATION_TYPES.items():
                    print(f"{name}: {routing_weight[0, i].item():.2%} | ", end="")
                print("\b\b ")
                
                # Step 3: Generate restoration using routing weight
                restored = net_g(img_tensor_padded, routing_weight=routing_weight)
                
                # Remove padding
                restored = restored[:, :, :h, :w]
                
                # Post-process and save
                restored = restored.clamp(0, 1).cpu().squeeze(0)
                restored_img = to_pil(restored)
                
                out_path = os.path.join(output_folder, file)
                restored_img.save(out_path, **save_kwargs)
                
                degradation_counts[degradation_name] += 1
                processed_count += 1
                
                elapsed = time.time() - start_time
                total_time += elapsed
                print(f"  ✓ Saved to: {out_path}")
                print(f"  Time: {elapsed:.2f}s")
                
            except Exception as e:
                print(f"  ✗ Error processing {file}: {str(e)}")
                continue

# ===== Summary =====
print("\n" + "=" * 70)
print(f"[Inference Complete]")
print(f"Total images processed: {processed_count}")
print(f"Total time: {total_time:.2f}s")
if processed_count > 0:
    print(f"Average time per image: {total_time / processed_count:.2f}s")
print(f"\nDegradation type distribution:")
for deg_type, count in degradation_counts.items():
    if count > 0:
        print(f"  {deg_type}: {count} images")

# Cleanup hooks
for hook in hooks:
    hook.remove()

print("\nDone!")
