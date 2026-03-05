"""
Tạo dataset test từ dataset_train/val - mỗi degradation 200 images
"""
import os
import shutil
from pathlib import Path

SOURCE_TRAIN = "./data/dataset_train"
SOURCE_VAL = "./data/dataset_val"
TEST_TRAIN_ROOT = "./data/dataset_test_train"
TEST_VAL_ROOT = "./data/dataset_test_val"

DEGRADATION_TYPES = ["Blur", "Haze", "Lowlight", "Rain", "Snow"]
NUM_TEST_IMAGES = 200

def get_image_files(folder):
    """Lấy danh sách file ảnh trong folder, đã sort."""
    if not os.path.exists(folder):
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted([
        f for f in os.listdir(folder)
        if Path(f).suffix.lower() in exts
    ])

def create_test_dataset():
    """Tạo dataset test từ train/val."""
    print("="*60)
    print("TẠO TEST DATASET")
    print("="*60)
    
    # Xóa folder cũ nếu tồn tại
    for folder in [TEST_TRAIN_ROOT, TEST_VAL_ROOT]:
        if os.path.exists(folder):
            print(f"[CLEAN] Xóa folder: {folder}")
            shutil.rmtree(folder)
    
    total_test_train = total_test_val = 0
    
    for deg in DEGRADATION_TYPES:
        print(f"\n[{deg}]")
        
        # ===== TRAIN =====
        train_lq_src = os.path.join(SOURCE_TRAIN, deg, "LQ")
        train_gt_src = os.path.join(SOURCE_TRAIN, deg, "GT")
        
        if os.path.isdir(train_lq_src) and os.path.isdir(train_gt_src):
            files = get_image_files(train_lq_src)
            files = files[:NUM_TEST_IMAGES]  # Lấy 200 ảnh đầu
            
            print(f"  Train: copy {len(files)} images")
            for fname in files:
                for sub in ["LQ", "GT"]:
                    src_folder = train_lq_src if sub == "LQ" else train_gt_src
                    src = os.path.join(src_folder, fname)
                    dst = os.path.join(TEST_TRAIN_ROOT, deg, sub, fname)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
            total_test_train += len(files)
        
        # ===== VAL =====
        val_lq_src = os.path.join(SOURCE_VAL, deg, "LQ")
        val_gt_src = os.path.join(SOURCE_VAL, deg, "GT")
        
        if os.path.isdir(val_lq_src) and os.path.isdir(val_gt_src):
            files = get_image_files(val_lq_src)
            files = files[:NUM_TEST_IMAGES]  # Lấy 200 ảnh đầu
            
            print(f"  Val:   copy {len(files)} images")
            for fname in files:
                for sub in ["LQ", "GT"]:
                    src_folder = val_lq_src if sub == "LQ" else val_gt_src
                    src = os.path.join(src_folder, fname)
                    dst = os.path.join(TEST_VAL_ROOT, deg, sub, fname)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
            total_test_val += len(files)
    
    print(f"\n{'='*60}")
    print(f"HOÀN THÀNH!")
    print(f"Test Train: {total_test_train} images => {TEST_TRAIN_ROOT}")
    print(f"Test Val:   {total_test_val} images => {TEST_VAL_ROOT}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    create_test_dataset()
