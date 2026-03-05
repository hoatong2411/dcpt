import os
import random
import shutil
from pathlib import Path

SOURCE_ROOT = "./data"       
TRAIN_ROOT  = "./data/dataset_train" # output train
VAL_ROOT    = "./data/dataset_val"   # output val

DEGRADATION_TYPES = ["Blur", "Haze", "Lowlight", "Rain", "Snow"]

VAL_RATIO   = 0.1      
SEED        = 42
COPY        = False    # Set False để di chuyển (move) thay vì copy - tiết kiệm dung lượng
CLEAN_EXIST = False    # Set True để xóa output folder cũ
# ============================================================

def get_image_files(folder):
    """Lấy danh sách file ảnh trong folder, đã sort."""
    if not os.path.exists(folder):
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted([
        f for f in os.listdir(folder)
        if Path(f).suffix.lower() in exts
    ])

def transfer(src, dst, use_copy=True):
    """Copy hoặc move file từ src đến dst."""
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            return  # Skip nếu file đã tồn tại (resume support)
        if use_copy:
            shutil.copy2(src, dst)
        else:
            shutil.move(src, dst)
    except OSError as e:
        print(f"[ERROR] Không thể transfer {src}: {e}")
        raise

def clean_output_folders():
    """Xóa các folder output nếu đã tồn tại."""
    for folder in [TRAIN_ROOT, VAL_ROOT]:
        if os.path.exists(folder):
            print(f"[CLEAN] Xóa folder: {folder}")
            shutil.rmtree(folder)

def split_dataset():
    """Split dataset thành train và val folders."""
    # Kiểm tra source root
    if not os.path.isdir(SOURCE_ROOT):
        print(f"[ERROR] Source root không tồn tại: {SOURCE_ROOT}")
        return
    
    # Làm sạch folder output nếu cần
    if CLEAN_EXIST:
        clean_output_folders()
    
    print("="*60)
    print("BẮTĐẦU SPLIT DATASET")
    print("="*60)
    print(f"Source:  {SOURCE_ROOT}")
    print(f"Train:   {TRAIN_ROOT}")
    print(f"Val:     {VAL_ROOT}")
    print(f"Val Ratio: {VAL_RATIO*100}%")
    print(f"Seed:    {SEED}")
    print(f"Copy:    {COPY}")
    print("="*60)
    
    random.seed(SEED)
    total_train = total_val = 0
    processed_degs = []

    for deg in DEGRADATION_TYPES:
        lq_src = os.path.join(SOURCE_ROOT, deg, "LQ")
        gt_src = os.path.join(SOURCE_ROOT, deg, "GT")

        if not os.path.isdir(lq_src) or not os.path.isdir(gt_src):
            print(f"[SKIP] {deg}: LQ hoặc GT folder không tồn tại.")
            continue

        # Lấy danh sách file từ LQ (dùng làm reference)
        files = get_image_files(lq_src)

        # Kiểm tra GT có đủ file tương ứng không
        gt_files = set(get_image_files(gt_src))
        files = [f for f in files if f in gt_files]

        if len(files) == 0:
            print(f"[SKIP] {deg}: Không tìm thấy cặp LQ/GT hợp lệ.")
            continue

        # Shuffle và split
        random.shuffle(files)
        n_val   = max(1, int(len(files) * VAL_RATIO))
        n_train = len(files) - n_val

        val_files   = files[:n_val]
        train_files = files[n_val:]

        print(f"\n[{deg}]  Total: {len(files):4d}  =>  Train: {n_train:4d}  |  Val: {n_val:3d}")

        # Move/copy từng split
        for split_name, split_files, split_root in [
            ("train", train_files, TRAIN_ROOT),
            ("val",   val_files,   VAL_ROOT),
        ]:
            for fname in split_files:
                for sub in ["LQ", "GT"]:
                    src_folder = lq_src if sub == "LQ" else gt_src
                    src = os.path.join(src_folder, fname)
                    dst = os.path.join(split_root, deg, sub, fname)
                    try:
                        transfer(src, dst, use_copy=COPY)
                    except OSError as e:
                        print(f"[WARNING] Lỗi {split_name}/{deg}/{sub}/{fname}")
                        if "No space left" in str(e):
                            print("[FATAL] Hết dung lượng ổ đĩa! Chạy lại script sau khi dọn dẹp.")
                            raise
                        continue

        processed_degs.append(deg)
        total_train += n_train
        total_val   += n_val

    print(f"\n{'='*60}")
    print(f"HOÀN THÀNH!")
    print(f"{'='*60}")
    print(f"Đã xử lý: {', '.join(processed_degs)}")
    print(f"Tổng train: {total_train}")
    print(f"Tổng val:   {total_val}")
    print(f"Train => {TRAIN_ROOT}")
    print(f"Val   => {VAL_ROOT}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    split_dataset()