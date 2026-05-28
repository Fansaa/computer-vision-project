"""
Pipeline Persiapan Data Aksara Jawa - 120 Kelas
================================================
Implementasi 4 langkah:
  Langkah 1 - Stratified Split (70:15:15)
  Langkah 2 - Integrasi sub-folder variations ke training set
  Langkah 3 - Targeted augmentasi aman (fotometrik + morfologis) pada kelas minoritas
  Langkah 4 - Perhitungan class weight untuk weighted loss function

Cara pakai:
  python jawa_pipeline.py

Output:
  split/train/, split/val/, split/test/   <- hasil split + integrasi
  augmented_train/                         <- training set setelah augmentasi
  class_weights.json                       <- bobot kelas untuk loss function
"""

import os
import json
import shutil
import random
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.utils.class_weight import compute_class_weight
import albumentations as A

# ============================================================
# KONFIGURASI - sesuaikan path dengan environment Anda
# ============================================================
JAWA_ROOT      = Path("Jawa")
ALL_CLASS_DIR  = JAWA_ROOT / "all_class"
VARIATIONS_DIR = JAWA_ROOT / "variations"

OUTPUT_ROOT    = Path("output_pipeline")
SPLIT_DIR      = OUTPUT_ROOT / "split"
AUG_DIR        = OUTPUT_ROOT / "augmented_train"
WEIGHTS_FILE   = OUTPUT_ROOT / "class_weights.json"

TRAIN_RATIO    = 0.70
VAL_RATIO      = 0.15
# TEST_RATIO   = 0.15 (sisa otomatis)

# Target minimum citra per kelas di training set SEBELUM augmentasi diterapkan
# Kelas minoritas (~21 citra asli) akan diaugmentasi hingga mencapai angka ini
AUGMENT_TARGET = 300

IMG_EXT        = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SEED           = 42

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# UTILITAS
# ============================================================

def collect_images(folder: Path) -> list[Path]:
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT])


def copy_images(src_list: list[Path], dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(src_list):
        # Tambahkan prefix nomor urut agar tidak ada nama file yang bentrok
        dst = dst_dir / f"{i:05d}_{src.name}"
        shutil.copy2(src, dst)


def class_label(vowel_folder_name: str, consonant_folder_name: str) -> str:
    """
    Membuat label kelas yang konsisten: 'VokalA_ba', 'VokalE_be', dst.
    """
    return f"{vowel_folder_name}_{consonant_folder_name}"


# ============================================================
# LANGKAH 1 - STRATIFIED SPLIT (70:15:15)
# ============================================================

def step1_stratified_split():
    print("\n" + "="*60)
    print("LANGKAH 1: Stratified Split 70:15:15")
    print("="*60)

    # Kumpulkan semua gambar per kelas dari all_class
    class_images: dict[str, list[Path]] = {}

    for vowel_dir in sorted(ALL_CLASS_DIR.iterdir()):
        if not vowel_dir.is_dir():
            continue
        for cons_dir in sorted(vowel_dir.iterdir()):
            if not cons_dir.is_dir():
                continue
            label = class_label(vowel_dir.name, cons_dir.name)
            imgs  = collect_images(cons_dir)
            if imgs:
                class_images[label] = imgs

    print(f"Total kelas ditemukan : {len(class_images)}")
    print(f"Total citra           : {sum(len(v) for v in class_images.values()):,}")

    # Split per kelas agar distribusi terjaga (stratified)
    split_registry: dict[str, dict[str, list[Path]]] = {
        "train": {}, "val": {}, "test": {}
    }

    for label, imgs in class_images.items():
        shuffled = imgs.copy()
        random.shuffle(shuffled)

        n       = len(shuffled)
        n_train = max(1, int(n * TRAIN_RATIO))
        n_val   = max(1, int(n * VAL_RATIO))
        # Pastikan minimal 1 citra di test
        n_val   = min(n_val, n - n_train - 1)

        split_registry["train"][label] = shuffled[:n_train]
        split_registry["val"][label]   = shuffled[n_train : n_train + n_val]
        split_registry["test"][label]  = shuffled[n_train + n_val :]

    # Tulis ke disk
    for subset in ["train", "val", "test"]:
        total = 0
        for label, imgs in split_registry[subset].items():
            dst = SPLIT_DIR / subset / label
            copy_images(imgs, dst)
            total += len(imgs)
        print(f"  {subset:5s}: {total:,} citra di {len(split_registry[subset])} kelas")

    print("Langkah 1 selesai.")
    return split_registry


# ============================================================
# LANGKAH 2 - INTEGRASI VARIATIONS KE TRAINING SET
# ============================================================

def step2_integrate_variations(split_registry: dict):
    print("\n" + "="*60)
    print("LANGKAH 2: Integrasi sub-folder variations ke training set")
    print("="*60)

    # Pemetaan nama folder variations ke label kelas Vokal A di all_class
    # Folder variations: 'ba', 'ga', 'ha', 'ka', 'la', 'nya', 'ra', 'ta'
    # Semua berkorespondensi dengan Vokal A karena nama mengandung vokal 'a'
    VOKAL_A_DIR_NAME = None
    for d in ALL_CLASS_DIR.iterdir():
        if "vokal" in d.name.lower() and d.name.lower().endswith("a") and "è" not in d.name.lower():
            VOKAL_A_DIR_NAME = d.name
            break

    if VOKAL_A_DIR_NAME is None:
        # Fallback: cari folder yang konsonannya ber-suffix 'a'
        for d in ALL_CLASS_DIR.iterdir():
            cons_dirs = list(d.iterdir())
            if cons_dirs and cons_dirs[0].name.endswith("a"):
                VOKAL_A_DIR_NAME = d.name
                break

    added_total = 0
    missing     = []

    for var_cons_dir in sorted(VARIATIONS_DIR.iterdir()):
        if not var_cons_dir.is_dir():
            continue

        cons_name = var_cons_dir.name          # e.g. 'ba', 'ha', 'nya'
        # Kelas Vokal A yang berkorespondensi
        target_label = class_label(VOKAL_A_DIR_NAME, cons_name) if VOKAL_A_DIR_NAME else None

        if target_label is None or target_label not in split_registry["train"]:
            # Coba cari berdasarkan suffix konsonan yang cocok
            matched = [lbl for lbl in split_registry["train"]
                       if lbl.split("_", 1)[-1] == cons_name]
            if matched:
                target_label = matched[0]
            else:
                missing.append(cons_name)
                continue

        var_imgs = collect_images(var_cons_dir)
        dst      = SPLIT_DIR / "train" / target_label

        # Salin dengan prefix 'var_' agar bisa dibedakan jika diperlukan
        dst.mkdir(parents=True, exist_ok=True)
        for i, src in enumerate(var_imgs):
            shutil.copy2(src, dst / f"var_{i:04d}_{src.name}")

        added_total += len(var_imgs)
        print(f"  {cons_name:6s} -> {target_label}: +{len(var_imgs)} citra variasi")

    print(f"\nTotal citra variations yang ditambahkan: {added_total}")
    if missing:
        print(f"Konsonan tidak ditemukan padanannya   : {missing}")
    print("Langkah 2 selesai.")


# ============================================================
# LANGKAH 3 - TARGETED AUGMENTASI AMAN (FOTOMETRIK + MORFOLOGIS)
# ============================================================

# Pipeline augmentasi yang tidak mengubah karakteristik aksara
SAFE_AUG = A.Compose([
    A.RandomBrightnessContrast(
        brightness_limit=0.30,
        contrast_limit=0.30,
        p=0.6
    ),
    A.GaussianBlur(blur_limit=(1, 3), p=0.3),
    A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
    A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2), p=0.3),
    A.OneOf([
        A.Morphological(scale=(1, 2), operation="dilation", p=1.0),
        A.Morphological(scale=(1, 2), operation="erosion",  p=1.0),
    ], p=0.4),
    A.RandomCrop(height=200, width=200, p=0.3),
    A.PadIfNeeded(
        min_height=224, min_width=224,
        border_mode=cv2.BORDER_CONSTANT,
        fill=255,
        p=1.0
    ),
    A.Resize(height=224, width=224),
])


def augment_image(img_path: Path) -> np.ndarray:
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result  = SAFE_AUG(image=img_rgb)
    return cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)


def step3_targeted_augmentation():
    print("\n" + "="*60)
    print(f"LANGKAH 3: Augmentasi targeted (target >= {AUGMENT_TARGET} per kelas)")
    print("="*60)

    train_dir = SPLIT_DIR / "train"
    AUG_DIR.mkdir(parents=True, exist_ok=True)

    summary = {"augmented": 0, "skipped": 0, "classes_augmented": 0}

    for class_dir in sorted(train_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        imgs = collect_images(class_dir)
        n    = len(imgs)

        dst_class_dir = AUG_DIR / class_dir.name
        dst_class_dir.mkdir(parents=True, exist_ok=True)

        # Salin semua gambar asli ke augmented_train tanpa modifikasi
        for img_path in imgs:
            shutil.copy2(img_path, dst_class_dir / img_path.name)

        if n >= AUGMENT_TARGET:
            # Kelas sudah cukup, tidak perlu augmentasi
            summary["skipped"] += 1
            continue

        # Hitung berapa gambar sintetis yang dibutuhkan
        needed = AUGMENT_TARGET - n
        summary["classes_augmented"] += 1

        aug_count = 0
        # Iterasi berulang atas gambar asli hingga target terpenuhi
        cycle_iter = 0
        while aug_count < needed:
            for img_path in imgs:
                if aug_count >= needed:
                    break
                aug_img = augment_image(img_path)
                if aug_img is None:
                    continue
                save_name = f"aug_{cycle_iter:02d}_{aug_count:04d}_{img_path.stem}.jpg"
                cv2.imwrite(str(dst_class_dir / save_name), aug_img)
                aug_count += 1
            cycle_iter += 1

        summary["augmented"] += aug_count

    final_counts = [
        len(collect_images(d))
        for d in AUG_DIR.iterdir() if d.is_dir()
    ]
    print(f"  Kelas yang diaugmentasi   : {summary['classes_augmented']}")
    print(f"  Kelas yang dilewati       : {summary['skipped']} (sudah >= {AUGMENT_TARGET})")
    print(f"  Total citra sintetis dibuat: {summary['augmented']:,}")
    print(f"  Total citra training akhir : {sum(final_counts):,}")
    if final_counts:
        print(f"  Min / Max per kelas       : {min(final_counts)} / {max(final_counts)}")
        print(f"  Imbalance ratio baru      : {max(final_counts)/min(final_counts):.2f}x")
    print("Langkah 3 selesai.")


# ============================================================
# LANGKAH 4 - CLASS WEIGHT UNTUK WEIGHTED LOSS FUNCTION
# ============================================================

def step4_compute_class_weights():
    print("\n" + "="*60)
    print("LANGKAH 4: Perhitungan class weight untuk weighted loss")
    print("="*60)

    aug_train_dir = AUG_DIR

    labels_all = []
    class_names = sorted([d.name for d in aug_train_dir.iterdir() if d.is_dir()])
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    for class_dir in aug_train_dir.iterdir():
        if not class_dir.is_dir():
            continue
        idx   = class_to_idx[class_dir.name]
        count = len(collect_images(class_dir))
        labels_all.extend([idx] * count)

    labels_arr = np.array(labels_all)
    classes    = np.unique(labels_arr)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels_arr
    )

    weight_dict = {class_names[i]: float(round(w, 6)) for i, w in zip(classes, weights)}

    WEIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(WEIGHTS_FILE, "w") as f:
        json.dump(
            {"class_to_idx": class_to_idx, "class_weights": weight_dict},
            f, indent=2, ensure_ascii=False
        )

    w_values = list(weight_dict.values())
    print(f"  Total kelas  : {len(class_names)}")
    print(f"  Weight min   : {min(w_values):.4f} (kelas mayoritas)")
    print(f"  Weight max   : {max(w_values):.4f} (kelas minoritas)")
    print(f"  Weight ratio : {max(w_values)/min(w_values):.2f}x")
    print(f"  Disimpan di  : {WEIGHTS_FILE}")
    print("Langkah 4 selesai.")

    return weight_dict, class_to_idx


# ============================================================
# CONTOH PENGGUNAAN WEIGHT DI PYTORCH DAN TENSORFLOW/KERAS
# ============================================================

USAGE_SNIPPET = '''
# ----------------------------------------------------------------
# CARA MENGGUNAKAN class_weights.json SAAT TRAINING
# ----------------------------------------------------------------

import json
import torch
import numpy as np

with open("output_pipeline/class_weights.json") as f:
    meta = json.load(f)

class_to_idx = meta["class_to_idx"]
weight_dict  = meta["class_weights"]

# --- PyTorch ---
num_classes    = len(class_to_idx)
weight_tensor  = torch.zeros(num_classes)
for cls_name, idx in class_to_idx.items():
    weight_tensor[idx] = weight_dict[cls_name]

criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor.to(device))

# --- Keras / TensorFlow ---
# compute_class_weight sudah menghasilkan format yang kompatibel
keras_weights = {idx: weight_dict[cls_name]
                 for cls_name, idx in class_to_idx.items()}

model.fit(
    train_dataset,
    class_weight=keras_weights,
    epochs=50
)
'''


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("Pipeline Persiapan Data Aksara Jawa - 120 Kelas")
    print(f"Root dataset : {JAWA_ROOT.resolve()}")
    print(f"Output       : {OUTPUT_ROOT.resolve()}")

    # Bersihkan output lama jika ada
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True)

    # Jalankan 4 langkah secara berurutan
    split_registry = step1_stratified_split()
    step2_integrate_variations(split_registry)
    step3_targeted_augmentation()
    weight_dict, class_to_idx = step4_compute_class_weights()

    print("\n" + "="*60)
    print("PIPELINE SELESAI")
    print("="*60)
    print(f"Direktori split asli  : {SPLIT_DIR}")
    print(f"Training siap training: {AUG_DIR}")
    print(f"Class weights         : {WEIGHTS_FILE}")
    print()
    print("Petunjuk penggunaan class_weights.json:")
    print(USAGE_SNIPPET)
