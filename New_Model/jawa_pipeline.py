"""
jawa_pipeline.py  (revisi New_Model)
========================================================================
Pipeline persiapan data Aksara Jawa 120 kelas -> menghasilkan data
siap-latih yang dibaca notebook train_*.ipynb.

Perbedaan dari versi lama:
  - Path benar ke dataset bersarang: <root>/Jawa/Jawa/{all_class,variations}
  - Output ke <root>/output_pipeline/ (terjangkau ../output_pipeline dari New_Model)
  - Augmentasi PIL yang PERSIS Tabel II paper (1-3 transform acak, non-geometris),
    tanpa ketergantungan Albumentations (menghindari error beda versi).

4 langkah:
  1. Stratified split per-kelas 70:15:15
  2. Integrasi sub-folder variations -> kelas Vokal A
  3. Targeted augmentation (hanya non-A -> 300/kelas), preservasi struktur
  4. Class weights (compute_class_weight 'balanced') untuk weighted loss

Jalankan:  python jawa_pipeline.py
"""
import json
import shutil
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from sklearn.utils.class_weight import compute_class_weight

# ============================================================
# KONFIGURASI (path absolut, tahan terhadap cwd)
# ============================================================
PROJECT_ROOT   = Path(__file__).resolve().parent.parent   # induk dari New_Model
JAWA_ROOT      = PROJECT_ROOT / "Jawa" / "Jawa"            # dataset bersarang
ALL_CLASS_DIR  = JAWA_ROOT / "all_class"
VARIATIONS_DIR = JAWA_ROOT / "variations"

OUTPUT_ROOT  = PROJECT_ROOT / "output_pipeline"
SPLIT_DIR    = OUTPUT_ROOT / "split"
AUG_DIR      = OUTPUT_ROOT / "augmented_train"
WEIGHTS_FILE = OUTPUT_ROOT / "class_weights.json"

TRAIN_RATIO    = 0.70
VAL_RATIO      = 0.15      # test = 0.15 (sisa)
AUGMENT_TARGET = 300       # target citra/kelas minoritas sebelum augmentasi
IMG_EXT        = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SEED           = 42

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# UTILITAS
# ============================================================
def collect_images(folder: Path):
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT])


def copy_images(src_list, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(src_list):
        shutil.copy2(src, dst_dir / f"{i:05d}_{src.name}")


def class_label(vowel_name: str, consonant_name: str) -> str:
    return f"{vowel_name}_{consonant_name}"


# ============================================================
# AUGMENTASI PRESERVASI STRUKTUR (PERSIS TABEL II PAPER)
# 1-3 transform acak; fotometrik + morfologis + crop <=10%.
# DILARANG: rotasi, flip, shear, afin, elastic (geometris).
# ============================================================
TRANSFORMS = ["brightness", "contrast", "blur", "noise",
              "sharpen", "dilation", "erosion", "crop_pad"]


def structure_preserving_augment(img_pil, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed % (2**31))

    img = img_pil.copy()
    chosen = random.sample(TRANSFORMS, random.randint(1, 3))   # 1-3 transform acak

    for t in chosen:
        if t == "brightness":
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.70, 1.30))
        elif t == "contrast":
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.70, 1.30))
        elif t == "blur":
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        elif t == "noise":
            arr = np.array(img, dtype=np.float32)
            sigma = random.uniform(0.01, 0.05) * 255.0
            arr = np.clip(arr + np.random.normal(0, sigma, arr.shape), 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
        elif t == "sharpen":
            img = ImageEnhance.Sharpness(img).enhance(random.uniform(1.2, 2.0))
        elif t == "dilation":
            img = img.filter(ImageFilter.MaxFilter(size=random.choice([3, 5])))
        elif t == "erosion":
            img = img.filter(ImageFilter.MinFilter(size=random.choice([3, 5])))
        elif t == "crop_pad":
            w, h = img.size
            mc = 0.10
            l  = random.randint(0, max(1, int(w * mc)))
            tp = random.randint(0, max(1, int(h * mc)))
            r  = random.randint(0, max(1, int(w * mc)))
            b  = random.randint(0, max(1, int(h * mc)))
            right = max(w - r, l + 1)
            lower = max(h - b, tp + 1)
            img = img.crop((l, tp, right, lower)).resize((w, h), Image.LANCZOS)
    return img


# ============================================================
# LANGKAH 1 - STRATIFIED SPLIT 70:15:15
# ============================================================
def step1_stratified_split():
    print("\n" + "=" * 60)
    print("LANGKAH 1: Stratified Split 70:15:15")
    print("=" * 60)

    class_images = {}
    for vowel_dir in sorted(ALL_CLASS_DIR.iterdir()):
        if not vowel_dir.is_dir():
            continue
        for cons_dir in sorted(vowel_dir.iterdir()):
            if not cons_dir.is_dir():
                continue
            label = class_label(vowel_dir.name, cons_dir.name)
            imgs = collect_images(cons_dir)
            if imgs:
                class_images[label] = imgs

    print(f"Total kelas : {len(class_images)}")
    print(f"Total citra : {sum(len(v) for v in class_images.values()):,}")

    split_registry = {"train": {}, "val": {}, "test": {}}
    for label, imgs in class_images.items():
        shuffled = imgs.copy()
        random.shuffle(shuffled)
        n = len(shuffled)
        n_train = max(1, int(n * TRAIN_RATIO))
        n_val = max(1, int(n * VAL_RATIO))
        n_val = min(n_val, n - n_train - 1)        # pastikan minimal 1 di test
        split_registry["train"][label] = shuffled[:n_train]
        split_registry["val"][label] = shuffled[n_train:n_train + n_val]
        split_registry["test"][label] = shuffled[n_train + n_val:]

    for subset in ["train", "val", "test"]:
        total = 0
        for label, imgs in split_registry[subset].items():
            copy_images(imgs, SPLIT_DIR / subset / label)
            total += len(imgs)
        print(f"  {subset:5s}: {total:,} citra di {len(split_registry[subset])} kelas")
    return split_registry


# ============================================================
# LANGKAH 2 - INTEGRASI VARIATIONS -> VOKAL A
# ============================================================
def step2_integrate_variations(split_registry):
    print("\n" + "=" * 60)
    print("LANGKAH 2: Integrasi variations -> training set Vokal A")
    print("=" * 60)

    # Cari folder Vokal A (mengandung 'vokal' & berakhiran 'a', bukan 'è')
    vokal_a_name = None
    for d in ALL_CLASS_DIR.iterdir():
        nm = d.name.lower()
        if "vokal" in nm and nm.endswith("a") and "è" not in nm:
            vokal_a_name = d.name
            break

    added_total = 0
    missing = []
    for var_dir in sorted(VARIATIONS_DIR.iterdir()):
        if not var_dir.is_dir():
            continue
        cons = var_dir.name
        target = class_label(vokal_a_name, cons) if vokal_a_name else None
        if target is None or target not in split_registry["train"]:
            matched = [l for l in split_registry["train"]
                       if l.split("_", 1)[-1] == cons]
            if matched:
                target = matched[0]
            else:
                missing.append(cons)
                continue
        var_imgs = collect_images(var_dir)
        dst = SPLIT_DIR / "train" / target
        dst.mkdir(parents=True, exist_ok=True)
        for i, src in enumerate(var_imgs):
            shutil.copy2(src, dst / f"var_{i:04d}_{src.name}")
        added_total += len(var_imgs)
        print(f"  {cons:6s} -> {target}: +{len(var_imgs)} citra")

    print(f"\nTotal variations ditambahkan: {added_total}")
    if missing:
        print(f"Konsonan tak ditemukan padanannya: {missing}")


# ============================================================
# LANGKAH 3 - TARGETED AUGMENTATION (hanya non-A -> 300/kelas)
# ============================================================
def step3_targeted_augmentation():
    print("\n" + "=" * 60)
    print(f"LANGKAH 3: Targeted augmentation (target >= {AUGMENT_TARGET}/kelas)")
    print("=" * 60)

    train_dir = SPLIT_DIR / "train"
    AUG_DIR.mkdir(parents=True, exist_ok=True)
    n_aug_cls = 0
    n_skip_cls = 0
    n_synth = 0

    class_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    for class_dir in class_dirs:
        imgs = collect_images(class_dir)
        n = len(imgs)
        dst_dir = AUG_DIR / class_dir.name
        dst_dir.mkdir(parents=True, exist_ok=True)

        # Salin semua gambar asli apa adanya
        for p in imgs:
            shutil.copy2(p, dst_dir / p.name)

        if n >= AUGMENT_TARGET:
            n_skip_cls += 1
            continue

        n_aug_cls += 1
        needed = AUGMENT_TARGET - n
        made = 0
        cycle = 0
        while made < needed:
            for p in imgs:
                if made >= needed:
                    break
                try:
                    img = Image.open(p).convert("RGB")
                    aug = structure_preserving_augment(
                        img, seed=SEED + made + (abs(hash(class_dir.name)) % 9973))
                    out = dst_dir / f"aug_{cycle:02d}_{made:04d}_{p.stem}.jpg"
                    aug.save(out, quality=95)
                    made += 1
                except Exception as e:
                    print(f"  WARN gagal augment {p.name}: {e}")
            cycle += 1
        n_synth += made

    final_counts = [len(collect_images(d)) for d in AUG_DIR.iterdir() if d.is_dir()]
    print(f"  Kelas diaugmentasi : {n_aug_cls}")
    print(f"  Kelas dilewati     : {n_skip_cls} (sudah >= {AUGMENT_TARGET})")
    print(f"  Citra sintetis     : {n_synth:,}")
    print(f"  Total train akhir  : {sum(final_counts):,}")
    if final_counts:
        print(f"  Min/Max per kelas  : {min(final_counts)}/{max(final_counts)}  "
              f"(IR {max(final_counts)/min(final_counts):.2f}x)")


# ============================================================
# LANGKAH 4 - CLASS WEIGHTS
# ============================================================
def step4_compute_class_weights():
    print("\n" + "=" * 60)
    print("LANGKAH 4: Class weights untuk weighted loss")
    print("=" * 60)

    class_names = sorted([d.name for d in AUG_DIR.iterdir() if d.is_dir()])
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    labels = []
    for name in class_names:
        cnt = len(collect_images(AUG_DIR / name))
        labels.extend([class_to_idx[name]] * cnt)
    labels = np.array(labels)
    classes = np.unique(labels)

    weights = compute_class_weight("balanced", classes=classes, y=labels)
    weight_dict = {class_names[i]: float(round(w, 6)) for i, w in zip(classes, weights)}

    WEIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(WEIGHTS_FILE, "w", encoding="utf-8") as f:
        json.dump({"class_to_idx": class_to_idx, "class_weights": weight_dict},
                  f, indent=2, ensure_ascii=False)

    w = list(weight_dict.values())
    print(f"  Total kelas : {len(class_names)}")
    print(f"  Weight min/max/ratio : {min(w):.4f} / {max(w):.4f} / {max(w)/min(w):.2f}x")
    print(f"  Disimpan di : {WEIGHTS_FILE}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("Pipeline Aksara Jawa 120 kelas (revisi New_Model)")
    print(f"Dataset : {JAWA_ROOT}")
    print(f"Output  : {OUTPUT_ROOT}")

    if not ALL_CLASS_DIR.exists():
        raise SystemExit(f"ERROR: {ALL_CLASS_DIR} tidak ditemukan. "
                         f"Pastikan dataset ada di Jawa/Jawa/all_class.")

    if OUTPUT_ROOT.exists():
        print("Menghapus output_pipeline lama...")
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True)

    reg = step1_stratified_split()
    step2_integrate_variations(reg)
    step3_targeted_augmentation()
    step4_compute_class_weights()

    print("\n" + "=" * 60)
    print("PIPELINE SELESAI")
    print("=" * 60)
    print(f"  Split  : {SPLIT_DIR}")
    print(f"  Train  : {AUG_DIR}")
    print(f"  Weights: {WEIGHTS_FILE}")
