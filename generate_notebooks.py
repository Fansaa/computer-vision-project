import json
from pathlib import Path

def create_notebook(model_name):
    # Set hyperparameter defaults per model
    if model_name == "EfficientNet-B0":
        lr_phase1 = "1e-3"
        lr_phase2 = "1e-4"
        weight_decay = "1e-5"
        patience_es = 10
    elif model_name == "MobileNetV3-Large":
        lr_phase1 = "1e-3"
        lr_phase2 = "1e-4"
        weight_decay = "1e-6"
        patience_es = 8
    elif model_name == "ResNet-18":
        lr_phase1 = "2e-3"
        lr_phase2 = "2e-4"
        weight_decay = "1e-5"
        patience_es = 12
    elif model_name == "VGG-16":
        lr_phase1 = "5e-4"
        lr_phase2 = "2e-5"
        weight_decay = "1e-4"
        patience_es = 15
    else:
        lr_phase1 = "1e-3"
        lr_phase2 = "1e-4"
        weight_decay = "1e-5"
        patience_es = 10

    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    # 1. Header
    title = f"# 🔏 Pengenalan Aksara Jawa 120 Kelas — Arsitektur {model_name}"
    desc = (
        "Notebook ini melakukan transfer learning menggunakan PyTorch untuk mengklasifikasi "
        "120 kelas silabik Aksara Jawa dari dataset Indonesian Local Script Characters.\n\n"
        "### Alur Utama:\n"
        "1. **Mempersiapkan Dataloader** dengan normalisasi ImageNet dan resize ke 224x224.\n"
        "2. **Penerapan Class Weights** dari `output_pipeline/class_weights.json` ke Weighted CrossEntropyLoss.\n"
        "3. **Fase 1 - Feature Extraction**: Melatih classifier head dengan parameter backbone dibekukan.\n"
        "4. **Fase 2 - Full Fine-Tuning**: Melatih seluruh parameter jaringan secara end-to-end dengan learning rate lebih rendah.\n"
        "5. **Evaluasi Akhir & Visualisasi**: Mengukur performa per kelompok Vokal A vs Vokal non-A serta visualisasi Confusion Matrix."
    )
    
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"{title}\n", f"{desc}\n"]
    })
    
    # 2. Imports and Configuration
    model_key = model_name.lower().replace("-", "_")
    results_dir = f"results_{model_key}"
    model_filename = f"{model_key}_aksara_jawa_120class.pth"
    
    imports_code = (
        "import os\n"
        "import sys\n"
        "import time\n"
        "import json\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "from sklearn.metrics import classification_report, confusion_matrix, f1_score\n\n"
        "import torch\n"
        "import torch.nn as nn\n"
        "import torch.optim as optim\n"
        "from torch.utils.data import DataLoader\n"
        "from torchvision import datasets, transforms, models\n\n"
        "# Konfigurasi Path\n"
        "OUTPUT_ROOT = Path(\"output_pipeline\")\n"
        "TRAIN_DIR = OUTPUT_ROOT / \"augmented_train\"\n"
        "VAL_DIR = OUTPUT_ROOT / \"split\" / \"val\"\n"
        "TEST_DIR = OUTPUT_ROOT / \"split\" / \"test\"\n"
        "WEIGHTS_FILE = OUTPUT_ROOT / \"class_weights.json\"\n\n"
        f"RESULTS_DIR = Path(\"{results_dir}\")\n"
        f"MODEL_SAVE_PATH = RESULTS_DIR / \"{model_filename}\"\n"
        "HISTORY_SAVE_PATH = RESULTS_DIR / \"training_history.json\"\n"
        "REPORT_SAVE_PATH = RESULTS_DIR / \"evaluation_report.json\"\n\n"
        "RESULTS_DIR.mkdir(parents=True, exist_ok=True)\n"
        "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n"
        "print(\"🖥️ Device:\", device)\n\n"
        "# ============================================================\n"
        f"# HYPERPARAMETERS KHUSUS ARSITEKTUR {model_name.upper()}\n"
        "# ============================================================\n"
        f"LR_PHASE1 = {lr_phase1}\n"
        f"LR_PHASE2 = {lr_phase2}\n"
        f"WEIGHT_DECAY = {weight_decay}\n"
        f"PATIENCE_ES = {patience_es}"
    )
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in imports_code.split("\n")]
    })
    
    # 3. Data transforms and loading
    loading_desc = (
        "## 📁 1. Memuat Dataset dan Preprocessing\n"
        "Citra diubah menjadi grayscale lalu dikonversi ke 3-channel RGB agar cocok dengan "
        "pretrained model ImageNet, diikuti dengan normalisasi piksel standard."
    )
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [loading_desc + "\n"]
    })
    
    loading_code = (
        "img_transforms = transforms.Compose([\n"
        "    transforms.Resize((224, 224)),\n"
        "    transforms.Grayscale(num_output_channels=3),\n"
        "    transforms.ToTensor(),\n"
        "    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n"
        "])\n\n"
        "print(\"📂 Memuat datasets...\")\n"
        "train_dataset = datasets.ImageFolder(root=str(TRAIN_DIR), transform=img_transforms)\n"
        "val_dataset = datasets.ImageFolder(root=str(VAL_DIR), transform=img_transforms)\n"
        "test_dataset = datasets.ImageFolder(root=str(TEST_DIR), transform=img_transforms)\n\n"
        "class_names = train_dataset.classes\n"
        "num_classes = len(class_names)\n\n"
        "print(f\"   - Jumlah kelas : {num_classes}\")\n"
        "print(f\"   - Training     : {len(train_dataset):,} citra\")\n"
        "print(f\"   - Validasi     : {len(val_dataset):,} citra\")\n"
        "print(f\"   - Testing      : {len(test_dataset):,} citra\")\n\n"
        "num_workers = 0 if os.name == 'nt' else 4\n"
        "train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=num_workers)\n"
        "val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=num_workers)\n"
        "test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=num_workers)"
    )
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in loading_code.split("\n")]
    })
    
    # 4. Class Weights
    weights_desc = (
        "## ⚖️ 2. Penyetelan Class Weights\n"
        "Memuat hasil class weights dari pipeline untuk diumpankan ke loss function CrossEntropyLoss."
    )
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [weights_desc + "\n"]
    })
    
    weights_code = (
        "print(f\"📖 Membaca bobot kelas dari: {WEIGHTS_FILE}\")\n"
        "with open(WEIGHTS_FILE, \"r\", encoding=\"utf-8\") as f:\n"
        "    meta = json.load(f)\n"
        "class_to_idx = meta[\"class_to_idx\"]\n"
        "weight_dict = meta[\"class_weights\"]\n\n"
        "weight_tensor = torch.zeros(num_classes)\n"
        "for cls_name, idx in class_to_idx.items():\n"
        "    weight_tensor[idx] = weight_dict[cls_name]\n\n"
        "class_weights = weight_tensor.to(device)\n"
        "criterion = nn.CrossEntropyLoss(weight=class_weights)\n"
        "print(\"✅ Class weights berhasil dimuat dan diterapkan ke loss function.\")"
    )
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in weights_code.split("\n")]
    })
    
    # 5. Model Initialization
    model_desc = (
        f"## 🏗️ 3. Inisialisasi Arsitektur {model_name}\n"
        "Mengunduh bobot pretrained ImageNet dan menyesuaikan classifier head untuk 120 kelas Aksara Jawa."
    )
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [model_desc + "\n"]
    })
    
    if model_name == "EfficientNet-B0":
        model_init_code = (
            "print(\"🏗️ Memuat EfficientNet-B0 Pretrained...\")\n"
            "try:\n"
            "    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)\n"
            "except AttributeError:\n"
            "    model = models.efficientnet_b0(pretrained=True)\n\n"
            "# Ganti classifier head\n"
            "in_features = model.classifier[1].in_features\n"
            "model.classifier[1] = nn.Linear(in_features, num_classes)\n"
            "model = model.to(device)\n"
            "print(\"✅ Model siap.\")"
        )
    elif model_name == "MobileNetV3-Large":
        model_init_code = (
            "print(\"🏗️ Memuat MobileNetV3-Large Pretrained...\")\n"
            "try:\n"
            "    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)\n"
            "except AttributeError:\n"
            "    model = models.mobilenet_v3_large(pretrained=True)\n\n"
            "# Ganti classifier head (linear di indeks 3 dari classifier)\n"
            "in_features = model.classifier[3].in_features\n"
            "model.classifier[3] = nn.Linear(in_features, num_classes)\n"
            "model = model.to(device)\n"
            "print(\"✅ Model siap.\")"
        )
    elif model_name == "ResNet-18":
        model_init_code = (
            "print(\"🏗️ Memuat ResNet-18 Pretrained...\")\n"
            "try:\n"
            "    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)\n"
            "except AttributeError:\n"
            "    model = models.resnet18(pretrained=True)\n\n"
            "# Ganti classifier head\n"
            "in_features = model.fc.in_features\n"
            "model.fc = nn.Linear(in_features, num_classes)\n"
            "model = model.to(device)\n"
            "print(\"✅ Model siap.\")"
        )
    elif model_name == "VGG-16":
        model_init_code = (
            "print(\"🏗️ Memuat VGG-16 Pretrained...\")\n"
            "try:\n"
            "    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)\n"
            "except AttributeError:\n"
            "    model = models.vgg16(pretrained=True)\n\n"
            "# Ganti classifier head (linear di indeks 6 dari classifier)\n"
            "in_features = model.classifier[6].in_features\n"
            "model.classifier[6] = nn.Linear(in_features, num_classes)\n"
            "model = model.to(device)\n"
            "print(\"✅ Model siap.\")"
        )
        
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in model_init_code.split("\n")]
    })
    
    # 6. Training Functions
    fn_desc = (
        "## 🛠️ 4. Definisi Helper Training (Epoch Loop, Eval, Early Stopping)\n"
        "Menulis callback `EarlyStopping` dan fungsi standar epoch training."
    )
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [fn_desc + "\n"]
    })
    
    fn_code = (
        "class EarlyStopping:\n"
        "    def __init__(self, patience=10, min_delta=0.0):\n"
        "        self.patience = patience\n"
        "        self.min_delta = min_delta\n"
        "        self.best_loss = float('inf')\n"
        "        self.counter = 0\n"
        "        self.early_stop = False\n\n"
        "    def __call__(self, val_loss):\n"
        "        if val_loss < self.best_loss - self.min_delta:\n"
        "            self.best_loss = val_loss\n"
        "            self.counter = 0\n"
        "        else:\n"
        "            self.counter += 1\n"
        "            if self.counter >= self.patience:\n"
        "                self.early_stop = True\n\n"
        "def train_epoch(model, loader, criterion, optimizer):\n"
        "    model.train()\n"
        "    running_loss = 0.0\n"
        "    correct = 0\n"
        "    total = 0\n"
        "    for images, labels in loader:\n"
        "        images, labels = images.to(device), labels.to(device)\n"
        "        optimizer.zero_grad()\n"
        "        outputs = model(images)\n"
        "        loss = criterion(outputs, labels)\n"
        "        loss.backward()\n"
        "        optimizer.step()\n"
        "        running_loss += loss.item() * images.size(0)\n"
        "        _, preds = outputs.max(1)\n"
        "        correct += preds.eq(labels).sum().item()\n"
        "        total += labels.size(0)\n"
        "    return running_loss / total, correct / total\n\n"
        "def evaluate_epoch(model, loader, criterion):\n"
        "    model.eval()\n"
        "    running_loss = 0.0\n"
        "    correct = 0\n"
        "    total = 0\n"
        "    all_preds = []\n"
        "    all_labels = []\n"
        "    with torch.no_grad():\n"
        "        for images, labels in loader:\n"
        "            images, labels = images.to(device), labels.to(device)\n"
        "            outputs = model(images)\n"
        "            loss = criterion(outputs, labels)\n"
        "            running_loss += loss.item() * images.size(0)\n"
        "            _, preds = outputs.max(1)\n"
        "            correct += preds.eq(labels).sum().item()\n"
        "            total += labels.size(0)\n"
        "            all_preds.extend(preds.cpu().numpy())\n"
        "            all_labels.extend(labels.cpu().numpy())\n"
        "    epoch_loss = running_loss / total\n"
        "    epoch_acc = correct / total\n"
        "    epoch_f1 = f1_score(all_labels, all_preds, average='macro')\n"
        "    return epoch_loss, epoch_acc, epoch_f1"
    )
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in fn_code.split("\n")]
    })
    
    # 7. Phase 1 Training
    p1_desc = (
        "## 🚀 5. FASE 1: FEATURE EXTRACTION (Melatih Kepala Klasifikasi saja)\n"
        "Membekukan parameter *backbone* dan hanya melatih *classifier head* dengan learning rate $10^{-3}$ selama 10 epoch awal."
    )
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [p1_desc + "\n"]
    })
    
    p1_code = (
        "# Freeze backbone & Unfreeze classifier head (model-agnostic)\n"
        "for param in model.parameters():\n"
        "    param.requires_grad = False\n\n"
        "if hasattr(model, 'classifier'):\n"
        "    for param in model.classifier.parameters():\n"
        "        param.requires_grad = True\n"
        "    optimizer_fase1 = optim.Adam(model.classifier.parameters(), lr=LR_PHASE1, weight_decay=WEIGHT_DECAY)\n"
        "elif hasattr(model, 'fc'):\n"
        "    for param in model.fc.parameters():\n"
        "        param.requires_grad = True\n"
        "    optimizer_fase1 = optim.Adam(model.fc.parameters(), lr=LR_PHASE1, weight_decay=WEIGHT_DECAY)\n\n"
        "epochs_phase1 = 10\n\n"
        "history = {\n"
        "    \"train_loss\": [], \"train_acc\": [],\n"
        "    \"val_loss\": [], \"val_acc\": [], \"val_f1\": []\n"
        "}\n\n"
        "print(\"🚀 Memulai Fase 1...\")\n"
        "for epoch in range(epochs_phase1):\n"
        "    t0 = time.time()\n"
        "    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer_fase1)\n"
        "    val_loss, val_acc, val_f1 = evaluate_epoch(model, val_loader, criterion)\n"
        "    elapsed = time.time() - t0\n\n"
        "    history[\"train_loss\"].append(train_loss)\n"
        "    history[\"train_acc\"].append(train_acc)\n"
        "    history[\"val_loss\"].append(val_loss)\n"
        "    history[\"val_acc\"].append(val_acc)\n"
        "    history[\"val_f1\"].append(val_f1)\n\n"
        "    print(f\"Epoch [{epoch+1:02d}/{epochs_phase1:02d}] ({elapsed:.1f}s) | \"\n"
        "          f\"Train Loss: {train_loss:.4f} - Acc: {train_acc*100:.2f}% | \"\n"
        "          f\"Val Loss: {val_loss:.4f} - Acc: {val_acc*100:.2f}% - F1: {val_f1:.4f}\")"
    )
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in p1_code.split("\n")]
    })
    
    # 8. Phase 2 Training
    p2_desc = (
        "## 🚀 6. FASE 2: FULL FINE-TUNING (Melatih Seluruh Jaringan)\n"
        "Membuka pembekuan parameter backbone untuk melatih seluruh jaringan secara end-to-end dengan learning rate yang lebih rendah ($10^{-4}$)."
    )
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [p2_desc + "\n"]
    })
    
    p2_code = (
        "# Unfreeze all parameters\n"
        "for param in model.parameters():\n"
        "    param.requires_grad = True\n\n"
        "optimizer_fase2 = optim.Adam(model.parameters(), lr=LR_PHASE2, weight_decay=WEIGHT_DECAY)\n"
        "scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_fase2, mode='min', factor=0.5, patience=3)\n"
        "early_stopping = EarlyStopping(patience=PATIENCE_ES)\n"
        "epochs_phase2 = 90\n"
        "best_val_loss = float('inf')\n\n"
        "print(\"🚀 Memulai Fase 2 (Fine-Tuning)...\")\n"
        "for epoch in range(epochs_phase2):\n"
        "    actual_epoch = epochs_phase1 + epoch + 1\n"
        "    t0 = time.time()\n"
        "    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer_fase2)\n"
        "    val_loss, val_acc, val_f1 = evaluate_epoch(model, val_loader, criterion)\n"
        "    elapsed = time.time() - t0\n\n"
        "    scheduler.step(val_loss)\n"
        "    history[\"train_loss\"].append(train_loss)\n"
        "    history[\"train_acc\"].append(train_acc)\n"
        "    history[\"val_loss\"].append(val_loss)\n"
        "    history[\"val_acc\"].append(val_acc)\n"
        "    history[\"val_f1\"].append(val_f1)\n\n"
        "    curr_lr = optimizer_fase2.param_groups[0]['lr']\n"
        "    print(f\"Epoch [{actual_epoch:02d}/{epochs_phase1+epochs_phase2:02d}] ({elapsed:.1f}s) | LR: {curr_lr:.2e} | \"\n"
        "          f\"Train Loss: {train_loss:.4f} - Acc: {train_acc*100:.2f}% | \"\n"
        "          f\"Val Loss: {val_loss:.4f} - Acc: {val_acc*100:.2f}% - F1: {val_f1:.4f}\")\n\n"
        "    if val_loss < best_val_loss:\n"
        "        best_val_loss = val_loss\n"
        "        torch.save(model.state_dict(), str(MODEL_SAVE_PATH))\n"
        "        print(f\"   💾 Checkpoint model terbaik disimpan ke: {MODEL_SAVE_PATH}\")\n\n"
        "    early_stopping(val_loss)\n"
        "    if early_stopping.early_stop:\n"
        "        print(f\"🛑 EARLY STOPPING dipicu pada epoch {actual_epoch}!\")\n"
        "        break\n\n"
        "# Simpan training history\n"
        "with open(HISTORY_SAVE_PATH, \"w\") as f:\n"
        "    json.dump(history, f, indent=2)"
    )
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in p2_code.split("\n")]
    })
    
    # 9. Curves plotting
    curves_desc = (
        "## 📊 7. Visualisasi Kurva Training\n"
        "Menggambar kurva loss dan akurasi untuk memantau kemajuan model."
    )
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [curves_desc + "\n"]
    })
    
    curves_code = (
        "epochs_range = range(1, len(history[\"train_loss\"]) + 1)\n"
        "plt.figure(figsize=(12, 5))\n\n"
        "# Loss Curve\n"
        "plt.subplot(1, 2, 1)\n"
        "plt.plot(epochs_range, history[\"train_loss\"], 'r-', label='Train Loss')\n"
        "plt.plot(epochs_range, history[\"val_loss\"], 'b-', label='Val Loss')\n"
        "plt.title('Kurva Loss')\n"
        "plt.xlabel('Epoch')\n"
        "plt.ylabel('Loss')\n"
        "plt.legend()\n"
        "plt.grid(True)\n\n"
        "# Accuracy Curve\n"
        "plt.subplot(1, 2, 2)\n"
        "plt.plot(epochs_range, history[\"train_acc\"], 'r-', label='Train Acc')\n"
        "plt.plot(epochs_range, history[\"val_acc\"], 'b-', label='Val Acc')\n"
        "plt.title('Kurva Akurasi')\n"
        "plt.xlabel('Epoch')\n"
        "plt.ylabel('Akurasi')\n"
        "plt.legend()\n"
        "plt.grid(True)\n\n"
        "plt.tight_layout()\n"
        "plt.savefig(RESULTS_DIR / \"training_curves.png\", dpi=150)\n"
        "plt.show()"
    )
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in curves_code.split("\n")]
    })
    
    # 10. Evaluation and Vowels division
    eval_desc = (
        "## 🎯 8. Evaluasi Model pada Test Dataset (Kelompok Vokal A vs Vokal non-A)\n"
        "Memuat parameter bobot terbaik yang disimpan, kemudian menguji pada data uji test set "
        "dan membagi metrik performa secara terpisah antara kelas Vokal A (mayoritas) dan Vokal non-A (minoritas)."
    )
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [eval_desc + "\n"]
    })
    
    eval_code = (
        "print(\"💾 Memuat model terbaik untuk evaluasi test set...\")\n"
        "model.load_state_dict(torch.load(str(MODEL_SAVE_PATH)))\n"
        "model.eval()\n\n"
        "all_preds = []\n"
        "all_labels = []\n\n"
        "with torch.no_grad():\n"
        "    for images, labels in test_loader:\n"
        "        images = images.to(device)\n"
        "        outputs = model(images)\n"
        "        _, preds = outputs.max(1)\n"
        "        all_preds.extend(preds.cpu().numpy())\n"
        "        all_labels.extend(labels.cpu().numpy())\n\n"
        "# Identifikasi kelompok vokal\n"
        "vokal_a_idx = [idx for name, idx in class_to_idx.items() if \"vokal a_\" in name.lower()]\n"
        "vokal_a_indices = set(vokal_a_idx)\n\n"
        "labels_a, preds_a = [], []\n"
        "labels_non_a, preds_non_a = [], []\n\n"
        "for l, p in zip(all_labels, all_preds):\n"
        "    if l in vokal_a_indices:\n"
        "        labels_a.append(l)\n"
        "        preds_a.append(p)\n"
        "    else:\n"
        "        labels_non_a.append(l)\n"
        "        preds_non_a.append(p)\n\n"
        "f1_overall = f1_score(all_labels, all_preds, average='macro')\n"
        "f1_a = f1_score(labels_a, preds_a, average='macro') if labels_a else 0.0\n"
        "f1_non_a = f1_score(labels_non_a, preds_non_a, average='macro') if labels_non_a else 0.0\n"
        "full_report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)\n\n"
        "summary = {\n"
        "    \"overall_accuracy\": full_report[\"accuracy\"],\n"
        "    \"overall_macro_f1\": f1_overall,\n"
        "    \"vokal_a_macro_f1\": f1_a,\n"
        "    \"vokal_non_a_macro_f1\": f1_non_a\n"
        "}\n\n"
        "with open(REPORT_SAVE_PATH, \"w\") as f:\n"
        "    json.dump({\"summary\": summary, \"full_report\": full_report}, f, indent=2)\n\n"
        "print(\"\\n\" + \"=\"*60)\n"
        f"print(\"📊 LAPORAN EVALUASI AKHIR — {model_name.upper()}\")\n"
        "print(\"=\"*60)\n"
        "print(f\"  Akurasi Keseluruhan          : {summary['overall_accuracy']*100:.2f}%\")\n"
        "print(f\"  Macro F1-Score Keseluruhan   : {summary['overall_macro_f1']:.4f}\")\n"
        "print(\"-\"*60)\n"
        "print(f\"  F1-Score Kelompok Vokal A     (Mayoritas): {summary['vokal_a_macro_f1']:.4f}\")\n"
        "print(f\"  F1-Score Kelompok Vokal non-A (Minoritas): {summary['vokal_non_a_macro_f1']:.4f}\")\n"
        "print(\"=\"*60)\n"
        "print(f\"Laporan tersimpan di: {REPORT_SAVE_PATH}\")"
    )
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in eval_code.split("\n")]
    })
    
    # 11. Confusion Matrix
    cm_desc = (
        "## 🎨 9. Visualisasi Confusion Matrix\n"
        "Menampilkan visualisasi persebaran prediksi yang benar dan kesalahan klasifikasi."
    )
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [cm_desc + "\n"]
    })
    
    cm_code = (
        "cm = confusion_matrix(all_labels, all_preds)\n"
        "plt.figure(figsize=(20, 16))\n"
        "sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)\n"
        f"plt.title('Confusion Matrix Heatmap — {model_name}', fontsize=16, fontweight='bold')\n"
        "plt.xlabel('Prediksi Kelas', fontsize=12)\n"
        "plt.ylabel('Label Sebenarnya', fontsize=12)\n"
        "plt.xticks(rotation=90, fontsize=6)\n"
        "plt.yticks(rotation=0, fontsize=6)\n"
        "plt.tight_layout()\n"
        "plt.savefig(RESULTS_DIR / \"confusion_matrix.png\", dpi=150)\n"
        "plt.show()"
    )
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in cm_code.split("\n")]
    })
    
    # Write file
    filepath = Path(f"train_{model_key}.ipynb")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    print(f"Success: {filepath}")

if __name__ == "__main__":
    create_notebook("EfficientNet-B0")
    create_notebook("MobileNetV3-Large")
    create_notebook("ResNet-18")
    create_notebook("VGG-16")

    

