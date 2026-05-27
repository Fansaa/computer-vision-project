# Dokumen Perencanaan Penelitian
# Pengenalan Aksara Jawa 120 Kelas dengan Strategi Augmentasi Berbasis Preservasi Struktur pada Dataset Tidak Seimbang

---

## Opsi Judul Penelitian

**Judul Utama (disarankan):**
> **Handwritten Javanese Script Recognition for 120-Class Syllabic Classification Using Structure-Preserving Augmentation and Weighted Loss on an Imbalanced Dataset**

**Alternatif 1 (lebih ringkas):**
> **CNN-Based Javanese Script Recognition with Structure-Preserving Augmentation for Imbalanced 120-Class Dataset**

**Alternatif 2 (menekankan transfer learning):**
> **Transfer Learning for 120-Class Handwritten Javanese Script Recognition under Severe Class Imbalance with Non-Geometric Augmentation**

Catatan pemilihan judul: judul utama dipilih karena memuat tiga unsur pembeda utama penelitian ini, yaitu cakupan 120 kelas silabik, strategi augmentasi yang tidak merusak struktur aksara, dan kondisi dataset tidak seimbang. Ketiganya secara bersama belum pernah dibahas dalam satu penelitian.

---

## 1. Latar Belakang Permasalahan

Aksara Jawa, yang dikenal juga sebagai Hanacaraka atau Carakan, merupakan warisan budaya tulisan yang memiliki nilai historis dan identitas signifikan bagi masyarakat Jawa di Indonesia. Aksara ini telah digunakan selama berabad-abad dalam sastra, dokumen resmi, dan naskah keagamaan (Ifriza et al., 2026). Namun seiring perkembangan teknologi modern dan dominasi aksara Latin dalam sistem pendidikan, penggunaan dan minat mempelajari Aksara Jawa mengalami penurunan drastis. Kondisi ini menyebabkan berkurangnya generasi yang mampu membaca dan menulis Aksara Jawa, sehingga ancaman kepunahan terhadap warisan budaya ini menjadi nyata (Faizin et al., 2025; Wisesty et al., 2026).

Salah satu upaya pelestarian yang dipandang strategis adalah digitalisasi Aksara Jawa melalui sistem pengenalan karakter otomatis berbasis Optical Character Recognition (OCR). Pendekatan ini memungkinkan naskah-naskah kuno yang tertulis dalam Aksara Jawa untuk dikonversi menjadi format digital yang dapat diakses, dilestarikan, dan dikembangkan lebih lanjut untuk aplikasi seperti mesin penerjemah, sistem pembelajaran digital, dan basis data warisan budaya.

Aksara Jawa merupakan sistem tulisan jenis abugida, yaitu sistem penulisan di mana setiap simbol dasar merepresentasikan konsonan dengan vokal inheren, dan vokal lain diindikasikan melalui diakritik tambahan yang disebut sandhangan (Wicaksono et al., 2024; Widiarti dan Adji, 2026). Dalam sistem ini, 20 aksara dasar nglegena (Carakan) menjadi fondasi, dan kombinasinya dengan lima sandhangan swara (vokal A, I, U, E, E-taling) menghasilkan 120 kelas silabik yang berbeda secara visual. Penelitian oleh Widiarti dan Adji (2026) mencatat bahwa kombinasi antara 20 aksara dasar, pasangan, dan sandhangan dapat menghasilkan lebih dari 11.000 unit silabik yang berbeda, yang menggarisbawahi kompleksitas inheren sistem penulisan ini untuk keperluan OCR.

Kompleksitas ini diperparah oleh dua tantangan teknis utama dalam pengembangan sistem pengenalan Aksara Jawa. Pertama, terdapat masalah kemiripan visual tinggi antar karakter tertentu. Penelitian Nindya et al. (2025) di Telkom University mengidentifikasi bahwa pasangan karakter "ha" dan "la" memiliki Likeness Rate sebesar 25,3%, artinya keduanya hampir identik secara visual dan sering menyebabkan misklasifikasi. Pasangan konfusif lain yang telah teridentifikasi adalah "sa"/"da" dan "nga"/"nya". Kedua, dataset tulisan tangan Aksara Jawa yang tersedia secara publik seringkali memiliki distribusi kelas yang sangat tidak seimbang, terutama ketika cakupan diperluas dari 20 kelas dasar ke 120 kelas silabik penuh.

---

## 2. Rumusan Masalah

Berdasarkan latar belakang di atas, permasalahan penelitian ini dirumuskan sebagai berikut:

1. Dataset Indonesian Local Script Characters (Mendeley Data) untuk Aksara Jawa menunjukkan imbalance ratio yang ekstrem sebesar 33,10x antara kelas mayoritas (Vokal A, 426-695 citra per kelas) dan kelas minoritas (Vokal E, I, O, U, E-taling, dengan hanya 21 citra per kelas), sehingga teknik training konvensional akan bias secara signifikan terhadap kelas mayoritas.

2. Augmentasi geometris yang umum digunakan seperti rotasi dan flip, yang umumnya direkomendasikan untuk menangani ketidakseimbangan data, justru terbukti menurunkan performa pengenalan Aksara Jawa (Wisesty et al., 2026) karena mengubah karakteristik struktural aksara. Pasangan karakter "ha"/"la" yang merupakan cerminan horizontal satu sama lain menjadi contoh nyata mengapa flip horizontal tidak dapat diterapkan.

3. Penelitian yang mencakup 120 kelas silabik Aksara Jawa sebelumnya hanya dilakukan pada dataset yang seimbang secara artifisial (Susanto et al., 2023 menggunakan 14.400 citra dengan 120 per kelas yang dikumpulkan secara terstruktur dari 30 penulis). Belum ada penelitian yang membahas pengenalan 120 kelas silabik dengan dataset nyata yang tidak seimbang secara alami.

---

## 3. Tujuan Penelitian

1. Membangun pipeline persiapan data yang menangani ketidakseimbangan ekstrem pada dataset 120 kelas Aksara Jawa melalui kombinasi stratified split, integrasi data variasi gaya tulisan, augmentasi berbasis preservasi struktur, dan weighted loss function.

2. Melatih dan mengevaluasi model deep learning berbasis transfer learning untuk klasifikasi 120 kelas silabik Aksara Jawa pada kondisi dataset tidak seimbang secara nyata.

3. Menganalisis performa model secara terpisah antara kelas Vokal A (mayoritas) dan kelas vokal lainnya (minoritas) untuk mendapatkan pemahaman yang lebih jujur tentang kemampuan generalisasi model.

4. Mengidentifikasi pasangan karakter dengan tingkat kesalahan klasifikasi tertinggi dan mengaitkannya dengan karakteristik visual aksara yang bersangkutan.

---

## 4. Tinjauan Pustaka (Literature Review)

### 4.1 Pengenalan Aksara Jawa 20 Kelas Dasar

Sebagian besar penelitian terdahulu berfokus pada 20 aksara dasar Carakan. Susanto et al. (IJECE, 2023) mengembangkan custom CNN dengan 4 lapisan konvolusi dan 2 lapisan fully connected untuk mengenali 120 kelas (20 konsonan x 6 vokal) Aksara Jawa. Dataset yang digunakan terdiri dari 14.400 citra yang dikumpulkan dari 30 penulis, masing-masing menulis 4 kali, menghasilkan distribusi sempurna 120 citra per kelas. Model ini mencapai akurasi 97,29% dan merupakan salah satu penelitian pertama yang mencakup 120 kelas silabik.

Susanto et al. (IJAI, 2023) mengembangkan Deep CNN 12 lapisan dengan augmentasi afin, proyektif, dan scaling minimal (rotasi hanya ±3 derajat untuk menghindari perubahan makna aksara). Dengan dataset 480 citra asli yang diperbesar melalui augmentasi menjadi 3.360 citra untuk 120 kelas, model ini mencapai akurasi 99,65% pada split 80:20. Hasil ini menegaskan potensi augmentasi selektif dan arsitektur yang dioptimalkan untuk aksara tulisan tangan.

Ifriza et al. (IJAI, 2026) menggunakan fine-tuned ResNet-18 dengan transfer learning untuk mengenali Aksara Jawa dari dataset Hanacaraka yang terdiri dari 1.562 citra untuk 20 kelas dasar, dengan split 80:10:10. Model ini mencapai akurasi 95,53%.

### 4.2 Pengenalan Multi-Aksara Nusantara

Wisesty et al. (IJAI, 2026) dari Telkom University mengembangkan MNetNCR berbasis MobileNetV3 untuk mengenali 5 aksara Nusantara (Jawa, Bali, Sunda, Lontara, Batak). Untuk Aksara Jawa dengan 20 kelas dasar dan 9.995 citra, model mencapai F1-score 0,9788 **tanpa augmentasi**. Yang sangat penting, augmentasi justru menurunkan performa Aksara Jawa dari F1 0,9788 menjadi 0,9579, dengan penjelasan bahwa augmentasi agresif mengubah struktur visual khas Aksara Jawa dan memperburuk kemampuan model dalam mengenali pola yang sudah dipelajari dengan baik.

Nindya et al. (ICoDSA, 2025) dari Telkom University menggunakan VGG-16 untuk klasifikasi tujuh kombinasi Aksara Bali, Jawa, dan Sunda. Untuk Aksara Jawa 20 kelas, akurasi mencapai 93,19%. Penelitian ini secara eksplisit mengidentifikasi pasangan karakter konfusif dengan Character Likeness Rate (CLR) tertinggi, yaitu "la"/"ha" (25,3%), "sa"/"da" (15,8%), dan memberikan kerangka analisis kesalahan yang relevan.

Sulistiyo et al. (RESTI, 2025) dari Telkom University membandingkan berbagai metode shallow learning (LightGBM, XGBoost, Random Forest, SVM, CatBoost) pada dataset Aksarawa yang mencakup Aksara Sunda, Bali, dan Jawa. Untuk Aksara Jawa, LightGBM mencapai F1-score tertinggi 77,15%, mengkonfirmasi bahwa untuk dataset multi-kelas yang kompleks, deep learning tetap lebih unggul dibanding pendekatan shallow learning.

### 4.3 Penanganan Ketidakseimbangan Data pada Aksara Jawa

Faizin et al. (RESTI, 2025) dari ITS mengembangkan SkelBAGAN (Skeleton-based Balancing GAN) untuk menangani ketidakseimbangan data pada dataset naskah Aksara Jawa (HJCS_DETC). Dataset ini terdiri dari 20 kelas dasar dengan Imbalance Ratio berkisar antara 1,00 hingga 81,78x, dengan Imbalance Degree (ID) sebesar 9,33-10,33 (dikategorikan "mid-imbalanced"). Pendekatan GAN dengan komponen skeleton autoencoder dan skeleton adversarial loss terbukti menghasilkan citra sintetis dengan FID lebih rendah dan SSIM lebih tinggi dibanding metode sebelumnya, meningkatkan performa pengenalan kelas minoritas.

Penelitian ini menjadi satu-satunya referensi yang secara eksplisit membahas imbalance pada Aksara Jawa dan memberikan landasan metodologis yang relevan untuk penanganan imbalance tanpa GAN (menggunakan weighted loss dan augmentasi selektif).

### 4.4 Sistem Penulisan Abugida Asia Tenggara

Wicaksono et al. (ICTIIA, 2024) menerbitkan systematic literature review tentang word segmentation untuk aksara Abugida Asia Tenggara (Thai, Burmese, Lao, Khmer, Jawa, Sunda). Temuan penting: **tidak ada satu pun penelitian dalam 5 tahun terakhir yang membahas word segmentation untuk Aksara Jawa atau Sunda**, meskipun kedua aksara ini masih digunakan dalam konteks tertentu. Ini mengkonfirmasi bahwa riset Aksara Jawa masih sangat terbuka untuk kontribusi baru.

---

## 5. Research Gap yang Diselesaikan

Berdasarkan analisis literatur, terdapat tiga gap penelitian yang secara spesifik diselesaikan oleh penelitian ini:

**Gap 1: Tidak ada penelitian 120 kelas dengan dataset tidak seimbang secara nyata.**
Satu-satunya penelitian yang mencakup 120 kelas silabik Aksara Jawa (Susanto et al., 2023; Susanto et al., IJAI 2023) menggunakan dataset yang dirancang seimbang secara artifisial (120 atau 4 citra per kelas dari proses pengumpulan terstruktur). Dataset publik seperti Indonesian Local Script Characters memiliki ketidakseimbangan alami yang ekstrem (imbalance ratio 33,10x) dan belum pernah digunakan untuk cakupan 120 kelas. Penelitian ini mengisi gap tersebut dengan membuktikan apakah model transfer learning dapat berhasil pada kondisi yang lebih realistis ini.

**Gap 2: Tidak ada strategi augmentasi yang dirancang khusus untuk mempertahankan integritas struktural Aksara Jawa.**
Penelitian terdahulu yang menggunakan augmentasi pada Aksara Jawa umumnya menerapkan teknik geometris standar (rotasi, flip, shear), padahal Wisesty et al. (2026) dan Susanto et al. (IJAI, 2023) secara eksplisit memperingatkan bahwa transformasi geometris mengubah karakteristik visual aksara. Belum ada penelitian yang secara sistematis mendefinisikan dan mengevaluasi augmentasi berbasis preservasi struktur (fotometrik dan morfologis) sebagai alternatif untuk Aksara Jawa.

**Gap 3: Analisis performa per-kelompok vokal belum pernah dilaporkan.**
Semua penelitian yang ada melaporkan metrik agregat (akurasi, F1-score keseluruhan). Ketika dataset memiliki ketidakseimbangan ekstrem antar kelompok vokal, angka agregat menyembunyikan kelemahan model pada kelas minoritas. Penelitian ini melaporkan metrik secara terpisah per kelompok vokal (Vokal A vs. vokal lainnya) sebagai kontribusi analisis yang lebih transparan.

---

## 6. Solusi yang Diusulkan

Solusi penelitian ini dibangun di atas empat langkah yang dieksekusi secara berurutan sebelum proses training:

**Langkah 1: Stratified Split (70:15:15)**
Split dataset dilakukan per kelas untuk memastikan distribusi kelas terjaga secara proporsional di setiap subset (training, validasi, testing). Ini mencegah kelas minoritas yang hanya memiliki 21 citra hilang sepenuhnya dari validation atau test set.

**Langkah 2: Integrasi Sub-folder Variations**
Sub-folder variations dari dataset yang berisi 1.066 citra variasi gaya tulisan untuk 8 konsonan (ba, ga, ha, ka, la, nya, ra, ta) diintegrasikan ke dalam training set kelas Vokal A yang bersesuaian. Langkah ini meningkatkan keberagaman gaya tulisan tangan untuk 8 kelas tersebut tanpa memerlukan pengumpulan data baru.

**Langkah 3: Targeted Augmentasi Berbasis Preservasi Struktur**
Augmentasi diterapkan secara selektif hanya pada kelas yang jumlah training image-nya di bawah target 300 citra (yaitu 100 kelas dari Vokal non-A). Teknik yang digunakan terbatas pada transformasi yang tidak mengubah karakteristik visual aksara:
- Fotometrik: brightness/contrast jitter (±30%), Gaussian blur (kernel 1-3 px), Gaussian noise (sigma 0,01-0,05), sharpening
- Morfologis: dilation dan erosion (kernel 2x2 atau 3x3) untuk mensimulasikan variasi ketebalan goresan
- Positional minimal: random crop (max 10% tepi) + padding

Rotasi, flip, shear, dan elastic distortion tidak digunakan.

**Langkah 4: Weighted Loss Function**
Class weight dihitung berdasarkan distribusi final setelah augmentasi menggunakan formula `compute_class_weight('balanced')`. Bobot ini diterapkan pada loss function selama training untuk memberikan penalti lebih besar pada kesalahan klasifikasi kelas minoritas.

---

## 7. Dataset: Apa dan Bagaimana

### 7.1 Sumber Dataset

**Nama dataset:** Indonesian Local Script Characters
**Sumber:** Mendeley Data, dikurasi oleh Aditya Firman Ihsan et al. (Telkom University)
**Referensi dataset:** digunakan juga oleh Wisesty et al. (IJAI, 2026) dan Sulistiyo et al. (RESTI, 2025)

### 7.2 Struktur Dataset Aksara Jawa

Dataset untuk Aksara Jawa terdiri dari dua sub-folder:

| Sub-folder | Struktur | Konten | Jumlah Kelas | Jumlah Citra |
|---|---|---|---|---|
| `all_class/` | Vokal X / konsonan / *.png | Kombinasi vokal x konsonan | 120 | 12.191 |
| `variations/` | konsonan / *.jpg | Variasi gaya tulisan 8 konsonan | 8 | 1.066 |
| **Total** | | | | **13.257** |

### 7.3 Karakteristik Distribusi

Analisis EDA mengungkapkan distribusi yang sangat tidak seimbang:

| Kelompok | Jumlah kelas | Citra per kelas | Keterangan |
|---|---|---|---|
| Vokal A (20 konsonan) | 20 | 426 - 695 | Mayoritas |
| Vokal E (20 konsonan) | 20 | ~21 | Minoritas |
| Vokal I (20 konsonan) | 20 | ~21 | Minoritas |
| Vokal O (20 konsonan) | 20 | ~21 | Minoritas |
| Vokal U (20 konsonan) | 20 | ~21 | Minoritas |
| Vokal E-taling (20 konsonan) | 20 | ~21 | Minoritas |

- Imbalance Ratio (all_class): **33,10x** (max 695 / min 21)
- Median per kelas: **22 citra** (jauh di bawah mean 101,59)
- Standar deviasi: 188,94 (menunjukkan dispersi ekstrem)
- 5 kelas terbesar: seluruhnya dari Vokal A (ya, pa, sa, ja, ma)
- 5 kelas terkecil: seluruhnya dari Vokal E dan E-taling (b, c, dh, g)

Sub-folder variations memiliki imbalance ratio yang jauh lebih rendah (2,87x, max 175 / min 61) dan dapat langsung diintegrasikan ke training set.

### 7.4 Karakteristik Citra

- Ukuran asli: dominan 362-363 x 347-348 piksel
- Format: mayoritas JPEG (91%), sebagian PNG (9%)
- Channel: mayoritas RGB (98,3%), sebagian RGBA (1,7%)
- Ukuran input ke model: distandarisasi ke 224 x 224 piksel

### 7.5 Alur Penggunaan Data

```
Dataset Mentah
    |
    +-- all_class/ (12.191 citra, 120 kelas)
    +-- variations/ (1.066 citra, 8 konsonan)
          |
          v
[Langkah 1] Stratified Split (70:15:15) per kelas
          |
          v
[Langkah 2] Integrasi variations ke training set
          |
          v
[Langkah 3] Targeted augmentasi pada 100 kelas minoritas
            (hanya fotometrik + morfologis)
            Target: minimal 300 citra per kelas
          |
          v
Training Set Final     Validation Set    Test Set
(~36.000+ citra)       (~1.800 citra)    (~1.800 citra)
          |
          v
[Langkah 4] Weighted Loss saat training
```

---

## 8. Metodologi

### 8.1 Preprocessing

1. Konversi ke grayscale (untuk menyederhanakan komputasi; warna tidak membawa informasi makna pada aksara tulisan tangan)
2. Resize ke 224 x 224 piksel (kompatibel dengan arsitektur ImageNet pretrained)
3. Normalisasi piksel (mean dan std ImageNet: [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225])
4. Untuk citra RGBA, konversi ke RGB terlebih dahulu

### 8.2 Pipeline Data (4 Langkah)

Telah diimplementasikan dalam file `jawa_pipeline.py` dengan parameter:
- TRAIN_RATIO: 0.70
- VAL_RATIO: 0.15
- AUGMENT_TARGET: 300 citra per kelas
- Seed: 42 (untuk reproducibility)

### 8.3 Arsitektur Model

Penelitian ini menggunakan pendekatan transfer learning dari model pretrained ImageNet. Kandidat arsitektur yang akan dibandingkan:

| Arsitektur | Parameter | Keunggulan | Referensi terkait |
|---|---|---|---|
| EfficientNet-B0 | ~5,3 juta | Efisien, akurat untuk ukuran kecil | - |
| MobileNetV3-Large | ~5,4 juta | Sangat ringan, cocok deployment | Wisesty et al. (2026) |
| ResNet-18 | ~11,7 juta | Stabil, residual connection | Ifriza et al. (2026) |
| VGG-16 | ~138 juta | Proven untuk aksara regional | Nindya et al. (2025) |

Rekomendasi: mulai dengan EfficientNet-B0 sebagai baseline utama, kemudian bandingkan dengan MobileNetV3 (mengikuti Wisesty et al.) dan ResNet-18 (mengikuti Ifriza et al.).

### 8.4 Konfigurasi Training

| Hyperparameter | Nilai | Justifikasi |
|---|---|---|
| Optimizer | Adam | Konsensus literatur (semua paper terkait) |
| Learning rate | 0,001 (awal), dengan ReduceLROnPlateau | Best practice transfer learning |
| Batch size | 32 atau 64 | Bergantung memori GPU |
| Epochs | Maks 100 dengan EarlyStopping (patience=10) | Mencegah overfitting |
| Loss function | CrossEntropyLoss dengan class_weight | Langkah 4 pipeline |
| Input size | 224 x 224 | Standar pretrained ImageNet |

Strategi fine-tuning: freeze lapisan backbone awal, train hanya classifier head pada 10 epoch pertama, kemudian unfreeze seluruh jaringan untuk fine-tuning penuh.

### 8.5 Evaluasi

**Metrik utama (keseluruhan):**
- Accuracy, Precision, Recall, F1-score (macro dan weighted)
- Confusion matrix 120 x 120

**Metrik per kelompok vokal (kontribusi analisis utama):**

| Kelompok | Kelas | Tujuan analisis |
|---|---|---|
| Vokal A | 20 kelas (kelas mayoritas) | Mengukur performa pada data melimpah |
| Vokal non-A | 100 kelas (kelas minoritas) | Mengukur kemampuan generalisasi pada data terbatas |

**Analisis confusion karakter:**
Mengidentifikasi 10 pasangan karakter dengan confusion rate tertinggi, dikaitkan dengan temuan literatur (pasangan ha/la, sa/da, dll.).

---

## 9. Scope Penelitian

| Aspek | Scope |
|---|---|
| Jenis aksara | Aksara Jawa (Hanacaraka) handwritten |
| Jumlah kelas | 120 (6 vokal x 20 konsonan dasar) |
| Jenis tugas | Isolated character recognition (klasifikasi satu karakter per citra) |
| Dataset | Indonesian Local Script Characters (Mendeley Data), subset Jawa |
| Platform | Python 3.x, PyTorch atau TensorFlow/Keras |
| Evaluasi | Accuracy, Precision, Recall, F1-score; analisis per kelompok vokal |
| Di luar scope | Word-level recognition, segmentasi kata, pasangan aksara (pasangan), murda |

---

## 10. Ringkasan Kontribusi Penelitian

1. **Kontribusi metodologis:** Penerapan strategi augmentasi berbasis preservasi struktur (non-geometris) yang secara eksplisit menghindari transformasi yang dapat mengubah makna Aksara Jawa, serta integrasi sub-folder variations sebagai sumber keberagaman gaya tulisan tangan.

2. **Kontribusi empiris:** Evaluasi pertama pengenalan 120 kelas silabik Aksara Jawa pada dataset tidak seimbang secara nyata (IR 33,10x), melengkapi penelitian Susanto et al. (2023) yang menggunakan dataset seimbang artifisial.

3. **Kontribusi analitis:** Pelaporan metrik performa secara terpisah antara kelas mayoritas (Vokal A) dan kelas minoritas (Vokal non-A), memberikan gambaran yang lebih transparan tentang kemampuan dan keterbatasan model dalam kondisi imbalance nyata.

---

## 11. Referensi Utama

1. Faizin, M. A., Suciati, N., & Fatichah, C. (2025). Handling Imbalance in Javanese Manuscript Character Dataset using Skeleton-based Balancing Generative Adversarial Networks. *Jurnal RESTI (Rekayasa Sistem dan Teknologi Informasi)*, 9(4), 820-831.

2. Ifriza, Y. N., et al. (2026). Optimizing Javanese script recognition using fine-tuned ResNet-18 and transfer learning. *International Journal of Artificial Intelligence (IJ-AI)*, 15(1), 443-453.

3. Nindya, T. P., Sulistiyo, M. D., & Wisesty, U. N. (2025). Recognition of Balinese, Javanese, and Sundanese Scripts Using CNN with VGG-16 Architecture. *2025 International Conference on Data Science and Its Applications (ICoDSA)*, 1221-1226.

4. Sulistiyo, M. D., et al. (2025). Aksarawa: Comparative Study of Shallow Learning for OCR of Nusantara Scripts. *Jurnal RESTI*, 9(6), 1537-1546.

5. Susanto, A., Mulyono, I. U. W., Sari, C. A., Rachmawanto, E. H., Setiadi, D. R. I. M., & Sarker, M. K. (2023). Handwritten Javanese script recognition method based 12-layers deep convolutional neural network and data augmentation. *International Journal of Artificial Intelligence (IJ-AI)*, 12(3), 1448-1458.

6. Susanto, A., Mulyono, I. U. W., Sari, C. A., Rachmawanto, E. H., & Setiadi, D. R. I. M. (2023). Improved Javanese script recognition using custom model of convolution neural network. *International Journal of Electrical and Computer Engineering (IJECE)*, 13(6), 6629-6636.

7. Wicaksono, B. A., Hantono, B. S., & Adji, T. B. (2024). Word Segmentation Task for Southeast Asian Abugida Scripts: A Systematic Literature Review. *2024 2nd International Conference on Technology Innovation and Its Applications (ICTIIA)*.

8. Widiarti, R., & Adji, F. T. (2026). A Baseline Evaluation of OCR Segmentation and Classification Methods for Printed Javanese Script. *Engineering, Technology & Applied Science Research (ETASR)*, 16(1), 31699-31705.

9. Wisesty, U. N., et al. (2026). MNetNCR: MobileNet model for efficient traditional Nusantara script character recognition. *International Journal of Artificial Intelligence (IJ-AI)*, 15(2), 1513-1528.
