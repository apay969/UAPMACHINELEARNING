# 🧠 UAP – Image Classification  
### CNN • ResNet50 • VGG16

Aplikasi **klasifikasi citra** berbasis **Deep Learning** menggunakan **TensorFlow & Streamlit**.  
Project ini membandingkan performa **Custom CNN**, **ResNet50**, dan **VGG16** pada dataset citra yang di-*upload* dalam format ZIP.

---

## 🚀 Fitur Utama
- Upload dataset citra (.zip)
- Pembagian data otomatis:
  - **Train:** 70%
  - **Validation:** 20%
  - **Test:** 10%
- Pilihan model:
  - CNN (from scratch)
  - ResNet50 (pretrained ImageNet)
  - VGG16 (pretrained ImageNet)
- Visualisasi:
  - Accuracy & Loss
  - Evaluasi Test Set
- Antarmuka interaktif dengan **Streamlit**

---

## 🧩 Arsitektur Model

### 1️⃣ CNN (Custom)
- Conv2D: 32 → 64 → 128
- MaxPooling
- Dense 256 + Dropout 0.5
- Optimizer: Adam (lr = 0.0001)

### 2️⃣ ResNet50
- Pretrained ImageNet
- `include_top=False`
- Global Average Pooling
- Dense 512
- Base model **dibekukan (freeze)**

### 3️⃣ VGG16
- Pretrained ImageNet
- `include_top=False`
- Global Average Pooling
- Dense 64
- 10 layer terakhir **trainable**

---

## ⚙️ Konfigurasi Training
| Parameter | Nilai |
|---------|------|
| Image Size | 224 × 224 |
| Channels | 3 (RGB) |
| Epoch | 1 – 30 |
| Batch Size | 8, 16, 32 |
| Loss | Sparse Categorical Crossentropy |
| Optimizer | Adam (0.0001) |

---

## 📊 Hasil Evaluasi Model

> Berikut adalah ringkasan hasil evaluasi model berdasarkan **Test Dataset (10%)**

### 🔍 Perbandingan Performa

| Model | Train Accuracy | Validation Accuracy | Test Accuracy | Test Loss |
|------|---------------|---------------------|--------------|-----------|
| CNN | **92.46%** | **95.20%** | **95.63%** | **0.1274** |
| ResNet50 | **53.90%** | **52.06%** | **50.60%** | **1.3762** |
| VGG16 | **97.57%** | **98.32%** | **98.04%** | **0.0660** |

> 🔧 *Isi nilai di atas sesuai dengan hasil evaluasi pada file PDF.*

---

## 📈 Visualisasi
Aplikasi menampilkan grafik:
- **Accuracy vs Epoch**
- **Loss vs Epoch**
  
Grafik dihasilkan otomatis setelah proses training selesai.

---

## 🧪 Evaluasi
Evaluasi dilakukan menggunakan **test dataset** yang belum pernah dilihat model selama training maupun validation.

Output evaluasi:
- Test Accuracy
- Test Loss

---

## 📦 Struktur Dataset
```text
dataset/
├── class_1/
│   ├── img1.jpg
│   └── img2.jpg
├── class_2/
│   ├── img1.jpg
│   └── img2.jpg
