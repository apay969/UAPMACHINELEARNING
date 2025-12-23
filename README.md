# 🌿 Klasifikasi Daun Tanaman Menggunakan Deep Learning  
### CNN • ResNet50 • VGG16

Project ini merupakan implementasi **klasifikasi citra daun tanaman** menggunakan pendekatan **Deep Learning** berbasis **Convolutional Neural Network (CNN)** dan **Transfer Learning**.

Dataset yang digunakan berasal dari **Kaggle**, dengan jumlah **lebih dari 6.000 citra daun tanaman**, yang terdiri dari berbagai kelas dan kondisi daun.  
Aplikasi dibangun menggunakan **TensorFlow** dan **Streamlit** untuk menyediakan proses **training, evaluasi, dan visualisasi model secara interaktif**.

---

## 📂 Deskripsi Dataset

- 📌 **Sumber Dataset:** Kaggle  
- 📊 **Jumlah Data:** > 6.000 citra
- 🖼️ **Format:** JPG / PNG
- 🌱 **Objek:** Daun tanaman
- 🏷️ **Label:** Folder-based (per kelas)
- 📐 **Ukuran Input:** 224 × 224 pixel (RGB)

### 📊 Pembagian Dataset
Dataset dibagi secara otomatis menggunakan TensorFlow `tf.data` dengan rasio:

| Split | Persentase |
|------|------------|
| Training | 70% |
| Validation | 20% |
| Testing | 10% |

Pembagian ini bertujuan untuk memastikan model dapat:
- Belajar dari data training
- Disesuaikan melalui validation
- Dievaluasi secara objektif pada test set

---

# 🧠 UAP – Image Classification  
### CNN • ResNet50 • VGG16

Aplikasi **klasifikasi citra** berbasis **Deep Learning** menggunakan **TensorFlow & Streamlit**.  
Project ini membandingkan performa **Custom CNN**, **ResNet50**, dan **VGG16** pada dataset citra daun tanaman yang di-*upload* dalam format ZIP.

---

## 🚀 Fitur Utama
- Upload dataset citra (.zip)
- Pembagian dataset otomatis (Train / Validation / Test)
- Pilihan model:
  - CNN (from scratch)
  - ResNet50 (pretrained ImageNet)
  - VGG16 (pretrained ImageNet)
- Visualisasi performa model:
  - Accuracy & Loss
  - Evaluasi Test Set
- Antarmuka interaktif berbasis **Streamlit**

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
| Loss Function | Sparse Categorical Crossentropy |
| Optimizer | Adam (0.0001) |

---

## 📊 Hasil Evaluasi Model

> Ringkasan hasil evaluasi berdasarkan **Test Dataset (10%)**

### 🔍 Perbandingan Performa Model

| Model | Train Accuracy | Validation Accuracy | Test Accuracy | Test Loss |
|------|---------------|---------------------|--------------|-----------|
| CNN | **92.46%** | **95.20%** | **95.63%** | **0.1274** |
| ResNet50 | **53.90%** | **52.06%** | **50.60%** | **1.3762** |
| VGG16 | **97.57%** | **98.32%** | **98.04%** | **0.0660** |

---

## 📈 Visualisasi
Aplikasi menampilkan grafik:
- **Accuracy vs Epoch**
- **Loss vs Epoch**

Grafik dihasilkan otomatis setelah proses training selesai.

---

## 🧪 Evaluasi Model
Evaluasi dilakukan menggunakan **test dataset** yang belum pernah digunakan pada tahap training maupun validation.

Output evaluasi:
- Test Accuracy
- Test Loss

---

## 📦 Struktur Dataset
```text
dataset/
├── Apple___Apple_scab/
│   ├── img1.jpg
│   └── img2.jpg
├── Blueberry___healthy/
│   ├── img1.jpg
│   └── img2.jpg
