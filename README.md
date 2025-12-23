# 🌿 Klasifikasi Daun Tanaman Menggunakan Deep Learning  
### CNN • ResNet50 • VGG16

Project ini merupakan implementasi **klasifikasi citra daun tanaman** menggunakan pendekatan  
**Deep Learning** berbasis **Convolutional Neural Network (CNN)** dan **Transfer Learning**.

Aplikasi dikembangkan menggunakan **TensorFlow** sebagai framework deep learning dan  
**Streamlit** sebagai antarmuka website interaktif.

Project ini bertujuan untuk:
- Membandingkan performa **CNN from scratch** dan **model pretrained**
- Menganalisis pengaruh arsitektur model terhadap akurasi klasifikasi citra
- Menyediakan sistem klasifikasi berbasis website yang mudah digunakan

---

## 📌 Deskripsi Proyek

Klasifikasi daun tanaman merupakan salah satu penerapan **Computer Vision** di bidang pertanian,  
khususnya untuk:
- Identifikasi jenis tanaman
- Deteksi penyakit daun
- Monitoring kondisi tanaman secara otomatis

Dalam project ini digunakan **3 model deep learning**, yaitu:
1. Custom CNN
2. ResNet50
3. VGG16  

Model dilatih dan dievaluasi menggunakan dataset citra daun tanaman, kemudian dibandingkan berdasarkan performanya.

---

## 📂 Deskripsi Dataset

- 📌 **Sumber Dataset:** Kaggle  
- 📊 **Jumlah Data:** > 6.000 citra
- 🖼️ **Format File:** JPG / PNG
- 🌱 **Objek:** Daun tanaman
- 🏷️ **Label:** Folder-based (setiap folder merepresentasikan 1 kelas)
- 📐 **Ukuran Input Model:** 224 × 224 pixel (RGB)

### 📊 Pembagian Dataset

Dataset dibagi secara otomatis menggunakan TensorFlow dengan rasio:

| Dataset | Persentase |
|-------|------------|
| Training | 70% |
| Validation | 20% |
| Testing | 10% |

Pembagian ini bertujuan agar:
- Model belajar dari data **training**
- Parameter disesuaikan menggunakan **validation**
- Evaluasi dilakukan secara objektif menggunakan **test set**

---

## 🔄 Preprocessing Data

Tahapan preprocessing citra meliputi:

1. **Resize gambar** ke ukuran 224 × 224 pixel  
2. **Normalisasi pixel** (0–255 → 0–1)  
3. **Batching & Prefetching** untuk efisiensi training  
4. **Label encoding otomatis** berdasarkan nama folder  

📸 **Screenshot Preprocessing**  
> Tambahkan screenshot hasil preprocessing di bawah ini:

assets/Screenshot 2025-12-23 185554.png
assets/preprocessing_normalization.png




---

## 🧠 Penjelasan Model Deep Learning

### 1️⃣ Convolutional Neural Network (CNN)

#### 📖 Sejarah Singkat
CNN pertama kali diperkenalkan pada tahun **1998** oleh **Yann LeCun** melalui arsitektur **LeNet-5**,  
yang digunakan untuk pengenalan tulisan tangan.

CNN dirancang khusus untuk:
- Mengekstraksi fitur spasial dari citra
- Mengurangi jumlah parameter dibanding neural network biasa

#### 🧩 Arsitektur CNN pada Project
- Conv2D: 32 → 64 → 128 filter
- MaxPooling setiap layer
- Fully Connected (Dense 256)
- Dropout 0.5 untuk mencegah overfitting
- Optimizer: Adam (lr = 0.0001)

✅ **Kelebihan:**
- Fleksibel
- Cocok untuk dataset menengah
- Mudah dikustomisasi

❌ **Kekurangan:**
- Butuh tuning manual
- Kurang optimal untuk dataset sangat besar

---

### 2️⃣ ResNet50

#### 📖 Sejarah Singkat
ResNet diperkenalkan oleh **Microsoft Research** pada tahun **2015** dan memenangkan  
kompetisi **ImageNet ILSVRC**.

Inovasi utama ResNet adalah:
> **Residual Connection (Skip Connection)**

Teknik ini memungkinkan jaringan yang sangat dalam  
(hingga ratusan layer) tetap stabil saat training.

#### 🧩 Konfigurasi ResNet50
- Pretrained: ImageNet
- `include_top=False`
- Global Average Pooling
- Dense 512
- Base model **dibekukan (freeze)**

✅ **Kelebihan:**
- Stabil untuk jaringan dalam
- Baik untuk dataset besar

❌ **Kekurangan:**
- Perlu fine-tuning agar optimal
- Kurang cocok jika data sangat berbeda dari ImageNet

---

### 3️⃣ VGG16

#### 📖 Sejarah Singkat
VGG16 dikembangkan oleh **Visual Geometry Group (Oxford)** pada tahun **2014**  
dan terkenal dengan arsitekturnya yang **sederhana namun efektif**.

VGG menggunakan:
- Kernel kecil (3×3)
- Arsitektur berlapis yang konsisten

#### 🧩 Konfigurasi VGG16
- Pretrained: ImageNet
- `include_top=False`
- Global Average Pooling
- Dense 64
- 10 layer terakhir **trainable**

✅ **Kelebihan:**
- Fitur sangat kuat
- Cocok untuk transfer learning

❌ **Kekurangan:**
- Parameter besar
- Konsumsi memori tinggi

---

## ⚙️ Konfigurasi Training

| Parameter | Nilai |
|--------|------|
| Image Size | 224 × 224 |
| Channels | RGB |
| Epoch | 1 – 30 |
| Batch Size | 8, 16, 32 |
| Loss Function | Sparse Categorical Crossentropy |
| Optimizer | Adam (0.0001) |

---

## 📊 Hasil Evaluasi Model

### 🔍 Tabel Evaluasi

| Model | Train Acc | Val Acc | Test Acc | Test Loss |
|-----|----------|---------|----------|-----------|
| CNN | 92.46% | 95.20% | 95.63% | 0.1274 |
| ResNet50 | 53.90% | 52.06% | 50.60% | 1.3762 |
| VGG16 | **97.57%** | **98.32%** | **98.04%** | **0.0660** |

---

## 📈 Analisis Perbandingan Model

### 🔹 CNN
Akurasi tinggi karena:
- Dataset cukup representatif
- Arsitektur CNN disesuaikan dengan kebutuhan dataset

Namun berpotensi overfitting jika dataset lebih kecil.

---

### 🔹 ResNet50
Akurasi rendah karena:
- Base model terlalu umum
- Layer dibekukan sehingga tidak beradaptasi optimal
- Dataset daun berbeda signifikan dari ImageNet

---

### 🔹 VGG16
Performa terbaik karena:
- Transfer learning efektif
- Fine-tuning layer akhir
- Ekstraksi fitur sangat kuat untuk citra daun

---




