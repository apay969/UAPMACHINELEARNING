# 🌿 Klasifikasi Daun Tanaman Menggunakan Deep Learning  
### CNN • ResNet50 • VGG16
<p align="center">
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white" />
  </a>
  <a href="https://www.tensorflow.org/install">
    <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white" />
  </a>
  <a href="https://streamlit.io/">
    <img src="https://img.shields.io/badge/Streamlit-Web_App-red?logo=streamlit&logoColor=white" />
  </a>
  <a href="https://keras.io/">
    <img src="https://img.shields.io/badge/Keras-Deep_Learning-D00000?logo=keras&logoColor=white" />
  </a>
  <a href="https://numpy.org/">
    <img src="https://img.shields.io/badge/NumPy-Scientific_Computing-013243?logo=numpy&logoColor=white" />
  </a>
  <a href="https://matplotlib.org/">
    <img src="https://img.shields.io/badge/Matplotlib-Visualization-11557C" />
  </a>
</p>


<img width="987" height="490" alt="Screenshot 2025-12-23 190231" src="https://github.com/user-attachments/assets/d1ec6511-43ed-47c9-9e1a-94e1004b689c" />


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
> 
<img width="647" height="611" alt="Screenshot 2025-12-23 185554" src="https://github.com/user-attachments/assets/3c7ac5f5-2d5e-4dbe-8f1c-db705b11d8b7" />
<img width="820" height="720" alt="Screenshot 2025-12-23 185603" src="https://github.com/user-attachments/assets/9af9a4c8-4544-4807-bbff-598d56ab0e97" />

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


## 🖥️ Penjelasan Aplikasi (Website)

Aplikasi **UAP – Klasifikasi Citra Daun Tanaman** merupakan sebuah **website interaktif berbasis Streamlit** yang dirancang untuk mempermudah proses **training, evaluasi, dan analisis performa model deep learning** dalam tugas UAP.

Aplikasi ini memungkinkan pengguna untuk melakukan eksperimen klasifikasi citra **tanpa perlu menulis kode secara langsung**, sehingga cocok digunakan sebagai media pembelajaran dan evaluasi algoritma deep learning.

Dengan tampilan antarmuka yang sederhana, informatif, dan responsif, aplikasi ini mampu menampilkan seluruh proses mulai dari **unggah dataset hingga hasil evaluasi model** secara terstruktur.

---

## 🔄 Alur Kerja Aplikasi

Alur kerja aplikasi dapat dijelaskan sebagai berikut:

1. **Upload Dataset**
   - Pengguna mengunggah dataset citra dalam format **ZIP**
   - Dataset harus memiliki struktur folder berdasarkan kelas
   - Aplikasi akan membaca dan memvalidasi dataset secara otomatis

   📸 *Screenshot Upload Dataset*  
   <img width="163" height="148" alt="Screenshot 2025-12-23 191002" src="https://github.com/user-attachments/assets/d90ea356-89b7-4c5b-98f3-02c18fe958ae" />

2. **Konfigurasi Dataset**
   - Aplikasi menghitung jumlah kelas secara otomatis
   - Dataset dibagi menjadi:
     - Training: **70%**
     - Validation: **20%**
     - Testing: **10%**

   📸 *Screenshot Konfigurasi Dataset*  
  <img width="832" height="281" alt="Screenshot 2025-12-24 110834" src="https://github.com/user-attachments/assets/f07dce9b-91e6-492a-bb8d-f06f6ec7aa3f" />


3. **Pemilihan Model Deep Learning**
   Pengguna dapat memilih salah satu dari tiga model berikut:
   - **CNN (Custom)** – model yang dibangun dari awal
   - **ResNet50** – pretrained model dengan residual connection
   - **VGG16** – pretrained model dengan arsitektur konvolusi berlapis

   📸 *Screenshot Pemilihan Model*  
   <img width="172" height="165" alt="Screenshot 2025-12-24 110930" src="https://github.com/user-attachments/assets/06f2cf9d-b0af-4323-8fca-8f49c88c7864" />


4. **Pengaturan Parameter Training**
   - Pengguna dapat mengatur:
     - Jumlah epoch
     - Batch size
   - Parameter ini mempengaruhi kecepatan dan performa training

   📸 *Screenshot Pengaturan Training*  
   <img width="173" height="175" alt="Screenshot 2025-12-24 111001" src="https://github.com/user-attachments/assets/ab9d3c24-4bbb-410e-abe2-9227d8861763" />


5. **Proses Training Model**
   - Setelah konfigurasi selesai, pengguna dapat menjalankan training
   - Aplikasi menampilkan status training secara real-time
   - Setelah training selesai, akan muncul notifikasi keberhasilan

   📸 *Screenshot Proses Training*  
   <img width="797" height="288" alt="Screenshot 2025-12-24 111042" src="https://github.com/user-attachments/assets/4d434d3b-730a-412f-82bc-b3f4db0d91fd" />


---

## 📊 Visualisasi dan Evaluasi Model

Setelah proses training selesai, aplikasi menyediakan tiga tab utama untuk analisis hasil:

### 📌 Ringkasan
Menampilkan ringkasan performa model berupa:
- Train Accuracy
- Validation Accuracy

📸 *Screenshot Ringkasan Hasil*  
<img width="1920" height="1080" alt="Screenshot 2025-12-23 160830" src="https://github.com/user-attachments/assets/a264b9d0-b837-411f-887f-fda1f9be0b82" />
<img width="1920" height="1080" alt="Screenshot 2025-12-23 163657" src="https://github.com/user-attachments/assets/502938a7-7159-4992-93f3-63843f96920e" />
<img width="1920" height="1080" alt="Screenshot 2025-12-23 172908" src="https://github.com/user-attachments/assets/6e0fe2c6-8cb4-448a-a7dd-65241366993d" />


---

### 📈 Grafik
Menampilkan grafik:
- **Accuracy vs Epoch**
- **Loss vs Epoch**

Grafik ini membantu pengguna dalam menganalisis proses pembelajaran model dan potensi overfitting atau underfitting.

📸 *Screenshot Grafik Training*  
<img width="1920" height="1080" alt="Screenshot 2025-12-23 160843" src="https://github.com/user-attachments/assets/bc766065-c6c5-4c83-b9cd-706efa1e025d" />
<img width="1920" height="1080" alt="Screenshot 2025-12-23 163705" src="https://github.com/user-attachments/assets/99c2cc0f-b76b-4979-976b-a434ef5d73b1" />
<img width="1920" height="1080" alt="Screenshot 2025-12-23 172917" src="https://github.com/user-attachments/assets/a79de4f3-380c-4104-8b69-a07ce05e4d58" />


---

### 🧪 Evaluasi
Menampilkan hasil evaluasi akhir model pada **test dataset**, berupa:
- Test Accuracy
- Test Loss

Hasil evaluasi ini digunakan sebagai dasar perbandingan performa antar model.

📸 *Screenshot Evaluasi Model*  
<img width="1920" height="1080" alt="Screenshot 2025-12-23 160856" src="https://github.com/user-attachments/assets/721dea34-0f43-4a40-98f0-238cd60002c8" />
<img width="1920" height="1080" alt="Screenshot 2025-12-23 163715" src="https://github.com/user-attachments/assets/a41f7365-4392-43e8-bf69-b98bb1a11df6" />
<img width="1920" height="1080" alt="Screenshot 2025-12-23 172932" src="https://github.com/user-attachments/assets/b7cac88e-69fb-48c2-b17f-0b1ccfeef112" />


---

## ✨ Keunggulan Aplikasi

Beberapa keunggulan dari aplikasi ini antara lain:
- Antarmuka interaktif dan mudah digunakan
- Mendukung eksperimen berbagai model deep learning
- Visualisasi hasil training secara real-time
- Cocok untuk keperluan akademik dan pembelajaran

Aplikasi ini diharapkan dapat membantu pengguna memahami **alur kerja deep learning untuk klasifikasi citra** secara praktis dan aplikatif.


---
---

## 📬 Kontak, Kritik & Saran

Jika Anda memiliki **kritik, saran, atau pertanyaan** terkait aplikasi maupun project ini,  
silakan menghubungi saya melalui:

📱 **WhatsApp:**  
👉 [Klik untuk chat via WhatsApp](https://wa.me/6281547190242)


---

Terima kasih telah mengunjungi dan menggunakan project ini 🙏  
Semoga aplikasi ini dapat memberikan manfaat dalam pembelajaran **Deep Learning & Computer Vision**.




