# Laporan Proyek Machine Learning - Halim Sajidi

## Project Overview

Recommender systems telah menjadi bagian integral dari berbagai platform digital, seperti YouTube, Netflix, dan Amazon. Sistem ini bertujuan untuk memberikan saran produk atau konten yang relevan kepada pengguna berdasarkan preferensi mereka, yang dapat meningkatkan keterlibatan pengguna dan meningkatkan pendapatan platform. Dalam proyek ini, saya akan menggunakan dataset Book-Crossing yang mencakup data pengguna, buku, dan rating, untuk mengembangkan sistem rekomendasi buku.

**Rubrik/Kriteria Tambahan (Opsional)**:
**Mengapa proyek ini penting?** Dalam era informasi yang melimpah, pengguna sering kali kesulitan menemukan konten yang relevan tanpa bimbingan. Sistem rekomendasi yang baik dapat memecahkan masalah ini dengan menyaring data yang tidak relevan dan menyarankan item yang sesuai dengan preferensi individu, sehingga memberikan pengalaman pengguna yang lebih personal (Ricci _et al._ 2021).
  
  Format Referensi: [Recommender Systems: Techniques, Applications, and Challenges](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Recommender+Systems%3A+Techniques%2C+Applications%2C+and+Challenges&btnG=) 

## Business Understanding
### Problem Statements
- Pernyataan Masalah 1: Bagaimana cara merekomendasikan buku kepada pengguna berdasarkan preferensi yang ada dalam data rating?
- Pernyataan Masalah 2: Bagaimana meningkatkan akurasi rekomendasi agar pengguna mendapatkan buku yang paling relevan dengan minat mereka?

### Goals
- Tujuan 1: Mengembangkan sistem rekomendasi buku berbasis data rating pengguna.
- Tujuan 2: Mengoptimalkan model agar dapat memberikan rekomendasi dengan tingkat akurasi yang lebih tinggi.

**Rubrik/Kriteria Tambahan (Opsional)**:
### Solution statements
Untuk mencapai tujuan, saya akan menggunakan dua pendekatan berikut:
- **Collaborative Filtering:** Menggunakan informasi tentang rating pengguna terhadap buku untuk merekomendasikan buku kepada pengguna yang memiliki preferensi serupa.
- **Content-Based Filtering:** Menggunakan fitur konten buku seperti publisher untuk merekomendasikan buku berdasarkan kesamaan konten dengan buku yang sudah diberi rating oleh pengguna.

## Data Understanding
Dataset pada proyek ini adalah dataset books recommendation dataset yang dapat di akses melalui situs kaggle. [Kaggle dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset/data).

Variabel-variabel pada books recommendation dataset adalah sebagai berikut:
- Books: berisi data tentang buku, termasuk ISBN, Book-Title, Book-Author, Year-Of-Publication, dan Publisher. Data URL gambar buku juga disertakan dalam tiga ukuran (S, M, L). Dengan banyak data 271360.
- Ratings: berisi data rating dari pengguna untuk buku-buku tertentu dengan skala 1-10 untuk rating eksplisit dan 0 untuk rating implisit. Dengan banyak data 340556.
- Users: berisi informasi tentang pengguna seperti User-ID, Location, dan Age. Beberapa kolom memiliki nilai NULL. Dengan banyak data 278858.

**Contoh variabel:**
- User-ID: ID anonim pengguna.
- ISBN: Nomor identifikasi buku.
- Book-Rating: Rating eksplisit atau implisit yang diberikan oleh pengguna terhadap buku.

**Rubrik/Kriteria Tambahan (Opsional)**:

- Penulis Teratas Berdasarkan Jumlah Buku
![image](https://github.com/user-attachments/assets/ba403ac0-d668-441d-a9f7-77e591bccf30)

Berdasarkan gambar di atas, terdapat 10 author dengan jumlah buku terbanyak. Walaupun sebenarnya masih belum bisa disimpulkan karena dari label jumlah buku nilai tertingginya di angka 10 ribu, dengan demikian simpulan awalnya adalah masih banyak data duplikat yang nantinya perlu di lakukan perbaikkan.

- Distribusi Peringkat Buku
![image](https://github.com/user-attachments/assets/0a02048a-2c33-4fe4-8c0e-4396413c12ac)

Dapat dilihat berdasarkan grafik di atas, kebanyakkan user memberikan nilai rendah, banyak sekali user yang memberikan nilai 0 atau 1

- Tabel deskripsi dari dataset users

![image](https://github.com/user-attachments/assets/dec85631-8968-4e3b-9e21-cfdee240856d)

Dapat dilihat kolom Age atau umur memiliki data yang tidak masuk akal, nilai minimum adalah 0 dan maksimumnya adalah 244

## Data Preparation
Pada tahap Data Preparation, saya menerapkan beberapa langkah penting untuk mempersiapkan data sebelum dimodelkan, yaitu:
- **Mengatasi Missing Value:** Saya memeriksa data untuk melihat apakah terdapat nilai kosong. Untuk dataset yang saya miliki terdapat nilai NaN di beberapa kolom dan dilakukan drop data, hal ini dilakukan karena jumlah data terbilang banyak sehingga dipilih untuk drop data dibandingkan menggantinya dengan mean atau mode.
- **Membagi Data Menjadi Training dan Validation Set:** Data dibagi menjadi dua bagian, yaitu data training untuk melatih model dan data validasi untuk mengevaluasi performa model. Pembagian data ini dilakukan pada pengembangan model collaborative filtering. Proporsi pembagiannya sendiri sebesar 80% dan 20% untuk data train dan test. 
- **Menggabungkan Variabel**: Variabel-variabel yang memiliki hubungan dengan ID yang bersifat unik digabungkan untuk mendapatkan informasi yang lebih lengkap.
- **Mengurutkan Data:** Data diurutkan berdasarkan ISBN secara ascending untuk mempermudah proses analisis dan visualisasi.
 -**Mengatasi Duplikasi Data:** Data yang memiliki nilai atau isi yang sama diidentifikasi dan dihapus untuk memastikan tidak ada pengulangan yang bisa memengaruhi hasil analisis.
- **Konversi Data Menjadi List:** Beberapa data diubah menjadi format list agar lebih mudah dikelola pada proses selanjutnya.
- **Membuat Dictionary:** Saya membuat dictionary dari data yang ada untuk memudahkan pemetaan dan pencarian informasi.
- **Menggunakan TfidfVectorizer:** Teknik TfidfVectorizer digunakan untuk melakukan pembobotan teks, yang membantu dalam mengukur seberapa penting suatu kata dalam dokumen.
- **Melakukan Preprocessing:** Langkah preprocessing dilakukan untuk mengatasi masalah yang dapat mengganggu hasil analisis data, seperti penghapusan karakter khusus dan normalisasi teks.
- **Mapping Data:** Data dipetakan untuk menyesuaikan antara variabel dan ID unik, sehingga informasi yang relevan dapat lebih mudah diakses dalam proses selanjutnya.

**Rubrik/Kriteria Tambahan (Opsional)**: 
Terdapat beberapa concern utama pada tahap data preparation ini. Pada dataset yang digunakan terdapat missing value dari dataset dan missing value ini pun muncul ketika menggabungkan dua dataset berdasarkan ID, sehingga perlu dilakukan drop rows, kenapa dihapus? Karena proporsi missing valuenya sendiri kurang dari 10%. Di samping itu dilakukan pula pengolahan data untuk data-data typo atau tidak sesuai penulisan misalnya pada penulisan nama author, publisher, dan nama buku. Di samping  itu dilakukan pula penghapusan tanda titik (.), koma (,), dan petik (') karena akan menggangu pada saat tokenisasi. Terakhir, hal kecil tapi sangat penting yaitu pergantian nama kolom, contohnya terdapat kolom Book-Title, hal ini cukup ambigu sehingga dilakukan rename menjadi book_title, sehingga pada saat peroses pemrograman tidak terjadi kesalahartian dari sebuah kolom.

## Modeling
### Algoritma 1: Content-Based Filtering
Pendekatan ini menggunakan informasi konten buku seperti genre atau penulis untuk memberikan rekomendasi. Model ini cocok untuk pengguna baru yang belum memberikan rating banyak buku.

**Kelebihan:**

Bisa memberikan rekomendasi tanpa data dari pengguna lain.

**Kekurangan:**

Rentan terhadap masalah overspecialization, di mana pengguna hanya disarankan item yang sangat mirip dengan yang sudah dilihat.

**Output dari model ini adalah top-N recommendation berupa daftar buku yang paling relevan untuk setiap pengguna.**
Berikut adalah outputnya:

![image](https://github.com/user-attachments/assets/18ccc58c-c976-4f17-81a3-f5ee1e5c2d07)

Tabel di atas adalah user yang menyukai buku berjudul aftermath yang berasal dari publisher fireside.

![image](https://github.com/user-attachments/assets/e24acd5e-01ec-4606-b13e-055961e2ee8f)

Berdasarkan hasil rekomendasi di atas, judul buku aftermah merupakan buku yang diterbitkan oleh publisher Fireside. Kemudian dari 5 rekomendasi yang diberikan, 4 dari 5 memiliki kesesuaian publisher


### Algoritma 2: Collaborative Filtering
Collaborative filtering memanfaatkan informasi dari pengguna lain dengan preferensi yang sama untuk memberikan rekomendasi. Saya menggunakan algoritma Singular Value Decomposition (SVD) untuk mendekomposisi matriks rating.

**Kelebihan:**

Mampu memprediksi buku yang relevan meskipun tidak ada informasi konten buku.

**Kekurangan:**

Membutuhkan banyak data rating dari pengguna untuk memberikan rekomendasi yang akurat.

**Output dari model ini adalah top-N recommendation berupa daftar buku yang paling tinggi ratingnya.**
Berikut adalah outputnya:

![image](https://github.com/user-attachments/assets/ad7f943a-6398-4262-99ca-c19ef43adc49)

hasil di atas adalah rekomendasi untuk user dengan id 105979. Dari output tersebut, kita dapat membandingkan antara books with high ratings from user dan Top 10 books recommendation untuk user.

## Evaluation
### 1. Evaluasi Content Based Filtering
Precision mengukur seberapa banyak dari semua prediksi positif yang benar-benar positif. Dengan kata lain, ini mengevaluasi ketepatan model dalam memprediksi kelas positif, dengan mengabaikan prediksi negatif yang salah.

![image](https://github.com/user-attachments/assets/b67f4366-7cb2-482e-a0e6-b8c774cc0f57)

- True Positives (TP): Jumlah kasus positif yang diprediksi benar oleh model.
- False Positives (FP): Jumlah kasus negatif yang diprediksi sebagai positif oleh model (disebut juga sebagai "false alarms").

**Berdasarkan hasil yang didapatkan**

TP = 4

FP = 1

Precision = 4/5 = 0.8

Artinya, precision model adalah 80%. Dari semua prediksi yang dikatakan positif oleh model, 80% adalah benar-benar positif.

Content-Based Filtering memberikan rekomendasi berdasarkan kesamaan konten buku, seperti penulis atau publisher. Pendekatan ini efektif untuk pengguna baru yang memiliki sedikit data rating, namun rentan terhadap masalah overspecialization. Dari hasil evaluasi, pendekatan ini memiliki precision sebesar 80%, yang menunjukkan ketepatan yang cukup baik dalam merekomendasikan buku sesuai preferensi pengguna.

### 2. Evaluasi Collaborative Filtering
Untuk mengevaluasi performa model, saya menggunakan metrik Root Mean Square Error (RMSE) untuk mengukur seberapa dekat prediksi rating dengan rating sebenarnya. RMSE digunakan karena model sistem rekomendasi biasanya prediksi rating kuantitatif.

![image](https://github.com/user-attachments/assets/40828e65-255c-430f-84de-3fd8efc14fbc)

Di mana:
- 𝑦𝑖 adalah rating aktual pengguna.
- y^ i adalah rating yang diprediksi oleh model.
- n adalah jumlah total observasi.

**hasil dari evaluasi model**

![image](https://github.com/user-attachments/assets/e1c3fe72-6d1c-4c21-b3d4-c159c5064e87)

Collaborative Filtering menggunakan informasi dari pengguna lain yang memiliki preferensi serupa untuk memberikan rekomendasi. Dengan menerapkan Singular Value Decomposition (SVD), model ini mampu memberikan prediksi rating yang cukup akurat, meskipun membutuhkan banyak data rating pengguna untuk memberikan hasil yang optimal. Evaluasi menggunakan Root Mean Square Error (RMSE) menunjukkan bahwa model ini mampu mendekati rating sebenarnya, meskipun ada ruang untuk peningkatan lebih lanjut.

### Kesimpulan 
**1. Untuk merekomendasikan buku kepada pengguna berdasarkan preferensi yang ada dalam data rating, dua pendekatan utama dapat digunakan:**

- Collaborative Filtering: Pendekatan ini merekomendasikan buku berdasarkan rating yang diberikan oleh pengguna lain yang memiliki preferensi serupa. Algoritma seperti Singular Value Decomposition (SVD) digunakan untuk menganalisis pola rating dari sejumlah besar pengguna dan mendeteksi kemiripan di antara mereka. Hasilnya, pengguna akan mendapatkan rekomendasi buku yang disukai oleh orang lain dengan preferensi yang mirip.
- Content-Based Filtering: Pendekatan ini fokus pada atribut buku seperti penulis, genre, atau penerbit untuk merekomendasikan buku yang mirip dengan buku-buku yang sebelumnya disukai oleh pengguna. Jika seorang pengguna memberikan rating tinggi pada buku dari penulis atau genre tertentu, sistem akan menyarankan buku lain yang memiliki karakteristik serupa.

**2. Untuk meningkatkan akurasi rekomendasi agar pengguna mendapatkan buku yang paling relevan dengan minat mereka, beberapa strategi dapat diterapkan:**
- Hybrid Recommender System: Menggabungkan pendekatan Collaborative Filtering dan Content-Based Filtering dapat meningkatkan akurasi dengan mengatasi kelemahan masing-masing metode. Collaborative Filtering membantu menemukan pola kesamaan antar pengguna, sementara Content-Based Filtering mengatasi masalah cold start (ketika data pengguna sangat sedikit).
- Penerapan Feedback Loop: Menggunakan feedback dari pengguna secara terus menerus (misalnya melalui fitur like/dislike) dapat meningkatkan relevansi rekomendasi dengan memperbarui preferensi pengguna secara real-time. Walapun belum dilakukan dalam proyek ini, tetapi saran ini bisa digunakan untuk pengembangan kedepannya.

**---Ini adalah bagian akhir laporan---**
