# Laporan Proyek Machine Learning - Syifa Azzahro

## Domain Proyek

Industri telekomunikasi merupakan sektor yang sangat kompetitif, di mana perusahaan berlomba-lomba menjaga loyalitas pelanggan mereka. Salah satu tantangan utama adalah tingginya tingkat customer churn, yaitu kondisi ketika pelanggan berhenti menggunakan layanan. Tingginya churn dapat berdampak langsung pada penurunan pendapatan dan meningkatnya biaya akuisisi pelanggan baru.

Mengetahui lebih awal pelanggan mana yang kemungkinan besar akan churn memungkinkan perusahaan untuk melakukan tindakan preventif seperti memberikan penawaran khusus, meningkatkan kualitas layanan, atau melakukan pendekatan yang lebih personal. Strategi ini tidak hanya meningkatkan tingkat retensi, tetapi juga membantu dalam menghemat biaya operasional perusahaan.

Seiring berkembangnya teknologi, pendekatan berbasis machine learning terbukti efektif dalam memprediksi kemungkinan churn. Model prediktif yang dibangun dari data historis pelanggan (misalnya pola penggunaan, data demografis, dan riwayat keluhan) memungkinkan perusahaan mengambil keputusan berbasis data yang lebih tepat dan efisien.

Dengan solusi ini, perusahaan telekomunikasi dapat meningkatkan kepuasan pelanggan, memperkuat daya saing, dan mempertahankan stabilitas bisnis jangka panjang.

Referensi:
[Analysis of customer churn prediction using machine learning and deep learning algorithms](https://doi.org/10.53730/ijhs.v6nS1.7861)
[Implementing machine learning techniques for customer retention and churn prediction in telecommunications](https://doi.org/10.51594/csitrj.v5i8.1489)

## Business Understanding

Dalam industri telekomunikasi, kehilangan pelanggan (churn) merupakan masalah yang serius karena dapat berdampak langsung pada pendapatan dan pertumbuhan perusahaan. Untuk mengatasinya, diperlukan pendekatan prediktif yang mampu mengidentifikasi pelanggan yang berisiko tinggi churn serta memahami faktor-faktor utama yang memengaruhi keputusan mereka.

### Problem Statements

1. Bagaimana mengidentifikasi pelanggan yang memiliki potensi tinggi untuk churn?
Perusahaan perlu mengetahui pelanggan mana yang cenderung berhenti menggunakan layanan berdasarkan pola perilaku dan atribut mereka.

2. Fitur apa saja yang paling berpengaruh terhadap perilaku churn pelanggan?
Penting untuk memahami variabel-variabel mana yang menjadi indikator kuat terhadap kemungkinan pelanggan berhenti, seperti durasi berlangganan, jumlah keluhan, atau jenis layanan yang digunakan.


### Goals

1. Membangun model prediksi churn yang mampu mengklasifikasikan pelanggan berdasarkan risiko mereka untuk churn.
2. Mengidentifikasi fitur-fitur utama yang paling berkontribusi terhadap churn untuk membantu pengambilan keputusan strategis oleh manajemen.


### Solution statements
1. Mengembangkan model prediktif menggunakan algoritma Machine Learning seperti Random Forest, XGBoost, dan Logistic Regression untuk membandingkan performa dan memilih model terbaik berdasarkan metrik evaluasi seperti accuracy, precision, recall, dan AUC.

2. Melakukan feature importance analysis untuk mengetahui variabel-variabel paling berpengaruh terhadap perilaku churn.

3. Meningkatkan performa model baseline dengan teknik hyperparameter tuning dan oversampling (SMOTE) untuk menangani ketidakseimbangan kelas dalam dataset.

4. Mengimplementasikan pipeline evaluasi model yang sistematis guna memastikan model dapat diandalkan sebelum digunakan dalam pengambilan keputusan bisnis.

## Data Understanding
Dataset yang digunakan adalah Telcom Customer Churn dari Kaggle. Dataset ini berisi informasi pelanggan seperti jenis kontrak, lamanya berlangganan, hingga apakah mereka memiliki layanan tambahan. Dataset berisi 7043 baris.
[WA_Fn-UseC_-Telco-Customer-Churn](https://www.kaggle.com/datasets/palashfendarkar/wa-fnusec-telcocustomerchurn).


### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- customerID – ID unik pelanggan

- gender – Jenis kelamin (Male or Female)

- SeniorCitizen – Status lansia (1= Yes, 0 = No)

- Partner – Apakah pelanggan mempunyai pasangan (Yes/No)

- Dependents – Apakah pelanggan mempunyai tanggungan (Yes/No)

- tenure – Lama pelanggan berlanggan 

- PhoneService – Apakah pelanggan memiliki layanan telepon (Yes/No)

- MultipleLines – Apakah pelanggan memiliki banyak jalur untuk layanan telepon (Yes/No/No phone service)

- InternetService – Jenis layanan internet (DSL, Fiber optic, No)

- OnlineSecurity – Apakah keamanan online diaktifkan (Yes/No/No internet service)

- OnlineBackup – Apakah pencadangan online diaktifkan (Yes/No/No internet service)

- DeviceProtection – Apakah perlindungan perangkat diaktifkan (Yes/No/No internet service)

- TechSupport – Apakah dukungan teknis diaktifkan (Yes/No/No internet service)

- StreamingTV – Apakah pelanggan memiliki layanan TV streaming (Yes/No/No internet service)

- StreamingMovies – Apakah pelanggan memiliki layanan streaming film (Yes/No/No internet service)

- Contract – Jenis kontrak pelanggan (Month-to-month, One year, Two year)

- PaperlessBilling – Apakah pelanggan memiliki penagihan tanpa kertas (Yes/No)

- PaymentMethod – Metode pembayaran(Electronic check, Mailed check, etc.)

- MonthlyCharges – Jumlah bulanan yang dibebankan kepada pelanggan

- TotalCharges – Jumlah total yang dibebankan kepada pelanggan

- Churn – Variabel target (Yes/No) yang menunjukkan apakah pelanggan pergi

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation
Tahapan data preparation dilakukan untuk memastikan bahwa data siap digunakan oleh model machine learning dan menghasilkan performa yang optimal. Beberapa langkah yang dilakukan pada tahap ini meliputi:

1. Mengubah Tipe Data
Beberapa kolom pada dataset memiliki tipe data yang tidak sesuai, seperti angka yang terbaca sebagai objek. Tipe data diubah agar lebih mudah diinterpretasikan dan diproses, misalnya mengubah tipe kolom “TotalCharges” dari string menjadi numerik. Ini diperlukan untuk memastikan semua data berada dalam format yang dapat diproses oleh algoritma.

2. Memisahkan Kolom Kategorikal dan Numerik
Pemisahan ini dilakukan untuk mempermudah proses transformasi data sesuai dengan jenisnya. Kolom kategorikal akan diproses dengan encoding, sedangkan kolom numerik akan distandarisasi. Tahapan ini diperlukan untuk mnghindari kesalahan pemrosesan karena perbedaan tipe data.

3. Encoding dan Standarisasi

- One-Hot Encoding diterapkan pada kolom kategorikal agar model dapat memahami data kategorikal dalam bentuk numerik tanpa mengasumsikan hubungan ordinal.

- StandardScaler digunakan pada kolom numerik untuk menstandarisasi nilai agar berada dalam skala yang sama, sehingga mempercepat konvergensi model dan meningkatkan akurasi.

Tahapan ini sangat penting untuk meningkatkan kemampuan generalisasi model melalui transformasi dan penyeimbangan kelas.

4. Menangani Ketidakseimbangan Kelas (Imbalanced Data)
Dataset menunjukkan distribusi kelas yang tidak seimbang antara pelanggan yang churn dan tidak churn. Oleh karena itu, teknik SMOTE (Synthetic Minority Over-sampling Technique) digunakan untuk melakukan oversampling pada kelas minoritas. Hal ini bertujuan agar model tidak bias terhadap kelas mayoritas dan dapat belajar dengan seimbang dari kedua kelas.

5. Membagi Data Train dan Test
Dataset dibagi menjadi data latih dan data uji dengan proporsi yang umum digunakan (misalnya 80:20) agar model dapat dievaluasi secara objektif terhadap data yang belum pernah dilihat sebelumnya. 


## Modeling
Pada tahap ini, dilakukan pemodelan menggunakan beberapa algoritma machine learning untuk memprediksi customer churn. Proses pemodelan dimulai dari algoritma yang sederhana sebagai baseline, kemudian dilanjutkan dengan model yang lebih kompleks untuk mengeksplorasi performa yang lebih optimal.

1. Logistic Regression (Baseline Model)
Model Logistic Regression digunakan sebagai baseline karena memiliki karakteristik berikut:

- Sederhana dan mudah diinterpretasikan.

- Cepat dalam proses pelatihan.

- Memberikan estimasi probabilitas churn yang berguna untuk pengambilan keputusan bisnis.

Model dilatih menggunakan parameter default tanpa tuning terlebih dahulu. Tujuannya adalah untuk mendapatkan baseline performa awal yang nantinya akan dibandingkan dengan model lain. Evaluasi dilakukan menggunakan metrik accuracy, precision, recall, dan AUC.

Kelebihan:

- Interpretasi koefisien fitur mudah dilakukan.

- Cepat dan efisien pada dataset kecil hingga menengah.

Kekurangan:

- Asumsi hubungan linear antara fitur dan target seringkali tidak sesuai untuk data kompleks.

- Performa dapat menurun jika data memiliki korelasi antar fitur atau relasi non-linear.

2. Random Forest (Model Kompleks)
Model Random Forest digunakan untuk mengeksplorasi model yang lebih kompleks. Algoritma ini merupakan ensemble model berbasis decision tree yang mampu menangkap hubungan non-linear dan bekerja baik terhadap dataset dengan banyak fitur.

Parameter awal yang digunakan:

- n_estimators = 100: jumlah pohon dalam model.

- max_depth = None: tidak ada batasan kedalaman pohon, sehingga pohon dapat tumbuh penuh.

- random_state = 42: menjaga konsistensi hasil.

- class_weight = 'balanced': untuk menangani ketidakseimbangan kelas churn dan non-churn.

Setelah melatih model awal, dilakukan proses hyperparameter tuning menggunakan GridSearchCV untuk meningkatkan performa model. Parameter yang dieksplorasi dalam tuning antara lain:

n_estimators: jumlah pohon.

max_depth: kedalaman maksimum pohon.

min_samples_split: jumlah minimal sampel untuk membagi node.

Kelebihan:

Dapat menangani data yang tidak teratur dan hubungan antar fitur yang kompleks.

- Tidak mudah overfitting karena menggunakan voting dari banyak pohon.

- Memberikan estimasi feature importance yang berguna dalam analisis bisnis.

Kekurangan:

- Waktu pelatihan dan prediksi lebih lama dibanding model sederhana.

- Interpretasi tidak sejelas Logistic Regression.

Pemilihan Model Terbaik
Berdasarkan hasil evaluasi pada data validasi menggunakan metrik seperti recall dan AUC, model Random Forest menunjukkan performa yang lebih baik dibanding Logistic Regression. Oleh karena itu, Random Forest dipilih sebagai model terbaik karena:

- Mampu menangkap kompleksitas data dengan lebih baik.

- Memiliki skor recall dan AUC yang lebih tinggi, yang sangat penting dalam konteks churn prediction, di mana mengidentifikasi pelanggan yang benar-benar akan churn lebih penting daripada hanya mencapai akurasi tinggi.


## Evaluation
Dalam proyek ini, evaluasi model dilakukan dengan menggunakan beberapa metrik evaluasi yang relevan untuk masalah klasifikasi churn, yaitu:

- Accuracy: Mengukur persentase prediksi yang benar dari seluruh data. Namun, karena dataset imbalance (jumlah pelanggan churn vs tidak churn tidak seimbang).

![alt text](Churn Analysis\asset\akurasi.png)

Misal diketahui confussion matrix seperti di bawah ini:
![alt text](asset\confusion_matrix.png)

maka akurasi model tersbut adalah (834+897) / (834+199+136+897) = 0.84.
Akurasi hanya cocok digunakan pada saat perbandingan jumlah label data sebenarnya relatif sama.

- Precision: Mengukur seberapa banyak dari pelanggan yang diprediksi akan churn, yang benar-benar churn. Berguna untuk menghindari false positive, yaitu salah mengira pelanggan tetap sebagai pelanggan churn.

![alt text](asset\presisi.png)

- Recall: Mengukur seberapa banyak dari pelanggan yang benar-benar churn, yang berhasil terdeteksi oleh model. Metode ini penting karena perusahaan ingin mengidentifikasi sebanyak mungkin pelanggan yang berisiko churn agar bisa segera mengambil tindakan.

- F1-Score: Harmonik rata-rata dari precision dan recall. Metrik ini cocok digunakan saat kita ingin menjaga keseimbangan antara false positive dan false negative.

- AUC (Area Under the ROC Curve): Metrik yang menunjukkan seberapa baik model dalam membedakan antara pelanggan yang churn dan tidak churn. Semakin tinggi nilai AUC (maksimal 1), semakin baik performa klasifikasi model.

📊 Hasil Evaluasi
Model Random Forest memberikan hasil sebagai berikut:

Metrik	Nilai
Accuracy	0.84
Precision	0.82
Recall	    0.87
F1-Score	0.84
AUC	        0.91

Nilai recall yang tinggi (0.87) menunjukkan bahwa model cukup efektif dalam mengidentifikasi pelanggan yang akan churn, sesuai dengan tujuan bisnis. Precision yang cukup baik juga menunjukkan bahwa model tidak terlalu banyak memberikan alarm palsu (false positive).

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

