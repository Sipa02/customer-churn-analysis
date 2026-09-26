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
1. Mengembangkan model prediktif menggunakan algoritma Machine Learning seperti Random Forest dan Logistic Regression untuk membandingkan performa dan memilih model terbaik berdasarkan metrik evaluasi seperti accuracy, precision, recall, dan f1-score.



## Data Understanding
Dataset yang digunakan adalah [Telcom Customer Churn](https://www.kaggle.com/datasets/palashfendarkar/wa-fnusec-telcocustomerchurn) dari Kaggle. Dataset ini terdiri dari 7043 baris dan 21 kolom. Berdasarkan eksplorasi awal, tidak ditemukan missing value maupun data duplikat. Namun, terdapat indikasi outlier pada kolom tenure, yang perlu diperhatikan dalam proses analisis dan pemodelan selanjutnya.


### Variabel-variabel pada Telcom Customer Churn dataset adalah sebagai berikut:
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

### EDA

Sebelum masuk ke tahap pemodelan, dilakukan eksplorasi data secara menyeluruh untuk memahami karakteristik dan pola dalam data. Tahapan ini penting untuk mengidentifikasi potensi masalah dalam data, hubungan antar fitur, dan insight awal yang berguna dalam proses modeling.

1. **Mengubah Tipe Data**

    Beberapa kolom pada dataset memiliki tipe data yang tidak sesuai, seperti angka yang terbaca sebagai objek dan sebaliknya. Tipe data diubah agar lebih mudah diinterpretasikan dan diproses, misalnya mengubah tipe kolom “TotalCharges” dari string menjadi numerik dan mengubah tipe kolom "SeniorCitizen" dari numerik menjadi string. Ini diperlukan untuk memastikan semua data berada dalam format yang dapat diproses oleh algoritma. 
    Mengubah representasi variabel Churn dari Yes dan No menjadi 1 dan 0 menggunakan binary mapping. Tujuannya agar lebih mudah dibaca.

2. **Menangani Missing Value**

    Ditemukan sebanyak 11 baris data yang mengandung nilai hilang (missing value) setelah mengubah tipe data pada kolom TotalCharges dari object menjadi float. Karena jumlahnya relatif sangat kecil dan tidak signifikan terhadap keseluruhan data, baris-baris tersebut dihapus Langkah ini dilakukan untuk memastikan kualitas data tetap terjaga dan siap digunakan pada tahap pemodelan selanjutnya.

3. **Memisahkan Kolom Kategorikal dan Numerik**

    Pemisahan ini dilakukan untuk mempermudah proses transformasi data sesuai dengan jenisnya. Kolom kategorikal akan diproses dengan encoding, sedangkan kolom numerik akan distandarisasi. Tahapan ini diperlukan untuk mnghindari kesalahan pemrosesan karena perbedaan tipe data.

4. **Univariate dan Multivariate Analysis terhadap Fitur Churn**

    Analisis ini dilakukan dengan memvisualisasikan seluruh fitur baik kategorikal maupun numerikal untuk memahami distribusi data dan hubungan awal antar fitur terhadap target Churn. Tahapan ini membantu mengidentifikasi pola dasar dan potensi hubungan antar variabel.

    Hasil Temuan:
    - Distribusi pelanggan berdasarkan jenis kelamin (laki-laki dan perempuan) relatif seimbang, baik dari sisi jumlah pelanggan maupun distribusi churn. Artinya, jenis kelamin tidak menunjukkan pengaruh signifikan terhadap kecenderungan churn.

    - Meskipun sebagian besar pelanggan berasal dari kalangan usia muda, persentase churn lebih tinggi pada pelanggan lansia (senior citizen). Hal ini dapat dijelaskan oleh proporsi pelanggan lansia yang relatif kecil, namun dengan tingkat churn yang lebih besar secara persentase.

    - Pelanggan yang tidak menggunakan layanan seperti Tech Support, Device Protection, Online Backup, dan Online Security memiliki tingkat churn yang lebih tinggi dibandingkan pelanggan yang menggunakan layanan-layanan tersebut. Hal ini menunjukkan bahwa penggunaan layanan tambahan mungkin berkontribusi terhadap loyalitas pelanggan.

    - Distribusi pelanggan yang menggunakan maupun tidak menggunakan layanan streaming (TV maupun Movies) relatif seimbang. Tingkat churn pada kedua kelompok ini juga serupa. Namun, pelanggan yang tidak menggunakan layanan streaming sama sekali menunjukkan tingkat churn yang sangat rendah, meskipun jumlahnya sedikit.

    - Pelanggan dengan kontrak jangka panjang (satu hingga dua tahun) menunjukkan tingkat churn yang lebih rendah secara signifikan dibandingkan pelanggan dengan kontrak bulanan. Hal ini mengindikasikan bahwa komitmen jangka panjang dapat menurunkan risiko churn.

    - Metode pembayaran menggunakan electronic check lebih sering dikaitkan dengan tingkat churn yang tinggi dibandingkan metode pembayaran lainnya, seperti automatic bank transfer atau credit card.


    - Pelanggan yang menggunakan layanan Fiber Optic memiliki kecenderungan churn yang lebih tinggi dibandingkan pelanggan dengan layanan DSL atau tanpa layanan internet. Hal ini dapat disebabkan oleh persepsi terhadap harga atau kualitas layanan.

5. **Korelasi Antar Variabel Numerik**

    Heatmap korelasi digunakan untuk melihat hubungan antar fitur numerik. Fitur seperti tenure, monthly charges, dan total charges menunjukkan korelasi tertentu terhadap churn. Tenure (lama berlangganan) memiliki korelasi negatif dengan churn: semakin lama pelanggan menggunakan layanan, semakin kecil kemungkinan mereka churn.

    


## Data Preparation
Tahapan data preparation dilakukan untuk memastikan bahwa data siap digunakan oleh model machine learning dan menghasilkan performa yang optimal. Beberapa langkah yang dilakukan pada tahap ini meliputi:


1. **Encoding dan Standarisasi**

    - One-Hot Encoding diterapkan pada kolom kategorikal agar model dapat memahami data kategorikal dalam bentuk numerik tanpa mengasumsikan hubungan ordinal.

    - RobustScaler digunakan pada kolom numerik untuk menstandarisasi nilai agar berada dalam skala yang sama, sehingga mempercepat konvergensi model dan meningkatkan akurasi.

2. **Memisahkan Fitur X dan Target y**
    Menghapus kolom Churn dan CustomerID untuk fitur X karna Churn akan digunakan untuk target y, sedangkan CustomerID dihapus karna tidak dibutuhkan.


4. **Menangani Ketidakseimbangan Kelas (Imbalanced Data)**

    Dataset menunjukkan distribusi kelas yang tidak seimbang antara pelanggan yang churn dan tidak churn. Oleh karena itu, teknik SMOTE (Synthetic Minority Over-sampling Technique) digunakan untuk melakukan oversampling pada kelas minoritas. Hal ini bertujuan agar model tidak bias terhadap kelas mayoritas dan dapat belajar dengan seimbang dari kedua kelas.

5. **Membagi Data Train dan Test**

    Dataset dibagi menjadi data latih dan data uji dengan proporsi 80:20 agar model dapat dievaluasi secara objektif terhadap data yang belum pernah dilihat sebelumnya. 


## Modeling
Pada tahap ini, dilakukan pemodelan menggunakan beberapa algoritma machine learning untuk memprediksi customer churn. Proses pemodelan dimulai dari algoritma yang sederhana sebagai baseline, kemudian dilanjutkan dengan model yang lebih kompleks untuk mengeksplorasi performa yang lebih optimal.

1. **Logistic Regression**

    Model Logistic Regression digunakan sebagai baseline karena memiliki karakteristik berikut:

    - Sederhana dan mudah diinterpretasikan.

    - Cepat dalam proses pelatihan.

    - Memberikan estimasi probabilitas churn yang berguna untuk pengambilan keputusan bisnis.

    Model dilatih menggunakan parameter default tanpa tuning terlebih dahulu. Tujuannya adalah untuk mendapatkan baseline performa awal yang nantinya akan dibandingkan dengan model lain. Evaluasi dilakukan menggunakan metrik accuracy, precision, recall. 

    Parameter awal yang digunakan:
    - max_iter = 100 : Batas maksimum iterasi selama proses optimisasi model.

    Setelah melatih model awal, dilakukan proses hyperparameter tuning menggunakan GridSearchCV untuk meningkatkan performa model. Parameter yang dieksplorasi dalam tuning antara lain:

    - C = 100 : mengontrol seberapa besar kita menghindari overfitting.
    
    - class_weight = 'balanced : untuk menangani ketidakseimbangan kelas churn dan non-churn.

    - max_iter = 500 : Batas maksimum iterasi selama proses optimisasi model.

    - penalty = 'l2' : jenis regularisasi yang digunakan.

    - solver = 'liblinear' :  optimisasi yang digunakan untuk menemukan bobot terbaik.

    **Kelebihan**:

    - Interpretasi koefisien fitur mudah dilakukan.

    - Cepat dan efisien pada dataset kecil hingga menengah.

    **Kekurangan:**

    - Asumsi hubungan linear antara fitur dan target seringkali tidak sesuai untuk data kompleks.

    - Performa dapat menurun jika data memiliki korelasi antar fitur atau relasi non-linear.


2. **Random Forest** 
    
    Model Random Forest digunakan untuk mengeksplorasi model yang lebih kompleks. Algoritma ini merupakan ensemble model berbasis decision tree yang mampu menangkap hubungan non-linear dan bekerja baik terhadap dataset dengan banyak fitur.

    Parameter awal yang digunakan:

    - n_estimators = 100: jumlah pohon dalam model.

    - max_depth = None: kedalaman maksimum pohon.

    - random_state = 42: menjaga konsistensi hasil.

    - class_weight = 'balanced': untuk menangani ketidakseimbangan kelas churn dan non-churn.

    Setelah melatih model awal, dilakukan proses hyperparameter tuning menggunakan GridSearchCV untuk meningkatkan performa model. Parameter yang dieksplorasi dalam tuning antara lain:

    - n_estimators = 200: jumlah pohon.

    - max_depth = None: kedalaman maksimum pohon.

    - min_samples_split = 2 : jumlah minimal sampel untuk membagi node.

    - min_samples_leaf = 2 : jumlah minimal sampel yang harus ada di leaf node.

    - max_features = 'log2' : jumlah fitur yang dipertimbangkan secara acak ketika membagi node adalah logaritma basis 2 dari total fitur

    **Kelebihan**:

    - Dapat menangani data yang tidak teratur dan hubungan antar fitur yang kompleks.

    - Tidak mudah overfitting karena menggunakan voting dari banyak pohon.


    **Kekurangan**:

    - Waktu pelatihan dan prediksi lebih lama dibanding model sederhana.

    - Interpretasi tidak sejelas Logistic Regression.<br />

#### Pemilihan Model Terbaik
Berdasarkan hasil evaluasi pada data validasi menggunakan metrik seperti recall, model Random Forest menunjukkan performa yang lebih baik dibanding Logistic Regression. Oleh karena itu, Random Forest dipilih sebagai model terbaik karena:

- Mampu menangkap kompleksitas data dengan lebih baik.

- Memiliki skor recall yang lebih tinggi, yang sangat penting dalam konteks churn prediction, di mana mengidentifikasi pelanggan yang benar-benar akan churn lebih penting daripada hanya mencapai akurasi tinggi.


## Evaluation
Dalam proyek ini, evaluasi model dilakukan dengan menggunakan beberapa metrik evaluasi yang relevan untuk masalah klasifikasi churn, yaitu:

- **Accuracy**: Mengukur persentase prediksi yang benar dari seluruh data. Namun, karena dataset imbalance (jumlah pelanggan churn vs tidak churn tidak seimbang).

    ![akurasi](https://github.com/user-attachments/assets/bc29ece2-7e88-4581-8b81-d335ac272fda)


    Misal diketahui confussion matrix seperti di bawah ini:

    ![confusion_matrix](https://github.com/user-attachments/assets/632f7485-927e-415d-ac8d-7dfc81cdaa6d)


    Maka akurasi model tersebut adalah (834+897) / (834+199+136+897) = 0.84.
    
    Akurasi hanya cocok digunakan pada saat perbandingan jumlah label data sebenarnya relatif sama.<br />
      

    
- **Precision**: Mengukur seberapa banyak dari pelanggan yang diprediksi akan churn, yang benar-benar churn. Berguna untuk menghindari false positive, yaitu salah mengira pelanggan tetap sebagai pelanggan churn.

    ![presisi](https://github.com/user-attachments/assets/2212b659-7c49-4417-b00b-1d2abe2aed10)


    Misal diketahui confussion matrix seperti di bawah ini:

    ![confusion_matrix](https://github.com/user-attachments/assets/87c10852-e41e-4cd4-83d8-059b3302e27d)


    Maka precision model tersebut adalah (834+897) / (834+199+136+897) = 0.84.<br />

- **Recall**: Mengukur seberapa banyak dari pelanggan yang benar-benar churn, yang berhasil terdeteksi oleh model. Metode ini penting karena perusahaan ingin mengidentifikasi sebanyak mungkin pelanggan yang berisiko churn agar bisa segera mengambil tindakan.

    ![recall](https://github.com/user-attachments/assets/bd273eee-735c-4220-8fb7-687e91c2d588)


    Misal diketahui confussion matrix seperti di bawah ini:

    ![confusion_matrix](https://github.com/user-attachments/assets/aae41104-01f5-49f8-b60b-3f464df61495)


    Maka recall model tersebut adalah (834) / (834+136) = 0.87.<br />

- **F1-Score**: Harmonik rata-rata dari precision dan recall. Metrik ini cocok digunakan saat kita ingin menjaga keseimbangan antara false positive dan false negative.

    ![f1_score](https://github.com/user-attachments/assets/bc515156-c11a-4ad1-a886-d61443d5dd19)


    Misal diketahui confussion matrix seperti di bawah ini:

    ![confusion_matrix](https://github.com/user-attachments/assets/69195fee-b790-41ce-96f7-2478b1f1664b)


    Maka F1-score model tersebut adalah (2*0.87*0.84) / (0.87+0.84) = 0.84.<br />



#### Hasil Evaluasi Model 

1. **Logistic Regression**
   
    Model dilatih dengan menggunakan parameter default
    <br />
      ![lr_cr_default](https://github.com/user-attachments/assets/044aa8ba-f99d-48c4-913d-80fe4829138f)
 
    

    Model setelah menggunakan grid search untuk menemukan kombinasi parameter terbaik
     <br />
      ![lr_cr_best](https://github.com/user-attachments/assets/58ca37db-b144-409e-998e-6e7fab265618)


    Confusion Matrix:
    <br />
      ![lr_cm](https://github.com/user-attachments/assets/f0115542-ac96-4a60-a293-f26bfc46785b)

 Nilai recall dan precision belum terlalu bagus walaupun sudah menggunakan kombinasi parameter terbaik yang ditemukan grid search.


3. **Random Forest**
   <br />
    Model dilatih dengan menggunakan parameter default
    <br />
      ![rf_cr_default](https://github.com/user-attachments/assets/976405b5-dd48-4e0b-b3e5-cdc198fb494d)


    Model setelah menemukan kombinasi parameter terbaik menggunakan grid search
    <br />
      ![rf_cr_best](https://github.com/user-attachments/assets/5e01a564-2613-48ba-bd9b-c0d7a4237fec)


    Confusion Matrix:
    <br />
      ![rf_cm](https://github.com/user-attachments/assets/76ce01b5-c380-4245-9517-07c1e2d01d6b)




Nilai recall yang tinggi (0.87) menunjukkan bahwa model cukup efektif dalam mengidentifikasi pelanggan yang akan churn, sesuai dengan tujuan bisnis. Precision yang cukup baik juga menunjukkan bahwa model tidak terlalu banyak memberikan alarm palsu (false positive).
<br />
<br />
#### Fitur yang Berkontribusi Besar pada Model 

1. **Logistic Regression**
    <br />
     ![lr_fitur](https://github.com/user-attachments/assets/f1cab15e-9589-4094-ad80-299f92810685)


3. **Random Forest**
    <br />
     ![rf_fitur](https://github.com/user-attachments/assets/1c49129c-fc0a-43ce-a354-770b8d167836)

<br />
<br />
Seluruh proses analisis dan pemodelan yang telah dilakukan berhasil menjawab solution statement yang dirumuskan di awal, yaitu mengembangkan model prediktif untuk mengidentifikasi pelanggan yang berisiko churn serta memahami faktor-faktor yang memengaruhinya.

Hasil eksplorasi data telah mengungkap fitur-fitur penting yang berkorelasi dengan churn, seperti jenis kontrak, metode pembayaran, layanan tambahan, hingga usia pelanggan. Proses pemodelan juga menunjukkan bahwa metrik recall menjadi pertimbangan utama dalam memilih model terbaik, karena dapat meminimalkan risiko pelanggan yang churn tetapi tidak terdeteksi.

Dengan demikian, keseluruhan tahapan ini diharapkan dapat menjadi langkah awal bagi perusahaan dalam memanfaatkan machine learning untuk mendukung pengambilan keputusan strategis, khususnya dalam upaya meningkatkan retensi pelanggan dan mengurangi churn secara proaktif.



