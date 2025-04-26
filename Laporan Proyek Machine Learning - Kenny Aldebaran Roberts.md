# Laporan Proyek Machine Learning - Kenny Aldebaran Roberts

## Domain Proyek

Kemampuan untuk memprediksi harga rumah secara akurat memiliki peran yang signifikan dalam dunia ekonomi. Bagi individu, prediksi ini menginformasikan keputusan penting tentang pembelian, penjualan, dan investasi dalam real estat. Bagi lembaga keuangan, prediksi ini penting untuk menilai risiko dan mengelola portofolio. Secara tradisional, penilaian harga rumah sangat bergantung pada penilaian manual, analisis pasar komparatif, dan penilaian ahli. Meskipun metode ini memberikan wawasan yang berharga, metode ini dapat bersifat subjektif, memakan waktu, dan mungkin tidak sepenuhnya menangkap interaksi kompleks dari berbagai faktor yang memengaruhi nilai properti. Meningkatnya ketersediaan dataset rumah yang lengkap, ditambah dengan kemajuan dalam machine learning, dapat merevolusi proses ini. Proyek ini mengeksplorasi penggunaan algoritma regresi untuk membangun model prediktif untuk harga rumah. Analisis regresi sangat cocok untuk tugas ini karena memungkinkan kita mengukur hubungan antara variabel dependen berkelanjutan (harga rumah) dan serangkaian variabel independen (fitur properti). Tujuan proyek adalah untuk membuat model yang kuat dan akurat yang dapat secara efektif memprediksi harga rumah berdasarkan kumpulan data yang diberikan.

Referensi riset terkait: [Abdul-Rahman, S., & Mutalib, S. (2021). Advanced Machine Learning Algorithms for House Price Prediction: Case Study in Kuala Lumpur. International Journal of Advanced Computer Science and Applications, 12(12).](https://thesai.org/Downloads/Volume12No12/Paper_91-Advanced_Machine_Learning_Algorithms.pdf)

## Business Understanding

### Problem Statements

- Bagaimana hubungan harga rumah dengan fitur-fitur tertentu?
- Berapakah harga rumah dengan karakteristik atau fitur tertentu?

### Goals

- Mengetahui hubungan fitur-fitur dengan harga rumah.
- Membuat model machine learning yang dapat memprediksi harga rumah seakurat mungkin berdasarkan fitur-fitur yang ada.

### Solution statements
- Menggunakan pairplot dan heatmap untuk melihat hubungan fitur dengan harga rumah.
- Menggunakan tiga jenis algoritma regresi untuk menggunakan model yang mencapai performa yang terbaik.

## Data Understanding
Dataset yang digunakan pada proyek ini adalah [House Price Regression Dataset](https://www.kaggle.com/datasets/prokshitha/home-value-insights/data). Dataset ini merupakan dataset kumpulan data mengenai harga rumah dan fitur properti tersebut sebesar 1000 baris, dengan setiap baris mewakili sebuah rumah dan berbagai atribut yang memengaruhi harganya.

### Variabel-variabel pada dataset ini adalah sebagai berikut:

- Square_Footage: Ukuran rumah dalam kaki persegi. Rumah yang lebih besar biasanya memiliki harga yang lebih tinggi.

- Num_Bedrooms: Jumlah kamar tidur di rumah. Lebih banyak kamar tidur umumnya meningkatkan nilai rumah.

- Num_Bathrooms: Jumlah kamar mandi di rumah. Rumah dengan lebih banyak kamar mandi biasanya memiliki harga yang lebih tinggi.

- Year_Built: Tahun rumah dibangun. Rumah yang lebih tua mungkin memiliki harga yang lebih rendah karena keausan.

- Lot_Size: Ukuran tanah tempat rumah dibangun, diukur dalam hektar. Tanah yang lebih besar cenderung menambah nilai properti.

- Garage_Size: Jumlah mobil yang dapat muat di garasi. Rumah dengan garasi yang lebih besar biasanya lebih mahal.

- Neighborhood_Quality: Peringkat kualitas lingkungan pada skala 1-10, di mana 10 menunjukkan lingkungan yang berkualitas tinggi. Lingkungan yang lebih baik biasanya memiliki harga yang lebih tinggi.

- House_Price (Variabel Target): Harga rumah, yang merupakan variabel dependen yang ingin diprediksi.

Untuk tahap exploratory data analysis (EDA) dan visualisasi data dapat dilihat pada notebook yang dilampirkan.

## Data Preparation
Pada proyek ini, akan digunakan teknik Train-Test Split dan Standardisasi Data.

**1. Train-Test Split**
Disini kita membagi data menjadi set training dan test, dengan rasio yang digunakan berupa 80/20, yang berarti 80 persen data digunakan untuk pelatihan, dan 20 untuk pengujian.

Membagi dataset menjadi data latih (train) dan data uji (test) merupakan hal yang harus kita lakukan sebelum membuat model. Kita perlu mempertahankan sebagian data yang ada untuk menguji seberapa baik generalisasi model terhadap data baru.

**2. Standardisasi Data**
Algoritma machine learning memiliki performa lebih baik dan konvergen lebih cepat ketika dimodelkan pada data dengan skala relatif sama atau mendekati distribusi normal. Proses scaling dan standardisasi membantu untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma.

Standardisasi adalah teknik transformasi yang paling umum digunakan dalam tahap persiapan pemodelan, yang pada proyek ini menggunakan teknik StandarScaler dari library Scikitlearn.

StandardScaler melakukan proses standardisasi fitur dengan mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi.  StandardScaler menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0. Sekitar 68% dari nilai akan berada di antara -1 dan 1.

Untuk menghindari kebocoran informasi pada data uji, kita hanya akan menerapkan standardisasi pada data latih. Kemudian, pada tahap evaluasi, kita akan melakukan standardisasi pada data uji. 

Pada proyek ini, tahap ini hanya digunakan bagi algoritma KNN, karena algoritma Random Forest dan AdaBoost tidak mempedulikan skala.

## Modeling
Model development adalah tahapan di mana kita menggunakan algoritma machine learning untuk menjawab problem statement dari tahap business understanding.

Pada tahap ini, kita akan mengembangkan model machine learning dengan tiga algoritma. Kemudian, kita akan mengevaluasi performa masing-masing algoritma dan menentukan algoritma mana yang memberikan hasil prediksi terbaik. Ketiga algoritma yang akan kita gunakan adalah:

**1. K-Nearest Neighbor**
K-Nearest Neighbor (KNN) adalah algoritma yang relatif sederhana dibandingkan dengan algoritma lain. Algoritma KNN menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. Dengan kata lain, setiap data baru diberi nilai berdasarkan seberapa mirip titik tersebut dalam set pelatihan. KNN bisa digunakan untuk kasus klasifikasi dan regresi. Pada proyek ini, KNN akan digunakan untuk kasus regresi. 

Dalam regresi KNN, algoritma memprediksi nilai kontinu untuk titik data baru dengan menemukan 'K' titik data terdekat (tetangga) dalam set pelatihan. Prediksi tersebut kemudian merupakan rata-rata nilai target dari K tetangga ini. Pada dasarnya, KNN memperkirakan nilai titik data baru berdasarkan kemiripannya dengan titik data terdekat.

"Kedekatan" titik data ditentukan oleh metrik jarak. Metrik umum meliputi:
- Jarak Euclidean: Jarak garis lurus antara dua titik.
- Jarak Manhattan: Jumlah selisih absolut koordinat dua titik.
- Jarak Minkowski: Generalisasi jarak Euclidean dan Manhattan.

**Kekuatan**:
- Mudah dipahami dan diterapkan: KNN secara konseptual mudah dipahami dan dipraktikkan.
- Tidak ada fase pelatihan yang eksplisit: KNN tidak memerlukan langkah pelatihan terpisah. Model dibangun sesuai kebutuhan selama prediksi.
- Nonparametrik: KNN tidak mengasumsikan bentuk fungsional tertentu untuk hubungan antara fitur dan variabel target, sehingga cocok untuk data kompleks dan nonlinier.

**Kelemahan**:
- Mahal secara komputasi: Memprediksi nilai untuk titik data baru memerlukan penghitungan jaraknya ke semua titik data dalam set pelatihan, yang dapat sangat memakan waktu untuk set data besar.
- Sensitif terhadap fitur yang tidak relevan: KNN mempertimbangkan semua fitur secara setara saat menghitung jarak, yang dapat menyebabkan kinerja yang buruk jika data berisi banyak fitur yang tidak relevan.
- Memerlukan penskalaan fitur: Fitur dengan skala yang lebih besar dapat mendominasi kalkulasi jarak, yang mengarah pada prediksi yang bias. Oleh karena itu, sangat penting untuk menskalakan fitur sebelum menerapkan KNN.
- Menentukan nilai K yang optimal: Pilihan K sangat penting untuk kinerja KNN. K yang kecil dapat menyebabkan prediksi yang tidak jelas, sementara K yang besar dapat memperhalus pola penting dalam data. K yang optimal sering ditemukan melalui eksperimen dan validasi.
- Data yang tidak seimbang: KNN dapat bias terhadap kelas mayoritas dalam kumpulan data yang tidak seimbang.

Cara pelatihan model KNN sangatlah mudah, hanya perlu menggunakan kode berikut:

knn = KNeighborsRegressor()
knn.fit(X_train_scaled, y_train)

knn = KNeighborsRegressor(): Membuat objek model KNN yang disebut knn menggunakan kelas KNeighborsRegressor. Objek ini menggunakan parameter default untuk model.
knn.fit(X_train_scaled, y_train): Kode ini melatih model KNN menggunakan dua argumen:
- X_train_scaled: Data pelatihan yang berisi fitur (seperti luas persegi, tahun pembangunan, dll.) yang telah diskalakan menggunakan StandardScaler. Penskalaan penting untuk KNN karena sensitif terhadap skala fitur.
- y_train: Data pelatihan yang berisi variabel target, yaitu harga rumah. Model mempelajari hubungan antara fitur dan harga rumah dari data ini.

**2. Random Forest**
Algoritma random forest adalah salah satu algoritma supervised learning yang termasuk ke dalam kategori ensemble (group) learning. Model ensemble sendiri merupakan model prediksi yang terdiri dari beberapa model yang bekerja sama untuk menyelesaikan masalah. Pada model ensemble, setiap model harus membuat prediksi secara independen yang nanti digabungkan untuk membuat prediksi akhir. 

Random Forest Regressor menggunakan banyak decision tree untuk meningkatkan akurasi regresi dan mengurangi overfitting. Decision Tree adalah struktur seperti diagram alir di mana setiap node mewakili pengujian pada atribut (fitur), setiap cabang mewakili hasil pengujian, dan setiap node daun mewakili prediksi (dalam regresi, nilai kontinu). Decision Tree untuk regresi biasanya bertujuan untuk meminimalkan varians atau mean squared error (MSE) pada setiap pemisahan. Untuk membuat prediksi untuk titik data baru, setiap decision tree memprediksi suatu nilai secara independen. Random Forest Regressor kemudian merata-ratakan prediksi dari semua decision tree untuk mendapatkan prediksi akhir.

**Kekuatan**:
- Akurasi tinggi: Dengan merata-ratakan prediksi beberapa decision tree, Random Forest umumnya mencapai akurasi yang lebih tinggi daripada satu decision tree saja. 
- Kuat terhadap overfitting: Kombinasi banyak decision tree membantu mengurangi overfitting.
- Menangani dimensionalitas tinggi: Random Forest dapat menangani set data dengan sejumlah besar fitur secara efektif.
- Menyediakan estimasi kepentingan fitur: Random Forest dapat memperkirakan pentingnya setiap fitur dalam set data, yang dapat berguna untuk pemilihan fitur dan pemahaman data.
- Menangani hubungan non-linier: Random Forest dapat memodelkan hubungan non-linier yang kompleks antara fitur dan variabel target.
- Tidak sensitif terhadap outlier: Proses perataan mengurangi dampak outlier.
- Bekerja dengan fitur kategoris dan numerik: Random Forest dapat menangani kedua jenis fitur tanpa memerlukan preprocessing.
**Kelemahan**:
- Komputasi intensif: Melatih Random Forest dapat memakan banyak komputasi, terutama untuk dataset besar. 
- Kurang dapat diinterpretasikan dibandingkan Decision Tree: Meskipun Random Forest dapat memberikan estimasi pentingnya fitur, namun secara umum Random Forest kurang dapat diinterpretasikan dibandingkan satu Decision Tree saja.
• Dapat menjadi "black box": Karena kompleksitasnya, proses prediksi Random Forest dapat susah dipahami.
• Potensi bias: Jika kumpulan data asli memiliki data yang bias, model random forest juga akan bias.

Cara pelatihan model Random Forest sangatlah mudah, hanya perlu menggunakan kode berikut:

rf = RandomForestRegressor()
rf.fit(X_train, y_train) # No need for standardized data

rf = RandomForestRegressor(): Membuat objek model Random Forest yang disebut rf menggunakan kelas RandomForestRegressor dengan parameter default.
rf.fit(X_train, y_train): Kode ini melatih model Random Forest menggunakan dua argumen:
- X_train: Data pelatihan yang berisi fitur. Data pelatihan yang tidak diskalakan digunakan di sini karena model Random Forest tidak terpengaruh oleh penskalaan fitur.
- y_train: Data pelatihan yang berisi harga rumah.

**3. AdaBoost**
AdaBoost, singkatan dari Adaptive Boosting, adalah metode pembelajaran ensemble yang menggabungkan prediksi beberapa pembelajar lemah untuk menciptakan pembelajar yang kuat. Tidak seperti Random Forest, yang melatih model secara paralel, AdaBoost melatihnya secara berurutan atau iteratif, dengan setiap model mencoba memperbaiki kesalahan pendahulunya. 

Awalnya, semua kasus dalam data latih memiliki weight atau bobot yang sama. Algoritme AdaBoost melatih secara berulang serangkaian model dan mengevaluasi performa mereka. Bobot (kepentingan) diberikan kepada model berdasarkan performanya. Model yang lebih baik mempelajari data mendapatkan bobot yang lebih tinggi. Prediksi akhir diperoleh dengan mengambil rata-rata tertimbang dari prediksi individual, di mana bobotnya adalah bobot kepentingan model. Proses iteratif ini berlanjut sampai model mencapai akurasi yang diinginkan.

Kekuatan:
- Akurasi yang bagus: AdaBoost memiliki akurasi tinggi karena merupakan metode ensemble.
- Kepentingan fitur: AdaBoost dapat memberikan estimasi pentingnya fitur.
- Relatif mudah diimplementasikan: Algoritma inti AdaBoost relatif mudah diimplementasikan.

Kelemahan:
-  Sensitif terhadap noise dan outlier: AdaBoost mencoba menyesuaikan setiap titik data dengan tepat, yang membuatnya sensitif terhadap noise dan outlier. Outlier dapat memengaruhi pembobotan titik data secara tidak proporsional, yang membuat model yang kurang optimal.
- Komputasi intensif: Seperti metode boosting lainnya, AdaBoost menggunakan komputasi intensif, terutama untuk dataset besar. 
- Dapat overfitting: Jika pelatihan terlalu rumit, AdaBoost dapat melakukan overfitting pada data pelatihan.

Cara pelatihan model AdaBoost sangatlah mudah, hanya perlu menggunakan kode berikut:

ada = AdaBoostRegressor()
ada.fit(X_train, y_train) # No need for standardized data

ada = AdaBoostRegressor(): Membuat objek model AdaBoost yang disebut ada menggunakan kelas AdaBoostRegressor dengan parameter default.
ada.fit(X_train, y_train): Kode ini melatih model AdaBoost menggunakan dua argumen. 
- X_train: Data pelatihan yang berisi fitur. Mirip dengan Random Forest, metode ini menggunakan data pelatihan yang tidak diskalakan (X_train) karena model AdaBoost tidak terpengaruh oleh penskalaan fitur.
- y_train: Data pelatihan yang berisi harga rumah.


**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation

Dalam analisis regresi ketiga model, digunakan metrik khusus untuk mengevaluasi seberapa baik model kami memprediksi variabel target yang berupa harga rumah. Berikut ini penjelasan dari tiga metrik yang digunakan: Mean Squared Error (MSE), Mean Absolute Error (MAE), dan R-squared (R²).

**Mean Squared Error (MSE)**
MSE menghitung rata-rata perbedaan kuadrat antara nilai yang diprediksi dan nilai aktual.

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

Dimana:
- $$(n)$$ adalah jumlah titik data.
- $$(y_i)$$ adalah nilai aktual dari variabel target untuk titik data ke-(i).
- $$(\hat{y}_i)$$ adalah nilai prediksi dari variabel target untuk titik data ke-(i).

MSE mengukur besarnya rata-rata kesalahan kuadrat, yang membuat MSE sensitif terhadap outlier. Nilai MSE yang lebih rendah menunjukkan kinerja model yang lebih baik. MSE 0 berarti model memprediksi dengan sempurna.

**Mean Absolute Error (MAE)**
MAE menghitung rata-rata perbedaan absolut antara nilai yang diprediksi dan nilai aktual.

$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

Dimana:
- $$(n)$$ adalah jumlah titik data.
- $$(y_i)$$ adalah nilai aktual dari variabel target untuk titik data ke-(i).
- $$(\hat{y}_i)$$ adalah nilai prediksi dari variabel target untuk titik data ke-(i).

MAE kurang sensitif terhadap outlier dibandingkan dengan MSE karena menggunakan nilai absolut daripada kesalahan kuadrat. Nilai MAE yang lebih rendah menunjukkan kinerja model yang lebih baik. MAE 0 berarti model memprediksi dengan sempurna.

**R-kuadrat (R²)**
R-kuadrat mengukur proporsi varians dalam variabel dependen yang dijelaskan oleh variabel independen dalam model. Ini menunjukkan seberapa baik model tersebut dalam memprediksi variabel target dibandingkan dengan hanya menggunakan rata-rata variabel target.

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$

Dimana:
- $$(SS_{res})$$ adalah jumlah kuadrat residual: $$(\sum_{i=1}^{n} (y_i - \hat{y}_i)^2)$$
- $$(SS_{tot})$$ adalah jumlah kuadrat total: $$(\sum_{i=1}^{n} (y_i - \bar{y})^2)$$
- $$(\bar{y})$$ adalah rata-rata nilai aktual variabel target.

R² berkisar dari 0 hingga 1, dengan 1 menunjukkan bahwa model menjelaskan semua varians dalam variabel target, dan 0 menunjukkan bahwa model tidak menjelaskan varians apa pun dalam variabel target.

**Evaluasi Hasil Proyek**
|        | train_mse    | test_mse     | train_mae    | test_mae     | train_r2     | test_r2      |
| :------- | :------------- | :------------- | :------------- | :------------- | :------------- | :------------- |
| KNN      | 1379094432.787155 | 2285279964.636393 | 29263.346088   | 38271.001851   | 0.978511       | 0.964547       |
| RandomForest | 81912686.95175  | 476631737.055197 | 7057.040997    | 17970.394885   | 0.998724       | 0.992606       |
| Boosting   | 880410352.054793 | 948741444.830329 | 24261.081325   | 25087.878261   | 0.986281       | 0.985281       |

Dapat dilihat bahwa performa model Random Forest adalah yang paling baik, dengan MAE dan MSE yang paling kecil, dan R2 yang paling besar, baik pada train set dan test set. AdaBoost memiliki performa kedua paling baik, dan KNN menempati posisi terakhir, dengan performa paling buruk. 

Maka, dari hasil ujicoba tersebut, model Random Forest yang paling baik untuk memprediksi harga rumah.


