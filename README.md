# Implementasi FIS Sugeno Orde-Nol & FIS Mamdani (Prediksi Jumlah Produksi)

Tugas Kelompok 2 Logika Fuzzy: implementasi sistem inferensi fuzzy untuk memprediksi rata-rata jumlah produksi berdasarkan suhu ($^\circ\text{C}$), kebisingan ($\text{dB}$), dan pencahayaan ($\text{lux}$).

## Struktur Proyek

```text
Implementasi_Fuzzy_Produksi/
├── data_produksi.py               # Dataset 27 kombinasi eksperimen, validasi semesta, dan 27 aturan (R1-R27)
├── fuzzy_sugeno_produksi.py       # FIS Sugeno Orde-Nol (Product & Weighted Average) + --interaktif
├── fuzzy_mamdani_produksi.py      # FIS Mamdani (Min-Max & Centroid) + rincian Data 10 + --interaktif
├── evaluasi_perbandingan.py       # Evaluasi Poin 3, Poin 4, Poin 5, dan analisis anomali keanggotaan
├── simulasi_interaktif.py         # CLI perbandingan Sugeno vs Mamdani untuk pengujian input baru
├── requirements.txt               # Daftar pustaka Python
├── README.md                      # Dokumentasi proyek dan ringkasan hasil
└── output/                        # Hasil CSV dan grafik
    ├── grafik_fungsi_keanggotaan.png
    ├── grafik_mamdani_centroid_data10.png
    ├── grafik_evaluasi_perbandingan.png
    ├── tabel_27_aturan_fuzzy.csv
    ├── hasil_defuzzifikasi_sugeno.csv
    ├── hasil_defuzzifikasi_mamdani.csv
    ├── evaluasi_segitiga_vs_trapesium.csv
    ├── evaluasi_sugeno_vs_mamdani.csv
    └── analisis_anomali_keanggotaan.csv
```

## Cara Menjalankan

1. Instal dependensi:
   ```bash
   pip install -r requirements.txt
   ```
2. Jalankan FIS Sugeno Orde-Nol (tambahkan `--interaktif` untuk uji input manual):
   ```bash
   python fuzzy_sugeno_produksi.py
   ```
3. Jalankan FIS Mamdani (tambahkan `--interaktif` untuk uji input manual):
   ```bash
   python fuzzy_mamdani_produksi.py
   ```
4. Jalankan evaluasi perbandingan dan deteksi anomali:
   ```bash
   python evaluasi_perbandingan.py
   ```
5. Jalankan simulasi perbandingan interaktif Sugeno vs Mamdani:
   ```bash
   python simulasi_interaktif.py --suhu 25 --kebisingan 65 --pencahayaan 250
   ```

## Ringkasan Perbandingan FIS Sugeno vs FIS Mamdani

| Aspek | FIS Sugeno Orde-Nol | FIS Mamdani |
| :--- | :--- | :--- |
| Konsekuen | Konstanta riil ($Z_i$) | Himpunan linguistik (*Rendah*, *Sedang*, *Tinggi*) |
| Implikasi & Agregasi | *Product* & penjumlahan terbobot | *Clipping* (`MIN`) & *Union* (`MAX`) |
| Defuzzifikasi | *Weighted Average* | *Centroid* |
| Prediksi Data No. 10 | `149.3840` (Aktual: `149.73`) | `152.4445` (Aktual: `149.73`) |
| Total Prediksi ($\sum Z_i$) | `3871.2900` produk | `3868.1904` produk |
| Rata-rata ($\bar{Z}$) | `143.3811` produk | `143.2663` produk |
| MAE | `0.2778` produk | `2.3768` produk |
| RMSE | `0.5011` produk | `2.7143` produk |
| Waktu Eksekusi (27 Data) | `~0.15 ms` | `~25.90 ms` |

## Pembahasan Evaluasi (Poin 3, Poin 4, dan Anomali Keanggotaan)

### Poin 3: Evaluasi Kurva Segitiga vs Trapesium
Hasil pengujian dari `output/evaluasi_segitiga_vs_trapesium.csv`:

| Rancangan Kurva | Mean $Z$ | MAE | RMSE | Max Error |
| :--- | :---: | :---: | :---: | :---: |
| Kombinasi Trapesium-Segitiga (Utama) | `143.3811` | `0.2778` | `0.5011` | `1.1260` |
| Full Segitiga | `143.4895` | `0.1694` | `0.3056` | `0.6866` |
| Full Trapesium | `143.4274` | `0.2315` | `0.4176` | `0.9383` |

- **Puncak tunggal segitiga**: nilai $\mu = 1.0$ hanya tercapai pada satu titik puncak, sedangkan sisi kiri dan kanannya langsung turun linear. Pada varian *Full Segitiga*, transisi antarhimpunan berlangsung lebih landai sehingga menghasilkan galat terkecil (`MAE = 0.1694`).
- **Inti datar (*core*) trapesium**: rentang $[b, c]$ mempertahankan $\mu = 1.0$ secara penuh untuk kondisi ekstrem (sangat dingin/panas, sunyi/bising, gelap/terang). Karena itu, kombinasi Trapesium-Segitiga tetap dipakai sebagai rancangan utama karena lebih sesuai untuk kondisi batas lingkungan kerja.

### Poin 4: Evaluasi FIS Sugeno vs FIS Mamdani
- **Akurasi**: FIS Sugeno Orde-Nol menghasilkan `MAE = 0.2778` dan `RMSE = 0.5011`, jauh lebih kecil dibanding Mamdani (`MAE = 2.3768`, `RMSE = 2.7143`). Sugeno mempertahankan nilai numerik eksperimen $Z_i$ pada setiap aturan, sedangkan Mamdani mengelompokkan output ke tiga kategori linguistik lalu melakukan *smoothing* melalui perhitungan titik pusat luasan (*Centroid*).
- **Kecepatan komputasi**: Sugeno hanya memakai operasi aljabar langsung (`~0.15 ms`), sedangkan Mamdani menghitung pemotongan dan integrasi diskret di atas 1.401 titik semesta output (`~25.90 ms`).

### Analisis Anomali Keanggotaan (`output/analisis_anomali_keanggotaan.csv`)
1. **Titik batas bahu trapesium ($22^\circ\text{C}$, $55\text{ dB}$, $150\text{ lux}$)**: nilai batas tersebut berada tepat di sudut atas trapesium sehingga memperoleh $\mu = 1.0$ penuh tanpa transisi ke kategori tengah. Pada Data 1-9 ($22^\circ\text{C}$) dan Data 19-27 ($32^\circ\text{C}$), hanya satu aturan yang aktif sehingga prediksi Sugeno tepat sama dengan nilai aktual (`Selisih = 0.0000`).
2. **Penarikan prediksi pada Data 10-18 ($26^\circ\text{C}$)**: pada suhu $26^\circ\text{C}$, $\mu_{\text{Dingin}} = 0.2$ dan $\mu_{\text{Normal}} = 0.8$. Bobot kecil $\mu_{\text{Dingin}} = 0.2$ ikut mengaktifkan aturan tetangga R1-R9 yang nilai produksinya lebih rendah dibanding R10-R18, sehingga seluruh prediksi Data 10-18 bergeser turun di bawah nilai aktual (selisih `-0.3460` pada Data 10 hingga `-1.1260` pada Data 12).
