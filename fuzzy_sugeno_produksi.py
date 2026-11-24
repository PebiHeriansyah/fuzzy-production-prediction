import os
import sys
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_produksi import DATASET_PRODUKSI, bangun_27_aturan, validasi_input_fuzzy


# --- Bagian A: Fungsi Keanggotaan (Segitiga & Trapesium) ---

def fungsi_segitiga(x: float, a: float, b: float, c: float) -> float:
    """Hitung derajat keanggotaan kurva segitiga (a, b, c)."""
    if x <= a or x >= c:
        return 0.0  # Di luar rentang [a, c], nilai keanggotaan 0
    elif a < x <= b:
        return (x - a) / (b - a)  # Sisi naik dari a ke puncak b
    else:
        return (c - x) / (c - b)  # Sisi turun dari puncak b ke c


def fungsi_trapesium(x: float, a: float, b: float, c: float, d: float) -> float:
    """Hitung derajat keanggotaan kurva trapesium (a, b, c, d), termasuk bahu kiri/kanan."""
    if b <= x <= c:
        return 1.0  # Bagian datar atas (core) trapesium bernilai 1.0
    elif x <= a or x >= d:
        return 0.0  # Di luar domain [a, d] bernilai 0.0
    elif a < x < b:
        return (x - a) / (b - a)  # Sisi naik linear dari a ke b
    else:
        return (d - x) / (d - c)  # Sisi turun linear dari c ke d


# --- Bagian C: Fuzzifikasi Variabel Input ---

def fuzzifikasi_suhu(suhu: float) -> Dict[str, float]:
    """Mengubah nilai crisp Suhu (°C) menjadi derajat keanggotaan 3 himpunan."""
    return {
        "Dingin": fungsi_trapesium(suhu, 18, 18, 22, 27),  # Trapesium bahu kiri
        "Normal": fungsi_segitiga(suhu, 22, 27, 32),       # Segitiga tengah
        "Panas": fungsi_trapesium(suhu, 27, 32, 36, 36),   # Trapesium bahu kanan
    }


def fuzzifikasi_kebisingan(kebisingan: float) -> Dict[str, float]:
    """Mengubah nilai crisp Kebisingan (dB) menjadi derajat keanggotaan 3 himpunan."""
    return {
        "Rendah": fungsi_trapesium(kebisingan, 40, 40, 55, 75),   # Trapesium bahu kiri
        "Sedang": fungsi_segitiga(kebisingan, 55, 75, 90),        # Segitiga tengah
        "Tinggi": fungsi_trapesium(kebisingan, 75, 90, 105, 105), # Trapesium bahu kanan
    }


def fuzzifikasi_pencahayaan(pencahayaan: float) -> Dict[str, float]:
    """Mengubah nilai crisp Pencahayaan (lux) menjadi derajat keanggotaan 3 himpunan."""
    return {
        "Rendah": fungsi_trapesium(pencahayaan, 50, 50, 150, 300),   # Trapesium bahu kiri
        "Sedang": fungsi_segitiga(pencahayaan, 150, 300, 500),       # Segitiga tengah
        "Tinggi": fungsi_trapesium(pencahayaan, 300, 500, 650, 650), # Trapesium bahu kanan
    }


# --- Bagian D & E: Inferensi Sugeno & Defuzzifikasi Weighted Average ---

def inferensi_sugeno(
    suhu: float,
    kebisingan: float,
    pencahayaan: float,
    daftar_aturan: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Menjalankan tahapan inferensi Sugeno Orde-Nol:
    1. Fuzzifikasi input suhu, kebisingan, dan pencahayaan.
    2. Menghitung alpha-predikat tiap aturan: alpha_i = mu_suhu * mu_bising * mu_cahaya (Product).
    3. Menghitung defuzzifikasi Weighted Average: Z = sum(alpha_i * Z_i) / sum(alpha_i).
    """
    mu_s = fuzzifikasi_suhu(suhu)               # Derajat keanggotaan variabel Suhu
    mu_b = fuzzifikasi_kebisingan(kebisingan)   # Derajat keanggotaan variabel Kebisingan
    mu_c = fuzzifikasi_pencahayaan(pencahayaan) # Derajat keanggotaan variabel Pencahayaan

    pembilang = 0.0                          # Akumulator sum(alpha_i * Z_i)
    penyebut = 0.0                           # Akumulator bobot sum(alpha_i)
    aturan_aktif: List[Dict[str, Any]] = []  # Daftar aturan yang memiliki alpha > 0

    for aturan in daftar_aturan:
        # Operator AND menggunakan Product (perkalian ketiga derajat keanggotaan)
        alpha = (
            mu_s[aturan["suhu"]]
            * mu_b[aturan["kebisingan"]]
            * mu_c[aturan["pencahayaan"]]
        )

        # Hanya aturan aktif (alpha > 0) yang berkontribusi terhadap nilai output Z
        if alpha > 0:
            aturan_aktif.append({
                "kode_aturan": aturan["kode_aturan"],
                "alpha": round(alpha, 4),
                "konsekuen_z": aturan["konsekuen_z"],
            })
            pembilang += alpha * aturan["konsekuen_z"]
            penyebut += alpha

    # Defuzzifikasi Weighted Average (rata-rata terbobot konsekuen konstanta)
    hasil_z = pembilang / penyebut if penyebut != 0 else 0.0

    return {
        "mu_suhu": mu_s,
        "mu_kebisingan": mu_b,
        "mu_pencahayaan": mu_c,
        "aturan_aktif": aturan_aktif,
        "hasil_z": hasil_z,
    }


# --- Visualisasi Kurva Keanggotaan Input (Bagian A) ---

def plot_fungsi_keanggotaan(output_path: str = "output/grafik_fungsi_keanggotaan.png") -> None:
    """Menggambar 3 grafik kurva keanggotaan untuk Suhu, Kebisingan, dan Pencahayaan."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.suptitle("Grafik Fungsi Keanggotaan Variabel Input Fuzzy", fontsize=14, fontweight="bold")

    # 1. Variabel Suhu
    x_suhu = np.linspace(18, 36, 500)
    axes[0].plot(x_suhu, [fungsi_trapesium(x, 18, 18, 22, 27) for x in x_suhu], "b-", lw=2, label="Dingin - Trapesium (18,18,22,27)")
    axes[0].plot(x_suhu, [fungsi_segitiga(x, 22, 27, 32) for x in x_suhu], color="orange", lw=2, label="Normal - Segitiga (22,27,32)")
    axes[0].plot(x_suhu, [fungsi_trapesium(x, 27, 32, 36, 36) for x in x_suhu], "g-", lw=2, label="Panas - Trapesium (27,32,36,36)")
    axes[0].set_title("1. Variabel Suhu (°C)", fontweight="bold")
    axes[0].set_xlabel("Suhu (°C)")
    axes[0].set_ylabel("Derajat Keanggotaan, μ(x)")
    axes[0].set_xticks([18, 22, 27, 32, 36])
    axes[0].set_yticks([0.0, 0.5, 1.0])
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="best", fontsize=8)

    # 2. Variabel Kebisingan
    x_bising = np.linspace(40, 105, 500)
    axes[1].plot(x_bising, [fungsi_trapesium(x, 40, 40, 55, 75) for x in x_bising], "b-", lw=2, label="Rendah - Trapesium (40,40,55,75)")
    axes[1].plot(x_bising, [fungsi_segitiga(x, 55, 75, 90) for x in x_bising], color="orange", lw=2, label="Sedang - Segitiga (55,75,90)")
    axes[1].plot(x_bising, [fungsi_trapesium(x, 75, 90, 105, 105) for x in x_bising], "g-", lw=2, label="Tinggi - Trapesium (75,90,105,105)")
    axes[1].set_title("2. Variabel Kebisingan (dB)", fontweight="bold")
    axes[1].set_xlabel("Kebisingan (dB)")
    axes[1].set_ylabel("Derajat Keanggotaan, μ(x)")
    axes[1].set_xticks([40, 55, 75, 90, 105])
    axes[1].set_yticks([0.0, 0.5, 1.0])
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="best", fontsize=8)

    # 3. Variabel Pencahayaan
    x_cahaya = np.linspace(50, 650, 500)
    axes[2].plot(x_cahaya, [fungsi_trapesium(x, 50, 50, 150, 300) for x in x_cahaya], "b-", lw=2, label="Rendah - Trapesium (50,50,150,300)")
    axes[2].plot(x_cahaya, [fungsi_segitiga(x, 150, 300, 500) for x in x_cahaya], color="orange", lw=2, label="Sedang - Segitiga (150,300,500)")
    axes[2].plot(x_cahaya, [fungsi_trapesium(x, 300, 500, 650, 650) for x in x_cahaya], "g-", lw=2, label="Tinggi - Trapesium (300,500,650,650)")
    axes[2].set_title("3. Variabel Pencahayaan (lux)", fontweight="bold")
    axes[2].set_xlabel("Pencahayaan (lux)")
    axes[2].set_ylabel("Derajat Keanggotaan, μ(x)")
    axes[2].set_xticks([50, 150, 300, 500, 650])
    axes[2].set_yticks([0.0, 0.5, 1.0])
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend(loc="best", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def simulasi_interaktif(aturan_27: List[Dict[str, Any]]) -> None:
    """Mode opsional CLI untuk menguji kombinasi input baru secara dinamis pada FIS Sugeno."""
    print("\n--- MODE SIMULASI INTERAKTIF FIS SUGENO ---")
    try:
        s = float(input("Masukkan Suhu (°C) [18 - 36]         : "))
        b = float(input("Masukkan Kebisingan (dB) [40 - 105]  : "))
        c = float(input("Masukkan Pencahayaan (lux) [50 - 650]: "))
        validasi_input_fuzzy(s, b, c)
        res = inferensi_sugeno(s, b, c, aturan_27)
        print(f"Derajat Suhu       : {res['mu_suhu']}")
        print(f"Derajat Kebisingan : {res['mu_kebisingan']}")
        print(f"Derajat Pencahayaan: {res['mu_pencahayaan']}")
        print("Aturan Aktif       :", res["aturan_aktif"])
        print(f"Prediksi Produksi Z: {res['hasil_z']:.4f} produk")
    except ValueError as err:
        print(f"[ERROR] {err}")


def main() -> None:
    os.makedirs("output", exist_ok=True)
    aturan_27 = bangun_27_aturan()

    # Simpan daftar 27 aturan fuzzy ke CSV
    pd.DataFrame(aturan_27).to_csv("output/tabel_27_aturan_fuzzy.csv", index=False)

    # Evaluasi seluruh 27 kombinasi data
    hasil_rows = []
    for no, suhu, bising, cahaya, aktual, sd in DATASET_PRODUKSI:
        res = inferensi_sugeno(suhu, bising, cahaya, aturan_27)
        ringkasan_aturan = "; ".join(
            [f"{a['kode_aturan']}(α={a['alpha']}, Z={a['konsekuen_z']})" for a in res["aturan_aktif"]]
        )
        hasil_rows.append({
            "No": no,
            "Suhu_C": suhu,
            "Kebisingan_dB": bising,
            "Pencahayaan_lux": cahaya,
            "mu_Suhu_Dingin": round(res["mu_suhu"]["Dingin"], 4),
            "mu_Suhu_Normal": round(res["mu_suhu"]["Normal"], 4),
            "mu_Suhu_Panas": round(res["mu_suhu"]["Panas"], 4),
            "mu_Bising_Rendah": round(res["mu_kebisingan"]["Rendah"], 4),
            "mu_Bising_Sedang": round(res["mu_kebisingan"]["Sedang"], 4),
            "mu_Bising_Tinggi": round(res["mu_kebisingan"]["Tinggi"], 4),
            "mu_Cahaya_Rendah": round(res["mu_pencahayaan"]["Rendah"], 4),
            "mu_Cahaya_Sedang": round(res["mu_pencahayaan"]["Sedang"], 4),
            "mu_Cahaya_Tinggi": round(res["mu_pencahayaan"]["Tinggi"], 4),
            "Aturan_Aktif_dan_Alpha": ringkasan_aturan,
            "Produksi_Aktual": aktual,
            "Prediksi_Fuzzy_Z": round(res["hasil_z"], 4),
            "Selisih_Absolut": round(abs(aktual - res["hasil_z"]), 4),
        })

    df_hasil = pd.DataFrame(hasil_rows)
    df_hasil.to_csv("output/hasil_defuzzifikasi_sugeno.csv", index=False)

    total_z = df_hasil["Prediksi_Fuzzy_Z"].sum()
    rata_rata_z = df_hasil["Prediksi_Fuzzy_Z"].mean()
    mae = df_hasil["Selisih_Absolut"].mean()

    print("-" * 65)
    print("HASIL IMPLEMENTASI FIS SUGENO ORDE-NOL (WEIGHTED AVERAGE)")
    print("-" * 65)
    print(df_hasil[["No", "Suhu_C", "Kebisingan_dB", "Pencahayaan_lux", "Aturan_Aktif_dan_Alpha", "Produksi_Aktual", "Prediksi_Fuzzy_Z"]].to_string(index=False))
    print("-" * 65)
    print(f"Total Prediksi Produksi (sum Z_i)  : {total_z:.4f}")
    print(f"Rata-rata Jumlah Produksi (mean Z) : {rata_rata_z:.4f} produk")
    print(f"Rata-rata Error Absolut (MAE)      : {mae:.4f} produk")
    print("-" * 65)

    plot_fungsi_keanggotaan()

    # Jalankan mode interaktif hanya jika argumen --interaktif diberikan
    if "--interaktif" in sys.argv:
        simulasi_interaktif(aturan_27)


if __name__ == "__main__":
    main()
