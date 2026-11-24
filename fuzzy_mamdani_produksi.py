import os
import sys
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_produksi import DATASET_PRODUKSI, bangun_27_aturan, validasi_input_fuzzy
from fuzzy_sugeno_produksi import (
    fungsi_segitiga,
    fungsi_trapesium,
    fuzzifikasi_suhu,
    fuzzifikasi_kebisingan,
    fuzzifikasi_pencahayaan,
)

# Semesta pembicaraan variabel output Jumlah Produksi: [130, 158] unit (1401 titik, resolusi 0.02 unit)
SEMESTA_OUTPUT: np.ndarray = np.linspace(130.0, 158.0, 1401)


def kurva_himpunan_output() -> Dict[str, np.ndarray]:
    """
    Mendefinisikan 3 himpunan fuzzy output Jumlah Produksi pada semesta [130, 158]:
    - Rendah : Trapesium bahu kiri  (130, 130, 138, 143)
    - Sedang : Segitiga tengah      (138, 144, 150)
    - Tinggi : Trapesium bahu kanan (145, 150, 158, 158)
    """
    return {
        "Rendah": np.array([fungsi_trapesium(z, 130, 130, 138, 143) for z in SEMESTA_OUTPUT]),
        "Sedang": np.array([fungsi_segitiga(z, 138, 144, 150) for z in SEMESTA_OUTPUT]),
        "Tinggi": np.array([fungsi_trapesium(z, 145, 150, 158, 158) for z in SEMESTA_OUTPUT]),
    }


def inferensi_mamdani(
    suhu: float,
    kebisingan: float,
    pencahayaan: float,
    daftar_aturan: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Menjalankan proses inferensi FIS Mamdani lengkap (Fuzzifikasi -> Implikasi MIN ->
    Agregasi MAX -> Defuzzifikasi Centroid) untuk satu kombinasi input.
    """
    mu_s = fuzzifikasi_suhu(suhu)                # Fuzzifikasi Suhu (°C)
    mu_b = fuzzifikasi_kebisingan(kebisingan)    # Fuzzifikasi Kebisingan (dB)
    mu_c = fuzzifikasi_pencahayaan(pencahayaan)  # Fuzzifikasi Pencahayaan (lux)
    mf_out = kurva_himpunan_output()             # Kurva diskret himpunan output

    agregasi_kurva = np.zeros_like(SEMESTA_OUTPUT)  # Kurva gabungan hasil agregasi MAX
    aturan_aktif: List[Dict[str, Any]] = []

    for aturan in daftar_aturan:
        # Operator AND Mamdani menggunakan nilai minimum (MIN) dari ketiga derajat keanggotaan
        alpha_min = min(
            mu_s[aturan["suhu"]],
            mu_b[aturan["kebisingan"]],
            mu_c[aturan["pencahayaan"]],
        )

        # Aturan hanya memengaruhi daerah output jika alpha_min > 0
        if alpha_min > 0:
            label_out = aturan["konsekuen_linguistik"]
            # Implikasi Mamdani (Clipping): memotong kurva output setinggi alpha_min
            kurva_terpotong = np.minimum(alpha_min, mf_out[label_out])
            # Agregasi aturan (Union): mengambil nilai maksimum di setiap titik semesta
            agregasi_kurva = np.maximum(agregasi_kurva, kurva_terpotong)

            aturan_aktif.append({
                "kode_aturan": aturan["kode_aturan"],
                "alpha_min": round(alpha_min, 4),
                "output_linguistik": label_out,
                "konsekuen_z_ref": aturan["konsekuen_z"],
            })

    # Defuzzifikasi Centroid (Center of Gravity): Z* = sum(z * mu_agregasi(z)) / sum(mu_agregasi(z))
    luas_total = np.sum(agregasi_kurva)
    z_centroid = float(np.sum(SEMESTA_OUTPUT * agregasi_kurva) / luas_total) if luas_total > 0 else 0.0

    return {
        "mu_suhu": mu_s,
        "mu_kebisingan": mu_b,
        "mu_pencahayaan": mu_c,
        "aturan_aktif": aturan_aktif,
        "agregasi_kurva": agregasi_kurva,
        "z_centroid": z_centroid,
    }


def cetak_rincian_manual_data_10(daftar_aturan: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Menampilkan perhitungan langkah demi langkah FIS Mamdani untuk Data 10
    (Suhu = 26°C, Kebisingan = 55 dB, Pencahayaan = 150 lux) sebagai pembanding manual.
    """
    no, s, b, c, aktual, _ = DATASET_PRODUKSI[9]  # Indeks 9 adalah Data No. 10
    res = inferensi_mamdani(s, b, c, daftar_aturan)

    print("\n" + "-" * 65)
    print("RINCIAN PERHITUNGAN FIS MAMDANI (DATA NO. 10)")
    print("-" * 65)
    print(f"Input Crisp : Suhu = {s}°C, Kebisingan = {b} dB, Pencahayaan = {c} lux (Aktual = {aktual})")
    print("1. Fuzzifikasi:")
    print(f"   - Suhu ({s}°C)         : Dingin={res['mu_suhu']['Dingin']:.2f}, Normal={res['mu_suhu']['Normal']:.2f}, Panas={res['mu_suhu']['Panas']:.2f}")
    print(f"   - Kebisingan ({b} dB)  : Rendah={res['mu_kebisingan']['Rendah']:.2f}, Sedang={res['mu_kebisingan']['Sedang']:.2f}, Tinggi={res['mu_kebisingan']['Tinggi']:.2f}")
    print(f"   - Pencahayaan ({c} lux): Rendah={res['mu_pencahayaan']['Rendah']:.2f}, Sedang={res['mu_pencahayaan']['Sedang']:.2f}, Tinggi={res['mu_pencahayaan']['Tinggi']:.2f}")
    print("2. Implikasi MIN & Aturan Aktif:")
    for a in res["aturan_aktif"]:
        print(f"   - {a['kode_aturan']}: min(mu) = {a['alpha_min']:.2f} -> Output '{a['output_linguistik']}' (Ref Z={a['konsekuen_z_ref']})")
    print("3. Agregasi MAX & Defuzzifikasi Centroid:")
    print(f"   - Titik Pusat Luasan (Z* Centroid) = {res['z_centroid']:.4f} produk")
    print("-" * 65)
    return res


def plot_mamdani_output_dan_centroid(res_data_10: Dict[str, Any], output_path: str = "output/grafik_mamdani_centroid_data10.png") -> None:
    """Memvisualisasikan himpunan fuzzy output Mamdani serta daerah arsiran agregasi Data 10."""
    mf_out = kurva_himpunan_output()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    fig.suptitle("Visualisasi FIS Mamdani: Himpunan Output & Proses Centroid (Data No. 10)", fontsize=13, fontweight="bold")

    # Subplot 1: 3 Kurva Himpunan Output Mamdani
    axes[0].plot(SEMESTA_OUTPUT, mf_out["Rendah"], "b-", lw=2, label="Rendah - Trapesium (130,130,138,143)")
    axes[0].plot(SEMESTA_OUTPUT, mf_out["Sedang"], color="orange", lw=2, label="Sedang - Segitiga (138,144,150)")
    axes[0].plot(SEMESTA_OUTPUT, mf_out["Tinggi"], "g-", lw=2, label="Tinggi - Trapesium (145,150,158,158)")
    axes[0].set_title("1. Himpunan Fuzzy Output Jumlah Produksi", fontweight="bold")
    axes[0].set_xlabel("Jumlah Produksi (unit)")
    axes[0].set_ylabel("Derajat Keanggotaan, μ(z)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=8)

    # Subplot 2: Daerah Agregasi & Garis Centroid Data 10
    axes[1].plot(SEMESTA_OUTPUT, mf_out["Tinggi"], "g--", lw=1.2, alpha=0.7, label="Kurva Output Tinggi (Utuh)")
    axes[1].fill_between(SEMESTA_OUTPUT, 0, res_data_10["agregasi_kurva"], color="skyblue", alpha=0.65, label="Daerah Agregasi MAX (α_R1=0.2, α_R10=0.8)")
    axes[1].plot(SEMESTA_OUTPUT, res_data_10["agregasi_kurva"], "b-", lw=2)
    z_c = res_data_10["z_centroid"]
    axes[1].axvline(z_c, color="red", linestyle="-.", lw=2, label=f"Titik Centroid Z* = {z_c:.4f}")
    axes[1].set_title("2. Hasil Agregasi & Titik Centroid (Data No. 10)", fontweight="bold")
    axes[1].set_xlabel("Jumlah Produksi (unit)")
    axes[1].set_ylabel("Derajat Keanggotaan Terpotong")
    axes[1].set_ylim(0, 1.1)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def simulasi_interaktif(aturan_27: List[Dict[str, Any]]) -> None:
    """Mode opsional CLI untuk menguji kombinasi input baru secara dinamis pada FIS Mamdani."""
    print("\n--- MODE SIMULASI INTERAKTIF FIS MAMDANI ---")
    try:
        s = float(input("Masukkan Suhu (°C) [18 - 36]         : "))
        b = float(input("Masukkan Kebisingan (dB) [40 - 105]  : "))
        c = float(input("Masukkan Pencahayaan (lux) [50 - 650]: "))
        validasi_input_fuzzy(s, b, c)
        res = inferensi_mamdani(s, b, c, aturan_27)
        print(f"Derajat Suhu       : {res['mu_suhu']}")
        print(f"Derajat Kebisingan : {res['mu_kebisingan']}")
        print(f"Derajat Pencahayaan: {res['mu_pencahayaan']}")
        print("Aturan Aktif       :", res["aturan_aktif"])
        print(f"Prediksi Centroid Z*: {res['z_centroid']:.4f} produk")
    except ValueError as err:
        print(f"[ERROR] {err}")


def main() -> None:
    os.makedirs("output", exist_ok=True)
    aturan_27 = bangun_27_aturan()

    hasil_rows = []
    for no, suhu, bising, cahaya, aktual, _ in DATASET_PRODUKSI:
        res = inferensi_mamdani(suhu, bising, cahaya, aturan_27)
        ringkasan = "; ".join(
            [f"{a['kode_aturan']}(α={a['alpha_min']}->{a['output_linguistik']})" for a in res["aturan_aktif"]]
        )
        hasil_rows.append({
            "No": no,
            "Suhu_C": suhu,
            "Kebisingan_dB": bising,
            "Pencahayaan_lux": cahaya,
            "Aturan_Aktif_Mamdani": ringkasan,
            "Produksi_Aktual": aktual,
            "Prediksi_Mamdani_Centroid": round(res["z_centroid"], 4),
            "Selisih_Absolut": round(abs(aktual - res["z_centroid"]), 4),
        })

    df_mamdani = pd.DataFrame(hasil_rows)
    df_mamdani.to_csv("output/hasil_defuzzifikasi_mamdani.csv", index=False)

    print("-" * 65)
    print("HASIL IMPLEMENTASI FIS MAMDANI (MIN-MAX & CENTROID)")
    print("-" * 65)
    print(df_mamdani[["No", "Suhu_C", "Kebisingan_dB", "Pencahayaan_lux", "Aturan_Aktif_Mamdani", "Produksi_Aktual", "Prediksi_Mamdani_Centroid"]].to_string(index=False))
    print("-" * 65)
    print(f"Rata-rata Jumlah Produksi Mamdani : {df_mamdani['Prediksi_Mamdani_Centroid'].mean():.4f} produk")
    print(f"Rata-rata Error Absolut (MAE)     : {df_mamdani['Selisih_Absolut'].mean():.4f} produk")
    print("-" * 65)

    res_10 = cetak_rincian_manual_data_10(aturan_27)
    plot_mamdani_output_dan_centroid(res_10)

    # Jalankan mode interaktif hanya jika argumen --interaktif diberikan
    if "--interaktif" in sys.argv:
        simulasi_interaktif(aturan_27)


if __name__ == "__main__":
    main()
