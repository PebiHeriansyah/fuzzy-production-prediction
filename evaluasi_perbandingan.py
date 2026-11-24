import os
import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_produksi import DATASET_PRODUKSI, bangun_27_aturan
from fuzzy_sugeno_produksi import fungsi_segitiga, fungsi_trapesium, inferensi_sugeno
from fuzzy_mamdani_produksi import inferensi_mamdani


def evaluasi_segitiga_vs_trapesium() -> pd.DataFrame:
    """
    Membandingkan performa 3 rancangan fungsi keanggotaan input (Poin 3):
    1. Kombinasi Trapesium-Segitiga (Rancangan Utama)
    2. Full Segitiga (seluruh himpunan menggunakan kurva segitiga)
    3. Full Trapesium (seluruh himpunan menggunakan kurva trapesium)
    """
    aturan_27 = bangun_27_aturan()

    def mu_full_segitiga(
        s: float, b: float, c: float
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        # Varian Full Segitiga: puncak tunggal pada titik referensi tanpa area datar (core)
        ms = {
            "Dingin": fungsi_segitiga(s, 14, 18, 27) if s > 18 else 1.0,
            "Normal": fungsi_segitiga(s, 22, 27, 32),
            "Panas": fungsi_segitiga(s, 27, 36, 40) if s < 36 else 1.0,
        }
        mb = {
            "Rendah": fungsi_segitiga(b, 25, 40, 75) if b > 40 else 1.0,
            "Sedang": fungsi_segitiga(b, 55, 75, 90),
            "Tinggi": fungsi_segitiga(b, 75, 105, 120) if b < 105 else 1.0,
        }
        mc = {
            "Rendah": fungsi_segitiga(c, 0, 50, 300) if c > 50 else 1.0,
            "Sedang": fungsi_segitiga(c, 150, 300, 500),
            "Tinggi": fungsi_segitiga(c, 300, 650, 800) if c < 650 else 1.0,
        }
        return ms, mb, mc

    def mu_full_trapesium(
        s: float, b: float, c: float
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        # Varian Full Trapesium: seluruh himpunan termasuk kategori tengah memiliki area datar
        ms = {
            "Dingin": fungsi_trapesium(s, 18, 18, 22, 27),
            "Normal": fungsi_trapesium(s, 22, 25, 29, 32),
            "Panas": fungsi_trapesium(s, 27, 32, 36, 36),
        }
        mb = {
            "Rendah": fungsi_trapesium(b, 40, 40, 55, 75),
            "Sedang": fungsi_trapesium(b, 55, 70, 80, 90),
            "Tinggi": fungsi_trapesium(b, 75, 90, 105, 105),
        }
        mc = {
            "Rendah": fungsi_trapesium(c, 50, 50, 150, 300),
            "Sedang": fungsi_trapesium(c, 150, 250, 350, 500),
            "Tinggi": fungsi_trapesium(c, 300, 500, 650, 650),
        }
        return ms, mb, mc

    def hitung_z_kustom(
        ms: Dict[str, float], mb: Dict[str, float], mc: Dict[str, float]
    ) -> float:
        # Inferensi Sugeno Orde-Nol (Product & Weighted Average) untuk rancangan kurva kustom
        num, den = 0.0, 0.0
        for r in aturan_27:
            alpha = ms[r["suhu"]] * mb[r["kebisingan"]] * mc[r["pencahayaan"]]
            num += alpha * r["konsekuen_z"]
            den += alpha
        return num / den if den != 0 else 0.0

    rows = []
    for no, s, b, c, aktual, _ in DATASET_PRODUKSI:
        z_kombinasi = inferensi_sugeno(s, b, c, aturan_27)["hasil_z"]
        z_segitiga = hitung_z_kustom(*mu_full_segitiga(s, b, c))
        z_trapesium = hitung_z_kustom(*mu_full_trapesium(s, b, c))
        rows.append({
            "No": no,
            "Suhu": s,
            "Kebisingan": b,
            "Pencahayaan": c,
            "Aktual": aktual,
            "Kombinasi_Trap_Seg": round(z_kombinasi, 4),
            "Full_Segitiga": round(z_segitiga, 4),
            "Full_Trapesium": round(z_trapesium, 4),
        })

    df_mf = pd.DataFrame(rows)
    df_mf.to_csv("output/evaluasi_segitiga_vs_trapesium.csv", index=False)
    return df_mf


def evaluasi_sugeno_vs_mamdani() -> pd.DataFrame:
    """Membandingkan akurasi (MAE/RMSE) dan kecepatan komputasi antara FIS Sugeno dan FIS Mamdani (Poin 4)."""
    aturan_27 = bangun_27_aturan()

    # Pengukuran waktu eksekusi 27 data untuk FIS Sugeno (aljabar langsung)
    t0_sugeno = time.perf_counter()
    hasil_sugeno = [inferensi_sugeno(s, b, c, aturan_27)["hasil_z"] for _, s, b, c, _, _ in DATASET_PRODUKSI]
    waktu_sugeno_ms = (time.perf_counter() - t0_sugeno) * 1000

    # Pengukuran waktu eksekusi 27 data untuk FIS Mamdani (diskretisasi 1401 titik Centroid)
    t0_mamdani = time.perf_counter()
    hasil_mamdani = [inferensi_mamdani(s, b, c, aturan_27)["z_centroid"] for _, s, b, c, _, _ in DATASET_PRODUKSI]
    waktu_mamdani_ms = (time.perf_counter() - t0_mamdani) * 1000

    rows = []
    for idx, (no, s, b, c, aktual, _) in enumerate(DATASET_PRODUKSI):
        zs = hasil_sugeno[idx]
        zm = hasil_mamdani[idx]
        rows.append({
            "No": no,
            "Suhu": s,
            "Kebisingan": b,
            "Pencahayaan": c,
            "Aktual": aktual,
            "FIS_Sugeno_WA": round(zs, 4),
            "FIS_Mamdani_Centroid": round(zm, 4),
            "Error_Sugeno": round(abs(aktual - zs), 4),
            "Error_Mamdani": round(abs(aktual - zm), 4),
        })

    df_fis = pd.DataFrame(rows)
    df_fis.attrs["waktu_sugeno_ms"] = waktu_sugeno_ms
    df_fis.attrs["waktu_mamdani_ms"] = waktu_mamdani_ms
    df_fis.to_csv("output/evaluasi_sugeno_vs_mamdani.csv", index=False)
    return df_fis


def analisis_anomali_keanggotaan() -> pd.DataFrame:
    """Mendeteksi kasus anomali derajat keanggotaan pada titik batas dan area transisi."""
    aturan_27 = bangun_27_aturan()
    anomali_list: List[Dict[str, object]] = []

    for no, s, b, c, aktual, _ in DATASET_PRODUKSI:
        res = inferensi_sugeno(s, b, c, aturan_27)
        selisih = res["hasil_z"] - aktual

        # Kasus 1: Titik batas bahu trapesium langsung memperoleh mu = 1.0 tanpa transisi
        if (s == 22 and res["mu_suhu"]["Dingin"] == 1.0) or (b == 55 and res["mu_kebisingan"]["Rendah"] == 1.0):
            catatan = "Titik sudut bahu trapesium memberi mu=1.0 penuh pada kategori ekstrem tanpa transisi"
        # Kasus 2: Pada 26°C, mu_Dingin=0.2 ikut mengaktifkan aturan R1-R9 dan menarik turun prediksi
        elif abs(selisih) > 0.3:
            catatan = f"Tarikan aturan tetangga (mu_Dingin=0.2, mu_Normal=0.8) menggeser prediksi {selisih:+.4f}"
        else:
            catatan = "Konsisten"

        if catatan != "Konsisten":
            anomali_list.append({
                "No": no,
                "Input_(S,B,C)": f"({s}, {b}, {c})",
                "Aktual": aktual,
                "Prediksi_Sugeno": round(res["hasil_z"], 4),
                "Selisih": round(selisih, 4),
                "Keterangan_Anomali": catatan,
            })

    df_anomali = pd.DataFrame(anomali_list)
    df_anomali.to_csv("output/analisis_anomali_keanggotaan.csv", index=False)
    return df_anomali


def plot_perbandingan_model(df_mf: pd.DataFrame, df_fis: pd.DataFrame) -> None:
    """Membuat grafik perbandingan hasil prediksi antar metode."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))

    axes[0].plot(df_mf["No"], df_mf["Aktual"], "ko-", lw=1.8, label="Data Aktual")
    axes[0].plot(df_mf["No"], df_mf["Kombinasi_Trap_Seg"], "b--s", lw=1.5, label="Kombinasi Trapesium-Segitiga (Utama)")
    axes[0].plot(df_mf["No"], df_mf["Full_Segitiga"], "r:^", lw=1.5, label="Full Segitiga")
    axes[0].plot(df_mf["No"], df_mf["Full_Trapesium"], "g-.d", lw=1.5, label="Full Trapesium")
    axes[0].set_title("Perbandingan Kurva Keanggotaan: Kombinasi vs Full Segitiga vs Full Trapesium", fontweight="bold")
    axes[0].set_ylabel("Jumlah Produksi")
    axes[0].set_xticks(df_mf["No"])
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].plot(df_fis["No"], df_fis["Aktual"], "ko-", lw=1.8, label="Data Aktual")
    axes[1].plot(df_fis["No"], df_fis["FIS_Sugeno_WA"], "b--s", lw=1.5, label="FIS Sugeno Orde-Nol (Weighted Average)")
    axes[1].plot(df_fis["No"], df_fis["FIS_Mamdani_Centroid"], "m-o", lw=1.5, label="FIS Mamdani (Min-Max Centroid)")
    axes[1].set_title("Perbandingan Metode Inferensi: FIS Sugeno vs FIS Mamdani", fontweight="bold")
    axes[1].set_xlabel("Nomor Kombinasi Data (1 - 27)")
    axes[1].set_ylabel("Jumlah Produksi")
    axes[1].set_xticks(df_fis["No"])
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("output/grafik_evaluasi_perbandingan.png", dpi=200)
    plt.close()


def main() -> None:
    os.makedirs("output", exist_ok=True)
    df_mf = evaluasi_segitiga_vs_trapesium()
    df_fis = evaluasi_sugeno_vs_mamdani()
    df_anomali = analisis_anomali_keanggotaan()
    plot_perbandingan_model(df_mf, df_fis)

    print("=== RINGKASAN EVALUASI 3: SEGITIGA VS TRAPESIUM ===")
    for col in ["Kombinasi_Trap_Seg", "Full_Segitiga", "Full_Trapesium"]:
        err = np.abs(df_mf["Aktual"] - df_mf[col])
        rmse = np.sqrt(np.mean(err ** 2))
        print(f"{col:<22} | Mean Z = {df_mf[col].mean():.4f} | MAE = {err.mean():.4f} | RMSE = {rmse:.4f} | MaxErr = {err.max():.4f}")

    print("\n=== RINGKASAN EVALUASI 4: FIS SUGENO VS FIS MAMDANI ===")
    rmse_s = np.sqrt(np.mean(df_fis["Error_Sugeno"] ** 2))
    rmse_m = np.sqrt(np.mean(df_fis["Error_Mamdani"] ** 2))
    print(f"FIS Sugeno (Weighted Avg) | Mean Z = {df_fis['FIS_Sugeno_WA'].mean():.4f} | MAE = {df_fis['Error_Sugeno'].mean():.4f} | RMSE = {rmse_s:.4f} | Waktu = {df_fis.attrs['waktu_sugeno_ms']:.2f} ms")
    print(f"FIS Mamdani (Centroid)    | Mean Z = {df_fis['FIS_Mamdani_Centroid'].mean():.4f} | MAE = {df_fis['Error_Mamdani'].mean():.4f} | RMSE = {rmse_m:.4f} | Waktu = {df_fis.attrs['waktu_mamdani_ms']:.2f} ms")

    print("\n=== RINGKASAN DETEKSI ANOMALI KEANGGOTAAN (DATA 10-18) ===")
    print(df_anomali[df_anomali["No"].between(10, 18)].to_string(index=False))


if __name__ == "__main__":
    main()
