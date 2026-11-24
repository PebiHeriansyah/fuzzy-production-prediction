import argparse
from typing import Any, Dict, List
from data_produksi import bangun_27_aturan, validasi_input_fuzzy
from fuzzy_sugeno_produksi import inferensi_sugeno
from fuzzy_mamdani_produksi import inferensi_mamdani


def format_derajat(mu: Dict[str, float]) -> str:
    """Memformat dictionary derajat keanggotaan menjadi teks ringkas."""
    return ", ".join(f"{k}={v:.4f}" for k, v in mu.items())


def jalankan_perbandingan(
    suhu: float,
    kebisingan: float,
    pencahayaan: float,
    aturan_27: List[Dict[str, Any]],
) -> None:
    """Memvalidasi input lalu menampilkan hasil FIS Sugeno vs FIS Mamdani secara berdampingan."""
    validasi_input_fuzzy(suhu, kebisingan, pencahayaan)

    res_sugeno = inferensi_sugeno(suhu, kebisingan, pencahayaan, aturan_27)
    res_mamdani = inferensi_mamdani(suhu, kebisingan, pencahayaan, aturan_27)

    z_sugeno = float(res_sugeno["hasil_z"])
    z_mamdani = float(res_mamdani["z_centroid"])
    selisih = z_sugeno - z_mamdani

    print("\n" + "-" * 65)
    print("SIMULASI INTERAKTIF: FIS SUGENO ORDE-NOL VS FIS MAMDANI")
    print("-" * 65)
    print(f"Input Crisp : Suhu = {suhu:.2f}°C | Kebisingan = {kebisingan:.2f} dB | Pencahayaan = {pencahayaan:.2f} lux")
    print("-" * 65)
    print("1. Derajat Keanggotaan Input (Fuzzifikasi):")
    print(f"   - Suhu        : {format_derajat(res_sugeno['mu_suhu'])}")
    print(f"   - Kebisingan  : {format_derajat(res_sugeno['mu_kebisingan'])}")
    print(f"   - Pencahayaan : {format_derajat(res_sugeno['mu_pencahayaan'])}")
    print("-" * 65)
    print("2. Aturan Aktif:")
    print("   - Sugeno (Product):")
    for a in res_sugeno["aturan_aktif"]:
        print(f"     * {a['kode_aturan']}: alpha = {a['alpha']:.4f} -> Z_i = {a['konsekuen_z']:.2f}")
    print("   - Mamdani (MIN):")
    for a in res_mamdani["aturan_aktif"]:
        print(f"     * {a['kode_aturan']}: alpha = {a['alpha_min']:.4f} -> Output '{a['output_linguistik']}' (Ref Z={a['konsekuen_z_ref']:.2f})")
    print("-" * 65)
    print("3. Hasil Prediksi (Defuzzifikasi):")
    print(f"   {'FIS Sugeno (Weighted Average)':<32} : {z_sugeno:.4f} produk")
    print(f"   {'FIS Mamdani (Centroid)':<32} : {z_mamdani:.4f} produk")
    print(f"   {'Selisih (Sugeno - Mamdani)':<32} : {selisih:+.4f} produk")
    print("-" * 65)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulasi Interaktif Terpadu FIS Sugeno Orde-Nol vs FIS Mamdani"
    )
    parser.add_argument("--suhu", type=float, help="Nilai Suhu (°C) dalam rentang [18, 36]")
    parser.add_argument("--kebisingan", type=float, help="Nilai Kebisingan (dB) dalam rentang [40, 105]")
    parser.add_argument("--pencahayaan", type=float, help="Nilai Pencahayaan (lux) dalam rentang [50, 650]")
    args = parser.parse_args()

    aturan_27 = bangun_27_aturan()

    try:
        if args.suhu is not None and args.kebisingan is not None and args.pencahayaan is not None:
            s, b, c = args.suhu, args.kebisingan, args.pencahayaan
        else:
            print("\n--- INPUT SIMULASI TERPADU (SUGENO VS MAMDANI) ---")
            s = float(input("Masukkan Suhu (°C) [18 - 36]         : "))
            b = float(input("Masukkan Kebisingan (dB) [40 - 105]  : "))
            c = float(input("Masukkan Pencahayaan (lux) [50 - 650]: "))

        jalankan_perbandingan(s, b, c, aturan_27)
    except ValueError as err:
        print(f"[ERROR] {err}")


if __name__ == "__main__":
    main()
