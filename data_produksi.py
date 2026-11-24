from typing import Any, Dict, List, Tuple

# Batas semesta pembicaraan masing-masing variabel input (min, max)
BATAS_VARIABEL: Dict[str, Tuple[float, float]] = {
    "suhu": (18.0, 36.0),
    "kebisingan": (40.0, 105.0),
    "pencahayaan": (50.0, 650.0),
}

# Dataset 27 kombinasi rata-rata dari 30 pekerja (total 810 data observasi)
# Format tuple: (No, Suhu_C, Kebisingan_dB, Pencahayaan_lux, Rata_rata_Produksi, Standar_Deviasi)
DATASET_PRODUKSI: List[Tuple[int, float, float, float, float, float]] = [
    (1, 22, 55, 150, 148.00, 4.71),
    (2, 22, 55, 300, 150.90, 4.78),
    (3, 22, 55, 500, 146.50, 4.90),
    (4, 22, 75, 150, 143.10, 4.90),
    (5, 22, 75, 300, 146.53, 4.58),
    (6, 22, 75, 500, 142.73, 5.42),
    (7, 22, 90, 150, 136.73, 4.49),
    (8, 22, 90, 300, 140.77, 4.49),
    (9, 22, 90, 500, 135.97, 4.75),
    (10, 26, 55, 150, 149.73, 4.43),
    (11, 26, 55, 300, 153.27, 5.59),
    (12, 26, 55, 500, 152.13, 5.04),
    (13, 26, 75, 150, 148.00, 5.15),
    (14, 26, 75, 300, 150.63, 5.06),
    (15, 26, 75, 500, 147.63, 4.84),
    (16, 26, 90, 150, 141.47, 5.69),
    (17, 26, 90, 300, 145.67, 4.81),
    (18, 26, 90, 500, 140.20, 4.76),
    (19, 32, 55, 150, 142.10, 4.28),
    (20, 32, 55, 300, 146.53, 5.38),
    (21, 32, 55, 500, 142.17, 4.53),
    (22, 32, 75, 150, 138.70, 4.84),
    (23, 32, 75, 300, 141.40, 4.95),
    (24, 32, 75, 500, 138.30, 5.12),
    (25, 32, 90, 150, 133.33, 4.71),
    (26, 32, 90, 300, 138.53, 4.51),
    (27, 32, 90, 500, 137.77, 4.83),
]

# Label himpunan linguistik untuk masing-masing variabel input
HIMPUNAN_SUHU: List[str] = ["Dingin", "Normal", "Panas"]
HIMPUNAN_KEBISINGAN: List[str] = ["Rendah", "Sedang", "Tinggi"]
HIMPUNAN_PENCAHAYAAN: List[str] = ["Rendah", "Sedang", "Tinggi"]


def validasi_input_fuzzy(suhu: float, kebisingan: float, pencahayaan: float) -> None:
    """
    Memvalidasi bahwa seluruh nilai input berada di dalam rentang semesta pembicaraan:
    - Suhu        : [18, 36] °C
    - Kebisingan  : [40, 105] dB
    - Pencahayaan : [50, 650] lux
    Melempar ValueError jika terdapat nilai di luar rentang semesta.
    """
    kesalahan: List[str] = []
    if not (BATAS_VARIABEL["suhu"][0] <= suhu <= BATAS_VARIABEL["suhu"][1]):
        kesalahan.append(f"Suhu ({suhu}°C) di luar semesta [18, 36] °C")
    if not (BATAS_VARIABEL["kebisingan"][0] <= kebisingan <= BATAS_VARIABEL["kebisingan"][1]):
        kesalahan.append(f"Kebisingan ({kebisingan} dB) di luar semesta [40, 105] dB")
    if not (BATAS_VARIABEL["pencahayaan"][0] <= pencahayaan <= BATAS_VARIABEL["pencahayaan"][1]):
        kesalahan.append(f"Pencahayaan ({pencahayaan} lux) di luar semesta [50, 650] lux")

    if kesalahan:
        raise ValueError("Input di luar rentang semesta variabel: " + "; ".join(kesalahan))


def kategori_output_linguistik(nilai_produksi: float) -> str:
    """
    Memetakan nilai numerik jumlah produksi ke himpunan linguistik output (untuk FIS Mamdani):
    - Rendah : < 141.0 unit
    - Sedang : 141.0 s.d. 147.5 unit
    - Tinggi : > 147.5 unit
    """
    if nilai_produksi < 141.0:
        return "Rendah"
    elif nilai_produksi <= 147.5:
        return "Sedang"
    else:
        return "Tinggi"


def bangun_27_aturan() -> List[Dict[str, Any]]:
    """
    Membentuk 27 aturan Fuzzy (R1 - R27) dari hasil perkalian Kartesian 3x3x3 himpunan input.
    - Untuk FIS Sugeno Orde-Nol: konsekuen berupa konstanta riil (konsekuen_z).
    - Untuk FIS Mamdani        : konsekuen berupa himpunan linguistik (konsekuen_linguistik).
    """
    aturan_list: List[Dict[str, Any]] = []
    idx = 0
    for s in HIMPUNAN_SUHU:
        for b in HIMPUNAN_KEBISINGAN:
            for c in HIMPUNAN_PENCAHAYAAN:
                nilai_z = DATASET_PRODUKSI[idx][4]
                aturan_list.append({
                    "kode_aturan": f"R{idx + 1}",
                    "suhu": s,
                    "kebisingan": b,
                    "pencahayaan": c,
                    "konsekuen_z": nilai_z,
                    "konsekuen_linguistik": kategori_output_linguistik(nilai_z),
                })
                idx += 1
    return aturan_list
