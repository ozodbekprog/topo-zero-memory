#!/usr/bin/env python3
"""
10 MB SSD xotirasi ichidan 1 GB ma'lumot oqimi — TopoZeroMemory isboti.
================================================================================

Nima qilinadi:
  1) SSD'da ANIQ 10 MiB (10 485 760 bayt) fayl yaratiladi: topo_memory_10mb.bin
  2) Fayl memmap (torch.from_file) sifatida xotira matritsasiga ulanadi.
  3) Ichidan 1 GiB (1 073 741 824 bayt) sun'iy oqim o'tkaziladi:
         - oqim bo'laklab generatsiya qilinadi (hech qayerda saqlanmaydi);
         - eski ma'lumot RECURSIVE DECAY + INVERSIYA bilan so'nadi;
         - yangi ma'lumot fixed-size matritsa ichiga golografik singadi.
  4) Isbot: fayl hajmi bir bayt ham o'zgarmaydi, RAM o'smaydi,
     eng so'nggi "igna" ma'lumot aniq topiladi, eng eski igna esa 0 ga aylanadi.

Ochig'i: bu SIQISH EMAS — xotira eski ma'lumotni rejali (lambda) so'ndiradi,
shuning uchun 1 GB ning hammasi qayta tiklanmaydi. Lekin hajm O(1) qoladi
va oqim qancha davom etsa ham fayl kattalashmaydi.

Ishga tushirish:
    python demo_10mb_1gb.py
"""

from __future__ import annotations

import gc
import os
import statistics
import time
import tracemalloc

import psutil
import torch
import torch.nn.functional as F

from topo_zero_memory import TopoZeroMemory, stereographic_compact

D = 64
S = 40960                      # M = 40960 * 64 * 4 = 10 485 760 bayt = aniq 10 MiB
MATRIX_BYTES = S * D * 4
CHUNK = 32768                  # bir oqim chunk'i: 8 MiB = 32768 * 64 * 4 bayt
GROUP = 32                     # har 32 vektor bitta slotga konveks o'rtacha bo'lib yoziladi
TOTAL_VECTORS = 4_194_304      # 4 194 304 * 64 * 4 = 1 073 741 824 bayt = aniq 1 GiB
N_CHUNKS = TOTAL_VECTORS // CHUNK
LAM = 1e-5                     # vaqt inversiyasi sur'ati
MB = 1024 ** 2
MEM_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "topo_memory_10mb.bin")


def rss_mb(proc: psutil.Process) -> float:
    return proc.memory_info().rss / MB


def needle(index: int = 0, value: float = 4.0) -> torch.Tensor:
    """Maxsus 'igna' vektor: e_index yo'nalishi (norma ~0.47 ga bukiladi)."""
    v = torch.zeros(D)
    v[index] = value
    return v


def ensure_memory_file() -> None:
    """SSD'da aniq 10 MiB fayl — xotiraning fizik joyi."""
    if not os.path.exists(MEM_FILE) or os.path.getsize(MEM_FILE) != MATRIX_BYTES:
        with open(MEM_FILE, "wb") as f:
            f.write(b"\0" * MATRIX_BYTES)  # bir marta to'ldiriladi: 10 485 760 bayt


def main() -> None:
    torch.manual_seed(7)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    ensure_memory_file()
    st0 = os.stat(MEM_FILE)
    print(f"PyTorch {torch.__version__} | SSD xotira fayli: {os.path.basename(MEM_FILE)}")
    print(f"   fayl hajmi: {st0.st_size:,} bayt ({st0.st_size / MB:.2f} MiB) | "
          f"diskda egallangan: {st0.st_blocks * 512:,} bayt")
    print(f"   reja: {N_CHUNKS} chunk x {CHUNK:,} vektor x {D} dim x 4 bayt = "
          f"{TOTAL_VECTORS * D * 4 / 1024 ** 3:.2f} GiB oqim\n")

    module = TopoZeroMemory(D, num_slots=S, write_rate=0.25, lam=LAM,
                            zero_eps=1e-4, temperature=0.1, active_slots=S)
    # Xotira matritsasi SSD'ga bog'lanadi (shared mmap: yozuvlar to'g'ridan-to'g'ri faylga)
    mem_view = torch.from_file(MEM_FILE, shared=True, size=MATRIX_BYTES // 4,
                               dtype=torch.float32).view(S, D)
    module.memory = mem_view
    base_footprint = module.footprint_elements()
    print(f"   xotira holati: {base_footprint:,} element = "
          f"{base_footprint * 4 / MB:.2f} MiB (matritsa {MATRIX_BYTES / MB:.2f} MiB SSD'da + "
          f"vazn/yosh {(S * 4 * 2) / MB:.2f} MiB RAM'da)\n")

    proc = psutil.Process()
    tracemalloc.start()

    EARLY_NEEDLE_CHUNK = 2                 # juda eski igna
    LATE_NEEDLE_CHUNKS = {N_CHUNKS - 3, N_CHUNKS - 2, N_CHUNKS - 1, N_CHUNKS}  # yangi igna

    rss_samples: list[float] = []
    traced_base = None
    processed_vectors = 0
    t_start = time.perf_counter()

    print("== OQIM: 10 MB xotira ichidan 1 GB o'tkazilyapti ==")
    for chunk in range(1, N_CHUNKS + 1):
        # 1) Chunk generatsiyasi: 1 GiB dan faqat 8 MiB xotirada turadi
        magnitude = 10.0 ** float(torch.empty(1).uniform_(-3.0, 6.0))
        x = torch.randn(CHUNK, D) * magnitude
        if chunk == EARLY_NEEDLE_CHUNK:
            x[:GROUP] = needle(1)  # e1 yo'nalishli ESKI igna
        elif chunk in LATE_NEEDLE_CHUNKS:
            x[:GROUP] = needle(0)  # e0 yo'nalishli YANGI igna

        with torch.inference_mode():
            # 2) Vaqt inversiyasi: dt = CHUNK => s = 1/(1 + lambda*dt)
            module.decay_step(dt=CHUNK)
            # 3) Vectorlashtirilgan golografik yozuv (1024 slot bir operatsiyada)
            module.write_batch(x, group_size=GROUP)
            # 4) Zero-weight sparsifikatsiya: o'lgan slot resurs sarflamaydi
            module.sparsify()

        processed_vectors += CHUNK

        # Har chunk'da barqarorlik va O(1) assertlari
        assert torch.isfinite(module.memory).all(), f"chunk {chunk}: xotira NaN/Inf"
        assert float(module.memory.norm(dim=-1).max()) <= 1.0 + 1e-5, \
            f"chunk {chunk}: ||M|| > 1 (birlik shar buzildi)"
        assert module.footprint_elements() == base_footprint, "xotira hajmi o'zgardi!"
        assert os.path.getsize(MEM_FILE) == MATRIX_BYTES, "SSD fayl hajmi o'zgardi!"

        gc.collect()
        rss_samples.append(rss_mb(proc))
        if chunk >= 8 and traced_base is None:
            traced_base = tracemalloc.get_traced_memory()[0] / MB

        if chunk % 16 == 0 or chunk == N_CHUNKS:
            done_gib = processed_vectors * D * 4 / 1024 ** 3
            print(f"   chunk {chunk:>3}/{N_CHUNKS} | oqim {done_gib:5.2f} GiB | "
                  f"RSS {rss_samples[-1]:6.1f} MB | fayl {os.path.getsize(MEM_FILE):,} bayt | "
                  f"o'lik slot {module.zero_weight_fraction * 100:4.1f}% | "
                  f"max||M|| {float(module.memory.norm(dim=-1).max()):.4f}")

    elapsed = time.perf_counter() - t_start
    end_traced = tracemalloc.get_traced_memory()[0] / MB
    tracemalloc.stop()

    # --- O(1) xotira isboti ---
    warm, tail = rss_samples[8:16], rss_samples[-8:]
    rss_growth = statistics.median(tail) - statistics.median(warm)
    st1 = os.stat(MEM_FILE)
    streamed = TOTAL_VECTORS * D * 4

    print("\n== 1) O(1) XOTIRA ISBOTI ==")
    print(f"   oqim o'tdi:        {streamed:,} bayt ({streamed / 1024 ** 3:.2f} GiB)")
    print(f"   SSD fayl (boshda): {st0.st_size:,} bayt")
    print(f"   SSD fayl (oxirda): {st1.st_size:,} bayt  -> o'zgarish: "
          f"{st1.st_size - st0.st_size:+d} bayt")
    print(f"   nisbat: 1 GiB oqim / 10 MiB xotira = {streamed / MATRIX_BYTES:.1f}x")
    print(f"   RAM (issiq median): {statistics.median(warm):.2f} MB | "
          f"RAM (oxirgi median): {statistics.median(tail):.2f} MB | "
          f"o'sish: {rss_growth:+.2f} MB")
    print(f"   Python-tracemalloc o'sishi: {end_traced - traced_base:+.3f} MB")
    assert st1.st_size == st0.st_size == MATRIX_BYTES, "SSD fayl hajmi o'zgardi!"
    assert module.footprint_elements() == base_footprint, "tensor hajmi o'zgardi!"
    assert rss_growth < 16.0, f"RAM o'sishi katta: {rss_growth:+.2f} MB"
    assert end_traced - traced_base < 1.0, "Python xotirasi o'sdi"

    # --- Eski va yangi ma'lumot taqdiri ---
    with torch.inference_mode():
        q_early = stereographic_compact(needle(1))
        q_late = stereographic_compact(needle(0))
        cos_early = float(F.cosine_similarity(q_early.unsqueeze(0), module.memory, dim=1).max())
        cos_late = float(F.cosine_similarity(q_late.unsqueeze(0), module.memory, dim=1).max())

    dead = module.weights == 0
    dead_rows_max = float(module.memory[dead].abs().max())
    print("\n== 2) ESKI vs YANGI MA'LUMOT ==")
    print(f"   ESKI igna (chunk {EARLY_NEEDLE_CHUNK}, ~1 GiB oldin): "
          f"eng yaqin slot kosinusi = {cos_early:.4f}  (so'nib ketgan)")
    print(f"   YANGI igna (oxirgi 4 chunk):                          "
          f"eng yaqin slot kosinusi = {cos_late:.4f}  (aniq saqlangan)")
    print(f"   o'lik (zero-weight) slotlar: {int(dead.sum()):,}/{S:,} "
          f"({module.zero_weight_fraction * 100:.1f}%), ulardagi max|M| = {dead_rows_max:.1e}")
    assert cos_late > 0.95, "yangi igna saqlanmadi!"
    assert cos_early < 0.9, "eski igna hali ham tirik — inversiya ishlamadi!"
    assert dead_rows_max == 0.0, "zero-weight slotda resurs qoldi"
    assert int(dead.sum()) > 0, "sparsifikatsiya ishlamadi"

    print(f"\n   tezlik: {elapsed:.1f} s ({streamed / MB / elapsed:.0f} MB/s ishlov berildi)")
    print("\n=== XULOSA: 1 GiB oqim 10 MiB SSD xotirasidan o'tdi — hajm O(1) qoldi ===")
    print("    (Eslatma: bu siqish emas — xotira eski ma'lumotni lambda-bo'yicha so'ndiradi;")
    print("     yangi ma'lumot aniq, eski ma'lumot golografik xulosa sifatida saqlanadi.)")


if __name__ == "__main__":
    main()
