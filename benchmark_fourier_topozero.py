#!/usr/bin/env python3
"""
BENCHMARK: 1 MB Fourier-Lossless bufer + TopoZeroMemory O(1) gologramma.
================================================================================

Isbotlanadi:
  1) 10 000+ qadamli (10 240 qadam) cheksiz bayt oqimi qayta ishlanadi.
  2) Lossless FFT buferi (aniq 1 MiB koeffitsient) oqimning oxirgi 64 KiB
     oynasini saqlaydi va davriy ravishda IFFT bilan 100% aniq tiklanadi:
         assert original_window == reconstructed_window
  3) TopoZeroMemory gologrammasi cheksiz oqimni O(1) xotirada kuzatadi —
     RAM ham, bufer hajmi ham o'smaydi.
  4) Integer Wavelet (CDF 5/3) ham xuddi shu oynani residualsiz, matematik
     exact qayta tiklaydi.

Ishga tushirish:
    python benchmark_fourier_topozero.py
"""

from __future__ import annotations

import gc
import statistics
import time
import tracemalloc

import numpy as np
import psutil
import torch

from fourier_lossless import IntegerWaveletCodec, LosslessFourierBuffer
from topo_zero_memory import TopoZeroMemory

STEP_BYTES = 128                    # bir qadamda oqimga kiradigan baytlar
STEPS = 10_240                      # 10 000+ qadam
STREAM_BYTES = STEP_BYTES * STEPS   # 1 310 720 bayt = 1.25 MiB oqim
BLOCK = 512                         # FFT blok uzunligi (belgi)
BUFFER_BYTES = 1_048_576            # FFT bufer byudjeti = aniq 1 MiB
TZ_DIM = 64
TZ_SLOTS = 4096                     # 4096 * 64 * 4 = 1 MiB TopoZero matritsasi
CHECK_EVERY = 2_048                 # lossless tekshiruv davri
SAMPLE_EVERY = 1_024                # RAM o'lchov davri
MB = 1024 ** 2


def build_corpus(target_bytes: int) -> bytes:
    """Deterministik aralash matn korpusi (o'zbek + ingliz + emoji)."""
    lines: list[bytes] = []
    total = 0
    i = 0
    while total < target_bytes:
        line = (
            f"[{i:06d}] Salom, dunyo! TopoZero-Fourier xotirasi sinovi. "
            f"Hello world! Ma'lumot oqimi uzluksiz. 🌊📡 data={i % 97}\n"
        )
        b = line.encode("utf-8")
        lines.append(b)
        total += len(b)
        i += 1
    return b"".join(lines)[:target_bytes]  # byte-exact kesish


def rss_mb(proc: psutil.Process) -> float:
    return proc.memory_info().rss / MB


def main() -> None:
    torch.manual_seed(11)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    corpus = build_corpus(STREAM_BYTES)
    assert len(corpus) == STREAM_BYTES, "korpus uzunligi mos emas"

    module = TopoZeroMemory(TZ_DIM, num_slots=TZ_SLOTS, write_rate=0.25, lam=5e-5,
                            zero_eps=1e-4, temperature=0.1, active_slots=TZ_SLOTS)
    fbuf = LosslessFourierBuffer(BUFFER_BYTES, BLOCK)
    wave = IntegerWaveletCodec(levels=4)

    # Tekshiruv uchun etalon halqa (faqat test harness'da, o'lchami O(1): 512 KiB)
    ref = np.zeros((fbuf.num_slots, BLOCK), dtype=np.int64)

    base_footprint = module.footprint_elements()
    base_buffer_bytes = fbuf.storage_bytes

    print(f"PyTorch {torch.__version__} | CPU | qadamlar={STEPS:,} | oqim={STREAM_BYTES / MB:.2f} MiB")
    print(f"FFT lossless bufer: {base_buffer_bytes:,} bayt = {fbuf.num_slots} blok x {BLOCK} "
          f"-> {fbuf.window_bytes:,} belgi ({fbuf.window_bytes / 1024:.0f} KiB) lossless oyna")
    print(f"TopoZero xotira:   {base_footprint:,} element = {base_footprint * 4 / MB:.2f} MiB "
          f"(matritsa {TZ_SLOTS * TZ_DIM * 4 / MB:.2f} MiB)\n")

    proc = psutil.Process()
    rss: list[float] = []
    traced_base = None
    checks = 0
    pending = bytearray()
    t0 = time.perf_counter()
    tracemalloc.start()

    with torch.inference_mode():
        for step in range(1, STEPS + 1):
            b = corpus[(step - 1) * STEP_BYTES: step * STEP_BYTES]

            # --- 1) TopoZeroMemory: cheksiz oqim gologrammasi (O(1)) ---
            u8 = np.frombuffer(b, dtype=np.uint8).astype(np.float32) / 255.0
            vec = torch.from_numpy(u8).view(-1, TZ_DIM)          # (2, 64)
            module.decay_step(dt=STEP_BYTES)                     # inversiya: s = 1/(1+lambda*dt)
            module.write_batch(vec, group_size=2)                # golografik yozuv
            module.sparsify()                                    # zero-weight tozalash

            # --- 2) Lossless FFT buferi: blok to'lganda FFT qilib yozish ---
            pending += b
            if len(pending) == BLOCK:
                sym = torch.from_numpy(
                    np.frombuffer(bytes(pending), dtype=np.uint8).astype(np.int64)
                )
                slot = fbuf.head
                n_res = fbuf.append(sym)                          # encode + halqaga yozuv
                assert n_res == 0, "float64 FFT residualga muhtoj bo'lmadi deb kutilgan edi"
                ref[slot] = sym.numpy()                           # etalon nusxa
                pending.clear()

            # --- 3) Davriy tekshiruvlar ---
            if step % SAMPLE_EVERY == 0:
                assert module.footprint_elements() == base_footprint, "xotira hajmi o'zgardi!"
                assert fbuf.storage_bytes == base_buffer_bytes, "bufer hajmi o'zgardi!"
                assert float(module.memory.norm(dim=-1).max()) <= 1.0 + 1e-5, "||M|| > 1"
                gc.collect()
                rss.append(rss_mb(proc))
                if traced_base is None and step >= 2 * SAMPLE_EVERY:
                    traced_base = tracemalloc.get_traced_memory()[0] / MB

            if step % CHECK_EVERY == 0 and fbuf.blocks_written >= fbuf.num_slots:
                # IFFT bilan butun oynani tiklash va etalon bilan bayt-bayt solishtirish
                rec = fbuf.decode_ordered().numpy()
                expect = np.roll(ref, -fbuf.head, axis=0).reshape(-1)
                assert np.array_equal(rec, expect), f"qadam {step}: LOSSLESS BUZILDI!"
                checks += 1
                done_mib = step * STEP_BYTES / MB
                print(f"   qadam {step:>6} | oqim {done_mib:4.2f} MiB | "
                      f"lossless tekshiruv #{checks} OK | RSS {rss[-1]:6.1f} MB | "
                      f"o'lik slot {module.zero_weight_fraction * 100:4.1f}%")

    elapsed = time.perf_counter() - t0
    end_traced = tracemalloc.get_traced_memory()[0] / MB
    tracemalloc.stop()

    # ------------------------------------------------------------------
    # YAKUNIY ISBOT: 1 MB bufer ichidan IFFT bilan 100% qayta tiklash
    # ------------------------------------------------------------------
    window = fbuf.decode_ordered().numpy().astype(np.uint8)
    expect = np.roll(ref, -fbuf.head, axis=0).reshape(-1).astype(np.uint8)

    orig_bytes = expect.tobytes()
    rec_bytes = window.tobytes()
    orig_text = orig_bytes.decode("utf-8", errors="replace")
    rec_text = rec_bytes.decode("utf-8", errors="replace")

    print("\n== 1) LOSSLESS QAYTA TIKLASH (IFFT) ==")
    print(f"   oyna: {len(orig_bytes):,} belgi | bufer: {fbuf.storage_bytes:,} bayt "
          f"(1 MiB koeffitsient + {fbuf.has_residual.nbytes + fbuf.residual.nbytes} bayt residual kanali)")
    print(f"   residual (noldan farqli belgilar): {fbuf.residual_total} ta -> float64 FFT exact")
    assert orig_bytes == rec_bytes, "original != reconstructed (bayt)"
    print("   assert original_bytes == reconstructed_bytes  ->  O'TDI (100.00% lossless)")
    assert orig_text == rec_text, "original != reconstructed (matn)"
    print("   assert original_text  == reconstructed_text  ->  O'TDI (matn 100% mos)")
    sample = rec_text[:64].replace("\n", " ")
    print(f"   tiklangan matn namunasi: \"{sample}...\"")

    # ------------------------------------------------------------------
    # Wavelet (CDF 5/3) — residualsiz, matematik exact
    # ------------------------------------------------------------------
    approx, details = wave.encode(expect.astype(np.int64))
    wav_back = wave.decode(approx, details)
    assert np.array_equal(wav_back, expect.astype(np.int64)), "wavelet lossless emas!"
    wav_kib = IntegerWaveletCodec.storage_bytes(approx, details) / 1024
    print("\n== 2) INTEGER WAVELET (CDF 5/3) ==")
    print(f"   o'sha oyna residualsiz exact tiklandi | int64 saqlash: {wav_kib:.0f} KiB "
          f"(int16 bo'lsa {wav_kib / 4:.0f} KiB)")
    print("   assert wavelet_decode(wavelet_encode(x)) == x  ->  O'TDI")

    # ------------------------------------------------------------------
    # O(1) xotira isboti
    # ------------------------------------------------------------------
    warm = rss[1:5]
    tail = rss[-4:]
    rss_growth = statistics.median(tail) - statistics.median(warm)
    print("\n== 3) O(1) XOTIRA ==")
    print(f"   qadamlar: {STEPS:,} | oqim: {STREAM_BYTES:,} bayt | vaqt: {elapsed:.1f} s "
          f"({STREAM_BYTES / MB / elapsed:.1f} MiB/s)")
    print(f"   TopoZero elementlar: {module.footprint_elements():,} (o'zgarmadi) | "
          f"FFT bufer: {fbuf.storage_bytes:,} bayt (o'zgarmadi)")
    print(f"   RAM issiq median: {statistics.median(warm):.2f} MB | oxirgi median: "
          f"{statistics.median(tail):.2f} MB | o'sish: {rss_growth:+.2f} MB")
    print(f"   Python-tracemalloc o'sishi: {end_traced - traced_base:+.3f} MB")
    print(f"   lossless tekshiruvlar: {checks}/{checks} O'TDI")

    with torch.inference_mode():
        q = torch.from_numpy(
            np.frombuffer(corpus[-TZ_DIM:], dtype=np.uint8).astype(np.float32) / 255.0
        )
        _, probs = module.read(q)
        assert abs(float(probs.sum()) - 1.0) < 1e-4
    print(f"   RiemannCompactAttention: e'tibor yig'indisi = {float(probs.sum()):.4f} | "
          f"o'lik slot {module.zero_weight_fraction * 100:.1f}%")

    assert rss_growth < 16.0, f"RAM o'sishi katta: {rss_growth:+.2f} MB"
    assert end_traced - traced_base < 1.0, "Python xotirasi o'sdi"

    print("\n=== XULOSA: 1 MB bufer oxirgi 64 KiB oqimni 100% lossless saqlaydi; ===")
    print("=== cheksiz qism TopoZero gologrammasida O(1) xotirada yashaydi.      ===")
    print("(Halol izoh: cheksiz oqimni 1 MB da 100% lossless saqlash mumkin emas —")
    print(" pigeonhole prinsipi. 100% kafolat oynaga tegishli; eski qism λ-so'nishli.)")


if __name__ == "__main__":
    main()
