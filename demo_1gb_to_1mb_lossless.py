#!/usr/bin/env python3
"""
DEMO: 1 GB ma'lumotni 1 MB buferga sig'dirib, 100% lossless qaytarib olish.
================================================================================

Halol bayonot:
  - Tasodifiy (siqilmaydigan) 1 GB ni 1 MB ga sig'dirib bo'lmaydi — bu
    axborot nazariyasi bo'yicha mumkin emas (pigeonhole: 2^(8*1GB) xil holat
    uchun 2^(8*1MB) xil quti yetmaydi).
  - LEKIN takrorlanuvchi/ortiqcha (redundant) ma'lumot — matn, loglar, kod,
    audio, rasm — siqiladi. Unda 1 MB bufer 1 GB ni 100% lossless saqlaydi.

Bu skript qiladigan ish:
  1) 1 GiB (1 073 741 824 bayt) takrorlanuvchi-ortiqcha matn oqimini
     bo'laklab generatsiya qiladi (hech qayerda 1 GB saqlanmaydi — O(1) RAM).
  2) Shu oqimni LZMA (xz) streaming bilan siqib, <= 1 MiB faylga yozadi.
  3) Fayldan dekod qilib, SHA-256 va bayt soni bo'yicha ASL oqim bilan
     aynan teng ekanini isbotlaydi:
         assert sha256(original_stream) == sha256(reconstructed_stream)
         assert reconstructed_bytes == 1 GiB

Ishga tushirish:
    python demo_1gb_to_1mb_lossless.py
"""

from __future__ import annotations

import hashlib
import lzma
import os
import statistics
import time

import psutil

TOTAL = 1024 ** 3                     # 1 GiB = 1 073 741 824 bayt
CHUNK = 4 * 1024 ** 2                 # 4 MiB bo'laklar
N_CHUNKS = TOTAL // CHUNK             # 256 bo'lak
BUDGET = 1024 ** 2                    # 1 MB bufer byudjeti
MB = 1024 ** 2
OUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "one_gb_in_one_mb.lzma")

# Ortiqcha (redundant) matn asosi: 64 qator, ~5 KB tsikl — LZMA uchun oson takrorlanadi
_BASE = "".join(
    f"Bugun {i:04d}-qator: TopoZero kompressiya sinovi, takrorlanuvchi matn bloki. "
    f"Salom, dunyo! Hello world! data={i % 13}\n"
    for i in range(64)
).encode("utf-8")

FILTERS = [{"id": lzma.FILTER_LZMA2, "preset": 6}]  # dict 8 MiB, kuchli siqish


def make_chunk(ci: int) -> bytes:
    """i-chi 4 MiB bo'lak: noyob sarlavha + takrorlanuvchi matn tsikli."""
    header = f"===== CHUNK {ci:04d} | TopoZero-1GB `======\n".encode("utf-8")
    need = CHUNK - len(header)
    body = (_BASE * (need // len(_BASE) + 1))[:need]
    return header + body


def rss_mb(proc: psutil.Process) -> float:
    return proc.memory_info().rss / MB


def main() -> None:
    proc = psutil.Process()
    print(f"Maqsad: {TOTAL:,} bayt (1 GiB) -> <= {BUDGET:,} bayt (1 MiB) bufer")
    print(f"Bo'laklar: {N_CHUNKS} x {CHUNK / MB:.0f} MiB | siqish: LZMA2 preset 6\n")

    # ------------------------------------------------------------------
    # 1) OQIM -> SIQISH (streaming, O(1) RAM)
    # ------------------------------------------------------------------
    sha_in = hashlib.sha256()
    comp = lzma.LZMACompressor(format=lzma.FORMAT_RAW, filters=FILTERS)

    rss: list[float] = []
    t0 = time.perf_counter()
    written = 0
    with open(OUT_FILE, "wb") as f:
        for ci in range(N_CHUNKS):
            chunk = make_chunk(ci)
            sha_in.update(chunk)                      # original oqim hash'i
            out = comp.compress(chunk)                # siqish (streaming)
            if out:
                f.write(out)
                written += len(out)
            if ci % 32 == 0:
                rss.append(rss_mb(proc))
        tail = comp.flush()
        f.write(tail)
        written += len(tail)

    t_comp = time.perf_counter() - t0
    in_hash = sha_in.hexdigest()

    # ------------------------------------------------------------------
    # 2) BUFFER TEKSHIRUVI: siqilgan hajm 1 MB dan oshmadi
    # ------------------------------------------------------------------
    file_size = os.path.getsize(OUT_FILE)
    print("== 1) SIQISH ==")
    print(f"   original oqim:  {TOTAL:,} bayt ({TOTAL / 1024 ** 3:.2f} GiB)")
    print(f"   siqilgan fayl:  {file_size:,} bayt ({file_size / MB:.3f} MiB)")
    print(f"   nisbat:         {TOTAL / file_size:,.0f}x")
    print(f"   vaqt:           {t_comp:.1f} s ({TOTAL / MB / t_comp:.1f} MiB/s)")
    print(f"   SHA-256 (original): {in_hash}")
    assert file_size <= BUDGET, f"1 MB budjetdan oshdi: {file_size} bayt"

    # ------------------------------------------------------------------
    # 3) 1 MB BUFERDAN 100% LOSSLESS QAYTA TIKLASH (streaming decode)
    # ------------------------------------------------------------------
    with open(OUT_FILE, "rb") as f:
        blob = f.read()                               # 1 MB bufer o'zi
    assert len(blob) <= BUDGET

    dec = lzma.LZMADecompressor(format=lzma.FORMAT_RAW, filters=FILTERS)
    sha_out = hashlib.sha256()
    total_out = 0
    pos = 0
    piece_size = 4096
    t1 = time.perf_counter()
    while True:
        if dec.needs_input:
            if pos >= len(blob):
                break
            piece = blob[pos: pos + piece_size]
            pos += len(piece)
            out = dec.decompress(piece, max_length=1 << 20)
        else:
            # Chiqish 1 MiB dan oshib ketmasin — xotira O(1) qoladi
            out = dec.decompress(b"", max_length=1 << 20)
        if out:
            sha_out.update(out)
            total_out += len(out)
        if dec.eof:
            break
    t_dec = time.perf_counter() - t1
    out_hash = sha_out.hexdigest()

    print("\n== 2) QAYTA TIKLASH (1 MB buferdan) ==")
    print(f"   tiklangan oqim: {total_out:,} bayt ({total_out / 1024 ** 3:.2f} GiB) | "
          f"vaqt: {t_dec:.1f} s")
    print(f"   SHA-256 (tiklangan): {out_hash}")
    assert total_out == TOTAL, "tiklangan hajm mos emas!"
    assert out_hash == in_hash, "SHA-256 mos emas — lossless buzildi!"
    print("   assert sha256(original) == sha256(reconstructed)  ->  O'TDI")
    print("   assert reconstructed_bytes == 1 GiB             ->  O'TDI")
    print("   ==> 1 GiB  =>  1 MiB  =>  1 GiB : 100% LOSSLESS")

    # ------------------------------------------------------------------
    # 4) Xotira: butun jarayon O(1) bo'laklab bajarildi
    # ------------------------------------------------------------------
    growth = (statistics.median(rss[-3:]) - statistics.median(rss[:3])) if len(rss) >= 6 else 0.0
    print("\n== 3) XOTIRA ==")
    print(f"   RAM namunalari: {len(rss)} marta o'lchandi | o'sish: {growth:+.2f} MB "
          f"(1 GB hech qachon RAM'da to'liq turmadi)")
    print(f"   bufer fayl: {file_size:,} bayt = {os.path.basename(OUT_FILE)}")

    print("\n=== XULOSA ===")
    print("   1 GiB ortiqcha ma'lumot 1 MB buferda 100% lossless saqlandi va qaytarildi.")
    print("   Tasodifiy 1 GB uchun bunday QILIB BO'LMAYDI — faqat ortiqcha (redundant)")
    print("   ma'lumot siqiladi. Bu FFT-hologrammadan farqli: bu yerda hech narsa")
    print("   yo'qolmaydi, lekin ma'lumot hajmi kamaygani sari sig'im cheksiz emas.")


if __name__ == "__main__":
    main()
