#!/usr/bin/env python3
"""
TopoZeroMemory — sinov va isbot skripti (CPU, oddiy PC).
================================================================

Isbotlanadigan da'volar:
  1) t -> infinity da inversiya shkalasi s(t) -> 0 (cheksiz vaqt ta'siri yo'qoladi).
  2) Har qanday kattalikdagi vektor [0,1] / birlik sharga kompaktifikatsiya qilinadi.
  3) 12 000+ qadamli cheksiz oqimda RAM O'SMAYDI (O(1) Memory Complexity).
  4) Xotira tensorlarining elementlar soni oqim davomida QAT'IY o'zgarmaydi.
  5) Barcha qiymatlar [-1, 1] / [0, 1] oralig'ida barqaror qoladi (NaN/Inf yo'q).
  6) Zero-weight (sparse) slotlar paydo bo'ladi va ular 0 resurs egallaydi.

Ishga tushirish:
    python test_topo_zero_memory.py
"""

from __future__ import annotations

import gc
import statistics
import time
import tracemalloc

import psutil
import torch

from topo_zero_memory import (
    TopoZeroMemory,
    riemann_inversion,
    riemann_time_scale,
    stereographic_compact,
)

STEPS = 12000          # cheksiz oqim simulyatsiyasi (10 000+)
WARMUP = 1000          # allokatorlarni barqarorlashtirish qadamlari
DIM = 64               # vektor o'lchami
SLOTS = 64             # fixed-size xotira slotlari
BATCH = 8              # bir qadamdagi vektorlar soni
SAMPLE_EVERY = 500     # o'lchov namunasi davri
MB = 1024 ** 2


def rss_mb(proc: psutil.Process) -> float:
    """Joriy RSS (RAM) hajmi, MB."""
    return proc.memory_info().rss / MB


def check_time_infinity() -> None:
    print("== 1) Vaqt inversiyasi: t -> infinity,  s(t) = 1/(1+lambda*t) -> 0 ==")
    prev = None
    for t in (0.0, 1e2, 1e4, 1e6, 1e9, 1e12):
        # lambda = 1.0 bilan sof inversiya shkalasini tekshiramiz
        s = float(riemann_time_scale(torch.tensor(t, dtype=torch.float64), lam=1.0))
        print(f"   t={t:>12.0e}  ->  s(t)={s:.3e}")
        assert 0.0 < s <= 1.0, "shkala (0, 1] dan chiqdi"
        if prev is not None:
            assert s < prev, "shkala t o'sishi bilan kamaymadi"
        prev = s
    assert float(riemann_time_scale(1e12, lam=1.0)) < 1e-6, "cheksizlikda 0 ga intilmadi"
    print("   OK: cheksiz uzoq kontekst ta'siri matematik 0 ga intiladi.\n")


def check_compactification() -> None:
    print("== 2) Riemann kompaktifikatsiya: R^n -> birlik shar ([-1,1]) ==")
    scales = torch.tensor([[1e-2], [1.0], [1e3], [1e6], [1e9]], dtype=torch.float32)
    x = scales * torch.ones(5, DIM)
    for row in range(x.shape[0]):
        norm = float(x[row].norm())
        compact = stereographic_compact(x[row])      # f(x) = 2x/(1+||x||^2)
        inv = float(riemann_inversion(x[row]))       # f(x) = 1/(1+|x|)
        cnorm = float(compact.norm())
        print(f"   |x|={norm:10.1e} -> ||f(x)||={cnorm:.6f}   (1/(1+|x|)={inv:.3e})")
        assert cnorm <= 1.0 + 1e-5, "kompaktifikatsiya birlik shardan chiqdi"
        assert float(compact.abs().max()) <= 1.0 + 1e-5, "element [-1,1] dan chiqdi"
        assert 0.0 < inv <= 1.0, "inversiya (0,1] dan chiqdi"
    print("   OK: 1e9 kattalik ham chekli [-1,1] sharga bukildi.\n")


def main() -> None:
    torch.manual_seed(1234)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    print(f"PyTorch {torch.__version__} | qurilma: CPU | dim={DIM}, slot={SLOTS}, "
          f"qadam={STEPS}, batch={BATCH}\n")
    check_time_infinity()
    check_compactification()

    module = TopoZeroMemory(
        dim=DIM,
        num_slots=SLOTS,
        write_rate=0.25,
        lam=1e-3,
        zero_eps=1e-4,
        temperature=0.1,
        active_slots=48,   # K=48 < S=64 => 25% slot doimiy zero-weight (sparse)
    )
    proc = psutil.Process()

    def stream_batch() -> torch.Tensor:
        """Cheksiz oqim: kattaligi 1e-3 dan 1e6 gacha o'zgaruvchi vektorlar."""
        magnitude = 10.0 ** float(torch.empty(1).uniform_(-3.0, 6.0))
        return torch.randn(BATCH, DIM) * magnitude

    # --- Warmup: CPU allokatorini barqarorlashtirish ---
    with torch.inference_mode():
        for _ in range(WARMUP):
            module.step(stream_batch())
    gc.collect()

    # --- Baza (baseline) o'lchovlari ---
    tracemalloc.start()
    base_rss = rss_mb(proc)
    base_traced = tracemalloc.get_traced_memory()[0] / MB
    base_footprint = module.footprint_elements()
    base_state_numel = sum(t.numel() for t in module.state_dict().values())

    print("== 3) Cheksiz oqim: O(1) xotira isboti ==")
    print(f"   baza: RSS={base_rss:.2f} MB | tensor elementlari={base_footprint} | "
          f"state_dict numel={base_state_numel}")
    print("   " + "-" * 112)

    samples: list[tuple[int, float, float]] = []
    t_start = time.perf_counter()

    with torch.inference_mode():
        for step in range(1, STEPS + 1):
            x = stream_batch()
            read, probs = module.step(x)

            # --- Har qadamda [0,1] / [-1,1] barqarorlik assertlari ---
            assert torch.isfinite(read).all(), f"qadam {step}: read NaN/Inf"
            assert torch.isfinite(module.memory).all(), f"qadam {step}: xotira NaN/Inf"
            assert float(read.abs().max()) <= 1.0 + 1e-5, f"qadam {step}: read [-1,1] dan chiqdi"
            assert float(module.memory.norm(dim=-1).max()) <= 1.0 + 1e-5, \
                f"qadam {step}: ||M|| birlik shardan chiqdi"
            assert float(module.weights.min()) >= 0.0, f"qadam {step}: vazn < 0"
            assert float(module.weights.max()) <= 1.0 + 1e-6, f"qadam {step}: vazn > 1"
            psum = float(probs.sum(dim=-1).max())
            assert psum == 0.0 or abs(psum - 1.0) < 1e-4, f"qadam {step}: attention normasi buzildi"

            # --- O(1) xotira asserti: tensorlar hajmi qadamga bog'liq emas ---
            assert module.footprint_elements() == base_footprint, \
                f"qadam {step}: xotira elementlari soni O'ZGARDI (O(1) buzildi)"

            # --- Davriy RAM o'lchovi ---
            if step % SAMPLE_EVERY == 0:
                gc.collect()
                rss = rss_mb(proc)
                traced = tracemalloc.get_traced_memory()[0] / MB
                samples.append((step, rss, traced))
                print(
                    f"   qadam {step:>6} | RSS={rss:7.2f} MB | python-trace={traced:6.3f} MB | "
                    f"max||M||={float(module.memory.norm(dim=-1).max()):.4f} | "
                    f"o'lik slot={module.zero_weight_fraction * 100:5.1f}% | "
                    f"s(t)={float(riemann_time_scale(float(step), lam=1e-3)):.2e}"
                )

    elapsed = time.perf_counter() - t_start
    gc.collect()
    end_rss = rss_mb(proc)
    end_traced = tracemalloc.get_traced_memory()[0] / MB
    tracemalloc.stop()

    # --- Yakuniy isbot bloklari ---
    end_footprint = module.footprint_elements()
    end_state_numel = sum(t.numel() for t in module.state_dict().values())

    first_rss = statistics.median([s[1] for s in samples[:4]])
    last_rss = statistics.median([s[1] for s in samples[-4:]])
    tail_growth = last_rss - first_rss
    total_growth = end_rss - base_rss
    traced_growth = end_traced - base_traced

    print("\n== 4) NATIJA ==")
    print(f"   qadamlar: {STEPS} | vaqt: {elapsed:.2f} s | tezlik: {STEPS / elapsed:,.0f} qadam/s")
    print(f"   tensor elementlari: baza={base_footprint} -> oxir={end_footprint}")
    print(f"   state_dict numel:   baza={base_state_numel} -> oxir={end_state_numel}")
    print(f"   RSS (median dastlabki 4 namuna): {first_rss:.2f} MB")
    print(f"   RSS (median oxirgi 4 namuna):    {last_rss:.2f} MB  -> farq: {tail_growth:+.2f} MB")
    print(f"   RSS umumiy: {base_rss:.2f} -> {end_rss:.2f} MB (farq {total_growth:+.2f} MB)")
    print(f"   Python-tracemalloc o'sishi: {traced_growth:+.3f} MB")

    assert end_footprint == base_footprint, "O(1) buzildi: tensor hajmi o'zgardi"
    assert end_state_numel == base_state_numel, "O(1) buzildi: state_dict o'zgardi"
    assert tail_growth < 16.0, f"RAM o'sishi juda katta: {tail_growth:+.2f} MB"
    assert traced_growth < 0.5, f"Python xotirasi o'sdi: {traced_growth:+.3f} MB"

    # --- Zero-weight / sparsifikatsiya isboti ---
    dead = module.weights == 0
    dead_rows_max = float(module.memory[dead].abs().max()) if bool(dead.any()) else 0.0
    expected_dead = SLOTS - module.active_slots
    print(f"\n== 5) Zero-weight (sparse) mexanizmi ==")
    print(f"   tirik slotlar byudjeti: K={module.active_slots}/{SLOTS} "
          f"-> kafolatlangan o'lik: {expected_dead} slot")
    print(f"   joriy o'lik slotlar: {int(dead.sum())}/{SLOTS} "
          f"({module.zero_weight_fraction * 100:.1f}%)")
    print(f"   o'lik slotlar xotirasidagi max|M| = {dead_rows_max:.1e} (0 bo'lishi shart)")
    assert dead_rows_max == 0.0, "zero-weight slotda resurs qoldi"
    assert int(dead.sum()) == expected_dead, "top-K sparsifikatsiya soni mos emas"
    assert abs(module.zero_weight_fraction - expected_dead / SLOTS) < 1e-9, \
        "zero-weight ulushi noto'g'ri"

    x_inf = torch.tensor([1e12] * DIM)
    print(f"\n== 6) Cheksizlik chegarasi ==")
    print(f"   ||compact(1e12)| = {float(stereographic_compact(x_inf).norm()):.3e}  (~0)")
    print(f"   1/(1+|1e12|)     = {float(riemann_inversion(x_inf)):.3e}  (~0)")
    print(f"   s(t=12000, lam=1e-3) = {float(riemann_time_scale(12000.0, lam=1e-3)):.3e}")

    print("\n=== NATIJA: HAMMA TEKSHIRUVLAR O'TDI — O(1) XOTIRA TASDIQLANDI ===")


if __name__ == "__main__":
    main()
