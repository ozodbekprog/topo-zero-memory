"""
TopoZeroMemory — cheksiz ma'lumotlar oqimi uchun O(1) xotirali PyTorch moduli.
================================================================================

Matematik asos:
  1) Topologik kompaktifikatsiya (Riemann sferasi / stereografik proyeksiya):
         f(x) = 2x / (1 + ||x||^2)   ->   ||f(x)|| <= 1   (birlik shar)
         f(x) = 1 / (1 + |x|)        ->   f(x)  in (0, 1]
     Har qanday cheksiz katta vektor chekli geometrik yuzaga "bukiladi".
  2) Vaqt inversiyasi:
         s(t) = 1 / (1 + lambda * t),    t -> infinity  =>  s(t) -> 0
     Cheksiz uzoq kontekstning ta'siri matematik 0 ga intiladi.
  3) Rekursiv so'nish (Recursive Decay) + proyeksiya:
         M_slot <- (1 - a) * M_slot + a * v_compact
     Konveks kombinatsiya => xotira satri HAR DOIM birlik shar ichida qoladi.
  4) Zero-weight sparsifikatsiya:
         w < eps  yoki slot top-K energiyaga kirmasa  =>  w = 0  va  M_satr = 0
         (o'lik slot xotirada resurs sarflamaydi; K < S bo'lsa sparsifikatsiya kafolatlanadi)

Murakkablik:
  - Xotira: O(S * D) — oqim uzunligiga BOG'LIQ EMAS (S = slotlar soni, D = o'lcham).
  - Har qadam: O(S * D) hisoblash; tensorlar soni ham, hajmi ham o'zgarmaydi.

CPU'da ham to'liq ishlaydi (GPU talab qilinmaydi).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

__all__ = [
    "riemann_inversion",
    "stereographic_compact",
    "riemann_time_scale",
    "RiemannCompactAttention",
    "TopoZeroMemory",
]


# ---------------------------------------------------------------------------
# 1-qadam: skalyar inversiya masshtabi
# ---------------------------------------------------------------------------
def riemann_inversion(x: Tensor, dim: int = -1) -> Tensor:
    """f(x) = 1 / (1 + |x|) — teskari masshtab (inversiya).

    |x| -> infinity bo'lganda f -> 0; x = 0 da f = 1.
    Natija har doim (0, 1] oraliqda — cheksiz kattalik chekli o'lchovga bukiladi.
    """
    return 1.0 / (1.0 + x.norm(dim=dim, keepdim=True))


# ---------------------------------------------------------------------------
# 2-qadam: vektorli kompaktifikatsiya (Riemann sferasi)
# ---------------------------------------------------------------------------
def stereographic_compact(x: Tensor, dim: int = -1) -> Tensor:
    """f(x) = 2x / (1 + ||x||^2) — stereografik proyeksiya.

    Butun R^n fazo birlik shar ICHIGA (Riemann sferasi) bukiladi:
        ||f(x)|| <= 1,  bunda ||x|| = 1 da aynan sferaga tegadi.
    Cheksiz katta x uchun ham natija 0 atrofida barqaror qoladi.
    """
    # ||x||^2: normaning kvadrati (masofa o'lchovi)
    sq = (x * x).sum(dim=dim, keepdim=True)
    # floating-point himoyasi: x^2 overflow bo'lsa ham inf/NaN chiqmasin
    sq = sq.clamp(max=torch.finfo(x.dtype).max)
    # silliq geometrik yuzaga bukish: 2x / (1 + ||x||^2)
    return 2.0 * x / (1.0 + sq)


# ---------------------------------------------------------------------------
# 3-qadam: vaqt inversiyasi
# ---------------------------------------------------------------------------
def riemann_time_scale(t: Any, lam: float = 1.0) -> Any:
    """s(t) = 1 / (1 + lambda * t) — vaqt bo'yicha inversiya shkalasi.

    t -> infinity bo'lganda s(t) -> 0: cheksiz uzoq o'tmishning ta'siri
    matematik jihatdan yo'qoladi (qo'lda unutish shart emas).
    """
    if isinstance(t, Tensor):
        return 1.0 / (1.0 + lam * t)
    return 1.0 / (1.0 + lam * float(t))


# ---------------------------------------------------------------------------
# Riemann sferasidagi kompakt attention (o'qish mexanizmi)
# ---------------------------------------------------------------------------
class RiemannCompactAttention(nn.Module):
    """Xotirani Riemann sferasidagi masofa orqali o'qish.

    So'rov ham, xotira satrlari ham birlik sharda yotadi; o'xshashlik
    xordal masofa d^2 = |q|^2 + |m|^2 - 2 q·m orqali o'lchanadi.
    """

    def __init__(self, dim: int, temperature: float = 0.1, lam: float = 1e-3) -> None:
        super().__init__()
        self.dim = int(dim)
        self.temperature = float(temperature)
        self.lam = float(lam)

    def forward(
        self, query: Tensor, memory: Tensor, weights: Tensor, age: Tensor
    ) -> tuple[Tensor, Tensor]:
        squeeze = query.dim() == 1
        if squeeze:
            query = query.unsqueeze(0)

        # 4-qadam: so'rovni xotira bilan bir xil sferaga bukamiz: ||q|| <= 1
        q = stereographic_compact(query)

        # 5-qadam: xordal masofa^2 (sferadagi yaqinlik):
        #          d^2 = |q|^2 + |m|^2 - 2 q·m
        q_sq = (q * q).sum(dim=-1, keepdim=True)                     # [B, 1]
        m_sq = (memory * memory).sum(dim=-1).unsqueeze(0)            # [1, S]
        d2 = (q_sq + m_sq - 2.0 * q @ memory.t()).clamp_min_(0.0)    # [B, S]

        # 6-qadam: vaqt inversiyasi — age katta slotlarning ta'siri
        #          1/(1 + lambda*age) bilan 0 ga intiladi.
        t_scale = riemann_time_scale(age, self.lam).unsqueeze(0)     # [1, S]

        # 7-qadam: logit = -(d^2 / T) * s(age)  (yaqin = katta e'tibor,
        #          eski = kichik e'tibor)
        logits = -(d2 / self.temperature) * t_scale

        # 8-qadam: zero-weight slotlar butunlay maskalanadi (sparse attention)
        active = (weights > 0).unsqueeze(0)
        logits = logits.masked_fill(~active, float("-inf"))

        # 9-qadam: softmax — faqat tirik slotlar bo'yicha normallashadi
        probs = torch.softmax(logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

        # 10-qadam: o'qish = xotira satrlarining konveks kombinatsiyasi,
        #           shuning uchun natija ham doim birlik shar ichida qoladi.
        read = probs @ memory

        if squeeze:
            read = read.squeeze(0)
        return read, probs


# ---------------------------------------------------------------------------
# O(1) xotirali asosiy modul
# ---------------------------------------------------------------------------
class TopoZeroMemory(nn.Module):
    """Standart KV-Cache o'rnini bosuvchi fixed-size golografik xotira.

    Holat (hammasi oldindan ajratilgan, hajmi o'zgarmas):
      memory  : [S, D]  — bukilgan xotira satrlari (||M|| <= 1)
      weights : [S]     — slot muhimligi (0 = o'lik, resurs sarflamaydi)
      age     : [S]     — oxirgi yozuvdan beri o'tgan vaqt (inversiya uchun)
    """

    def __init__(
        self,
        dim: int,
        num_slots: int = 64,
        write_rate: float = 0.25,
        lam: float = 1e-3,
        zero_eps: float = 1e-4,
        temperature: float = 0.1,
        active_slots: int | None = None,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_slots = int(num_slots)
        self.write_rate = float(write_rate)   # a: yozish qadami (0 < a <= 1)
        self.lam = float(lam)                 # lambda: vaqt inversiyasi sur'ati
        self.zero_eps = float(zero_eps)       # eps: zero-weight chegarasi
        # K: bir vaqtda "tirik" slotlar byudjeti (None = hammasi).
        # K < S bo'lsa eng past vaznli S-K slot matematik 0 ga majburan o'tadi.
        self.active_slots = int(active_slots) if active_slots is not None else self.num_slots

        # FIXED-SIZE holat: oqim davomida numel O'ZGARMAYDI (O(1) xotira isboti)
        self.register_buffer("memory", torch.zeros(num_slots, dim))
        self.register_buffer("weights", torch.zeros(num_slots))
        self.register_buffer("age", torch.zeros(num_slots))

        self.attention = RiemannCompactAttention(dim, temperature=temperature, lam=lam)
        self._steps = 0                       # oddiy hisoblagich (tensor emas)

    # ------------------------------------------------------------------
    def decay_step(self, dt: float = 1.0) -> None:
        """Rekursiv so'nish: dt vaqt birligi davomida xotira inversiya shkalasi bilan kamayadi."""
        # 11-qadam: M <- M * s(dt),  bunda s(dt) = 1/(1 + lambda*dt) < 1.
        #           Konvekslik saqlanadi => ||M|| hech qachon 1 dan oshmaydi.
        s = 1.0 / (1.0 + self.lam * dt)
        self.memory.mul_(s)
        self.weights.mul_(s)
        self.age.add_(dt)
        self._steps += 1

    # ------------------------------------------------------------------
    def write(self, x: Tensor) -> int:
        """Yangi ma'lumotni fixed-size xotira ichiga golografik singdirish."""
        # 12-qadam: kirishni sferaga bukish (kattaligi qanday bo'lsa ham)
        v = stereographic_compact(x)
        if v.dim() == 2:
            v = v.mean(dim=0)  # konveks o'rtacha — baribir birlik shar ichida

        # Eng bo'sh slotni tanlash: vazni eng kichigi (eski/zero-weight nomzod).
        # Qidiruv O(S), xotira O(1) — oqim uzunligiga bog'liq emas.
        slot = int(torch.argmin(self.weights).item())

        # 13-qadam: rekursiv proyeksiya (konveks kombinatsiya):
        #           M_slot <- (1 - a) * M_slot + a * v
        self.memory[slot].mul_(1.0 - self.write_rate).add_(v, alpha=self.write_rate)

        # Slot "yangi" deb belgilanadi: vazn = 1, yoshi = 0
        self.weights[slot] = 1.0
        self.age[slot] = 0.0
        return slot

    # ------------------------------------------------------------------
    def write_batch(self, x: Tensor, group_size: int = 32) -> int:
        """Vectorlashtirilgan oqim yozuvi (katta chunk'lar uchun).

        B vektor group_size tadan guruhlanadi va har bir guruh konveks
        o'rtachasi K = B // g eng bo'sh slotga yoziladi — bir operatsiyada.
        Xotira baribir O(S * D), oqim hajmi hech narsani o'zgartirmaydi.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # 12-qadam: butun chunk'ni sferaga bukish (vektorlashtirilgan)
        v = stereographic_compact(x)
        group = max(1, min(int(group_size), v.shape[0]))
        slots_count = v.shape[0] // group
        # Guruh o'rtachasi — konveks kombinatsiya, birlik shar ichida qoladi
        v = v[: slots_count * group].view(slots_count, group, -1).mean(dim=1)

        # Eng bo'sh K slotni tanlash (top-K, O(S)) — o'lik slotlar birinchi
        idx = torch.topk(self.weights, slots_count, largest=False).indices

        # 13-qadam: rekursiv proyeksiya, barcha K slot uchun birdaniga:
        #           M[idx] <- (1 - a) * M[idx] + a * v
        self.memory[idx] = (1.0 - self.write_rate) * self.memory[idx] + self.write_rate * v
        self.weights[idx] = 1.0
        self.age[idx] = 0.0
        return int(slots_count)

    # ------------------------------------------------------------------
    def sparsify(self) -> int:
        """Zero-weight mexanizmi: o'lik slotlar xotiradan resurs olmaydi."""
        # 14-qadam: (a) vazni eps dan kichik slotlar (chirigan ma'lumot) va
        #           (b) top-K energiyaga kirmagan slotlar matematik 0 ga o'tadi:
        #           w = 0  va  M_satr = 0  (sparse / zero-weight).
        dead = self.weights < self.zero_eps
        if self.active_slots < self.num_slots:
            # Energiya reytingi: faqat eng katta K vazn tirik qoladi.
            keep_idx = torch.topk(self.weights, self.active_slots).indices
            keep = torch.zeros_like(dead)
            keep[keep_idx] = True
            dead = dead | ~keep
        self.weights.masked_fill_(dead, 0.0)
        self.memory.mul_(dead.logical_not().to(self.memory.dtype).unsqueeze(-1))
        return int(dead.sum().item())

    # ------------------------------------------------------------------
    def read(self, query: Tensor) -> tuple[Tensor, Tensor]:
        """Xotiradan o'qish (Riemann sferasidagi kompakt attention)."""
        return self.attention(query, self.memory, self.weights, self.age)

    # ------------------------------------------------------------------
    def step(self, x: Tensor, query: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Bitta to'liq qadam: so'nish -> yozish -> o'qish -> sparsifikatsiya.

        Xotira hajmi (elementlar soni) qadamlar soniga BOG'LIQ EMAS.
        """
        # 15-qadam: kompozitsiya — barcha amallar in-place, fixed-size holatda.
        self.decay_step()
        self.write(x)
        q = x if query is None else query
        read, probs = self.read(q)
        self.sparsify()
        return read, probs

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Xotirani nolga qaytarish (hajm o'zgarmaydi)."""
        self.memory.zero_()
        self.weights.zero_()
        self.age.zero_()
        self._steps = 0

    # ------------------------------------------------------------------
    def footprint_elements(self) -> int:
        """Doimiy holatdagi tensor elementlari soni — O(1) xotira isboti."""
        return self.memory.numel() + self.weights.numel() + self.age.numel()

    @property
    def zero_weight_fraction(self) -> float:
        """O'lik (zero-weight) slotlar ulushi — [0, 1]."""
        return float((self.weights == 0).to(torch.float32).mean().item())

    @property
    def steps(self) -> int:
        return self._steps
