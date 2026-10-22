"""
FourierLosslessEncoderDecoder va LosslessFourierBuffer.
================================================================================

Matematik asos:

  1) FFT (Diskret Furye Almashtirish):
         X_k = sum_{n=0}^{N-1} x_n * exp(-2*pi*i*k*n/N)
     Vaqt sohasidagi diskret belgilar -> chastota sohasidagi kompleks
     koeffitsientlarga o'tadi.

  2) IFFT (teskari almashtirish):
         x_n = (1/N) * sum_{k=0}^{N-1} X_k * exp(+2*pi*i*k*n/N)
     Koeffitsientlardan dastlabki signal qayta tiklanadi.

  3) 100% lossless kafolati:
     float64 FFT'da yaxlitlash xatosi (~1e-12) juda kichik, shuning uchun
         round(IFFT(X)) == x_original
     bo'ladi. Bunga QO'SHIMCHA ravishda
         residual = x_original - round(IFFT(X))
     hisoblanadi va nolga teng bo'lmasa payload ichida saqlanadi — shu tarzda
     qayta tiklash har qanday holatda bit-bit aniq (lossless) bo'lishi
     MATEMATIK KAFOLATLANADI.

  4) Integer Wavelet (CDF 5/3, lifting sxemasi):
         predict:  d_i = odd_i - floor((even_i + even_{i+1}) / 2)
         update:   s_i = even_i + floor((d_{i-1} + d_i + 2) / 4)
     Butun sonli arifmetika ishlatiladi — teskari almashtirish matematik
     jihatdan TO'LIQ aniq (residual talab qilinmaydi).

Muhim izoh: FFT o'zi siqmaydi — complex128 koeffitsient 1 belgi uchun 16 bayt
(32 bayt orqaga emas, lekin xom uint8 dan 16x katta). Shuning uchun 1 MiB
FFT buferi oqimning oxirgi 65 536 belgisini (64 KiB) 100% lossless saqlaydi.
Cheksiz oqimni 1 MB da 100% lossless saqlash mumkin emas (pigeonhole prinsipi);
undan eski ma'lumot TopoZeroMemory gologrammasida (lossy, O(1)) qoladi.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import torch

Symbols = Union[torch.Tensor, np.ndarray, Sequence[int]]

__all__ = [
    "FFTPayload",
    "FourierLosslessEncoderDecoder",
    "LosslessFourierBuffer",
    "IntegerWaveletCodec",
]


# ---------------------------------------------------------------------------
# FFT payload: koeffitsientlar + uzunlik + (kerak bo'lsa) residual
# ---------------------------------------------------------------------------
@dataclass
class FFTPayload:
    coeffs: torch.Tensor                  # complex128 (N,) — asosiy matematik xotira
    length: int                           # haqiqiy belgilar soni (blok to'lmagan bo'lishi mumkin)
    residual: Optional[torch.Tensor] = None  # exact tuzatish: x - round(IFFT(X))

    @property
    def storage_bytes(self) -> int:
        """Payload egallagan hajm: har koeffitsient 16 bayt (complex128)."""
        extra = 0 if self.residual is None else int(self.residual.numel()) * 8
        return int(self.coeffs.numel()) * 16 + extra


# ---------------------------------------------------------------------------
# 1) FFT orqali 100% lossless encode / decode
# ---------------------------------------------------------------------------
class FourierLosslessEncoderDecoder:
    """Diskret belgilarni FFT koeffitsientlariga aylantiradi va IFFT bilan
    bit-bit aniq qayta tiklaydi.

    Ishlatish:
        codec = FourierLosslessEncoderDecoder(block=512)
        payload = codec.encode(symbols)      # FFT: vaqt -> chastota
        restored = codec.decode(payload)     # IFFT: chastota -> vaqt (exact)
        assert torch.equal(restored, symbols)
    """

    def __init__(self, block: int = 512) -> None:
        self.block = int(block)

    # ------------------------------------------------------------------
    @staticmethod
    def _as_int_tensor(symbols: Symbols) -> torch.Tensor:
        """Kirishni int64 1D tensorga keltirish (matn baytlari, token id'lar...)."""
        if isinstance(symbols, torch.Tensor):
            return symbols.detach().to(torch.int64).reshape(-1).cpu()
        arr = np.asarray(symbols)
        return torch.from_numpy(arr.astype(np.int64).reshape(-1))

    # ------------------------------------------------------------------
    def encode(self, symbols: Symbols) -> FFTPayload:
        """Forward FFT + float64 yaxlitlash + residual nazorati."""
        s = self._as_int_tensor(symbols)
        n = int(s.numel())
        if n > self.block:
            raise ValueError(f"blok {self.block} dan katta: {n}")

        # Nol bilan to'ldirish: FFT uzunligi doim self.block (fixed-size xotira)
        x = torch.zeros(self.block, dtype=torch.float64)
        x[:n] = s.to(torch.float64)

        # 1-tenglama: FFT — X_k = sum x_n * exp(-2*pi*i*k*n/N)
        coeffs = torch.fft.fft(x)

        # 2-tenglama: IFFT tekshiruvi — x_n = (1/N) sum X_k * exp(+...)
        recon = torch.fft.ifft(coeffs).real
        recon_int = torch.round(recon).to(torch.int64)

        # 3-qadam: residual = x - round(IFFT(X)); nolga teng bo'lmasa qo'shib saqlanadi
        residual = (s - recon_int[:n]).to(torch.int64)
        residual = residual if bool((residual != 0).any()) else None
        return FFTPayload(coeffs=coeffs, length=n, residual=residual)

    # ------------------------------------------------------------------
    def decode(self, payload: FFTPayload) -> torch.Tensor:
        """IFFT orqali dastlabki belgilarni 100% aniq tiklash."""
        # IFFT: chastota -> vaqt; float64 xato ~1e-12, round() aniq beradi
        values = torch.fft.ifft(payload.coeffs).real
        out = torch.round(values).to(torch.int64)[: payload.length]
        # Residual bo'lsa — qo'shiladi (matematik kafolat, har doim exact)
        if payload.residual is not None:
            out = out + payload.residual.to(torch.int64)
        return out

    # ------------------------------------------------------------------
    def roundtrip(self, symbols: Symbols) -> torch.Tensor:
        """encode + decode (sinov uchun qulay)."""
        return self.decode(self.encode(symbols))


# ---------------------------------------------------------------------------
# 2) 1 MB fixed-size lossless FFT bufer (circular, O(1))
# ---------------------------------------------------------------------------
class LosslessFourierBuffer:
    """Fixed-size halqa: har bir 512-belgili blok o'z FFT koeffitsientlari
    sifatida oldindan ajratilgan tensorlarga yoziladi.

    Xotira:  num_slots * block * 16 bayt (complex128) — oqim uzunligiga
    BOG'LIQ EMAS (O(1)). Eski blok ustiga yangi blok yoziladi (halqa).
    """

    def __init__(self, budget_bytes: int = 1_048_576, block: int = 512) -> None:
        self.codec = FourierLosslessEncoderDecoder(block=block)
        self.block = int(block)
        self.slot_bytes = self.block * 16          # bitta blok koeffitsienti hajmi
        self.num_slots = int(budget_bytes) // self.slot_bytes

        # FIXED tensorlar: oqim davomida hajm o'zgarmaydi
        self.coeffs = torch.zeros(self.num_slots, self.block, dtype=torch.complex128)
        self.has_residual = np.zeros(self.num_slots, dtype=bool)
        self.residual = np.zeros((self.num_slots, self.block), dtype=np.int8)

        self.head = 0                # navbatdagi yozuv sloti (halqa boshi)
        self.blocks_written = 0      # nechta blok o'tdi (counter, tensor emas)
        self.residual_total = 0      # noldan farqli residual belgilar soni

    # ------------------------------------------------------------------
    @property
    def storage_bytes(self) -> int:
        """Bufer egallagan aniq hajm (bayt) — doimiy."""
        return (
            self.coeffs.numel() * 16
            + self.has_residual.nbytes
            + self.residual.nbytes
        )

    @property
    def window_bytes(self) -> int:
        """Buferda lossless saqlanadigan oyna uzunligi (belgi/bayt)."""
        return self.num_slots * self.block

    # ------------------------------------------------------------------
    def append(self, symbols: Symbols) -> int:
        """Bitta blokni FFT qilib halqaga yozish. Qaytaradi: residual soni."""
        payload = self.codec.encode(symbols)
        slot = self.head
        self.coeffs[slot] = payload.coeffs              # in-place yozuv (O(1))

        count = 0
        if payload.residual is not None:
            r = payload.residual.numpy()
            count = int((r != 0).sum())
            assert np.abs(r).max() <= 127, "residual int8 ga sig'madi"
            self.residual[slot, : r.size] = r
            self.has_residual[slot] = True
        else:
            self.has_residual[slot] = False

        self.residual_total += count
        self.head = (self.head + 1) % self.num_slots
        self.blocks_written += 1
        return count

    # ------------------------------------------------------------------
    def decode_ordered(self) -> torch.Tensor:
        """Buferdagi barcha bloklarni eng eskidan eng yangigacha decode qilish."""
        chunks = []
        for i in range(self.num_slots):
            slot = (self.head + i) % self.num_slots   # halqani vaqt tartibida yurish
            residual = None
            if self.has_residual[slot]:
                residual = torch.from_numpy(self.residual[slot].astype(np.int64))
            payload = FFTPayload(coeffs=self.coeffs[slot], length=self.block, residual=residual)
            chunks.append(self.codec.decode(payload))
        return torch.cat(chunks)


# ---------------------------------------------------------------------------
# 3) Integer wavelet (CDF 5/3) — butun sonli, matematik exact
# ---------------------------------------------------------------------------
class IntegerWaveletCodec:
    """CDF 5/3 lifting wavelet: butun sonlar ustida ishlaydi, shuning uchun
    decode(encode(x)) == x har qanday holatda (residualsiz) aniq.
    """

    def __init__(self, levels: int = 4) -> None:
        self.levels = int(levels)

    # ------------------------------------------------------------------
    @staticmethod
    def _forward_level(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        even, odd = x[0::2], x[1::2]
        # predict: d_i = odd_i - floor((even_i + even_{i+1}) / 2)
        e_r = np.concatenate([even[1:], even[-1:]])          # o'ng chegara nusxasi
        d = odd - (even + e_r) // 2
        # update: s_i = even_i + floor((d_{i-1} + d_i + 2) / 4)
        d_l = np.concatenate([d[:1], d[:-1]])                # chap chegara nusxasi
        a = even + (d_l + d + 2) // 4
        return a, d

    @staticmethod
    def _inverse_level(a: np.ndarray, d: np.ndarray) -> np.ndarray:
        # Teskari update, so'ng teskari predict (aynan teskari tartib)
        d_l = np.concatenate([d[:1], d[:-1]])
        even = a - (d_l + d + 2) // 4
        e_r = np.concatenate([even[1:], even[-1:]])
        odd = d + (even + e_r) // 2
        x = np.empty(even.size + odd.size, dtype=np.int64)
        x[0::2] = even
        x[1::2] = odd
        return x

    # ------------------------------------------------------------------
    def encode(self, symbols: Symbols) -> tuple[np.ndarray, list[np.ndarray]]:
        """Ko'p darajali wavelet: approx + details ro'yxati."""
        if isinstance(symbols, torch.Tensor):
            x = symbols.detach().cpu().numpy().astype(np.int64).reshape(-1)
        else:
            x = np.asarray(symbols, dtype=np.int64).reshape(-1)
        approx = x
        details: list[np.ndarray] = []
        for _ in range(self.levels):
            if approx.size < 4 or approx.size % 2:
                break
            approx, d = self._forward_level(approx)
            details.append(d)
        return approx, details

    def decode(self, approx: np.ndarray, details: list[np.ndarray]) -> np.ndarray:
        x = approx
        for d in reversed(details):
            x = self._inverse_level(x, d)
        return x

    def roundtrip(self, symbols: Symbols) -> np.ndarray:
        return self.decode(*self.encode(symbols))

    @staticmethod
    def storage_bytes(approx: np.ndarray, details: list[np.ndarray], dtype_bytes: int = 8) -> int:
        return int(approx.size + sum(d.size for d in details)) * dtype_bytes
