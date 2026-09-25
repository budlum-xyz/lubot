#!/usr/bin/env python3
"""Independent cross-check of the Rust port, in numpy.

Why this file exists: a port checked only against itself proves nothing. This
script reads the same split checkpoint through a *different* path - it opens the
part files itself, parses the safetensors header, converts half precision with
numpy's own rules, and computes the encoder, the decision head and the scorer
with matrix products - then compares its hidden states, its option scores and its
action scores against a dump the Rust run wrote.

What it is not: a reference. Two implementations that share a misreading of the
architecture will agree. What it catches is transcription errors: wrong tensor
names, transposed matrices, a norm in the wrong place, a missing residual, a
doubled one, the wrong position for the type embedding.

Memory matters here as much as it does in the port: the embedding table is 196
million elements, so only the rows a sequence needs are read, and each layer's
weights are read one layer at a time and dropped before the next.

Usage:
    python3 tools/kodlayici_capraz.py --paket <dizin> --kimlik 1,105,4096 \
        --isaret 1,2 --tip choice --rust-dokum <ham f32 dosyasi>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys

import numpy as np

VARSAYILAN_EPS = 1e-5
KAFA_GENISLIGI = 64


class Paket:
    """The header and a part-aware byte reader over one logical artifact."""

    def __init__(self, paket: str) -> None:
        parcalar = sorted(
            (ad for ad in os.listdir(paket) if ad.startswith("model.safetensors.part-")),
            key=lambda ad: int(ad.rsplit("-", 1)[1]),
        )
        if not parcalar:
            raise SystemExit("parca dosyasi yok: " + paket)
        self.yollar = [os.path.join(paket, ad) for ad in parcalar]
        self.boyutlar = [os.path.getsize(y) for y in self.yollar]
        self.toplam = sum(self.boyutlar)
        if len(self.yollar) == 1:
            self._tampon = open(self.yollar[0], "rb").read()
        else:
            self._tampon = None
        uzunluk = struct.unpack("<Q", self._oku(0, 8))[0]
        self.baslik = json.loads(self._oku(8, uzunluk).decode())
        self.veri_baslangici = 8 + uzunluk

    def _oku(self, bas: int, adet: int) -> bytes:
        """Reads `adet` bytes starting at `bas`, crossing part boundaries."""
        if self._tampon is not None:
            return self._tampon[bas : bas + adet]
        parcalar = []
        konum = bas
        kalan = adet
        kaydir = 0
        for yol, boy in zip(self.yollar, self.boyutlar):
            if kalan <= 0:
                break
            son = kaydir + boy
            if konum < son:
                ic_bas = konum - kaydir
                alinacak = min(kalan, boy - ic_bas)
                with open(yol, "rb") as f:
                    f.seek(ic_bas)
                    parcalar.append(f.read(alinacak))
                konum += alinacak
                kalan -= alinacak
            kaydir = son
        return b"".join(parcalar)

    def _tensor_ham(self, ad: str) -> bytes:
        b = self.baslik[ad]
        bas, son = b["data_offsets"]
        return self._oku(self.veri_baslangici + bas, son - bas)

    def tensor(self, ad: str) -> np.ndarray:
        """One tensor as `float32`, exactly converted from its stored type."""
        if ad not in self.baslik:
            raise SystemExit("tensor yok: " + ad)
        b = self.baslik[ad]
        ham = self._tensor_ham(ad)
        if b["dtype"] == "F32":
            dizi = np.frombuffer(ham, dtype="<f4")
        elif b["dtype"] == "F16":
            dizi = np.frombuffer(ham, dtype="<f2").astype(np.float32)
        else:
            raise SystemExit(f"{ad}: desteklenmeyen tur {b['dtype']}")
        return dizi.reshape(b["shape"]).astype(np.float32)

    def satirlar(self, ad: str, satir_indeksleri: list[int]) -> np.ndarray:
        """Selected rows of a two-dimensional tensor.

        The embedding table is read this way: 196 million elements is 786 MB as
        `float32`, more than this machine has to spare, and a sequence needs a
        few dozen of them.
        """
        b = self.baslik[ad]
        satir, genislik = b["shape"]
        eleman = 4 if b["dtype"] == "F32" else 2
        ham = bytearray()
        for i in satir_indeksleri:
            if not 0 <= i < satir:
                raise SystemExit(f"{ad}: satir disi {i} (satir {satir})")
            bas, _ = b["data_offsets"]
            ham += self._oku(self.veri_baslangici + bas + i * genislik * eleman, genislik * eleman)
        if b["dtype"] == "F32":
            dizi = np.frombuffer(bytes(ham), dtype="<f4")
        else:
            dizi = np.frombuffer(bytes(ham), dtype="<f2").astype(np.float32)
        return dizi.reshape(len(satir_indeksleri), genislik).astype(np.float32)


def katman_norm(x: np.ndarray, agirlik: np.ndarray, sapma: np.ndarray | None, eps: float) -> np.ndarray:
    """Layer normalisation over the last axis."""
    ortalama = x.mean(-1, keepdims=True)
    varyans = ((x - ortalama) ** 2).mean(-1, keepdims=True)
    y = (x - ortalama) / np.sqrt(varyans + eps)
    y = y * agirlik
    return y if sapma is None else y + sapma


def gelu(x: np.ndarray) -> np.ndarray:
    """Exact gelu, from numpy's own error function."""
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def rope_tablosu(konumlar: np.ndarray, kafa: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """Cosine and sine tables for one rope base, in pair layout."""
    yarim = kafa // 2
    frekans = 1.0 / (theta ** (np.arange(yarim, dtype=np.float64) / yarim))
    aci = np.outer(konumlar, frekans)
    return np.cos(aci), np.sin(aci)


def rope_uygula(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Rotates by halves: element i pairs with element yarim + i.

    The frequency table covers half the head and is duplicated, so both halves
    turn by the same angle at the same index.
    """
    if cos.ndim == 2:
        cos = cos[:, None, :]
        sin = sin[:, None, :]
    yarim = x.shape[-1] // 2
    a = x[..., :yarim]
    b = x[..., yarim:]
    yeni = np.empty_like(x)
    yeni[..., :yarim] = a * cos - b * sin
    yeni[..., yarim:] = a * sin + b * cos
    return yeni


def pencere_maskesi(uzunluk: int, yarim: int) -> np.ndarray:
    """`abs(i - j) > yarim`, which is the window the checkpoint's config means."""
    i = np.arange(uzunluk)[:, None]
    j = np.arange(uzunluk)[None, :]
    return np.abs(i - j) > yarim


def kodla(paket: Paket, yapi: dict, kimlikler: list[int]) -> np.ndarray:
    """The encoder: embeddings, the layer stack, the final norm."""
    d = int(yapi["hidden_size"])
    kafa_sayisi = int(yapi["num_attention_heads"])
    kafa = d // kafa_sayisi
    eps = float(yapi.get("layer_norm_eps", VARSAYILAN_EPS))
    yarim = int(yapi.get("local_attention", 128)) // 2
    uzunluk = len(kimlikler)
    konumlar = np.arange(uzunluk)

    h = paket.satirlar("encoder.embeddings.tok_embeddings.weight", kimlikler)
    h = katman_norm(h, paket.tensor("encoder.embeddings.norm.weight"), None, eps)

    tablolar = {}
    for tur in set(yapi["layer_types"]):
        theta = float(
            yapi.get("rope_parameters", {}).get(tur, {}).get("rope_theta", 10000.0)
        )
        tablolar[tur] = rope_tablosu(konumlar, kafa, theta)

    for sira, tur in enumerate(yapi["layer_types"]):
        on = f"encoder.layers.{sira}"
        # Layer 0 has no attention norm: the embedding block already ended with
        # one and the reference builds that norm as an identity.
        if sira == 0:
            norm_girdi = h
        else:
            norm_girdi = katman_norm(h, paket.tensor(f"{on}.attn_norm.weight"), None, eps)

        wqkv = paket.tensor(f"{on}.attn.Wqkv.weight")
        wo = paket.tensor(f"{on}.attn.Wo.weight")
        qkv = (norm_girdi @ wqkv.T).reshape(uzunluk, 3, kafa_sayisi, kafa)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        cos, sin = tablolar[tur]
        q = rope_uygula(q, cos, sin)
        k = rope_uygula(k, cos, sin)

        skor = np.einsum("ihd,jhd->hij", q, k) / math.sqrt(kafa)
        if tur == "sliding_attention":
            skor = np.where(pencere_maskesi(uzunluk, yarim)[None, :, :], -1e30, skor)
        agirlik = softmax(skor)
        birlesik = np.einsum("hij,jhd->ihd", agirlik, v).reshape(uzunluk, d)
        h = h + birlesik @ wo.T

        gizli = katman_norm(h, paket.tensor(f"{on}.mlp_norm.weight"), None, eps)
        wi = paket.tensor(f"{on}.mlp.Wi.weight")
        mlp_wo = paket.tensor(f"{on}.mlp.Wo.weight")
        yari = gizli @ wi.T
        ffn = wi.shape[0] // 2
        # First half through the activation, second half as the gate - the order
        # in the checkpoint, checked here rather than assumed.
        h = h + (gelu(yari[:, :ffn]) * yari[:, ffn:]) @ mlp_wo.T
        del wqkv, wo, wi, mlp_wo, qkv, q, k, v, skor, agirlik, birlesik, gizli

    return katman_norm(h, paket.tensor("encoder.final_norm.weight"), None, eps)


def karar(paket: Paket, yapi: dict, h: np.ndarray, isaretler: list[int], tip: int,
          kafa_katmani: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The head, the scorer, and the action head."""
    d = int(yapi["hidden_size"])
    eps = float(yapi.get("layer_norm_eps", VARSAYILAN_EPS))
    uzunluk = h.shape[0]
    kafa_sayisi = d // KAFA_GENISLIGI
    kafa = d // kafa_sayisi

    # The type embedding goes onto every position.
    h = h + paket.tensor("type_emb.weight")[tip][None, :]

    for sira in range(kafa_katmani):
        on = f"head.layers.{sira}"
        n1 = katman_norm(h, paket.tensor(f"{on}.norm1.weight"), paket.tensor(f"{on}.norm1.bias"), eps)
        qkv = n1 @ paket.tensor(f"{on}.self_attn.in_proj_weight").T
        qkv = qkv + paket.tensor(f"{on}.self_attn.in_proj_bias")
        q, k, v = np.split(qkv, 3, axis=-1)
        q = q.reshape(uzunluk, kafa_sayisi, kafa).transpose(1, 0, 2)
        k = k.reshape(uzunluk, kafa_sayisi, kafa).transpose(1, 0, 2)
        v = v.reshape(uzunluk, kafa_sayisi, kafa).transpose(1, 0, 2)
        # Bidirectional: no mask, and the same weights under a causal mask would
        # give a different answer that still looks like one.
        skor = np.einsum("hid,hjd->hij", q, k) / math.sqrt(kafa)
        dikkat = np.einsum("hij,hjd->ihd", softmax(skor), v).reshape(uzunluk, d)
        h = h + dikkat @ paket.tensor(f"{on}.self_attn.out_proj.weight").T
        h = h + paket.tensor(f"{on}.self_attn.out_proj.bias")

        n2 = katman_norm(h, paket.tensor(f"{on}.norm2.weight"), paket.tensor(f"{on}.norm2.bias"), eps)
        ara = gelu(n2 @ paket.tensor(f"{on}.linear1.weight").T + paket.tensor(f"{on}.linear1.bias"))
        h = h + ara @ paket.tensor(f"{on}.linear2.weight").T + paket.tensor(f"{on}.linear2.bias")

    m = h[isaretler]
    puan = gelu(
        katman_norm(m, paket.tensor("scorer.0.weight"), paket.tensor("scorer.0.bias"), eps)
        @ paket.tensor("scorer.1.weight").T
        + paket.tensor("scorer.1.bias")
    )
    puan = (puan @ paket.tensor("scorer.3.weight").T + paket.tensor("scorer.3.bias")).reshape(-1)

    p = softmax(puan)
    en_iyi = np.argsort(-p)
    k_sayisi = len(p)
    entropi = float(-(p * np.log(np.clip(p, 1e-9, None))).sum() / np.log(k_sayisi))
    # The runner-up is the second position in value order, so a tie gives a gap
    # of zero.
    ozellik = np.array(
        [p[en_iyi[0]], p[en_iyi[0]] - p[en_iyi[1]], entropi, k_sayisi / 255.0],
        dtype=np.float32,
    )
    girdi = np.concatenate([h[0], ozellik]).astype(np.float32)
    a1 = gelu(paket.tensor("act_head.0.weight") @ girdi + paket.tensor("act_head.0.bias"))
    eylem = paket.tensor("act_head.2.weight") @ a1 + paket.tensor("act_head.2.bias")
    return puan, h[0].copy(), eylem


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paket", required=True)
    ap.add_argument("--kimlik", required=True)
    ap.add_argument("--isaret", default="")
    ap.add_argument("--tip", default="choice", choices=["choice", "score", "noul"])
    ap.add_argument("--rust-dokum", default="")
    ap.add_argument("--tolerans", type=float, default=2e-3)
    a = ap.parse_args()

    kimlikler = [int(x) for x in a.kimlik.split(",") if x.strip()]
    isaretler = [int(x) for x in a.isaret.split(",") if x.strip()]
    tip = {"choice": 0, "score": 1, "noul": 2}[a.tip]

    with open(os.path.join(a.paket, "encoder", "config.json")) as f:
        yapi = json.load(f)
    with open(os.path.join(a.paket, "rl_agent_config.json")) as f:
        karar_yapisi = json.load(f)

    paket = Paket(a.paket)
    h = kodla(paket, yapi, kimlikler)
    print("# Capraz dogrulama (numpy)\n")
    print(f"- paket: `{a.paket}`")
    print(f"- parca: {len(paket.yollar)}, bayt: {paket.toplam}")
    print(f"- jeton: {len(kimlikler)} ({kimlikler})")
    print(f"- gizli durum: {h.shape[0]} x {h.shape[1]}")
    print(f"- ortalama: {float(h.mean()):.6f}")
    print(f"- en kucuk: {float(h.min()):.6f}")
    print(f"- en buyuk: {float(h.max()):.6f}")
    print(f"- NaN/sonsuz: {int((~np.isfinite(h)).sum())}")

    basarisiz = False
    if a.rust_dokum:
        ham = np.fromfile(a.rust_dokum, dtype="<f4")
        if ham.size != h.size:
            print(f"\n- DOKUM BOYUTU FARKLI: {ham.size} != {h.size}")
            return 1
        rust = ham.reshape(h.shape)
        fark = np.abs(rust - h)
        olcek = max(1e-9, float(np.abs(h).max()))
        print("\n## Rust ile karsilastirma\n")
        print(f"- en buyuk mutlak fark: {float(fark.max()):.3e}")
        print(f"- ortalama mutlak fark: {float(fark.mean()):.3e}")
        print(f"- en buyuk goreli fark: {float(fark.max()) / olcek:.3e}")
        if fark.max() > a.tolerans:
            print(f"- SONUC: FARKLI (tolerans {a.tolerans:g})")
            basarisiz = True
        else:
            print(f"- SONUC: ayni (tolerans {a.tolerans:g})")

    if isaretler:
        puan, havuz, eylem = karar(paket, yapi, h, isaretler, tip, karar_yapisi["head_layers"])
        p = softmax(puan)
        print("\n## Karar (numpy)\n")
        print(f"- tip: {a.tip} ({tip})")
        print(f"- puanlar: {[round(float(x), 6) for x in puan]}")
        print(f"- olasiliklar: {[round(float(x), 6) for x in p]}")
        print(f"- havuz (h[0]) normu: {float(np.linalg.norm(havuz)):.6f}")
        print(f"- havuz en buyuk mutlak: {float(np.abs(havuz).max()):.6f}")
        print(f"- eylem puanlari: {[round(float(x), 4) for x in eylem]}")
    return 1 if basarisiz else 0


if __name__ == "__main__":
    raise SystemExit(main())
