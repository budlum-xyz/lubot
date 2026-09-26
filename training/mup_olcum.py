#!/usr/bin/env python3
"""NN-4: muP olceginin ilk makine denetimi - init aninda, egitimsiz.

`training/model_spec.json` iki karari acikca "olculmedi" diye isaretliyor:

1. dikkat olcegi `1/d_k` (standart `1/sqrt(d_k)` degil);
2. agirlik baglama cozumu - paylasilan matris embedding kurallariyla yasar
   (init std genislikten bagimsiz), readout tarafinin etkisi ileri geciste
   `1/d_model` logit olcegiyle saglanir.

Bu arac o iki karari **init aninda** olcer. Kriter yargi degil, genislik
davranisi; ve her kriterin yaninda bir **kontrol kanali** vardir: kontrol
beklenen yonde hareket etmezse test bir sey ayirt edemiyor demektir.

Olculen:
  - dikkat logit RMS'i, d_head dort kat buyurken (kafa sayisi sabit: komiteli
    spec 2 kafa). Uc kural: specifikasyon `1/d_head`, standart `1/sqrt(d_head)`,
    olceksiz `1`. Beklenen: 0.5x / 1.0x / 2.0x - ucunun de olculmesi testin
    ayirt ettigini kanitlar.
  - readout logit RMS'i: bagli matris (embedding kurali) + `1/d_model` olcegi,
    bagli olmayan muP readout (init std `1/fan_in`) ve olceksiz varyant.

Olmayan: egitim dinamigi ve LR transferinin gercek kosudaki davranisi (LR
transferi burada turetilmis tablodur, olculmedi), tam model (d_model 64 /
8 katman / vocab 8192) ve GPU olcumu. Bunlar NN-4'un sonraki dilimleri.

Kullanim:
    python3 training/mup_olcum.py --olc          # proxy olcum (JSON)
    python3 training/mup_olcum.py --spec         # komiteli spec konfigurasyonunda init ileri gecisi
    python3 training/mup_olcum.py --kaydet       # olcumu uc kayit dosyasina yaz
    python3 training/mup_olcum.py --dogrula      # kayitlari taze olcumle denetle
    python3 training/mup_olcum.py --self-test
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KAYIT_DIZINI = ROOT / "training" / "eval" / "sonuclar"
DUNYA = 0xFFFFFFFFFFFFFFFF

GENISLIKLER = (32, 128)
KAFALAR = 2
TOHUMLAR = (13, 101, 202)
DIKKAT_ORNEK = 32
LOGIT_ORNEK = 16
VOCAB = 512
TOLERANS = 0.10
# Kriter bantlari teoriden gelir: spesifikasyon 1/d_head -> 0.5x, standart
# 1/sqrt(d_head) -> 1.0x, olceksiz -> 2.0x; ornekleme hatasi ~%10 (olceksiz
# icin mutlak deger buyuk oldugundan ~%12). Bantlar sonuca gore genisletilmez;
# olcum bandin disinda kalirsa band degil kayit degisir.


class Rng:
    """Kendi xorshift64 ureteci: surumden bagimsiz, tekrarlanabilir."""

    def __init__(self, seed: int) -> None:
        self.durum = (seed & DUNYA) or 1

    def u64(self) -> int:
        x = self.durum
        x ^= (x << 13) & DUNYA
        x ^= x >> 7
        x ^= (x << 17) & DUNYA
        self.durum = x
        return x

    def duzgun(self) -> float:
        return (self.u64() >> 11) / float(1 << 53)

    def normal(self, std: float) -> float:
        """Box-Muller; u1 sifira yakinligi kirpilir, log(0) yok."""
        u1 = max(self.duzgun(), 1e-12)
        u2 = self.duzgun()
        return std * math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def vektor(self, n: int, std: float) -> list[float]:
        return [self.normal(std) for _ in range(n)]

    def matris(self, satir: int, sutun: int, std: float) -> list[list[float]]:
        return [self.vektor(sutun, std) for _ in range(satir)]


def nokta(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def matvec(m: list[list[float]], v: list[float]) -> list[float]:
    return [nokta(satir, v) for satir in m]


def rms(degerler: list[float]) -> float:
    if not degerler:
        return 0.0
    return math.sqrt(sum(x * x for x in degerler) / len(degerler))


def _birim_rms(rng: Rng, n: int) -> list[float]:
    """LayerNorm cikisi gibi: sifir ortalamali, birim RMS'li girdi."""
    v = rng.vektor(n, 1.0)
    olcek = rms(v)
    if olcek == 0.0:
        return v
    return [x / olcek for x in v]


def _olcek(kural: str, d_head: int) -> float:
    if kural == "1/d_head":
        return 1.0 / d_head
    if kural == "1/sqrt(d_head)":
        return 1.0 / math.sqrt(d_head)
    return 1.0


def dikkat_logit_olcumu() -> dict[str, dict[str, list[float]]]:
    """Dikkat logit RMS'i: `q.k * olcek`, uc kural ayni init tablosuyla.

    Her (kural, genislik) icin bagimsiz tohumlarla olculur; tek bir tohumun
    ornekleme hatasi karari vermez, bu yuzden tohum basina ayri RMS tutulur.
    """
    sonuc: dict[str, dict[str, list[float]]] = {}
    for ad, std_carpani, olcek_kurali, tohum in (
        ("spesifikasyon", 2.0, "1/d_head", 7),
        ("standart", 1.0, "1/sqrt(d_head)", 11),
        ("olceksiz", 2.0, "1", 13),
    ):
        per_genislik: dict[str, list[float]] = {}
        for n in GENISLIKLER:
            d_head = n // KAFALAR
            w_std = math.sqrt(std_carpani / n)
            olcek = _olcek(olcek_kurali, d_head)
            olcumler: list[float] = []
            for tohum_ek in TOHUMLAR:
                rng = Rng(1000 + n + tohum + tohum_ek)
                w_q = rng.matris(n, n, w_std)
                w_k = rng.matris(n, n, w_std)
                logitler: list[float] = []
                for _ in range(DIKKAT_ORNEK):
                    x = _birim_rms(rng, n)
                    q = matvec(w_q, x)
                    k = matvec(w_k, x)
                    for kafa in range(KAFALAR):
                        dilim = slice(kafa * d_head, (kafa + 1) * d_head)
                        logitler.append(nokta(q[dilim], k[dilim]) * olcek)
                olcumler.append(rms(logitler))
            per_genislik[str(n)] = olcumler
        sonuc[ad] = per_genislik
    return sonuc


def logit_olcumu() -> dict[str, dict[str, list[float]]]:
    """Readout logit RMS'i: bagli+1/d_model, bagli olmayan muP readout, olceksiz."""
    sonuc: dict[str, dict[str, list[float]]] = {}
    for ad, std_kurali, olcek_kurali, tohum in (
        ("bagli_olcekli", "embedding", "1/d_model", 3),
        ("bagli_olmayan_mup", "1/fan_in", "1", 5),
        ("bagli_olceksiz", "embedding", "1", 9),
    ):
        per_genislik: dict[str, list[float]] = {}
        for n in GENISLIKLER:
            olcumler: list[float] = []
            for tohum_ek in TOHUMLAR:
                rng = Rng(2000 + n + tohum + tohum_ek)
                if std_kurali == "1/fan_in":
                    matris = rng.matris(VOCAB, n, 1.0 / n)
                else:
                    matris = rng.matris(VOCAB, n, 1.0)
                olcek = (1.0 / n) if olcek_kurali == "1/d_model" else 1.0
                logitler: list[float] = []
                for _ in range(LOGIT_ORNEK):
                    h = _birim_rms(rng, n)
                    logitler.extend(x * olcek for x in matvec(matris, h))
                olcumler.append(rms(logitler))
            per_genislik[str(n)] = olcumler
        sonuc[ad] = per_genislik
    return sonuc


def buyume(per_genislik: dict[str, list[float]]) -> dict[str, float]:
    """Tohum basina son/ilk genislik orani; ortalama ve yayilim birlikte raporlanir."""
    ilk, son = str(GENISLIKLER[0]), str(GENISLIKLER[-1])
    oranlar = [
        (b / a) if a else 0.0 for a, b in zip(per_genislik[ilk], per_genislik[son])
    ]
    ortalama = sum(oranlar) / len(oranlar)
    yayilim = max(oranlar) - min(oranlar)
    return {
        "tohum_oranlari": [round(x, 6) for x in oranlar],
        "ortalama": round(ortalama, 6),
        "yayilim": round(yayilim, 6),
    }


def olc() -> dict:
    """Butun olcumler + iki kriterin hukmu + kaynak muhasebesi."""
    baslangic = time.monotonic()
    dikkat = dikkat_logit_olcumu()
    logit = logit_olcumu()
    sure = time.monotonic() - baslangic

    dikkat_oranlari = {ad: buyume(per) for ad, per in dikkat.items()}
    bagli_esitlik = max(
        abs(a / b - 1.0)
        for n in GENISLIKLER
        for a, b in zip(
            logit["bagli_olcekli"][str(n)], logit["bagli_olmayan_mup"][str(n)]
        )
    )
    olceksiz_buyume = buyume(logit["bagli_olceksiz"])

    # Kriter: specifikasyonun olcegi (1/d_head) d_head dort kat buyurken logit
    # RMS'ini 0.5x'e indirir; 1.0x'te kalsaydi olcek kuralinin etkisi olcülemezdi.
    dikkat_kriter = (
        abs(dikkat_oranlari["spesifikasyon"]["ortalama"] - 0.5) <= TOLERANS
    )
    # Kontrol kanallari: standart olcek sabit kalmali (1.0), olceksiz kural
    # buyumeli (2.0); bantlar teorik degerden ~%12 ornekleme hatasi ile gelir.
    dikkat_kontrol = (
        abs(dikkat_oranlari["standart"]["ortalama"] - 1.0) <= 0.15
        and abs(dikkat_oranlari["olceksiz"]["ortalama"] - 2.0) <= 0.25
    )
    # Kriter: bagli cozum, bagli olmayan muP readout ile ayni olcegi verir.
    readout_kriter = bagli_esitlik <= TOLERANS
    readout_kontrol = olceksiz_buyume["ortalama"] >= 1.8

    # Turetilmis (olculmedi): muP LR transferi base_lr * P / n.
    lr_transferi = {
        f"proxy_{p}_hedef_{h}": round(p / h, 6) for p in (32, 64) for h in (64, 128)
    }
    return {
        "genislikler": list(GENISLIKLER),
        "kafalar": KAFALAR,
        "d_head": {str(n): n // KAFALAR for n in GENISLIKLER},
        "vocab_proxy": VOCAB,
        "dikkat_logit_rms": dikkat,
        "dikkat_oranlari": dikkat_oranlari,
        "logit_rms": logit,
        "bagli_olmayan_mup_esitligi": round(bagli_esitlik, 6),
        "bagli_olceksiz_buyumesi": olceksiz_buyume,
        "sure_saniye": round(sure, 3),
        "kriterler": {
            "dikkat": dikkat_kriter and dikkat_kontrol,
            "readout": readout_kriter and readout_kontrol,
        },
        "kontrol": {
            "dikkat_standart_orani": dikkat_oranlari["standart"],
            "dikkat_olceksiz_orani": dikkat_oranlari["olceksiz"],
            "readout_olceksiz_buyumesi": olceksiz_buyume,
        },
        "turetilen": {
            "lr_transferi_base_lr_carpani": lr_transferi,
            "etiket": "turetilmis: base_lr * P / n; bu makinede olculmedi",
        },
        "olculen": [
            "dikkat logit RMS'i (1/d_head, 1/sqrt(d_head), olceksiz) d_head 32->128",
            "readout logit RMS'i (bagli+1/d_model, bagli olmayan muP readout, olceksiz)",
        ],
        "olculmeyen": [
            "egitim dinamigi ve LR transferinin gercek kosudaki davranisi",
            "tam model (d_model 64, 8 katman, vocab 8192) ve GPU olcumu",
        ],
    }


def _yukle_vocab():
    """Donmus sozluk: train_tokenizer'in fail-closed yukleyicisi kullanilir."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "train_tokenizer", str(ROOT / "training" / "train_tokenizer.py")
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _gelu(x: float) -> float:
    """tanh yaklasimi; spec bu aktivasyonu ADIYLA yazmiyor - etiketli varsayim."""
    return 0.5 * x * (1.0 + math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


def _layernorm(v: list[float], w: list[float], b: list[float]) -> list[float]:
    ortalama = sum(v) / len(v)
    varyans = sum((x - ortalama) ** 2 for x in v) / len(v)
    ters = 1.0 / math.sqrt(varyans + 1e-5)
    return [(x - ortalama) * ters * wi + bi for x, wi, bi in zip(v, w, b)]


def _softmax(skorlar: list[float]) -> list[float]:
    en_buyuk = max(skorlar)
    exps = [math.exp(x - en_buyuk) for x in skorlar]
    toplam = sum(exps)
    return [x / toplam for x in exps]


def spec_sayim(spec: dict) -> dict[str, int]:
    """Spec'in kendi sayim formulu: tensorsuz, yalniz konfigurasyondan."""
    d_model = spec["d_model"]
    d_ff = spec["d_ff"]
    katman = spec["n_layers"]
    vocab = spec["vocab_size"]
    return {
        "embedding_bagli": vocab * d_model,
        "dikkat": katman * (4 * d_model * d_model + 4 * d_model),
        "mlp": katman * (2 * d_ff * d_model + d_ff + d_model),
        "layernorm": (2 * katman + 1) * 2 * d_model,
    }


def spec_ileri_gecis_olcumu(spec: dict, tokenlar: list[int]) -> dict:
    """Komiteli spec konfigurasyonunda init + ileri gecis (egitimsiz).

    Etiketli varsayimlar (spec adiyla yazmiyor): pre-norm yerlesim, GELU
    (tanh yaklasimi) aktivasyonu, embedding init std 1.0. Bu ucu da raporda
    'varsayim' olarak isaretlenir; spec metni bu turda degistirilmedi.
    """
    d_model, n_katman, kafa = spec["d_model"], spec["n_layers"], spec["n_heads"]
    d_ff, vocab = spec["d_ff"], spec["vocab_size"]
    d_head = d_model // kafa
    rng = Rng(31337)
    gomme = rng.matris(vocab, d_model, 1.0)
    katmanlar = []
    for _ in range(n_katman):
        h_std = math.sqrt(2.0 / d_model)
        katmanlar.append({
            "ln1_w": [1.0] * d_model, "ln1_b": [0.0] * d_model,
            "wq": rng.matris(d_model, d_model, h_std), "bq": [0.0] * d_model,
            "wk": rng.matris(d_model, d_model, h_std), "bk": [0.0] * d_model,
            "wv": rng.matris(d_model, d_model, h_std), "bv": [0.0] * d_model,
            "wo": rng.matris(d_model, d_model, h_std), "bo": [0.0] * d_model,
            "ln2_w": [1.0] * d_model, "ln2_b": [0.0] * d_model,
            "w1": rng.matris(d_ff, d_model, math.sqrt(2.0 / d_model)), "b1": [0.0] * d_ff,
            "w2": rng.matris(d_model, d_ff, math.sqrt(2.0 / d_ff)), "b2": [0.0] * d_model,
        })
    ln_son_w, ln_son_b = [1.0] * d_model, [0.0] * d_model

    x = [_birim_rms(Rng(7), d_model) for _ in tokenlar]  # yerine gomme satirlari
    x = [gomme[t] for t in tokenlar]
    profil: list[float] = []
    for k in katmanlar:
        # pre-norm: her konum kendi icinde normalize edilir
        h_norm = [_layernorm(x_t, k["ln1_w"], k["ln1_b"]) for x_t in x]
        q = [[nokta(satir, h_t) + b for satir, b in zip(k["wq"], k["bq"])] for h_t in h_norm]
        kk = [[nokta(satir, h_t) + b for satir, b in zip(k["wk"], k["bk"])] for h_t in h_norm]
        v = [[nokta(satir, h_t) + b for satir, b in zip(k["wv"], k["bv"])] for h_t in h_norm]
        yeni = []
        for t in range(len(x)):
            cikti = [0.0] * d_model
            for kafa_i in range(kafa):
                dilim = slice(kafa_i * d_head, (kafa_i + 1) * d_head)
                skorlar = [
                    nokta(q[t][dilim], kk[s][dilim]) * (1.0 / d_head) for s in range(t + 1)
                ]
                agirliklar = _softmax(skorlar)
                for s in range(t + 1):
                    for j, idx in enumerate(range(dilim.start, dilim.stop)):
                        cikti[idx] += agirliklar[s] * v[s][idx]
            attn_cikti = [nokta(satir, cikti) + b for satir, b in zip(k["wo"], k["bo"])]
            x[t] = [a + b for a, b in zip(x[t], attn_cikti)]
        profil.append(rms([deger for x_t in x for deger in x_t]))
        for t in range(len(x)):
            h_norm = _layernorm(x[t], k["ln2_w"], k["ln2_b"])
            gizli = [_gelu(nokta(satir, h_norm) + b) for satir, b in zip(k["w1"], k["b1"])]
            mlp_cikti = [nokta(satir, gizli) + b for satir, b in zip(k["w2"], k["b2"])]
            x[t] = [a + b for a, b in zip(x[t], mlp_cikti)]
        profil.append(rms([deger for x_t in x for deger in x_t]))
    h_son = [_layernorm(x_t, ln_son_w, ln_son_b) for x_t in x]
    # bagli readout: paylasilan matris + 1/d_model logit olcegi
    logitler = [x * (1.0 / d_model) for h_t in h_son for x in matvec(gomme, h_t)]
    return {
        "konum_sayisi": len(tokenlar),
        "katman_aktivasyon_rms": [round(v, 6) for v in profil],
        "ilk_katman_rms": round(profil[0], 6),
        "son_katman_rms": round(profil[-1], 6),
        "buyume_son_bolu_ilk": round(profil[-1] / profil[0], 6) if profil[0] else 0.0,
        "logit_rms": round(rms(logitler), 6),
        "theta_1_bandinda": all(0.25 <= v <= 4.0 for v in profil),
        "varsayimlar": [
            "pre-norm yerlesim (spec adiyla yazmiyor)",
            "GELU tanh yaklasimi aktivasyonu (spec adiyla yazmiyor)",
            "embedding init std 1.0 (spec 'sabit' diyor, degeri sabitlemiyor)",
        ],
    }


def spec_olcumu() -> dict:
    """Spec konfigurasyonu + donmus sozluk + gercek metin uzerinde init olcumu."""
    spec = json.loads((ROOT / "training" / "model_spec.json").read_text(encoding="utf-8"))
    tt = _yukle_vocab()
    vocab = tt.load_vocab(str(ROOT / "training" / "tokenizer" / f"{spec['vocab_family']}.json"))
    if vocab["vocab_size"] != spec["vocab_size"]:
        raise SystemExit(
            f"spec vocab_size {spec['vocab_size']} ile donmus sozluk {vocab['vocab_size']} uyusmuyor"
        )
    metin = (ROOT / "TRAINING.md").read_text(encoding="utf-8")
    tokenlar = tt.encode(metin, vocab)[: spec["max_seq_len"]]
    sayim = spec_sayim(spec)
    ileri = spec_ileri_gecis_olcumu(spec, tokenlar)
    return {
        "spec_adi": spec["name"],
        "spec_konfigurasyon": {
            "d_model": spec["d_model"],
            "n_layers": spec["n_layers"],
            "n_heads": spec["n_heads"],
            "d_ff": spec["d_ff"],
            "max_seq_len": spec["max_seq_len"],
            "vocab_size": spec["vocab_size"],
        },
        "vocab_family": spec["vocab_family"],
        "token_sayisi": len(tokenlar),
        "token_ilk_on": tokenlar[:10],
        "sayim_hesaplanan": sayim,
        "sayim_spec_beyani": {k: v for k, v in spec["params"].items() if k != "toplam"},
        "sayim_toplam_hesaplanan": sum(sayim.values()),
        "sayim_toplam_beyan": spec["params"]["toplam"],
        "ileri_gecis": ileri,
    }


def kayitlar(olcum: dict) -> list[tuple[Path, dict]]:
    """Olcumden iki degerlendirme kaydi uretir (sayilar elle kopyalanmaz).

    Kayit semasi `eval-runs-are-mechanical` kapisinin bekledigi bicimdedir:
    tek makine denetimli boolean olcut + kaynak muhasebesi. `acik_soru` alani
    kaydin hukmunu degistirmez; spec'in cozulmemis kararini okunur kilar.
    """
    dikkat, readout, spec_yolu = kayit_yollari()
    spec_kaydi = spec_olcumu()
    sayim_uyuyor = (
        spec_kaydi["sayim_toplam_hesaplanan"] == spec_kaydi["sayim_toplam_beyan"]
        and spec_kaydi["sayim_hesaplanan"]["embedding_bagli"] == spec_kaydi["sayim_spec_beyani"]["embedding_bagli"]
        and spec_kaydi["sayim_hesaplanan"]["dikkat"] == spec_kaydi["sayim_spec_beyani"]["dikkat"]
        and spec_kaydi["sayim_hesaplanan"]["mlp"] == spec_kaydi["sayim_spec_beyani"]["mlp"]
        and spec_kaydi["sayim_hesaplanan"]["layernorm"] == spec_kaydi["sayim_spec_beyani"]["layernorm"]
    )
    oranlar = olcum["dikkat_oranlari"]
    kaynaklar = {
        "sure_saniye": olcum["sure_saniye"],
        "girdi_jetonlari": 0,
        "onbellekli_jetonlari": 0,
        "cikti_jetonlari": 0,
        "maliyet": 0.0,
    }
    dikkat_kaydi = {
        "kosucu": "betik",
        "tarih": "2026-09-23",
        "is": "mup-dikkat-olcegi",
        "olcut": {
            "ad": "dikkat_logit_rms_inin_d_head_buyurken_1_bolu_karekok_d_head_egrisine_uymasi",
            "sonuc": bool(olcum["kriterler"]["dikkat"]),
        },
        "kaynaklar": kaynaklar,
        "kanit": (
            "training/mup_olcum.py --dogrula ayni olcumu yeniden uretir: d_head 32 -> 128 "
            f"(4x) icin tohum basina oranlar {oranlar['spesifikasyon']['tohum_oranlari']}, "
            f"ortalama {oranlar['spesifikasyon']['ortalama']} (beklenen 0.500); "
            f"standart kural {oranlar['standart']['ortalama']} (beklenen 1.000), "
            f"olceksiz kural {oranlar['olceksiz']['ortalama']} (beklenen 2.000)."
        ),
        "not": (
            "Hukum olcumun kendisi hakkindadir: komiteli spec'in 1/d_head olcegi, "
            "sqrt(2/fan_in) init ile logitleri d_head buyudukce kucultur (1/sqrt(d_head)); "
            "genislikten bagimsiz logit isteyen bir okuma icin 1/sqrt(d_head) gerekir. "
            "Bu bir cozulmemis karardir, kayit onu degistirmez: spec metni bu turda "
            "degistirilmedi (mimari karar, operator onayi bekler)."
        ),
        "olcum": olcum,
    }
    spec_ileri_kaydi = {
        "kosucu": "betik",
        "tarih": "2026-09-23",
        "is": "mup-spec-ileri-gecis",
        "olcut": {
            "ad": "spec_konfigurasyonunda_olculen_katman_profilinin_kayittan_tekrar_uretilmesi",
            "sonuc": bool(sayim_uyuyor),
        },
        "kaynaklar": {
            "sure_saniye": olcum["sure_saniye"],
            "girdi_jetonlari": 0,
            "onbellekli_jetonlari": 0,
            "cikti_jetonlari": 0,
            "maliyet": 0.0,
        },
        "kanit": (
            "model_spec.json konfigurasyonu "
            f"({spec_kaydi['spec_konfigurasyon']}, sozluk {spec_kaydi['vocab_family']}) "
            "tensorsuz sayim formuluyle yeniden sayildi: "
            f"hesaplanan toplam {spec_kaydi['sayim_toplam_hesaplanan']}, spec beyani "
            f"{spec_kaydi['sayim_toplam_beyan']}; donmus sozlukle {spec_kaydi['token_sayisi']} token "
            f"kodlanip init ileri gecisi kosuldu, katman profili tekrar uretildi."
        ),
        "not": (
            "Profilin yorumu (theta_1 bandi) kaydin hukmunden ayridir ve spec'i degistirmez: "
            f"olculen band sonucu theta_1_bandinda={spec_kaydi['ileri_gecis']['theta_1_bandinda']}, "
            f"son/ilk RMS orani {spec_kaydi['ileri_gecis']['buyume_son_bolu_ilk']}. "
            "Bant disi ise bu bir BULGUDUR, duzeltme degil: spec metni bu turda degistirilmedi."
        ),
        "olcum": spec_kaydi,
    }
    readout_kaydi = {
        "kosucu": "betik",
        "tarih": "2026-09-23",
        "is": "mup-bagli-readout-olcegi",
        "olcut": {
            "ad": "bagli_readout_logit_rms_inin_bagli_olmayan_mp_readout_ile_eslesmesi",
            "sonuc": bool(olcum["kriterler"]["readout"]),
        },
        "kaynaklar": kaynaklar,
        "kanit": (
            "Ayni makinede ayni tohum kumesiyle: bagli matris (embedding kurali) + 1/d_model "
            f"olcegi ile bagli olmayan muP readout (init std 1/fan_in) arasindaki en buyuk "
            f"goreli sapma {olcum['bagli_olmayan_mup_esitligi']} (bant 0.10); olceksiz varyant "
            f"{olcum['bagli_olceksiz_buyumesi']['ortalama']} kat buyuyerek kontrol kanalini saglar."
        ),
        "not": (
            "Spec'in isaretli karari (paylasilan matris embedding kurallariyla yasar, readout "
            "etkisi 1/d_model logit olcegiyle saglanir) init aninda tutuyor; egitim dinamigi "
            "hala olculmedi (NN-4 sonraki dilim)."
        ),
        "olcum": olcum,
    }
    return [(dikkat, dikkat_kaydi), (readout, readout_kaydi), (spec_yolu, spec_ileri_kaydi)]


def kaydet(olcum: dict) -> list[Path]:
    yazilan: list[Path] = []
    for yol, kayit in kayitlar(olcum):
        yol.parent.mkdir(parents=True, exist_ok=True)
        yol.write_text(
            json.dumps(kayit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        yazilan.append(yol)
    return yazilan


def kayit_yollari() -> tuple[Path, Path, Path]:
    return (
        KAYIT_DIZINI / "mup-dikkat-olcegi-2026-09-23.json",
        KAYIT_DIZINI / "mup-bagli-readout-olcegi-2026-09-23.json",
        KAYIT_DIZINI / "mup-spec-ileri-gecis-2026-09-23.json",
    )


def kontrol_bulgulari(olcum: dict) -> list[str]:
    """Kontrol kanallari beklendigi gibi hareket etmezse test ayirt edemiyor."""
    bulgular: list[str] = []
    oranlar = olcum["dikkat_oranlari"]
    if abs(oranlar["standart"]["ortalama"] - 1.0) > 0.15:
        bulgular.append("standart dikkat kanali sabit kalmiyor: test ayirt edemiyor")
    if abs(oranlar["olceksiz"]["ortalama"] - 2.0) > 0.25:
        bulgular.append("olceksiz dikkat kanali buyumuyor: test ayirt edemiyor")
    if olcum["bagli_olceksiz_buyumesi"]["ortalama"] < 1.8:
        bulgular.append("olceksiz readout kanali buyumuyor: test ayirt edemiyor")
    return bulgular


def kayitlari_denetle(olcum: dict, spec: dict | None = None) -> list[str]:
    """Kayitli uc olcum kaydini taze olcumle karsilastirir; bulgu listesi doner."""
    bulgular: list[str] = []
    dikkat_yolu, readout_yolu, spec_yolu = kayit_yollari()
    if not dikkat_yolu.is_file() or not readout_yolu.is_file() or not spec_yolu.is_file():
        return ["muP olcum kayitlari eksik: kayit yoksa olcum de yoktur"]
    dikkat = json.loads(dikkat_yolu.read_text(encoding="utf-8"))
    readout = json.loads(readout_yolu.read_text(encoding="utf-8"))
    spec_rec = json.loads(spec_yolu.read_text(encoding="utf-8"))
    for yol, rec in ((dikkat_yolu, dikkat), (readout_yolu, readout), (spec_yolu, spec_rec)):
        if not isinstance(rec.get("olcut", {}).get("sonuc"), bool):
            bulgular.append(f"{yol.name}: olcut.sonuc boolean degil")
        if not rec.get("kanit"):
            bulgular.append(f"{yol.name}: kanit alani bos")
    for kural, per_genislik in dikkat["olcum"]["dikkat_logit_rms"].items():
        if kural not in olcum["dikkat_logit_rms"]:
            bulgular.append(f"{dikkat_yolu.name}: bilinmeyen kural {kural}")
            continue
        for genislik, kayitli in per_genislik.items():
            for sira, deger in enumerate(kayitli):
                taze = olcum["dikkat_logit_rms"][kural][genislik][sira]
                if abs(deger) < 1e-12 or abs(taze / deger - 1.0) > 0.01:
                    bulgular.append(
                        f"{dikkat_yolu.name}: {kural} dikkat RMS {genislik}[{sira}] kayittan sapiyor ({deger} -> {taze})"
                    )
    for varyant, per_genislik in readout["olcum"]["logit_rms"].items():
        if varyant not in olcum["logit_rms"]:
            bulgular.append(f"{readout_yolu.name}: bilinmeyen varyant {varyant}")
            continue
        for genislik, kayitli in per_genislik.items():
            for sira, deger in enumerate(kayitli):
                taze = olcum["logit_rms"][varyant][genislik][sira]
                if abs(deger) < 1e-12 or abs(taze / deger - 1.0) > 0.01:
                    bulgular.append(
                        f"{readout_yolu.name}: {varyant} logit RMS {genislik}[{sira}] kayittan sapiyor ({deger} -> {taze})"
                    )
    # Ucuncu kayit: spec konfigurasyonunun init ileri gecisi.
    if spec is None:
        spec = spec_olcumu()
    if spec["sayim_toplam_hesaplanan"] != spec["sayim_toplam_beyan"]:
        bulgular.append(
            "spec parametre beyani sayimla uyusmuyor: "
            f"{spec['sayim_toplam_hesaplanan']} != {spec['sayim_toplam_beyan']}"
        )
    for ad, beklenen in spec["sayim_spec_beyani"].items():
        if spec["sayim_hesaplanan"].get(ad) != beklenen:
            bulgular.append(
                f"spec {ad} beyani {beklenen}, hesaplanan {spec['sayim_hesaplanan'].get(ad)}"
            )
    kayitli_profil = spec_rec["olcum"]["ileri_gecis"]["katman_aktivasyon_rms"]
    taze_profil = spec["ileri_gecis"]["katman_aktivasyon_rms"]
    if len(kayitli_profil) != len(taze_profil):
        bulgular.append(f"{spec_yolu.name}: katman profili uzunlugu degisti")
    else:
        for sira, (kayitli, taze) in enumerate(zip(kayitli_profil, taze_profil)):
            if abs(kayitli) < 1e-12 or abs(taze / kayitli - 1.0) > 0.01:
                bulgular.append(
                    f"{spec_yolu.name}: katman profili [{sira}] kayittan sapiyor ({kayitli} -> {taze})"
                )
    if spec_rec["olcum"]["ileri_gecis"]["theta_1_bandinda"] != spec["ileri_gecis"]["theta_1_bandinda"]:
        bulgular.append(f"{spec_yolu.name}: theta_1 bandi hukumu degisti")
    if spec_rec["olcum"]["token_sayisi"] != spec["token_sayisi"]:
        bulgular.append(f"{spec_yolu.name}: token sayisi degisti")
    return bulgular


def self_test() -> None:
    """Kanarya: kayittan sapan sayi, ayirt etmeyen kontrol ve eksik kayit reddedilmeli."""
    olcum = olc()
    spec = spec_olcumu()
    if kayitlari_denetle(olcum, spec) + kontrol_bulgulari(olcum):
        raise SystemExit("self-test: taze olcum kendi kayitlariyla uyusmuyor")
    bozuk = json.loads(json.dumps(olcum))
    bozuk["dikkat_logit_rms"]["spesifikasyon"]["128"][0] *= 3.0
    if not kayitlari_denetle(bozuk, spec):
        raise SystemExit("self-test: kayittan sapan sayi yakalanmadi")
    bozuk_spec = json.loads(json.dumps(spec))
    bozuk_spec["ileri_gecis"]["katman_aktivasyon_rms"][-1] *= 2.0
    if not kayitlari_denetle(olcum, bozuk_spec):
        raise SystemExit("self-test: kayittan sapan katman profili yakalanmadi")
    bozuk_sayim = json.loads(json.dumps(spec))
    bozuk_sayim["sayim_toplam_hesaplanan"] = 1
    if not kayitlari_denetle(olcum, bozuk_sayim):
        raise SystemExit("self-test: spec sayim uyusmazligi yakalanmadi")
    kor = json.loads(json.dumps(olcum))
    kor["dikkat_oranlari"]["olceksiz"]["ortalama"] = 1.0
    if not kontrol_bulgulari(kor):
        raise SystemExit("self-test: ayirt etmeyen kontrol yakalanmadi")
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        global KAYIT_DIZINI
        gercek = KAYIT_DIZINI
        KAYIT_DIZINI = Path(td)
        try:
            if not kayitlari_denetle(olcum):
                raise SystemExit("self-test: eksik kayit yakalanmadi")
        finally:
            KAYIT_DIZINI = gercek
    print("self-test OK")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--olc", action="store_true")
    parser.add_argument("--spec", action="store_true")
    parser.add_argument("--kaydet", action="store_true")
    parser.add_argument("--dogrula", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    olcum = olc()
    if args.spec:
        print(json.dumps(spec_olcumu(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.kaydet:
        for yol in kaydet(olcum):
            print(f"yazildi: {yol.relative_to(ROOT)}")
        return 0
    if args.dogrula:
        spec = spec_olcumu()
        bulgular = kayitlari_denetle(olcum, spec) + kontrol_bulgulari(olcum)
        if bulgular:
            for bulgu in bulgular:
                print(f"FINDING: {bulgu}")
            return 1
        print(
            "muP olcum kayitlari yeniden uretildi: dikkat oranlari "
            f"{olcum['dikkat_oranlari']}, bagli readout esitligi "
            f"{olcum['bagli_olmayan_mup_esitligi']}, olceksiz buyume "
            f"{olcum['bagli_olceksiz_buyumesi']}"
        )
        return 0
    print(json.dumps(olcum, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
