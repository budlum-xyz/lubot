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
    python3 training/mup_olcum.py --olc          # ham olcum (JSON)
    python3 training/mup_olcum.py --kaydet       # olcumu iki kayit dosyasina yaz
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


def kayitlar(olcum: dict) -> list[tuple[Path, dict]]:
    """Olcumden iki degerlendirme kaydi uretir (sayilar elle kopyalanmaz).

    Kayit semasi `eval-runs-are-mechanical` kapisinin bekledigi bicimdedir:
    tek makine denetimli boolean olcut + kaynak muhasebesi. `acik_soru` alani
    kaydin hukmunu degistirmez; spec'in cozulmemis kararini okunur kilar.
    """
    dikkat, readout = kayit_yollari()
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
    return [(dikkat, dikkat_kaydi), (readout, readout_kaydi)]


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


def kayit_yollari() -> tuple[Path, Path]:
    return (
        KAYIT_DIZINI / "mup-dikkat-olcegi-2026-09-23.json",
        KAYIT_DIZINI / "mup-bagli-readout-olcegi-2026-09-23.json",
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


def kayitlari_denetle(olcum: dict) -> list[str]:
    """Kayitli iki olcum kaydini taze olcumle karsilastirir; bulgu listesi doner."""
    bulgular: list[str] = []
    dikkat_yolu, readout_yolu = kayit_yollari()
    if not dikkat_yolu.is_file() or not readout_yolu.is_file():
        return ["muP olcum kayitlari eksik: kayit yoksa olcum de yoktur"]
    dikkat = json.loads(dikkat_yolu.read_text(encoding="utf-8"))
    readout = json.loads(readout_yolu.read_text(encoding="utf-8"))
    for yol, rec in ((dikkat_yolu, dikkat), (readout_yolu, readout)):
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
    return bulgular


def self_test() -> None:
    """Kanarya: kayittan sapan sayi, ayirt etmeyen kontrol ve eksik kayit reddedilmeli."""
    olcum = olc()
    if kayitlari_denetle(olcum) + kontrol_bulgulari(olcum):
        raise SystemExit("self-test: taze olcum kendi kayitlariyla uyusmuyor")
    bozuk = json.loads(json.dumps(olcum))
    bozuk["dikkat_logit_rms"]["spesifikasyon"]["128"][0] *= 3.0
    if not kayitlari_denetle(bozuk):
        raise SystemExit("self-test: kayittan sapan sayi yakalanmadi")
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
    parser.add_argument("--kaydet", action="store_true")
    parser.add_argument("--dogrula", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    olcum = olc()
    if args.kaydet:
        for yol in kaydet(olcum):
            print(f"yazildi: {yol.relative_to(ROOT)}")
        return 0
    if args.dogrula:
        bulgular = kayitlari_denetle(olcum) + kontrol_bulgulari(olcum)
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
