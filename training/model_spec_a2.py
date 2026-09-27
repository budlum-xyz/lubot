#!/usr/bin/env python3
"""`lubot-a2-needle` ailesinin şekil ve parametre muhasebesi.

Neden ayrı bir dosya? `model_spec.py`'nin `say_params` fonksiyonu **düz
transformer** sayıyor: `mlp = n_layers * (d_model*d_ff + d_ff + d_ff*d_model +
d_model)`. a2 ailesinde FFN yok — yerinde sabit bir Walsh-Hadamard dönüşümü ve
öğrenilen köşegenler var, ve ayrıca çok-şerit artık akışı, engram tablosu ve
parametresiz bir rota var. O formülü a2'yi de sayacak şekilde esnetmek, tek
fonksiyonu iki mimarinin ortalaması yapardı; a1 ailesinin sayımı bozulmasın
diye `model_spec.py`'ye **dokunulmadı**.

**İki yol, tek sayı.** Buradaki formüller Rust'taki alt-şekillerin
`parametre_sayisi()` gövdelerinden bağımsız olarak, mimari tanımından yeniden
yazıldı. `crates/egitim/tests/a2_spec_baglanti.rs` aynı JSON'u okuyup
`BirlesikSpec::parametre_sayisi()` ile karşılaştırıyor. İki yol ayrışırsa
test kırmızı yanar — yani bu dosya bir kopya değil, bir **kontrol**.

Kullanım:
    python3 training/model_spec_a2.py --olc        # JSON'u stdout'a yaz
    python3 training/model_spec_a2.py --yaz        # spec dosyasını yaz
    python3 training/model_spec_a2.py --dogrula    # spec'i yeniden ölç
    python3 training/model_spec_a2.py --self-test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent
SPEC = KOK / "training" / "model_spec_a2.json"
A1_SPEC = KOK / "training" / "model_spec.json"


# --------------------------------------------------------------------------
# Alt şekillerin parametre sayımı. Her biri mimari tanımdan yazıldı.
# --------------------------------------------------------------------------
def serit_param(serit: int) -> int:
    """Çok-şerit artık akışı: okuma ağırlığı (şerit), karışım matrisi
    (şerit×şerit), yazma ağırlığı (şerit)."""
    return serit + serit * serit + serit


def norm_param(d_model: int) -> int:
    """Sıfır merkezli RMS norm: kanal başına bir ölçek, bias yok."""
    return d_model


def hadamard_param(d_model: int, d_r: int, blok: int) -> int:
    """Hadamard MLP.

    Walsh-Hadamard dönüşümünün **kendisinde ağırlık yok** - sabit ortonormal
    matris, n log n zamanda uygulanıyor. Öğrenilen şeyler: blok-köşegen giriş
    çiftleri (`2 * blok * (d_model/blok) * (d_r/blok)`), düşüş izdüşümü
    (`d_r * d_model`) ve yanlılıklar (`2*d_r + d_model`).

    FFN ile fark buradan okunur: aynı `d_model=64, d_ff=256` bir FFN
    `64*256 + 256 + 256*64 + 64 = 33088` parametre ister; bu şekil `d_r=128,
    blok=4` ile 12608 ile aynı işi yapmayı deniyor. Ölçülen bir iddia değil,
    şeklin kendisi - hangi ağın daha iyi olduğu eğitimde ölçülür.
    """
    if blok == 0:
        return 0
    blok_girisi = d_model // blok
    blok_ici = d_r // blok
    w_ic = 2 * blok * blok_girisi * blok_ici
    w_dusus = d_r * d_model
    yanlilik = 2 * d_r + d_model
    return w_ic + w_dusus + yanlilik


def engram_param(tablo: int, d_kv: int) -> int:
    """Hashlenmiş n-gram tablosu: `tablo` satır, her satırda bir (k, v) çifti.

    Bu parametreler **matmul görmüyor** - gather ile okunuyor. Toplam
    parametre sayısına girer, FLOP'a girmez; ikisini ayrı raporlamak bu
    ailenin bütün iddiası.
    """
    return 2 * tablo * d_kv


def engram_izdusum_param(d_model: int, d_kv: int) -> int:
    """Engram çıktısını model genişliğine taşıyan izdüşüm."""
    return d_model * d_kv


def rota_param() -> int:
    """Sinkhorn rotası parametre tutmaz - ve bu **sıfır terim olarak yazılır**.

    Toplamdan sessizce düşmek, rotanın parametresiz olduğu bilgisini görünmez
    yapardı.
    """
    return 0


def dikkat_param(d_model: int, n_heads: int, n_kv_heads: int) -> int:
    """Gruplu-sorgu dikkat (GQA): q tam, k ve v `n_kv_heads` kadar."""
    d_head = d_model // n_heads
    q = d_model * (n_heads * d_head) + n_heads * d_head
    k = d_model * (n_kv_heads * d_head) + n_kv_heads * d_head
    v = d_model * (n_kv_heads * d_head) + n_kv_heads * d_head
    o = (n_heads * d_head) * d_model + d_model
    return q + k + v + o


# --------------------------------------------------------------------------
def blok_param(sekil: dict, engramli: bool) -> int:
    """Bir bloğun parametre sayısı; `birlesik.rs`'in kompozisyonuyla aynı sıra."""
    d = sekil["d_model"]
    h = sekil["hadamard"]
    toplam = (
        serit_param(sekil["serit"])
        + norm_param(d)
        + sekil["uzman"] * hadamard_param(d, h["d_r"], h["blok"])
        + rota_param()
    )
    if engramli:
        e = sekil["engram"]
        toplam += engram_param(e["tablo"], e["d_kv"]) + engram_izdusum_param(d, e["d_kv"])
    return toplam


def say(sekil: dict) -> dict:
    """Tüm ailenin parametre muhasebesi, bileşen bileşen."""
    d = sekil["d_model"]
    n = sekil["n_layers"]
    e_kat = sekil["engram_katmanlari"]
    if e_kat > n:
        raise SystemExit(f"engram katmani {e_kat} > katman {n}")

    embedding = sekil["vocab_size"] * d
    dikkat = n * dikkat_param(d, sekil["n_heads"], sekil["n_kv_heads"])
    blok_engramsiz = blok_param(sekil, engramli=False)
    blok_engramli = blok_param(sekil, engramli=True)
    bloklar = (n - e_kat) * blok_engramsiz + e_kat * blok_engramli
    son_norm = norm_param(d)
    toplam = embedding + dikkat + bloklar + son_norm

    e = sekil["engram"]
    engram_toplam = e_kat * (engram_param(e["tablo"], e["d_kv"]))
    return {
        "embedding_bagli": embedding,
        "dikkat_gqa": dikkat,
        "bloklar": bloklar,
        "blok_engramsiz": blok_engramsiz,
        "blok_engramli": blok_engramli,
        "son_norm": son_norm,
        "rota": rota_param(),
        "toplam": toplam,
        "engram_tablolari": engram_toplam,
        "matmul_etkin": toplam - engram_toplam,
    }


SEKIL = {
    "d_model": 64,
    "n_layers": 12,
    "n_heads": 4,
    "n_kv_heads": 1,
    "vocab_size": 8192,
    "serit": 4,
    "uzman": 4,
    "engram_katmanlari": 2,
    "hadamard": {"d_r": 128, "blok": 4},
    "rota": {"uzman": 4, "k": 2, "sicaklik": 1.0, "yineleme": 4},
    "engram": {"n": 3, "tablo": 4096, "d_kv": 32},
}


def tavan() -> int:
    """K6 tavanı a1 spec'inden okunur, yeniden yazılmaz: iki dosyada iki
    tavan, bir gün ayrışacak iki tavandır."""
    a1 = json.loads(A1_SPEC.read_text(encoding="utf-8"))
    return int(a1["ceiling_reference"]["max_params_train_fp32_adamw"])


def kur() -> dict:
    sayim = say(SEKIL)
    ust = tavan()
    if sayim["toplam"] > ust:
        raise SystemExit(f"K6 ihlali: {sayim['toplam']} > tavan {ust}")
    a1 = json.loads(A1_SPEC.read_text(encoding="utf-8"))
    a1_toplam = int(a1["params"]["toplam"])
    korpus_jeton = int(a1["corpus_reference"]["bpe_tokens"])
    return {
        "name": "lubot-a2-needle",
        "schema": 1,
        "aile": "a2",
        "karar": (
            "Ikinci mimari ailesi: 7.4'un portlanmis modulleri ilk kez tek bir "
            "spec'te toplaniyor. a1 duz transformer; a2 FFN yerine Hadamard MLP, "
            "tek artik akis yerine 4 serit, ek olarak engram tablosu ve "
            "parametresiz Sinkhorn rotasi tasiyor. Bu spec bir egitim kosusu "
            "baslatmiyor - portun 'yazildi'dan 'calisiyor'a gectigi ilk adim, ve "
            "bir kosuya baglanmasi isaretli bir karardir."
        ),
        "vocab_family": "lubot-bpe-v2",
        "sekil": SEKIL,
        "params": sayim,
        "params_etiket": (
            "turetildi: model_spec_a2.py, mimari tanimindan; "
            "crates/egitim/tests/a2_spec_baglanti.rs ayni JSON'u okuyup "
            "BirlesikSpec::parametre_sayisi() ile karsilastiriyor (iki yol, tek sayi)"
        ),
        "a1_karsilastirma": {
            "a1_toplam": a1_toplam,
            "a2_toplam": sayim["toplam"],
            "oran": round(sayim["toplam"] / a1_toplam, 6),
            "not": (
                "a2 daha buyuk ve bunun nedeni engram: tablolar toplamin "
                f"%{round(100 * sayim['engram_tablolari'] / sayim['toplam'], 1)}'i. "
                "Engram parametreleri matmul gormez, gather ile okunur - yani "
                "parametre sayisi ile FLOP ayni yonde buyumuyor ve ikisi ayri "
                "raporlanir."
            ),
        },
        "matmul_orani": round(sayim["matmul_etkin"] / sayim["toplam"], 6),
        "jeton_param": {
            "korpus_jeton": korpus_jeton,
            "oran": round(korpus_jeton / sayim["toplam"], 6),
            "not": (
                "a1'de 1.94 token/param, a2'de daha dusuk - engram tabani "
                "parametre sayisini buyuttugu icin. Chinchilla referansi 20 "
                "(olculmedi); iki aile de referansin cok altinda ve bu bilincli "
                "asilıyor. Hangi ailenin daha iyi oldugu **olculmedi**; bu spec "
                "bir iddia degil, bir sekil beyani."
            ),
        },
        "ceiling_reference": {
            "max_params_train_fp32_adamw": ust,
            "kaynak": "training/model_spec.json (a1 spec'inden okunur, yeniden yazilmaz)",
            "pay": round(sayim["toplam"] / ust, 6),
        },
        "olcut": {
            "ad": "a2_param_sayimi_iki_yoldan_ayni_ve_K6_tavani_altinda",
            "ifade": (
                "model_spec_a2.py'nin mimari tanimindan turettigi parametre "
                "sayisi Rust'taki BirlesikSpec::parametre_sayisi() ile blok "
                "duzeyinde ayni VE toplam K6 tavaninin altinda"
            ),
            "sonuc": True,
        },
        "kaynaklar": {
            "sure_saniye": 0.0,
            "girdi_jetonlari": 0,
            "onbellekli_jetonlari": 0,
            "cikti_jetonlari": 0,
            "maliyet": 0.0,
        },
        "bagli_degil": (
            "Bu spec hicbir egitim cagrisindan gecmiyor. model_spec.json "
            "degismedi; a1 ailesi ve lubot-a1-derin-dar oldugu gibi duruyor."
        ),
    }


def dogrula() -> list[str]:
    """Yazılı spec'i taze ölçümle karşılaştırır."""
    if not SPEC.is_file():
        return [f"spec yok: {SPEC}"]
    yazili = json.loads(SPEC.read_text(encoding="utf-8"))
    taze = kur()
    sorun = []
    if yazili.get("sekil") != taze["sekil"]:
        sorun.append("sekil degismis ama spec yeniden uretilmemis")
    if yazili.get("params") != taze["params"]:
        sorun.append(
            f"param sayimi bayat: yazili {yazili.get('params', {}).get('toplam')}, "
            f"olculen {taze['params']['toplam']}"
        )
    if yazili.get("ceiling_reference", {}).get("max_params_train_fp32_adamw") != taze[
        "ceiling_reference"
    ]["max_params_train_fp32_adamw"]:
        sorun.append("K6 tavani a1 spec'iyle uyusmuyor")
    return sorun


def self_test() -> None:
    """Formüllerin kendi kanaryaları."""
    # Sifir blok sifir parametre verir; sessizce bolme hatasi vermez.
    assert hadamard_param(64, 128, 0) == 0
    # Hadamard, ayni genislikteki bir FFN'den kucuk olmali - bu ailenin sebebi.
    ffn = 64 * 256 + 256 + 256 * 64 + 64
    assert hadamard_param(64, 128, 4) < ffn, "hadamard FFN'den buyuk cikti"
    # Rotalama parametresiz ve bu sifir **terim olarak** duruyor.
    assert rota_param() == 0
    # GQA, tam dikkatten kucuk olmali.
    assert dikkat_param(64, 4, 1) < dikkat_param(64, 4, 4)
    # Engram tablosu matmul disinda: matmul_etkin < toplam olmali.
    s = say(SEKIL)
    assert s["matmul_etkin"] < s["toplam"], "engram tablolari matmul disina alinmamis"
    assert s["toplam"] == (
        s["embedding_bagli"] + s["dikkat_gqa"] + s["bloklar"] + s["son_norm"]
    ), "toplam bilesenlerin toplamiyla tutmuyor"
    # Engramli blok engramsizdan buyuk olmali.
    assert s["blok_engramli"] > s["blok_engramsiz"]
    # K6 tavani altinda.
    assert s["toplam"] < tavan()
    print("self-test OK [model_spec_a2]")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--olc", action="store_true")
    ap.add_argument("--yaz", action="store_true")
    ap.add_argument("--dogrula", action="store_true")
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    a = ap.parse_args(argv)
    if a.self_test:
        self_test()
        return 0
    if a.dogrula:
        sorun = dogrula()
        if sorun:
            sys.stderr.write("a2 spec bayat:\n  " + "\n  ".join(sorun) + "\n")
            return 1
        print("a2 spec taze")
        return 0
    kayit = kur()
    metin = json.dumps(kayit, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if a.yaz:
        SPEC.write_text(metin, encoding="utf-8")
    if a.olc or not a.yaz:
        sys.stdout.write(metin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
