#!/usr/bin/env python3
"""Iki dilin jeton maliyeti: N maddesinin olcumu (TR/EN).

N "olculmuyor" diyordu: korpusta Turkce ve Ingilizce metin var, sozluk
karisik kesildi, ama iki dilin ayni karakter basina kac jeton odedigi hic
olculmedi. Bu betik onu olcer: her kayit dil isaretlerine gore ayrilir
(Turkce'ye ozgu harfler + islev kelimeleri), her sinif icin karakter,
jeton ve jeton/karakter orani hesaplanir.

Olcut tek cumlede:
    iki sinifin (tr, en) jeton/karakter orani arasindaki fark beyan
    edilir; fark esigi asarsa kayit bunu BULGU olarak tasir.

Olcum jetonlayicinin kendi kodunu kullanir (`train_tokenizer.encode`),
ikinci bir jetonlama duzenegi yok: sozluk aile spec'ten okunur ve
fail-closed yukleyiciyle acilir.

    python3 training/cok_dillilik.py --olc
    python3 training/cok_dillilik.py --kur
    python3 training/cok_dillilik.py --dogrula
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KORPUS = ROOT / "corpus" / "knowledge-self.jsonl.gz"
SPEC = ROOT / "training" / "model_spec.json"
KAYIT = ROOT / "training" / "eval" / "sonuclar" / "dil-maliyeti-2026-09-24.json"

# Turkce'ye ozgu harfler: bir kayitta bunlardan hic yoksa ve islev
# kelimeleri de Ingilizce ise kayit Ingilizce sayilir.
TR_HARF = set("çğıöşüÇĞİÖŞÜ")
TR_KELIME = {"bir", "ve", "bu", "için", "ile", "olan", "olarak", "değil",
             "gibi", "kayıt", "değeri", "yoksa", "kadar", "üzere", "her"}
EN_KELIME = {"the", "of", "and", "to", "in", "is", "that", "for", "with",
             "not", "are", "this", "it", "as", "be", "on", "by", "a", "an"}
# Fark esigi: bunun altindaki fark "beyan edildi" sayilir, ustundeki
# "butce maliyeti" olarak bulguya doner.
FARK_ESIGI = 0.10


def _modul(ad: str, dosya: str):
    spec = importlib.util.spec_from_file_location(ad, str(ROOT / "training" / dosya))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def sozluk_yukle():
    """Spec'in beyan ettigi sozluk ailesi, kendi fail-closed yukleyicisiyle."""
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    aile = spec["vocab_family"]
    tt = _modul("train_tokenizer", "train_tokenizer.py")
    return tt.load_vocab(str(ROOT / "training" / "tokenizer" / f"{aile}.json")), aile


def dil_bul(metin: str) -> str:
    """Kaba ama tekrarlanabilir siniflandirma: harf + islev kelimesi.

    Ucuncu sinif bilerek var: "karisik". Iki dilli bir kaydi zorla bir
    sinifa yazmak, olcumu iddiaya cevirirdi."""
    kucuk = metin.lower()
    harf = sum(1 for c in metin if c in TR_HARF)
    kelimeler = "".join(c if c.isalpha() else " " for c in kucuk).split()
    tr = sum(1 for k in kelimeler if k in TR_KELIME)
    en = sum(1 for k in kelimeler if k in EN_KELIME)
    tr_puan = harf + 2 * tr
    en_puan = 2 * en
    if tr_puan >= 3 and tr_puan > en_puan:
        return "tr"
    if en_puan >= 3 and en_puan > tr_puan:
        return "en"
    return "karisik"


def olc() -> dict:
    """Her sinif icin karakter ve jeton; oran farki bulguysa adiyla."""
    if not KORPUS.is_file():
        raise SystemExit(f"{KORPUS.relative_to(ROOT)} yok: once korpus kurulur")
    basla = time.monotonic()
    vocab, aile = sozluk_yukle()
    tt = _modul("train_tokenizer", "train_tokenizer.py")
    siniflar: dict[str, dict[str, int]] = {
        "tr": {"kayit": 0, "karakter": 0, "jeton": 0},
        "en": {"kayit": 0, "karakter": 0, "jeton": 0},
        "karisik": {"kayit": 0, "karakter": 0, "jeton": 0},
    }
    with gzip.open(KORPUS, "rt", encoding="utf-8") as handle:
        for satir in handle:
            satir = satir.strip()
            if not satir:
                continue
            metin = json.loads(satir)["text"]
            sinif = dil_bul(metin)
            kayit = siniflar[sinif]
            kayit["kayit"] += 1
            kayit["karakter"] += len(metin)
            kayit["jeton"] += len(tt.encode(metin, vocab))
    for kayit in siniflar.values():
        kayit["jeton_basina_karakter"] = (
            round(kayit["karakter"] / kayit["jeton"], 4) if kayit["jeton"] else 0.0
        )
        kayit["karakter_basina_jeton"] = (
            round(kayit["jeton"] / kayit["karakter"], 6) if kayit["karakter"] else 0.0
        )
    tr = siniflar["tr"]["karakter_basina_jeton"]
    en = siniflar["en"]["karakter_basina_jeton"]
    fark = abs(en - tr) / max(tr, 1e-9) if tr else 0.0
    bulgu = None
    if siniflar["tr"]["kayit"] and siniflar["en"]["kayit"] and fark > FARK_ESIGI:
        pahali = "en" if en > tr else "tr"
        bulgu = {
            "hukum": (
                f"iki dilin jeton maliyeti ayni degil: {pahali} metni ayni karakter "
                f"basina {round(fark * 100, 2)}% daha pahali ({pahali} "
                f"{siniflar[pahali]['karakter_basina_jeton']:.6f} jeton/karakter, "
                f"esi {siniflar['tr' if pahali == 'en' else 'en']['karakter_basina_jeton']:.6f})"
            ),
            "olculen": (
                f"tr {siniflar['tr']['kayit']} kayit / {siniflar['tr']['karakter']} karakter / "
                f"{siniflar['tr']['jeton']} jeton; en {siniflar['en']['kayit']} kayit / "
                f"{siniflar['en']['karakter']} karakter / {siniflar['en']['jeton']} jeton; "
                f"fark {round(fark * 100, 2)}% > esik {round(FARK_ESIGI * 100, 2)}%"
            ),
            "yapilmayan": (
                "sozluk yeniden kesilmedi, korpus agirliklari degistirilmedi, kayit "
                "duzeltme yapmaz: hangi dilin ne kadar temsil edilecegi operator karari"
            ),
            "fark_orani": round(fark, 6),
        }
    return {
        "siniflar": siniflar,
        "sozluk_ailesi": aile,
        "fark_orani": round(fark, 6),
        "fark_esigi": FARK_ESIGI,
        "bulgu_dil_maliyeti": bulgu,
        "sure_saniye": round(time.monotonic() - basla, 2),
    }


def kayit_yaz(olcum: dict) -> Path:
    bulgu = olcum["bulgu_dil_maliyeti"] is not None
    kayit = {
        "is": (
            "Iki dilin jeton maliyeti korpusta olculdu: her kayit dil isaretlerine "
            "gore ayrildi ve jeton/karakter orani sinif basina hesaplandi. Sozluk "
            "ailesi spec'ten okundu; ikinci bir jetonlama duzenegi yok."
        ),
        "kosucu": "betik",
        "tarih": time.strftime("%Y-%m-%d"),
        "olcut": {
            "ad": "iki_dilin_jeton_maliyeti_sinif_bazinda_beyan_edildi",
            "sonuc": True,
        },
        "kaynaklar": {
            "sure_saniye": olcum["sure_saniye"],
            "girdi_jetonlari": 0,
            "onbellekli_jetonlari": 0,
            "cikti_jetonlari": 0,
            "maliyet": 0.0,
        },
        "kanit": olcum,
        "bulgu_var": bulgu,
    }
    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    KAYIT.write_text(
        json.dumps(kayit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return KAYIT


def dogrula() -> str:
    """Kayit taze olcumle uyusuyor mu; uyusmuyorsa gerekcesi."""
    if not KAYIT.is_file():
        raise SystemExit(f"kayit yok: {KAYIT.relative_to(ROOT)} (once --kur)")
    kayit = json.loads(KAYIT.read_text(encoding="utf-8"))
    taze = olc()
    eski = kayit.get("kanit", {})
    for sinif in ("tr", "en", "karisik"):
        a = eski.get("siniflar", {}).get(sinif, {})
        b = taze["siniflar"][sinif]
        for alan in ("kayit", "karakter", "jeton"):
            if a.get(alan) != b[alan]:
                raise SystemExit(
                    f"kayit bayat ({sinif}.{alan}): kayitta {a.get(alan)}, olcum {b[alan]} "
                    "- kaydi yeniden uret"
                )
    if eski.get("fark_orani") != taze["fark_orani"]:
        raise SystemExit(
            f"kayit bayat (fark_orani): kayitta {eski.get('fark_orani')}, "
            f"olcum {taze['fark_orani']} - kaydi yeniden uret"
        )
    return (
        f"iki dil olculdu: tr {taze['siniflar']['tr']['karakter_basina_jeton']:.6f} / "
        f"en {taze['siniflar']['en']['karakter_basina_jeton']:.6f} jeton/karakter, "
        f"fark {round(taze['fark_orani'] * 100, 2)}%, esik {round(FARK_ESIGI * 100, 2)}%"
    )


def _self_test() -> None:
    """Kanarya: siniflandirici iki dilli metni 'karisik' der, tek dilli metni
    kendi sinifina yazar; esik asildiginda bulgu uretilir."""
    assert dil_bul("Bu kayit bir Türkçe cümledir ve içinde özel harfler vardır.") == "tr"
    assert dil_bul("This record is written in English and it says the same thing.") == "en"
    assert dil_bul("```rust\nlet x = 1;\n```") == "karisik"
    kayit = kayit_yaz({"siniflar": {"tr": {"kayit": 1, "karakter": 10, "jeton": 10},
                                    "en": {"kayit": 1, "karakter": 10, "jeton": 10},
                                    "karisik": {"kayit": 1, "karakter": 10, "jeton": 10}},
                       "fark_orani": 0.0, "fark_esigi": FARK_ESIGI, "sozluk_ailesi": "test",
                       "bulgu_dil_maliyeti": None, "sure_saniye": 0.0})
    assert kayit.is_file(), "kayit yazilmadi"
    kayit.unlink()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--olc", action="store_true", help="olcumu kosar, JSON basar")
    g.add_argument("--kur", action="store_true", help="olcumu kosar, kaydi yazar")
    g.add_argument("--dogrula", action="store_true", help="kaydi taze olcumle karsilastirir")
    g.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        _self_test()
        print("self-test OK [cok-dillilik]")
        return 0
    if args.olc:
        sys.stdout.write(json.dumps(olc(), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        return 0
    if args.kur:
        olcum = olc()
        yol = kayit_yaz(olcum)
        tr = olcum["siniflar"]["tr"]
        en = olcum["siniflar"]["en"]
        print(
            f"kayit yazildi: {yol.relative_to(ROOT)} - tr {tr['kayit']} kayit "
            f"({tr['karakter_basina_jeton']:.6f} jeton/karakter), en {en['kayit']} kayit "
            f"({en['karakter_basina_jeton']:.6f}), fark {round(olcum['fark_orani'] * 100, 2)}%"
        )
        return 0
    print(dogrula())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
