#!/usr/bin/env python3
"""Kapışma protokolü: beyan edilen sınıfa karşı ikinci puanlayıcı koşusu (Adım 8c).

TRAINING.md'nin açık kalan son maddesi şuydu: koşucu ve kontrol noktası var,
ama beyan edilen sınıfa karşı puanlanmış bir karşılaştırma için bitmiş bir
koşu ve ikinci bir puanlayıcıya karşı protokol koşusu gerekiyordu. Bu betik
o protokolün Lubot tarafıdır ve `training/rekabet.py` ile aynı kilidi
kullanır: kartlar `kartlari_kur()` ile birebir aynı inşa edilir, böylece iki
taraf da aynı soruları ve aynı şıkları görmüş olur.

Ayrım şuradadır: `rekabet.py` Lubot'un kendi kontrol noktasını ve danışma
katmanını puanlar; bu betik, GG'nin beyan ettiği açık küçük model
sınıfından bir rakibi (AA protokolü) aynı kartlarda koşar. Rakip koşusunun
ağır bağımlılığı (torch/transformers) `tools/rakip_kosucu.py`'dedir; bu
dosya stdlib + kardeş modül sınırında kalır (MM kapısı).

K2/AA sınırı: rakip çıktısı hiçbir zaman corpus/'a veya
training/curriculum/'a girmez; tek yaşam alanı bu kayıttır. Bunu kapı değil
fizik söyler: betiğin yazabildiği tek yol --out ile verilendir ve --out
yalnızca training/eval/sonuclar/ altında kabul edilir.

Kullanım:
    python3 training/kapisma_protokolu.py --kos --model /yol/model
    python3 training/kapisma_protokolu.py --dogrula
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KOK / "training"))

import rekabet  # noqa: E402  kardes modul: kart kurgusu tek yerde kalsin

SONUCLAR = KOK / "training" / "eval" / "sonuclar"
KOSUCU = KOK / "tools" / "rakip_kosucu.py"


def kart_kimligi(kartlar: list[dict]) -> str:
    """The locked cards, hashed: a run that does not name its cards cannot be
    compared with another run, and a card edited afterwards is a new card."""
    ham = json.dumps(kartlar, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(ham.encode("utf-8")).hexdigest()


def agirlik_ozeti(model_dizini: Path) -> tuple[str, str]:
    """The weight file's digest and size: which bytes scored is part of the
    measurement, so the report carries the hash rather than a name."""
    adaylar = sorted(model_dizini.glob("*.safetensors"))
    if not adaylar:
        raise SystemExit(f"model dizininde safetensors yok: {model_dizini}")
    ozet = hashlib.sha256()
    boyut = 0
    with adaylar[0].open("rb") as dosya:
        while True:
            parca = dosya.read(1 << 20)
            if not parca:
                break
            ozet.update(parca)
            boyut += len(parca)
    return ozet.hexdigest(), adaylar[0].name


def lubot_tarafi() -> dict:
    """Lubot's own scores on the same cards, from the recorded rekabet run.

    This script does not re-score Lubot: the checkpoint is an operator
    artifact (gitignored by design), so the recorded measurement is the
    source and is cited by file rather than recomputed silently.
    """
    kayitlar = sorted(SONUCLAR.glob("rekabet-*.json"))
    if not kayitlar:
        raise SystemExit("rekabet kaydi yok: Lubot tarafi kaynaksiz kaldi")
    kayit = json.loads(kayitlar[-1].read_text(encoding="utf-8"))
    return {
        "kaynak": f"training/eval/sonuclar/{kayitlar[-1].name}",
        "top1": kayit["lubot_top1"],
        "dogru": kayit["lubot_dogru"],
        "kart_sayisi": kayit["kart_sayisi"],
        "kontrol_noktasi": "lubot-a1.ckpt",
    }


def sik_tutarli(kartlar: list[dict], lubot_kaynak: dict) -> str | None:
    """The rekabet record names each card's option order; if the rebuilt
    cards disagree, the two sides did not see the same match and the
    comparison is refused rather than averaged."""
    try:
        eski = json.loads(Path(KOK / lubot_kaynak["kaynak"]).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return f"Lubot kaynagi okunamadi: {e}"
    eski_kartlar = {k["kart"]: k for k in eski.get("kartlar", [])}
    for kart in kartlar:
        karsi = eski_kartlar.get(kart["kart"])
        if karsi is None:
            return f"{kart['kart']}: Lubot kaydinda yok"
        simdiki = [s["content_id"][:8] for s in kart["siklar"]]
        if simdiki != karsi.get("sik_sirasi"):
            return f"{kart['kart']}: sik sirasi Lubot kosusundan farkli"
    return None


def kos(model_dizini: Path) -> int:
    if not model_dizini.is_dir():
        raise SystemExit(f"model dizini yok: {model_dizini}")
    kartlar = rekabet.kartlari_kur()
    kimlik = kart_kimligi(kartlar)
    lubot = lubot_tarafi()
    sorun = sik_tutarli(kartlar, lubot)
    if sorun:
        raise SystemExit(f"karsilastirma reddedildi: {sorun}")
    agirlik_sha, agirlik_ad = agirlik_ozeti(model_dizini)

    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8"
    ) as gecici:
        json.dump(kartlar, gecici, ensure_ascii=False)
        gecici_yol = gecici.name
    try:
        calisma = subprocess.run(
            [sys.executable, str(KOSUCU), "--model", str(model_dizini),
             "--kartlar", gecici_yol],
            capture_output=True, text=True, check=False,
        )
    finally:
        Path(gecici_yol).unlink(missing_ok=True)
    if calisma.returncode != 0:
        raise SystemExit(
            f"rakip kosucu reddetti (cikis {calisma.returncode}): "
            f"{calisma.stderr.strip()[-300:]}"
        )
    rakip = json.loads(calisma.stdout)

    rakip_top1 = rakip["top1"]
    olcut_sonuc = lubot["top1"] >= rakip_top1
    bugun = datetime.date.today().isoformat()
    kayit = {
        "surum": 1,
        "tarih": bugun,
        "kosucu": "model",
        "is": "Kapisma protokolu (AA): beyan edilmis siniftan rakip, ayni "
              "kilitli kartlarda; Lubot tarafi kayitli rekabet kosusundan.",
        "olcut": {
            "ad": "lubot_top1_beyan_edilen_sinif_rakibinden_dusuk_degil",
            "sonuc": olcut_sonuc,
        },
        "kaynaklar": {
            "sure_saniye": rakip["toplam"]["sure_saniye"],
            "girdi_jetonlari": rakip["toplam"]["girdi_jetonlari"],
            "onbellekli_jetonlari": 0,
            "cikti_jetonlari": rakip["toplam"]["cikti_jetonlari"],
            "maliyet": 0.0,
        },
        "kanit": "Ayni kartlar (kartlari_kur) ve ayni sik sirasi iki tarafa "
                 "da verildi; rakip secimi kapali bir sayi olarak okundu, "
                 "Lubot top-1'i rekabet kaydindan alindi ve oranlar "
                 "karsilastirildi.",
        "rakip": {
            "model_dizini": str(model_dizini),
            "agirlik_dosyasi": agirlik_ad,
            "agirlik_sha256": agirlik_sha,
            "parametre": rakip["model"]["parametre"],
            "uretim": rakip["model"]["uretim"],
            "sinif_beyani": "training/eval/sonuclar/kiyas-sinifi-2026-09-23.json",
        },
        "lubot_tarafi": lubot,
        "kartlar_sha256": kimlik,
        "kart_sayisi": rakip["kart_sayisi"],
        "rakip_dogru": rakip["dogru"],
        "rakip_top1": rakip_top1,
        "kartlar": rakip["kartlar"],
        "k2_siniri": {
            "rakip_cikti_korpusa_girer": False,
            "rakip_cikti_curriculuma_girer": False,
            "rakip_ciktinin_tek_yasama_alani": "bu kayit (training/eval/sonuclar/)",
        },
        "olculmeyen": [
            "rakip modelin egitimi/surumu uzerinde hicbir kontrol yok; "
            "yalniz yayimlanmis agirliklarin bu kopyasi olculdu",
            "Lubot puani onceki rekabet kosusundan alindigi icin korpus "
            "buyumus olsa bile eski pasaj metinleriyle puanlanmistir; "
            "kimlikler ve sik sirasi ayni, metin zamani farkli",
            "12 kartlik kilitli set kucuktur; oranlar genelleme degil "
            "bu setin olcumudur",
            "enerji tuketimi olculmedi",
        ],
    }
    SONUCLAR.mkdir(parents=True, exist_ok=True)
    hedef = SONUCLAR / f"kapisma-sinifi-{bugun}.json"
    hedef.write_text(
        json.dumps(kayit, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(
        f"kayit yazildi: {hedef.relative_to(KOK)} - "
        f"Lubot {lubot['dogru']}/{lubot['kart_sayisi']} "
        f"({lubot['top1']:.3f}) vs rakip {rakip['dogru']}/{rakip['kart_sayisi']} "
        f"({rakip_top1:.3f}); olcut sonucu: {olcut_sonuc}"
    )
    return 0


def dogrula() -> int:
    kayitlar = sorted(SONUCLAR.glob("kapisma-sinifi-*.json"))
    if not kayitlar:
        raise SystemExit("kapisma sinifi kaydi yok")
    for yol in kayitlar:
        kayit = json.loads(yol.read_text(encoding="utf-8"))
        for alan in ("olcut", "kaynaklar", "rakip", "lubot_tarafi", "kartlar_sha256"):
            if alan not in kayit:
                raise SystemExit(f"{yol.name}: {alan} eksik")
        olcut = kayit["olcut"]
        if not isinstance(olcut.get("sonuc"), bool):
            raise SystemExit(f"{yol.name}: olcut sonucu bool degil")
        kaynak = kayit["kaynaklar"]
        for alan in ("sure_saniye", "girdi_jetonlari", "onbellekli_jetonlari",
                     "cikti_jetonlari", "maliyet"):
            if alan not in kaynak:
                raise SystemExit(f"{yol.name}: kaynak alani eksik: {alan}")
        if kayit["kosucu"] == "model" and kaynak["cikti_jetonlari"] <= 0:
            raise SystemExit(f"{yol.name}: model kosusu sifir cikti jetonu bildiriyor")
        if kayit["k2_siniri"]["rakip_cikti_korpusa_girer"]:
            raise SystemExit(f"{yol.name}: rakip cikti korpusa giriyor beyani")
        # The lock re-measured: the cards rebuilt today must be the cards the
        # run scored yesterday, or the record is about a different match.
        kartlar = rekabet.kartlari_kur()
        if kart_kimligi(kartlar) != kayit["kartlar_sha256"]:
            raise SystemExit(
                f"{yol.name}: kart kilidi degismis - kayit baska bir macin skoru"
            )
        sorun = sik_tutarli(kartlar, kayit["lubot_tarafi"])
        if sorun:
            raise SystemExit(f"{yol.name}: {sorun}")
    print(f"{len(kayitlar)} kapisma kaydi yeniden dogrulandi: kart kilidi ve "
          f"sik sirasi tutarli, sema alanlari tam")
    return 0


def main(argv: list[str]) -> int:
    ayristirici = argparse.ArgumentParser(description=__doc__)
    grup = ayristirici.add_mutually_exclusive_group(required=True)
    grup.add_argument("--kos", action="store_true")
    grup.add_argument("--dogrula", action="store_true")
    ayristirici.add_argument("--model", help="rakip modelin yerel dizini")
    args = ayristirici.parse_args(argv)
    if args.kos:
        if not args.model:
            raise SystemExit("--kos bir --model dizini istiyor")
        return kos(Path(args.model))
    return dogrula()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
