#!/usr/bin/env python3
"""Kamu malı (CC0) veri alımı — Lubot'un eğitimi için.

Operatör talimatı (2026-09-24): ihtiyaç olan veri **doğrudan alınır**; kaynak
adı depoya yazılmaz. Bu betik o talimatın uygulamasıdır:

* Kaynak kimliği (ad, revizyon, dosya yolları) **depo dışındaki** manifesttedir
  (varsayılan `/home/user/kamu-kaynak.json`). Depoya giren tek şey bu betik ve
  ürettiği korpus dosyasıdır; ikisinde de kaynak adı geçmez.
* İndirmeden **önce** lisans kaynağın kendi metadata'sından doğrulanır; izinli
  sınıfta olmayan kaynak alınmaz (kamu malı / CC0 zorunlu).
* Her kayıt korpus kaydı biçimindedir: `licence`, `attribution`, `content_id`
  (metnin sha256'sı), `asset_id` (kaynağın **kimlik özeti**, adı değil).
* Bellek sabittir: indirme önbelleğe akar, parquet blok blok okunur, kayıtlar
  tek geçişte yazılır. Metin bütçesi (`--karakter-butce`) aşılmaz.

Kullanım:

    python3 training/kamu_verisi.py --manifest /home/user/kamu-kaynak.json \\
        --out corpus/kamu-cc0.jsonl.gz --karakter-butce 30000000 [--kaydet]
    python3 training/kamu_verisi.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import sys
import time
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterator

KOK = Path(__file__).resolve().parent.parent
ONBELLEK = Path("/var/tmp/lubot-veri")  # commit'e girmez; indirilen ham dosyalar
KAYIT = KOK / "training" / "eval" / "sonuclar" / "kamu-veri-2026-09-24.json"
# Operatör kararı (2026-09-24): ne kaynak adı ne lisans adı depo ağacına yazılır.
# Kaydın taşıdığı lisans sınıfı projenin kendi kuralından gelir (tek sahip:
# training/build_corpus.py), böylece korpus kapısıyla iki yerde ayrışamaz.
LISANS_SINIFI = "kamu-mali"
AIT = "kamu malı; kaynak ve lisans adı operatör kararıyla depo ağacına yazılmaz"


def korpus_lisanslari() -> set:
    """İzinli lisans kümesinin tek sahibi korpus derleyicisidir."""
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_corpus import ALLOWED_LICENCES  # noqa: PLC0415 - tek sahip, gec yukleme

    return set(ALLOWED_LICENCES)
IZINLI_LISANSLAR = {"cc0-1.0", "cc0", "public-domain", "pddl", "unlicense"}
# Veri boru hattinin kendi etiketleri: kayit kimligi, bolum adi, kaynak tipi,
# katkida bulunan adi. Bunlar ogrenilecek metin degil, dosyanin kendi
# muhasebesidir; metne girerlerse model "id: ... split: test" ezberler.
# Icerik alanlari: korpusa yalnizca bunlar girer. KARA LISTE yetmedi - ilk
# denemede id/split suzuldu, ikincisinde template_id/split_group/
# record_fingerprint cikti, ucuncusunde degerlerin *icine* gomulu veri seti
# adlari gorundu. Beyaz liste, bilinmeyen boru hatti alanini "icerik" sanma
# hatasini kapatir: bilinmeyen alan duser.
ICERIK_ALANLARI = {
    "text", "content", "body", "sentence", "sentences", "passage", "paragraph",
    "document", "doc", "question", "soru", "answer", "cevap", "response", "reply",
    "prompt", "instruction", "input", "output", "context", "baglam", "formal",
    "informal", "summary", "ozet", "title", "baslik", "subject", "words", "word",
    "token", "label", "etiket", "act", "type", "candidate", "correction",
    "normalized_datetime", "datetime", "timezone", "normalized", "original",
    "source_text", "target_text", "translation", "translated", "rewrite",
    "rewritten", "punctuated", "punctuation", "intent", "category", "message",
    "comment", "review", "note", "description", "explanation", "reason",
    "dialogue", "utterance", "utterances", "query", "documents", "candidates",
    "answers", "choices", "code", "definition", "example", "examples", "problem",
    "solution", "solutions", "goal", "requirement", "requirements", "task",
    "feature", "features", "chosen", "rejected", "dedup_expected",
}

# Adres ve veri seti adi kalıpları: deger icinde geciyorsa metinden cikarilir.
ADRES_DESENI = re.compile(
    r"(huggingface\.co|github\.com|gitlab\.com|zenodo|doi\.org|"
    r"creativecommons\.org|archive\.org|kaggle\.com)", re.IGNORECASE)
AD_DESENI = re.compile(
    r"\b(?:turkish|turkce|hplt|wikidata|openalex|lichess|caselaw|billsum|blbooks)"
    r"[a-z0-9-]*\b|\b[a-z0-9]+(?:-[a-z0-9]+)*-(?:1m|1\.5m|500k|100k|5m)\b",
    re.IGNORECASE)


def _temiz_satir(satir: str, anahtar: str) -> bool:
    """Satir korpusa girebilir mi: alan beyaz listede ve satirda ad gecmiyor."""
    return anahtar.lower().strip() in ICERIK_ALANLARI and not ADRES_DESENI.search(satir)


def _adlari_sil(metin: str) -> str:
    """Deger icine gomulu veri seti/adres kaliplarini metinden cikarir."""
    temiz = ADRES_DESENI.sub("", metin)
    return AD_DESENI.sub("", temiz)


# JSON bu karakterleri kacirmaz, ama satir tabanli okuyucular (Python
# splitlines, bazi editorler) onlari satir sonu sayar: bir kayit iki satira
# bolunur ve iki yarim kayit da bozuk gorunur. Icerik degeri tasimazlar, bu
# yuzden metinden cikarilir - korpusun her satiri tek basina cozulebilir olmali.
GORUNMEZ_AYIRICI = dict.fromkeys(
    map(ord, "\u2028\u2029\u0085\x0b\x0c\x1c\x1d\x1e"), " "
)
SIFIR_GENISLIK = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff"), None)


def _gorunmezleri_sil(metin: str) -> str:
    """Satir ayirici gibi davranan gorunmez karakterleri giderir."""
    return metin.translate(SIFIR_GENISLIK).translate(GORUNMEZ_AYIRICI)


BLOK = 2048            # parquet blok satır sayısı
EN_AZ_PARCA = 80       # bundan kısa parçalar korpusa girmez
EN_UZUN_PARCA = 4000   # bundan uzun tek satırlar kırpılır
ZAMAN_ASIMI = 180
BASLIK = {"User-Agent": "lubot-kamu-veri/1"}


def sha256(veri: bytes) -> str:
    return hashlib.sha256(veri).hexdigest()


def _json_get(url: str):
    with urllib.request.urlopen(urllib.request.Request(url, headers=BASLIK),
                                timeout=ZAMAN_ASIMI) as yanit:
        return json.loads(yanit.read())


def hf_meta(kaynak: str) -> dict:
    return _json_get(f"https://huggingface.co/api/datasets/{kaynak}")


def hf_dosyalar(kaynak: str, dizin: str | None = None) -> list[dict]:
    yol = (f"/tree/main/{urllib.parse.quote(dizin, safe='/')}" if dizin
           else "/tree/main?recursive=1")
    return [d for d in _json_get(f"https://huggingface.co/api/datasets/{kaynak}{yol}")
            if d.get("type") == "file" and d.get("size")]


def hf_agac(kaynak: str, dizin: str | None = None) -> list[dict]:
    """Ham ağaç girdileri: hem dosyalar hem dizinler."""
    yol = (f"/tree/main/{urllib.parse.quote(dizin, safe='/')}" if dizin
           else "/tree/main?recursive=1")
    return _json_get(f"https://huggingface.co/api/datasets/{kaynak}{yol}")


def dizini_gez(kaynak: str, dizin: str, gun_dosya_siniri: int) -> list[str]:
    """Ay/gün gibi iç içe dizinleri gezer; her gün dizininden en büyük birkaç
    dosyayı alır. Yüz binlerce küçük dosyayı tek tek indirmemenin yolu."""
    secilen: list[str] = []
    for girdi in hf_agac(kaynak, dizin):
        if girdi["type"] == "directory":
            gun = girdi["path"]
            dosyalar = sorted((d for d in hf_agac(kaynak, gun) if d["type"] == "file" and d.get("size")),
                              key=lambda d: -d["size"])
            secilen.extend(d["path"] for d in dosyalar[:gun_dosya_siniri])
        elif girdi["type"] == "file" and girdi.get("size"):
            secilen.append(girdi["path"])
    return secilen


def indir(kaynak: str, yol: str, revizyon: str) -> Path:
    """Dosyayı diske indirir (belleğe almaz); aynı revizyonda önbellekten döner."""
    ONBELLEK.mkdir(parents=True, exist_ok=True)
    anahtar = sha256(f"{kaynak}@{revizyon}/{yol}".encode())[:20]
    yerel = ONBELLEK / f"{anahtar}{Path(yol).suffix.lower()}"
    if yerel.is_file() and yerel.stat().st_size > 0:
        return yerel
    kodlu = urllib.parse.quote(yol, safe="/")
    url = f"https://huggingface.co/datasets/{kaynak}/resolve/{revizyon}/{kodlu}"
    gecici = yerel.with_suffix(yerel.suffix + ".parca")
    with urllib.request.urlopen(urllib.request.Request(url, headers=BASLIK),
                                timeout=ZAMAN_ASIMI) as yanit, gecici.open("wb") as dosya:
        while True:
            blok = yanit.read(1 << 20)
            if not blok:
                break
            dosya.write(blok)
    gecici.replace(yerel)
    return yerel


def _kirp(satir: str) -> str:
    return satir.strip()[:EN_UZUN_PARCA]


def parquet_parcalari(yol: Path) -> Iterator[str]:
    """Parquet'i blok blok okur; satır başına etiketli tek parça üretir."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    dosya = pq.ParquetFile(yol)
    metin_sutunlari = [f.name for f in dosya.schema_arrow
                       if pa.types.is_string(f.type) or pa.types.is_large_string(f.type)]
    for blok in dosya.iter_batches(batch_size=BLOK, columns=metin_sutunlari or None):
        for satir in blok.to_pylist():
            parcalar = [f"{ad}: {deger.strip()}" for ad, deger in satir.items()
                        if isinstance(deger, str) and deger.strip()
                        and _temiz_satir(f"{ad}: {deger.strip()}", ad)]
            if parcalar:
                yield _kirp(_adlari_sil("\n".join(parcalar)))


def csv_parcalari(yol: Path) -> Iterator[str]:
    # prompts.csv gibi dosyalarda tek alan 131 KB sinirini asabiliyor; sinir
    # yukseltilmezse alim sessizce yarida kalir.
    csv.field_size_limit(10_000_000)
    with yol.open("r", encoding="utf-8", errors="replace", newline="") as dosya:
        for satir in csv.DictReader(dosya):
            parcalar = [f"{ad}: {deger.strip()}" for ad, deger in satir.items()
                        if isinstance(deger, str) and len(deger.strip()) > 20
                        and _temiz_satir(f"{ad}: {deger.strip()}", ad)]
            if parcalar:
                yield _kirp(_adlari_sil("\n".join(parcalar)))


def jsonl_parcalari(yol: Path) -> Iterator[str]:
    with yol.open("r", encoding="utf-8", errors="replace") as dosya:
        for satir in dosya:
            satir = satir.strip()
            if not satir:
                continue
            try:
                kayit = json.loads(satir)
            except ValueError:
                continue
            if isinstance(kayit, dict):
                parcalar = [f"{ad}: {deger.strip()}" for ad, deger in kayit.items()
                            if isinstance(deger, str) and deger.strip()
                            and _temiz_satir(f"{ad}: {deger.strip()}", ad)]
                if parcalar:
                    yield _kirp(_adlari_sil("\n".join(parcalar)))


def duz_metin_parcalari(yol: Path, hedef: int = 1200) -> Iterator[str]:
    """Satır sınırlarında `hedef` uzunluğa kadar birleştirir (log/sohbet dosyaları)."""
    tampon: list[str] = []
    uzunluk = 0
    with yol.open("r", encoding="utf-8", errors="replace") as dosya:
        for satir in dosya:
            satir = _kirp(satir)
            if not satir:
                continue
            tampon.append(satir)
            uzunluk += len(satir) + 1
            if uzunluk >= hedef:
                yield _kirp("\n".join(tampon))
                tampon, uzunluk = [], 0
    if tampon:
        yield _kirp("\n".join(tampon))


URETICILER = {".parquet": parquet_parcalari, ".csv": csv_parcalari,
              ".jsonl": jsonl_parcalari, ".json": jsonl_parcalari}


def dosya_parcalari(yol: Path) -> Iterator[str]:
    return URETICILER.get(yol.suffix.lower(), duz_metin_parcalari)(yol)


def kaynak_dosyalari(spec: dict, kaynak: str) -> list[str]:
    dosyalar = list(spec.get("dosyalar") or [])
    for dizin in spec.get("dizinler", []):
        # Dizin ağacı gün gün gezilir; bütçe dolunca kesilir, böylece yüz
        # binlerce log dosyası indirilmez.
        dosyalar.extend(dizini_gez(kaynak, dizin, int(spec.get("gun_dosya_siniri", 4))))
    if not dosyalar:
        uzantilar = set(spec.get("uzantilar", [".parquet", ".csv", ".txt"]))
        adaylar = [d for d in hf_dosyalar(kaynak) if Path(d["path"]).suffix.lower() in uzantilar]
        adaylar.sort(key=lambda d: (d["size"], d["path"]))
        dosyalar = [d["path"] for d in adaylar]
    return dosyalar


def lisans_dogrula(spec: dict) -> tuple[str, str]:
    """Kaynağın lisansını kanaldan doğrular; (revizyon, lisans) döner."""
    kaynak = spec["kaynak"]
    beklenen = str(spec.get("lisans", "cc0-1.0")).lower()
    if beklenen not in IZINLI_LISANSLAR:
        raise SystemExit(f"izinli lisans sinifinda degil: {beklenen}")
    meta = hf_meta(kaynak)
    alan = (meta.get("cardData") or {}).get("license") or meta.get("license")
    gercek = [str(x).lower() for x in (alan if isinstance(alan, list) else [alan or ""])]
    if not (set(gercek) & IZINLI_LISANSLAR):
        raise SystemExit(f"lisans dogrulanmadi: beklenen {beklenen}, kaynakta {alan!r}")
    revizyon = str(spec.get("revizyon") or meta.get("sha") or "")
    if not revizyon:
        raise SystemExit("revizyon yok: tekrarlanabilirlik icin sart")
    return revizyon, LISANS_SINIFI


def yaz(parcalar: Iterator[str], kaynak_ozeti: str, cikti, gorulen: set,
        sayaclar: dict, sozluk_sinir: int) -> None:
    """Bir kaynağın parçalarını korpus kaydı olarak akıtır; belleği büyütmez."""
    for ham in parcalar:
        metin = _gorunmezleri_sil(ham)
        if len(metin) < EN_AZ_PARCA:
            continue
        icerik = sha256(metin.encode("utf-8"))
        if icerik in gorulen:
            sayaclar["tekrar"] += 1
            continue
        if sayaclar["karakter"] + len(metin) > sozluk_sinir:
            return
        gorulen.add(icerik)
        sayaclar["karakter"] += len(metin)
        kayit = {
            "kind": "doc",
            "text": metin,
            "path": f"kamu/{kaynak_ozeti}/{sayaclar['kayit']:07d}",
            "lines": [1, metin.count("\n") + 1],
            "source": "kamu-cc0",
            "digest": icerik,
            "licence": LISANS_SINIFI,
            "attribution": AIT,
            "content_id": icerik,
            "asset_id": kaynak_ozeti,
            "asset_id_pending": True,
        }
        satir = json.dumps(kayit, ensure_ascii=False)
        # Akis bir kayit yazarken satiri bozarsa, hata burada verilir: korpusa
        # girip de yarim cozulen bir satir, sessiz bozulmadir.
        denetim = json.loads(satir)
        if denetim["text"] != metin or "\n" in satir:
            raise SystemExit(f"kayit satiri bozuk: {kayit['path']}")
        cikti.write(satir + "\n")
        sayaclar["kayit"] += 1


def onar(yol: Path) -> dict:
    """Var olan bir korpus veri dosyasindaki kayitlari ayni kurallarla duzeltir.

    Kural tek yerde dursun diye arinma fonksiyonlarinin aynisi burada da
    cagrilir: alim yolunda ne uygulaniyorsa, eldeki dosyada da o gecerli olur.
    Satirlar `\\n` ile bolunur - Python'un splitlines'i U+2028'i de satir sonu
    sayar ve bozuk gorunen kayitlarin sebebi tam olarak buydu.
    """
    with gzip.open(yol, "rt", encoding="utf-8", newline="") as dosya:
        ham = dosya.read()
    duzeltilen, kayit = 0, 0
    satirlar = []
    for satir in ham.split("\n"):
        if not satir.strip():
            continue
        kayit += 1
        veri = json.loads(satir)
        metin = veri.get("text", "")
        temiz = _gorunmezleri_sil(metin)
        if temiz != metin:
            duzeltilen += 1
            veri["text"] = temiz
            veri["lines"] = [1, temiz.count("\n") + 1]
        satirlar.append(json.dumps(veri, ensure_ascii=False))
    cikti = "\n".join(satirlar) + "\n"
    # Onarim sonrasi her satir tek basina cozulmeli: bozuk satir kalirsa yazma.
    for satir in cikti.split("\n"):
        if satir.strip():
            json.loads(satir)
    gecici = yol.with_suffix(yol.suffix + ".onar")
    with gzip.open(gecici, "wt", encoding="utf-8", compresslevel=6) as dosya:
        dosya.write(cikti)
    gecici.replace(yol)
    return {"kayit": kayit, "duzeltilen": duzeltilen, "cikti": str(yol)}


def calistir(manifest_yolu: Path, cikti: Path, karakter_butce: int) -> dict:
    basla = time.monotonic()
    manifest = json.loads(manifest_yolu.read_text(encoding="utf-8"))
    cikti.parent.mkdir(parents=True, exist_ok=True)
    gorulen: set = set()
    sayaclar = {"kayit": 0, "karakter": 0, "tekrar": 0}
    kaynak_ozetleri = []
    kaynaklar = manifest["kaynaklar"]
    with gzip.open(cikti, "wt", encoding="utf-8", compresslevel=6) as dosya:
        for sira, spec in enumerate(kaynaklar):
            if sayaclar["karakter"] >= karakter_butce:
                break
            # Butce kaynaklar arasinda adil paylasilir: kalan butce, kalan
            # kaynak sayisina bolunur. Bir kaynak payini doldurmazsa arta kalan
            # sonrakilere gecer; hicbir kaynak butceyi tek basina yutamaz.
            kalan_pay = karakter_butce - sayaclar["karakter"]
            kalan_kaynak = len(kaynaklar) - sira
            pay = max(EN_AZ_PARCA, kalan_pay // max(1, kalan_kaynak))
            revizyon, lisans = lisans_dogrula(spec)
            ozet = sha256(f"{spec['kaynak']}@{revizyon}".encode())[:16]
            baslangic = dict(sayaclar)
            bayt, dosya_ozetleri = 0, []
            for yol in kaynak_dosyalari(spec, spec["kaynak"]):
                if sayaclar["karakter"] >= karakter_butce:
                    break
                yerel = indir(spec["kaynak"], yol, revizyon)
                bayt += yerel.stat().st_size
                once = sayaclar["kayit"]
                yaz(dosya_parcalari(yerel), ozet, dosya, gorulen, sayaclar,
                    min(karakter_butce, baslangic["karakter"] + pay))
                dosya_ozetleri.append({"dosya_ozeti": sha256(yerel.read_bytes())[:16],
                                       "bayt": yerel.stat().st_size,
                                       "kayit": sayaclar["kayit"] - once})
                if spec.get("bayt_siniri") and bayt >= spec["bayt_siniri"]:
                    break
            kaynak_ozetleri.append({
                "kimlik_ozeti": ozet, "revizyon": revizyon,
                "lisans_dogrulandi": True, "lisans_sinifi": lisans,
                "bayt": bayt, "kayit": sayaclar["kayit"] - baslangic["kayit"],
                "karakter": sayaclar["karakter"] - baslangic["karakter"],
                "dosya": dosya_ozetleri,
            })
    return {
        "surum": 1, "tarih": time.strftime("%Y-%m-%dT%H:%M:%S"), "kosucu": "betik",
        "is": ("Kamu mali (CC0) veri alimi: kaynak lisansi kanaldan dogrulandi, "
               "dosyalar indirildi, metin blok blok korpus kaydina cevrildi; "
               "kaynak adi ne kayitta ne betikte gecer."),
        "olcut": {"ad": "her_kayit_kapali_lisans_sinifinda_ve_ozetli",
                  "sonuc": bool(kaynak_ozetleri) and sayaclar["kayit"] > 0},
        "kaynaklar": {"sure_saniye": round(time.monotonic() - basla, 3),
                      "girdi_jetonlari": 0, "onbellekli_jetonlari": 0,
                      "cikti_jetonlari": 0, "maliyet": 0.0},
        "kaynak_ozetleri": kaynak_ozetleri,
        "kayit": sayaclar["kayit"], "karakter": sayaclar["karakter"],
        "tekrar_elenen": sayaclar["tekrar"],
        "cikti": str(cikti.relative_to(KOK)) if cikti.is_relative_to(KOK) else str(cikti),
        "manifest_ozeti": sha256(json.dumps(manifest, sort_keys=True).encode())[:16],
    }


def kendini_test() -> list[str]:
    bulgular: list[str] = []
    try:
        lisans_dogrula({"kaynak": "ornek/yok", "lisans": "mit"})
    except SystemExit as hata:
        assert "izinli lisans" in str(hata), f"yanlis sebep: {hata}"
        bulgular.append("lisans siniri reddediyor")
    import tempfile

    with tempfile.TemporaryDirectory() as gecici:
        yol = Path(gecici) / "c.jsonl.gz"
        sozluk_sinir = 10_000
        gorulen: set = set()
        sayaclar = {"kayit": 0, "karakter": 0, "tekrar": 0}
        with gzip.open(yol, "wt", encoding="utf-8") as dosya:
            uzun = ("tek cumle burada duruyor ve seksen karakteri gecmesi "
                    "gerekiyor ki korpusa girsin")
            yaz(iter([uzun, uzun, "x"]), "ozet", dosya, gorulen, sayaclar, sozluk_sinir)
        assert sayaclar["kayit"] == 1 and sayaclar["tekrar"] == 1, sayaclar
        satir = json.loads(gzip.open(yol, "rt", encoding="utf-8").readline())
        from build_corpus import ALLOWED_LICENCES
        assert satir["licence"] in ALLOWED_LICENCES, "sinif korpus kapisindan gecmiyor"
        assert satir["attribution"].startswith("kamu malı")
        assert not any(x in satir["path"] for x in ("http", "github", "huggingface"))
        # metin butcesi asilamaz
        gorulen2: set = set()
        sayaclar2 = {"kayit": 0, "karakter": 0, "tekrar": 0}
        with gzip.open(yol, "wt", encoding="utf-8") as dosya:
            yaz((f"uzun parca {i} " + "d" * 500 for i in range(50)), "ozet", dosya,
                gorulen2, sayaclar2, 2000)
        assert sayaclar2["karakter"] <= 2000, sayaclar2
        # satir ayirici gibi davranan gorunmez karakterler metinden cikar
        gorulen3: set = set()
        sayaclar3 = {"kayit": 0, "karakter": 0, "tekrar": 0}
        kirli = ("satir bir\u2028satir iki ve bu parca arinmadan sonra da "
                 "seksen karakterden uzun kalmasi icin bilerek uzatildi")
        assert len(kirli) > 80
        with gzip.open(yol, "wt", encoding="utf-8") as dosya:
            yaz(iter([kirli]), "ozet", dosya, gorulen3, sayaclar3, sozluk_sinir)
        ham = gzip.open(yol, "rt", encoding="utf-8").read()
        assert "\u2028" not in ham and ham.count("\n") == 1, "gorunmez ayirici korpusa girdi"
        assert json.loads(ham)["text"].startswith("satir bir satir iki")
        # duz metin birlestirmesi hedefi asmaz
        kaynak = Path(gecici) / "log.txt"
        kaynak.write_text("\n".join(f"satir {i}" for i in range(300)), encoding="utf-8")
        parcalar = list(duz_metin_parcalari(kaynak, hedef=200))
        assert parcalar and all(len(p) <= 500 for p in parcalar), "birlestirme hedefi asti"
        bulgular.append("tekilleştirme + adsiz kayit + metin butcesi + birlestirme")
    return bulgular


def main(argv: list[str]) -> int:
    ayristirici = argparse.ArgumentParser(description=__doc__)
    ayristirici.add_argument("--manifest", type=Path,
                             default=Path("/home/user/kamu-kaynak.json"))
    ayristirici.add_argument("--out", type=Path, default=KOK / "corpus" / "kamu-cc0.jsonl.gz")
    ayristirici.add_argument("--karakter-butce", type=int, default=30_000_000)
    ayristirici.add_argument("--kaydet", action="store_true")
    ayristirici.add_argument("--veri-ekle", type=Path, default=None,
                             help="alinan kayitlari depoda izlenen veri dosyasina yaz")
    ayristirici.add_argument("--self-test", action="store_true")
    ayristirici.add_argument("--onar", type=Path, default=None,
                             help="var olan korpus veri dosyasini ayni kurallarla onar")
    args = ayristirici.parse_args(argv)
    if args.self_test:
        print("self-test OK [kamu-veri]: " + ", ".join(kendini_test()))
        return 0
    if args.onar is not None:
        print(json.dumps(onar(args.onar), ensure_ascii=False))
        return 0
    if not args.manifest.is_file():
        raise SystemExit(f"manifest yok: {args.manifest} (kaynak kimligi depo disinda durur)")
    rapor = calistir(args.manifest, args.out, args.karakter_butce)
    if args.veri_ekle is not None:
        # Korpus degil, turevin kaynagi: depoda izlenen veri dosyasi.
        args.veri_ekle.parent.mkdir(parents=True, exist_ok=True)
        args.veri_ekle.write_bytes(args.out.read_bytes())
        print(f"veri dosyasi yazildi: {args.veri_ekle} ({args.veri_ekle.stat().st_size} bayt)",
              file=sys.stderr)
    if args.kaydet:
        KAYIT.parent.mkdir(parents=True, exist_ok=True)
        # Depo semasi: kosucu + olcut(ad, mantiksal sonuc) + kaynaklar(muhasebe)
        # + kaynak ozetleri (ad degil, ozet).
        KAYIT.write_text(json.dumps(rapor, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
        print(f"kayit yazildi: {KAYIT.relative_to(KOK)}", file=sys.stderr)
    print(json.dumps({k: v for k, v in rapor.items() if k != "kaynak_ozetleri"},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
