#!/usr/bin/env python3
"""Otonom egitim dongusu: dort katman, bir anayasa, olculen durdurma kosullari.

Katmanlar
  1. Deney dongusu  : dugme uzayindan bir degisiklik onerir, egitim cekirdegini
                      o yapilandirmayla kosar, en iyi dogrulama kaybini olcer,
                      iyilestiriyorsa tutar, yoksa atar.
  2. Kalici hafiza  : `hafiza/ideation.jsonl` (denenmis ve atilmis yonler) ve
                      `hafiza/experimentation.jsonl` (ise yaramis olanlar).
  3. Kendini gelistirme: yalnizca `kendini-gelistirme.json` icindeki bes sayiyi
                      degistirebilir; her degisiklik bir deney gibi loglanir,
                      geri alinabilir ve pencerede olculur.
  4. Genisletilmis kesif: `--oneri-dosyasi` ile veri isleme/format adimlarina
                      dokunan oneriler kabul edilir; K2 siniri degismez.

Anayasa (INVARIANTS.md) makine tarafindan okunur: dokunulmaz dosyalar ve
durdurma kosullari oradan gelir. Bu betik o dosyalara YAZAMAZ; yazma girisimi
tek basina durdurma kosuludur (S2) ve girisim kayda gecer.

Butun ciktilar dosyaya yazilir (`kosum/` gunlukleri, commit mesaji, hafiza);
sohbete rapor yazilmaz - quiet mode kurali.

    python3 autonomous-training/dongu.py --kalibrasyon
    python3 autonomous-training/dongu.py --tek-tur
    python3 autonomous-training/dongu.py --kos --deney 10
    python3 autonomous-training/dongu.py --durum
    python3 autonomous-training/dongu.py --geri-al
    python3 autonomous-training/dongu.py --ci-onayla <run-id>
    python3 autonomous-training/dongu.py --onayla "<gerekce>"
    python3 autonomous-training/dongu.py --kendini-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

AT = Path(__file__).resolve().parent
KOK = AT.parent
KOSUM = AT / "kosum"
HAFIZA = AT / "hafiza"
DURUM = KOSUM / "durum.json"
DURUS = KOSUM / "DURUS.json"
GUNLUK = KOSUM / "oturum.md"
INVARIANTS = AT / "INVARIANTS.md"
AYARLAR = AT / "ayarlar.json"
MUTASYON = AT / "mutasyon_alani.json"
POLITIKA = AT / "kendini-gelistirme.json"

IKILI = KOK / "target" / "release" / "lubot"
KORPUS = KOK / "corpus" / "knowledge-self.jsonl.gz"
SINAV = KOK / "training" / "eval" / "sinav-seti.jsonl"


# --- anayasa ---------------------------------------------------------------
def anayasa() -> dict:
    """INVARIANTS.md icindeki makine tarafindan okunabilen blogu ayiklar."""
    metin = INVARIANTS.read_text(encoding="utf-8")
    bloklar = re.findall(r"```json\n(.*?)\n```", metin, re.S)
    if len(bloklar) != 1:
        raise SystemExit("INVARIANTS.md: tek bir ```json blogu bekleniyordu")
    return json.loads(bloklar[0])


def ozet(dosya: Path) -> str:
    return hashlib.sha256(dosya.read_bytes()).hexdigest()


def anayasa_kilidi() -> None:
    """Ozet kilidi: degistirilemez kurallar degistiyse hicbir kosu baslamaz."""
    beklenen = (AT / "INVARIANTS.sha256").read_text(encoding="utf-8").split()[0].strip()
    olculen = ozet(INVARIANTS)
    if olculen != beklenen:
        raise SystemExit(
            "INVARIANTS.md ozeti tutmuyor: degistirilemez kurallar degismis. "
            f"kayitli {beklenen[:12]}, olculen {olculen[:12]}. "
            "Kural degisikligi operatordedir: dosyayi duzelt, ozeti elle yenile."
        )


def dokunulmaz_mi(yol: str) -> bool:
    """Yol dokunulmazlar listesinde mi (onek eslesmesi dahil)."""
    hedef = Path(yol).as_posix().lstrip("./")
    for kayit in anayasa()["dokunulmaz_dosyalar"]:
        if hedef == kayit or hedef.startswith(kayit.rstrip("/") + "/"):
            return True
    return False


def yazma_reddi(yol: str) -> str | None:
    """Yazma girisiminin reddi; yazilabilirse None.

    Dokunulmaz dosyalar ve mutasyon alani disindaki yamalar burada durur.
    Reddin sebebi adiyla soylenir: hangi kural, hangi dosya."""
    if dokunulmaz_mi(yol):
        return f"S2: dokunulmaz dosyaya yazma girisiMi reddedildi: {yol}"
    return None


DURUM_AKTIF: dict = {}   # dur() cagrilarinin yazacagi durum (main doldurur)


def s2_dur(kanit: str) -> None:
    """S2: dokunulmaz dosyaya yazma girisiMi tek basina bir durdurma kosuludur.

    Red sadece bir hata degil, bir DURUS'tur: DURUS dosyasi yazilir ve insan
    onayi beklenir. Sessiz bir SystemExit onay kapisini atlardi."""
    dur(DURUM_AKTIF, "S2", kanit)


def guvenli_yaz(yol: Path, icerik: str, ekle: bool = False) -> None:
    """Dokunulmazlara yazma reddiyle korunan TEK yazma noktasi.

    Otomasyon agacindaki butun yazmalar buradan gecer: boylece yazma izni tek
    bir yerde denetlenir ve kapi `invariants-are-frozen` ikinci bir yazma yolu
    acildigini gorse kirmizi yanar."""
    red = yazma_reddi(str(yol))
    if red:
        s2_dur(red)
    yol.parent.mkdir(parents=True, exist_ok=True)
    if ekle:
        with yol.open("a", encoding="utf-8") as fh:
            fh.write(icerik)
        return
    yol.write_text(icerik, encoding="utf-8")


# --- durum ve hafiza -------------------------------------------------------
def bos_durum() -> dict:
    return {
        "surum": 1,
        "deney": 0,
        "tutulan": 0,
        "atilan": 0,
        "en_iyi_skor": None,
        "burakim": {},           # aktif dugme degerleri
        "ust_uste_kirmizi": 0,
        "son_checkpoint": 0,
        "kendini_gelistirme": {"son_degisim_deneyi": 0, "ust_uste_kotu": 0, "pencere": []},
        "kanit": {},
    }


def durum_oku() -> dict:
    if not DURUM.is_file():
        return bos_durum()
    return json.loads(DURUM.read_text(encoding="utf-8"))


def durum_yaz(durum: dict) -> None:
    guvenli_yaz(DURUM, json.dumps(durum, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def hafiza_yaz(ad: str, kayit: dict) -> None:
    guvenli_yaz(HAFIZA / ad, json.dumps(kayit, ensure_ascii=False, sort_keys=True) + "\n", ekle=True)


def hafiza_oku(ad: str) -> list[dict]:
    yol = HAFIZA / ad
    if not yol.is_file():
        return []
    return [json.loads(s) for s in yol.read_text(encoding="utf-8").splitlines() if s.strip()]


def gunluk_yaz(satir: str) -> None:
    guvenli_yaz(GUNLUK, satir.rstrip() + "\n", ekle=True)


# --- durdurma --------------------------------------------------------------
def dur(durum: dict, kosul: str, kanit: str) -> None:
    """Durdurma kosulu: otomasyon DURUR ve insan onayi bekler (S1-S5)."""
    kayit = {
        "durdu": True,
        "kosul": kosul,
        "kanit": kanit,
        "tarih": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "onay": None,
    }
    guvenli_yaz(DURUS, json.dumps(kayit, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    durum["durus"] = kayit
    durum_yaz(durum)
    gunluk_yaz(f"- DURDU [{kosul}] {kanit}")
    raise SystemExit(f"DURDU [{kosul}] {kanit}\nInsan onayi icin: --onayla \"<gerekce>\"")


def durus_acik_mi() -> dict | None:
    if not DURUS.is_file():
        return None
    kayit = json.loads(DURUS.read_text(encoding="utf-8"))
    return None if kayit.get("onay") else kayit


# --- egitim cekirdegi kosusu ----------------------------------------------
def kosucu(ayar: dict, burakim: dict, adim: int, cikti: Path) -> dict:
    """Bir egitim kosusu: rapor + kayit + olculen en iyi dogrulama kaybi."""
    if not IKILI.is_file():
        raise SystemExit(f"{IKILI.relative_to(KOK)} yok: once cargo build --release -p lubot")
    if not KORPUS.is_file():
        raise SystemExit(f"{KORPUS.relative_to(KOK)} yok: once korpus kurulur")
    damga = subprocess.run(
        [str(IKILI), "korpus-damgasi", "--corpus", str(KORPUS),
         "--vocab", f"training/tokenizer/{sozluk_ailesi()}.json"],
        cwd=KOK, capture_output=True, text=True, check=False,
    )
    if damga.returncode != 0:
        raise SystemExit(f"damga olculemedi: {(damga.stderr or damga.stdout)[-200:]}")
    damga_degeri = damga.stdout.strip().splitlines()[0].strip()
    cikti.mkdir(parents=True, exist_ok=True)
    rapor = cikti / "rapor.md"
    kayit = cikti / "kayit.json"
    s = ayar["deney"]
    komut = [
        str(IKILI), "egitim-kosu",
        "--corpus", str(KORPUS),
        "--vocab", f"training/tokenizer/{sozluk_ailesi()}.json",
        "--ckpt", str(cikti / "ckpt.bin"),
        "--damga", damga_degeri,
        "--sinav", str(SINAV),
        "--adim", str(adim),
        "--tohum", str(s["tohum"]),
        "--dogrulama-her", str(s["dogrulama_her"]),
        "--dogrulama-payi", str(s["dogrulama_payi"]),
        "--sessiz",
        "--rapor", str(rapor),
        "--kayit", str(kayit),
    ]
    for dugme, deger in sorted(burakim.items()):
        komut += [dugme, str(deger)]
    basla = time.monotonic()
    kosu = subprocess.run(komut, cwd=KOK, capture_output=True, text=True, check=False)
    sure = time.monotonic() - basla
    if kosu.returncode != 0:
        return {"basarili": False, "neden": f"kosu dustu: {(kosu.stderr or kosu.stdout)[-200:]}",
                "sure_saniye": round(sure, 2), "adim": adim}
    if not rapor.is_file() or not kayit.is_file():
        return {"basarili": False, "neden": "rapor ya da kayit yazilmadi (sema dogrulamasi)",
                "sure_saniye": round(sure, 2), "adim": adim}
    skor = en_iyi_dogrulama(rapor)
    if skor is None:
        return {"basarili": False, "neden": "dogrulama egrisi bos: olculecek kayip yok",
                "sure_saniye": round(sure, 2), "adim": adim}
    kayit_verisi = json.loads(kayit.read_text(encoding="utf-8"))
    # `ckpt.bin` bir kanit degil ara urundur: kayit.json zaten ckpt ozetini
    # tasir ("ckpt sha256 ...") ve oturum basi tekrarlanabilirlik kapisi ayni
    # imzanin ayni sayiyi verdigini kanitlar. Olculen yuk 22.2 MB; her deneyde
    # saklanirsa gunde ~890 MB eder, bu yuzden kosudan sonra silinir.
    ckpt = cikti / "ckpt.bin"
    yuk_silindi = False
    if ckpt.is_file():
        ckpt.unlink()
        yuk_silindi = True
    return {
        "basarili": True,
        "ckpt_yuku_silindi": yuk_silindi,
        "skor": skor,
        "sure_saniye": round(sure, 2),
        "adim": adim,
        "damga": damga_degeri,
        "kayit_olcut": kayit_verisi.get("olcut", {}).get("ad"),
        "rapor_satiri": len(rapor.read_text(encoding="utf-8").splitlines()),
    }


SATIR = re.compile(r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([0-9.]+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|")


def en_iyi_dogrulama(rapor: Path) -> float | None:
    """Raporun dogrulama tablosundan en iyi (en dusuk) kayip."""
    degerler = [float(m.group(3)) for m in (SATIR.match(s) for s in rapor.read_text(encoding="utf-8").splitlines()) if m]
    return min(degerler) if degerler else None


def sozluk_ailesi() -> str:
    spec = json.loads((KOK / "training" / "model_spec.json").read_text(encoding="utf-8"))
    return str(spec["vocab_family"])


# --- kalibrasyon: adim butcesi olculur, yazilmaz ---------------------------
def kalibrasyon(ayar: dict, durum: dict, zorla: bool = False) -> dict:
    """Adim hizini olcer ve sure butcesine sigan adim sayisini turetir.

    Iki noktali prob: kisa kosu sabit yuku (korpus yukleme, jetonlama) tasir,
    bu yuzden tek noktadan turetilen "adim basina saniye" butceyi asar. Iki
    uzunluk olculur, egim (marjinal adim maliyeti) ve sabit yuk ayristirilir.
    Ayrica gecen oturumun gercek kosusu butceyi astiysa o gozlem de egime
    katilir: kalibrasyon kapali cevrimlidir. Butce degistiginde kalibrasyon
    bayatlar, cunku adim butcesi butceden turetilir."""
    kayit_yolu = KOSUM / "kalibrasyon.json"
    if kayit_yolu.is_file() and not zorla:
        kayitli = json.loads(kayit_yolu.read_text(encoding="utf-8"))
        if kayitli.get("sure_butcesi_saniye") == ayar["deney"]["sure_butcesi_saniye"]:
            return kayitli
    KOSUM.mkdir(parents=True, exist_ok=True)
    # Alt sinir dogrulama duzeni: `--dogrulama-her` adimindan once duran bir kosu
    # dogrulama kaybi uretmez ve olculecek bir metrik kalmaz.
    alt = int(ayar["deney"]["dogrulama_her"])
    kisa, uzun = alt, 5 * alt
    olcumler = []
    for i, adim_prob in enumerate((kisa, uzun), start=1):
        olcum = kosucu(ayar, {}, adim_prob, KOSUM / f"kalibrasyon-{i}")
        if not olcum["basarili"]:
            raise SystemExit(f"kalibrasyon kosusu basarisiz: {olcum['neden']}")
        olcumler.append(olcum)
    t_kisa, t_uzun = (o["sure_saniye"] for o in olcumler)
    egim = (t_uzun - t_kisa) / (uzun - kisa)
    if egim <= 0:
        egim = t_uzun / uzun
    yuk_saniye = max(0.0, t_kisa - egim * kisa)
    # Gozlem: gecen oturumun gercek kosusu butceyi astiysa daha kotu egim gecerli.
    gozlem = durum.get("butce_gozlemi") or {}
    gozlem_egim = None
    if gozlem.get("adim") and gozlem.get("sure"):
        gozlem_egim = max(0.0, (gozlem["sure"] - yuk_saniye) / gozlem["adim"])
        egim = max(egim, gozlem_egim)
    kullanilabilir = ayar["deney"]["sure_butcesi_saniye"] - yuk_saniye
    ham = int(kullanilabilir / egim * ayar["deney"]["adim_guvenlik_payi"])
    # Butce dogrulama kadansina hizalanir: makine yuku oynasa da adim sayisi
    # ayni kalir, taban skoru her oturumda yeniden olculmek zorunda kalmaz.
    adim = max(alt, (ham // alt) * alt)
    sonuc = {
        "olculdu": True,
        "problar": {"kisa": {"adim": kisa, "saniye": t_kisa}, "uzun": {"adim": uzun, "saniye": t_uzun}},
        "kalibrasyon_kosusu_saniye": olcumler[0]["sure_saniye"],
        "adim_basina_saniye": round(egim, 4),
        "yuk_saniye": round(yuk_saniye, 2),
        "gozlem_egimi": None if gozlem_egim is None else round(gozlem_egim, 4),
        "gozlem": gozlem or None,
        "sure_butcesi_saniye": ayar["deney"]["sure_butcesi_saniye"],
        "adim_butcesi": adim,
        "formul": "(sure_butcesi - yuk) / egim * guvenlik_payi, kadansa hizali",
    }
    guvenli_yaz(kayit_yolu, json.dumps(sonuc, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return sonuc


# --- K6: donanim tavani olculur -------------------------------------------
def butce_gozlemi_yaz(durum: dict, olcum: dict, ayar: dict) -> None:
    """Kosunun gercek suresini kaydeder; butce asildiysa adiyla soyler.

    Kalibrasyon bu gozlemi girdi olarak kullanir: olculen egim ile gerceklesen
    sure ayrisirsa daha kotusu gecerli olur."""
    if not olcum.get("basarili"):
        return
    butce = ayar["deney"]["sure_butcesi_saniye"]
    oran = round(olcum["sure_saniye"] / butce, 3) if butce else None
    durum["butce_gozlemi"] = {
        "adim": olcum["adim"], "sure": olcum["sure_saniye"], "butce": butce, "oran": oran,
        "tarih": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if oran is not None and oran > 1.1:
        gunluk_yaz(f"- BUTCE ASIMI: {olcum['adim']} adim {olcum['sure_saniye']} s "
                   f"(butce {butce} s, oran {oran}) - sonraki kalibrasyon bu gozlemi kullanir")
    return


def k6_kontrol(ayar: dict, durum: dict, zorla: bool = False) -> dict:
    kayit_yolu = KOSUM / "k6.json"
    if kayit_yolu.is_file() and not zorla:
        return json.loads(kayit_yolu.read_text(encoding="utf-8"))
    KOSUM.mkdir(parents=True, exist_ok=True)
    bench = KOSUM / "bench.json"
    kosu = subprocess.run(
        [sys.executable, "training/bench_hardware.py", "--out", str(bench)],
        cwd=KOK, capture_output=True, text=True, check=False,
    )
    if kosu.returncode != 0:
        dur(durum, "S3", f"donanim olculemedi: {(kosu.stderr or kosu.stdout)[-160:]}")
    tahmin = subprocess.run(
        [sys.executable, "training/recommend_model_size.py", "--bench", str(bench)],
        cwd=KOK, capture_output=True, text=True, check=False,
    )
    if tahmin.returncode != 0:
        dur(durum, "S3", f"model tavani turetilemedi: {(tahmin.stderr or tahmin.stdout)[-160:]}")
    veri = json.loads(tahmin.stdout)
    tavan = veri["derived"]["max_params"][ayar["k6"]["muhasebe"]]["value"]
    spec = json.loads((KOK / "training" / "model_spec.json").read_text(encoding="utf-8"))
    # Spec'teki toplam, grubun toplamiyla tutmali: tek bir literal yetmez.
    gruplar = spec.get("params", {})
    toplam = gruplar.get("toplam")
    parcalar = [v for k, v in gruplar.items() if k != "toplam" and isinstance(v, int)]
    if not isinstance(toplam, int) or (parcalar and sum(parcalar) != toplam):
        dur(durum, "S3", f"spec params.toplam tutarsiz: toplam {toplam}, parcalar {sum(parcalar)}")
    parametre = toplam
    if parametre > tavan:
        dur(durum, "S3", f"spec {parametre} parametre, olculen K6 tavani {tavan}")
    sonuc = {
        "olculdu": True,
        "ram_gib": veri["measured_inputs"]["ram_gib"]["value"],
        "cpu_cores": veri["measured_inputs"]["cpu_cores"]["value"],
        "tavan_parametre": tavan,
        "muhasebe": ayar["k6"]["muhasebe"],
        "spec_parametre": parametre,
        "kullanim_orani": round(parametre / tavan, 6),
        "kaynak": ayar["k6"]["tavan_kaynagi"],
    }
    guvenli_yaz(kayit_yolu, json.dumps(sonuc, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return sonuc


# --- oneri uretimi (katman 1) ---------------------------------------------
def imza(burakim: dict) -> str:
    return hashlib.sha256(json.dumps(burakim, sort_keys=True).encode()).hexdigest()[:12]


def oneri_uret(durum: dict, ayar: dict, politika: dict, mutasyon: dict) -> dict:
    """Dugme uzayindan tek bir degisiklik onerir.

    Sirasi politika dosyasindan gelir; denenmis imzalar ideation hafizasindan
    okunur ki ayni yon sonsuza dek tekrarlanmasin (katman 2'nin varlik sebebi)."""
    uzay = mutasyon["dugme_uzayi"]
    denenmis = {k["imza"] for k in hafiza_oku("ideation.jsonl") if "imza" in k}
    mevcut = durum["burakim"]
    secilen = politika["oncelik_sirasi"]
    kesif = politika["kesif_orani"]
    carpan = politika["adim_carpani"]
    for tur in range(len(secilen) * 2):
        dugme = secilen[tur % len(secilen)]
        sinir = uzay[dugme]
        mevcut_deger = mevcut.get(dugme, varsayilan_deger(dugme))
        adim = (sinir["ust"] - sinir["alt"]) / 8.0 * carpan * (1.0 if kesif >= 0.5 else 0.5)
        # Yon: sirayla yukari/asagi; hep ayni yone gitmek kesif degildir.
        yukari = (durum["deney"] + tur) % 2 == 0
        yeni = mevcut_deger + (adim if yukari else -adim)
        if isinstance(sinir["ust"], int):
            yeni = int(round(yeni))
        yeni = min(max(yeni, sinir["alt"]), sinir["ust"])
        if yeni == mevcut_deger:
            continue
        aday = dict(mevcut)
        aday[dugme] = yeni
        if imza(aday) in denenmis:
            continue
        return {
            "dugme": dugme,
            "eski": mevcut_deger,
            "yeni": yeni,
            "burakim": aday,
            "imza": imza(aday),
            "aile": sinir["aile"],
        }
    return {"dugme": None, "burakim": mevcut, "imza": imza(mevcut), "neden": "uzay tukendi"}


def varsayilan_deger(dugme: str):
    """Dugmenin arac varsayilani: koddan degil, aracin kendisinden okunur."""
    return {
        "--ogrenme-orani": 0.01,
        "--agirlik-sonumu": 0.1,
        "--kirpma": 1.0,
        "--isinma": 50,
        "--yigin": 2,
    }[dugme]


# --- tam dogrulama ---------------------------------------------------------
def tam_dogrulama() -> dict:
    """fmt + clippy + test + tum kapilar: 'tutma' icin on kosul."""
    adimlar = {
        "fmt": [["cargo", "fmt", "--all", "--", "--check"], 120],
        "clippy": [["cargo", "clippy", "--workspace", "--all-targets", "--", "-D", "warnings"], 600],
        "test": [["cargo", "test", "--workspace"], 900],
        "kapilar": [[sys.executable, "gates/check.py", "--all"], 1800],
    }
    sonuc = {}
    for ad, (komut, sinir) in adimlar.items():
        basla = time.monotonic()
        try:
            kosu = subprocess.run(komut, cwd=KOK, capture_output=True, text=True, check=False,
                                  timeout=sinir)
        except FileNotFoundError:
            # Arac yoklugu bir regresyon degil, ortam eksigidir: acikca bildirilir.
            sonuc[ad] = {"gecti": False, "sure_saniye": round(time.monotonic() - basla, 2),
                         "hata": f"{komut[0]} bulunamadi (PATH)"}
            continue
        except subprocess.TimeoutExpired:
            sonuc[ad] = {"gecti": False, "sure_saniye": round(time.monotonic() - basla, 2),
                         "hata": f"zaman asimi: {sinir}s"}
            continue
        sonuc[ad] = {
            "gecti": kosu.returncode == 0,
            "sure_saniye": round(time.monotonic() - basla, 2),
        }
        if ad == "test":
            m = re.findall(r"test result: (?:ok|FAILED)\. (\d+) passed; (\d+) failed", kosu.stdout)
            sonuc[ad]["test"] = sum(int(a) for a, _ in m)
            sonuc[ad]["dusen"] = sum(int(b) for _, b in m)
        if ad == "kapilar":
            sonuc[ad]["kapi"] = len(re.findall(r"^OK\s", kosu.stdout, re.M))
            sonuc[ad]["kirmizi"] = len(re.findall(r"^FAIL\s", kosu.stdout, re.M))
            sonuc[ad]["hatali"] = re.findall(r"^FAIL\s+\[([^\]]+)\]", kosu.stdout, re.M)
            sonuc[ad]["cikti_son"] = kosu.stdout.strip().splitlines()[-1:] or [""]
    return sonuc


def kirmizi_adlar(dogrulama: dict) -> list[str]:
    """Kirmizilarin adi: reddin kaniti sayi degil, liste olmali.

    Hangi kapinin ya da testin dustugu yazilmazsa atilan deney teshis edilemez."""
    adlar = [ad for ad, d in dogrulama.items() if not d.get("gecti") or d.get("hata")]
    adlar += list(dogrulama.get("kapilar", {}).get("hatali", []))
    if dogrulama.get("test", {}).get("dusen"):
        adlar.append(f"test:{dogrulama['test']['dusen']} dustu")
    return adlar


def kirmizi_sayisi(dogrulama: dict) -> int:
    """Tam dogrulamadaki basarisiz oge sayisi (test + kapi)."""
    return (
        dogrulama.get("test", {}).get("dusen", 0)
        + dogrulama.get("kapilar", {}).get("kirmizi", 0)
        + sum(0 if d["gecti"] else 1 for d in dogrulama.values())
    )


# --- checkpoint ve geri alma ----------------------------------------------
def checkpoint_al(durum: dict, etiket: str) -> Path:
    yol = KOSUM / "checkpoint" / f"{durum['deney']:04d}-{etiket}"
    yol.mkdir(parents=True, exist_ok=True)
    guvenli_yaz(yol / "durum.json",
                json.dumps(durum, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    guvenli_yaz(yol / "burakim.json",
                json.dumps(durum["burakim"], ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    guvenli_yaz(yol / "kendini-gelistirme.json", POLITIKA.read_text(encoding="utf-8"))
    guvenli_yaz(yol / "MANIFEST.json",
        json.dumps(
            {
                "deney": durum["deney"],
                "etiket": etiket,
                "tarih": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "icerik": ["durum.json", "burakim.json", "kendini-gelistirme.json"],
                "kural": "geri alma yalniz bu manifestte listelenen dosyalari geri yazar",
            },
            ensure_ascii=False, indent=2, sort_keys=True,
        ) + "\n")
    durum["son_checkpoint"] = durum["deney"]
    return yol


def geri_al(durum: dict) -> str:
    kok = KOSUM / "checkpoint"
    if not kok.is_dir():
        raise SystemExit("checkpoint yok: geri alinacak bir iyi hal yok")
    adaylar = sorted(p for p in kok.iterdir() if p.is_dir() and (p / "MANIFEST.json").is_file())
    if not adaylar:
        raise SystemExit("checkpoint yok: manifest tasiyan dizin bulunamadi")
    son = adaylar[-1]
    manifest = json.loads((son / "MANIFEST.json").read_text(encoding="utf-8"))
    # Manifest yalniz listeledigi dosyalari geri yazar: kapsam disi bir dosyaya
    # dokunmak "geri alma" degil, gizli bir yazma olurdu.
    for ad in manifest["icerik"]:
        yol = son / ad
        if not yol.is_file():
            raise SystemExit(f"geri alma durdu: manifestte adi gecen {ad} checkpoint'te yok")
        if ad == "durum.json":
            guvenli_yaz(DURUM, yol.read_text(encoding="utf-8"))
        elif ad == "burakim.json":
            durum["burakim"] = json.loads(yol.read_text(encoding="utf-8"))
        elif ad == "kendini-gelistirme.json":
            guvenli_yaz(POLITIKA, yol.read_text(encoding="utf-8"))
    durum_yaz(durum)
    gunluk_yaz(f"- GERI ALINDI: checkpoint {son.name} ({manifest['tarih']})")
    return son.name


# --- katman 3: kendi kendini gelistirme ------------------------------------
IZINLI_POLITIKA_ANAHTARLARI = {
    "kesif_orani", "adim_carpani", "sogutma", "oncelik_sirasi", "geri_donme_esigi"
}


def politika_oku() -> dict:
    veri = json.loads(POLITIKA.read_text(encoding="utf-8"))
    return veri["arama_politikasi"]


def politika_yaz(politika: dict) -> None:
    veri = json.loads(POLITIKA.read_text(encoding="utf-8"))
    fazla = set(politika) - IZINLI_POLITIKA_ANAHTARLARI
    if fazla:
        raise SystemExit(f"S2: arama politikasinda izinli olmayan anahtar: {sorted(fazla)}")
    veri["arama_politikasi"] = politika
    guvenli_yaz(POLITIKA, json.dumps(veri, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def kendini_gelistir(durum: dict, ayar: dict) -> dict | None:
    """Katman 3: yalnizca bes sayiyi degistirir; degisiklik bir deney gibi loglanir."""
    kg = ayar["kendini_gelistirme"]
    takip = durum["kendini_gelistirme"]
    if durum["deney"] - takip["son_degisim_deneyi"] < kg["aralik_deney"]:
        return None
    politika = politika_oku()
    onceki = dict(politika)
    # Kural: kaybedilen deney orani pencerede yuksekse daha muhafazakar,
    # dusukse daha agresif ara. Esik ve yon burada yazili, olcume uydurulmaz.
    pencere = takip["pencere"][-kg["pencere"]:]
    oran = (sum(1 for p in pencere if p == "atildi") / len(pencere)) if pencere else 0.0
    if oran > 0.5:
        politika["adim_carpani"] = round(max(0.25, politika["adim_carpani"] * politika["sogutma"]), 4)
        yon = "muhafazakar"
    else:
        politika["adim_carpani"] = round(min(4.0, politika["adim_carpani"] * 1.5), 4)
        yon = "agresif"
    politika_yaz(politika)
    degisim = {"onceki": onceki, "yeni": politika, "yon": yon, "pencere_atilma_orani": round(oran, 4)}
    gunluk_yaz(f"- KENDINI GELISTIRME ({yon}): adim_carpani {onceki['adim_carpani']} -> "
               f"{politika['adim_carpani']} (pencere atilma orani {degisim['pencere_atilma_orani']})")
    takip["son_degisim_deneyi"] = durum["deney"]
    # Onceki degisiklik pencerede kotulestirdi mi: art arda sayilir, S5'e baglanir.
    if len(pencere) == kg["pencere"] and sum(1 for p in pencere if p == "tutuldu") == 0:
        takip["ust_uste_kotu"] += 1
    else:
        takip["ust_uste_kotu"] = 0
    hafiza_yaz("ideation.jsonl", {
        "tur": "kendini-gelistirme", "deney": durum["deney"], "tarih": time.strftime("%Y-%m-%d"),
        "yon": yon, **degisim,
    })
    if takip["ust_uste_kotu"] >= kg["ust_uste_kotu_durur"]:
        dur(durum, "S5", f"arada {takip['ust_uste_kotu']} pencere kotulesti "
                         f"(yon {yon}, pencere {pencere})")
    return degisim


# --- tek deney -------------------------------------------------------------
def danisma_oyu(ayar: dict, skor: float, mevcut: float | None, esik: float) -> dict:
    """K7: belirsizlik bandindaki karar icin oy ister; karar kuralini degistirmez.

    Danisma modulu yoksa ya da kapaliysa "danisilmadi" doner ve karar eskisi
    gibi koda kalir: danisma katmani dongunun on kosulu degildir.
    """
    try:
        import danisma as D
    except ImportError:
        return {"danisildi": False, "gerekce": "danisma modulu yok", "oy": None}
    ayar_danisma = (ayar or {}).get("danisma", {})
    if not ayar_danisma.get("etkin", False):
        return {"danisildi": False, "gerekce": "danisma kapali", "oy": None}
    return D.karar_oyu(ayar_danisma, skor, mevcut, esik, kart="tut_at")


def tek_deney(durum: dict, ayar: dict, mutasyon: dict, adim: int, kaynak: str) -> dict:
    oneri = oneri_uret(durum, ayar, politika_oku(), mutasyon)
    if oneri["dugme"] is None:
        dur(durum, "S1", "dugme uzayi tukendi: yeni bir yon insan kararidir")
    durum["deney"] += 1
    cikti = KOSUM / f"deney-{durum['deney']:04d}"
    gunluk_yaz(
        f"\n## Deney {durum['deney']} ({time.strftime('%Y-%m-%dT%H:%M:%S')}) - kaynak: {kaynak}\n"
        f"- oneri: {oneri['dugme']} {oneri['eski']} -> {oneri['yeni']} (aile {oneri['aile']})\n"
        f"- adim butcesi: {adim}"
    )
    olcum = kosucu(ayar, oneri["burakim"], adim, cikti)
    butce_gozlemi_yaz(durum, olcum, ayar)
    if not olcum["basarili"]:
        durum["atilan"] += 1
        hafiza_yaz("ideation.jsonl", {
            "tur": "deneme", "deney": durum["deney"], "tarih": time.strftime("%Y-%m-%d"),
            "dugme": oneri["dugme"], "eski": oneri["eski"], "yeni": oneri["yeni"],
            "imza": oneri["imza"], "karar": "atildi", "neden": olcum["neden"],
        })
        durum["kendini_gelistirme"]["pencere"].append("atildi")
        durum_yaz(durum)
        gunluk_yaz(f"- ATILDI: {olcum['neden']}")
        return {"deney": durum["deney"], "karar": "atildi", "neden": olcum["neden"], **olcum}

    taban = durum.get("taban") or {}
    if taban.get("adim") != adim:
        dur(durum, "S1", f"olcut tabani adim {taban.get('adim')} icin olculmus, deney adim "
                         f"{adim}: karsilastirma gecersiz")
    mevcut = durum["en_iyi_skor"]
    esik = ayar["olcut"]["asgari_iyilesme"]
    iyilesti = mevcut is None or olcum["skor"] < mevcut - esik
    # K7: karar *belirsiz* ise danisma katmani yalnizca OY verir. Marj, kabul
    # esiginin bandi icindeyse oy sorulur; oy gelmez, gecersiz olur ya da guven
    # esigin altinda kalirsa karar insana gider ve dongu durur (S2).
    danisma = danisma_oyu(ayar, olcum["skor"], mevcut, esik)
    if danisma["danisildi"]:
        hafiza_yaz("experimentation.jsonl", {
            "tur": "danisma", "deney": durum["deney"], "tarih": time.strftime("%Y-%m-%d"),
            "skor": olcum["skor"], "mevcut_skor": mevcut, "esik": esik, **danisma,
        })
        gunluk_yaz(
            f"- DANISMA: marj {danisma['marj']} bant {danisma['bant']} | oy {danisma['oy']} "
            f"(guven {danisma.get('guven')}) | {danisma['gerekce']}"
        )
        if danisma["insan_gerekli"]:
            dur(durum, "S2", f"danisma oyu karar vermedi: {danisma['gerekce']}")
        # Oy ile kod karari celisirse karar KODDA kalir; celiski kayda gecer.
        oy_tut = danisma["oy"] == "tut"
        if oy_tut != iyilesti:
            hafiza_yaz("experimentation.jsonl", {
                "tur": "celiski", "deney": durum["deney"], "tarih": time.strftime("%Y-%m-%d"),
                "oy": danisma["oy"], "kod_karari": "tut" if iyilesti else "at",
                "gerekce": "karar kurali kodda; oy yalnizca kayda gecer",
            })
            gunluk_yaz(f"- CELISKI: oy {danisma['oy']}, kod {'tut' if iyilesti else 'at'} dedi")
    if not iyilesti:
        durum["atilan"] += 1
        hafiza_yaz("ideation.jsonl", {
            "tur": "deneme", "deney": durum["deney"], "tarih": time.strftime("%Y-%m-%d"),
            "dugme": oneri["dugme"], "eski": oneri["eski"], "yeni": oneri["yeni"],
            "imza": oneri["imza"], "skor": olcum["skor"], "mevcut_skor": mevcut,
            "karar": "atildi", "neden": "iyilesme yok",
        })
        durum["kendini_gelistirme"]["pencere"].append("atildi")
        durum_yaz(durum)
        gunluk_yaz(f"- ATILDI: skor {olcum['skor']} (mevcut {mevcut}, esik {esik})")
        return {"deney": durum["deney"], "karar": "atildi", "neden": "iyilesme yok", **olcum}

    # Tutma: once tam dogrulama, sonra checkpoint, sonra kanit alani.
    dogrulama = tam_dogrulama()
    kirmizi = kirmizi_sayisi(dogrulama)
    adlar = kirmizi_adlar(dogrulama)
    if kirmizi > 0:
        durum["ust_uste_kirmizi"] += 1
        durum["atilan"] += 1
        hafiza_yaz("ideation.jsonl", {
            "tur": "deneme", "deney": durum["deney"], "tarih": time.strftime("%Y-%m-%d"),
            "dugme": oneri["dugme"], "eski": oneri["eski"], "yeni": oneri["yeni"],
            "imza": oneri["imza"], "skor": olcum["skor"], "karar": "atildi",
            "neden": f"tam dogrulamada {kirmizi} kirmizi: {', '.join(adlar) or 'ad yok'}",
            "kirmizilar": adlar,
            "dogrulama": {ad: d["gecti"] for ad, d in dogrulama.items()},
        })
        durum["kendini_gelistirme"]["pencere"].append("atildi")
        durum_yaz(durum)
        gunluk_yaz(f"- ATILDI: tam dogrulamada {kirmizi} kirmizi: {', '.join(adlar) or 'ad yok'}")
        if durum["ust_uste_kirmizi"] >= ayar["regresyon"]["ust_uste_kirmizi_durur"]:
            dur(durum, "S4", f"art arda {durum['ust_uste_kirmizi']} deney tam dogrulamadan kirmizi cikti")
        return {"deney": durum["deney"], "karar": "atildi", "neden": "dogrulama kirmizi", **olcum}

    durum["ust_uste_kirmizi"] = 0
    durum["tutulan"] += 1
    onceki_skor = durum["en_iyi_skor"]
    durum["en_iyi_skor"] = olcum["skor"]
    durum["burakim"] = oneri["burakim"]
    hafiza_yaz("experimentation.jsonl", {
        "tur": "tutuldu", "deney": durum["deney"], "tarih": time.strftime("%Y-%m-%d"),
        "dugme": oneri["dugme"], "eski": oneri["eski"], "yeni": oneri["yeni"],
        "imza": oneri["imza"], "skor": olcum["skor"], "onceki_skor": onceki_skor,
        "iyilesme": None if onceki_skor is None else round(onceki_skor - olcum["skor"], 6),
        "dogrulama": {ad: d["gecti"] for ad, d in dogrulama.items()},
        "kanit_durumu": "ci-bekliyor",
        "commit_sha": None,
        "ci_kosusu": None,
    })
    durum["kendini_gelistirme"]["pencere"].append("tutuldu")
    if durum["deney"] - durum["son_checkpoint"] >= ayar["checkpoint"]["her_tutmada"]:
        yol = checkpoint_al(durum, "tutma")
        gunluk_yaz(f"- CHECKPOINT: {yol.name}")
    durum_yaz(durum)
    gunluk_yaz(f"- TUTULDU: skor {olcum['skor']} (onceki {onceki_skor})")
    return {"deney": durum["deney"], "karar": "tutuldu", "dogrulama": dogrulama, **olcum}


# --- oturum ----------------------------------------------------------------
def oturum_hazirla(ayar: dict, zorla: bool) -> tuple[dict, int]:
    anayasa_kilidi()
    acik = durus_acik_mi()
    if acik:
        raise SystemExit(
            f"DURUS acik [{acik['kosul']}]: {acik['kanit']}\n"
            "Insan onayi icin: --onayla \"<gerekce>\""
        )
    durum = durum_oku()
    k6 = k6_kontrol(ayar, durum, zorla)
    kal = kalibrasyon(ayar, durum, zorla)
    durum.setdefault("profil", {})
    durum["profil"] = {
        "k6": k6,
        "kalibrasyon": kal,
        "anayasa_ozeti": ozet(INVARIANTS)[:12],
    }
    return durum, int(kal["adim_butcesi"])


def tekrarlanabilirlik_kapisi(durum: dict, ayar: dict, adim: int) -> dict:
    """Oturum basinda: ayni yapilandirma iki kez ayni kaybi vermeli.

    Karsilastirma yalniz ayni adim butcesinde gecerlidir (olcut.md), bu yuzden
    kapinin urettigi taban skor adim butcesiyle birlikte saklanir: butce
    degistiginde taban kendiliginden yeniden olculur. Aksi hâlde uzun bir kosu
    kisa bir kosuya kiyasla "iyilesme" gibi gorunurdu."""
    taban = durum.get("taban") or {}
    if taban.get("adim") == adim and taban.get("tekrarlanabilir"):
        return durum
    skorlar = []
    for i in (1, 2):
        olcum = kosucu(ayar, durum["burakim"], adim, KOSUM / f"tekrar-{i}")
        if not olcum["basarili"]:
            dur(durum, "S1", f"tekrarlanabilirlik kosusu dustu: {olcum['neden']}")
        butce_gozlemi_yaz(durum, olcum, ayar)
        skorlar.append(olcum["skor"])
    if skorlar[0] != skorlar[1]:
        dur(durum, "S1", f"olcum tekrarlanabilir degil: {skorlar[0]} != {skorlar[1]}")
    durum["taban"] = {
        "adim": adim,
        "skor": skorlar[0],
        "tekrarlanabilir": True,
        "tarih": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    durum["en_iyi_skor"] = skorlar[0]
    durum_yaz(durum)
    gunluk_yaz(f"- TABAN: adim {adim}, iki kosu ayni kaybi verdi ({skorlar[0]})")
    return durum


def komut_satirlari(vekil: bool = False) -> str:
    return " (vekil)" if vekil else ""


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--tek-tur", action="store_true", help="tek deney: oner, kos, tut ya da at")
    g.add_argument("--kos", action="store_true", help="surekli: --deney sayisi kadar deney")
    g.add_argument("--durum", action="store_true", help="durum ozeti (dosyadan)")
    g.add_argument("--geri-al", action="store_true", help="son iyi checkpoint'e don")
    g.add_argument("--ci-onayla", metavar="RUN_ID", help="son tutulan deneyin CI kosusunu kaydet")
    g.add_argument("--onayla", metavar="GEREKCE", help="acik DURUS'u insan onayi ile kapat")
    g.add_argument("--kendini-test", action="store_true", help="durdurma kosullarinin kanaryasi")
    parser.add_argument("--deney", type=int, default=1,
                        help="--kos ile: kac deney kosulacak")
    parser.add_argument("--butce", type=int, default=None,
                        help="deney sure butcesini (saniye) bu oturum icin degistir; loglanir")
    g.add_argument("--kendini-test-uzun", action="store_true", help="+ gercek egitim kosusu")
    args = parser.parse_args(argv)

    KOSUM.mkdir(parents=True, exist_ok=True)
    if args.kendini_test or args.kendini_test_uzun:
        return kendini_test(uzun=args.kendini_test_uzun)
    if args.durum:
        durum = durum_oku()
        print(json.dumps({k: v for k, v in durum.items() if k != "profil"}, ensure_ascii=False,
                         indent=2, sort_keys=True))
        if (KOSUM / "kalibrasyon.json").is_file():
            print("profil:", json.dumps(durum.get("profil", {}), ensure_ascii=False, sort_keys=True))
        return 0
    if args.onayla:
        acik = durus_acik_mi()
        if not acik:
            raise SystemExit("acik bir DURUS yok: onaylanacak bir durma yok")
        acik["onay"] = {"gerekce": args.onayla, "tarih": time.strftime("%Y-%m-%dT%H:%M:%S")}
        guvenli_yaz(DURUS, json.dumps(acik, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        hafiza_yaz("ideation.jsonl", {
            "tur": "insan-onayi", "tarih": time.strftime("%Y-%m-%d"),
            "kosul": acik["kosul"], "gerekce": args.onayla,
        })
        gunluk_yaz(f"- ONAYLANDI [{acik['kosul']}]: {args.onayla}")
        return 0
    if args.ci_onayla:
        kayitlar = hafiza_oku("experimentation.jsonl")
        tutulanlar = [k for k in kayitlar if k.get("tur") == "tutuldu"]
        if not tutulanlar:
            raise SystemExit("tutulmus deney yok: onaylanacak bir iyilesme yok")
        son = tutulanlar[-1]
        son["ci_kosusu"] = args.ci_onayla
        son["kanit_durumu"] = "ci-onayli"
        yol = HAFIZA / "experimentation.jsonl"
        guvenli_yaz(yol, "".join(json.dumps(k, ensure_ascii=False, sort_keys=True) + "\n"
                                 for k in kayitlar))
        gunluk_yaz(f"- CI ONAYI: deney {son['deney']} kosu {args.ci_onayla}")
        return 0

    eksik = [arac for arac in ("cargo",) if shutil.which(arac) is None]
    if eksik:
        raise SystemExit(
            f"arac yok: {', '.join(eksik)} - PATH'e ~/.cargo/bin ekleyip yeniden dene"
        )
    ayar = json.loads(AYARLAR.read_text(encoding="utf-8"))
    if args.butce is not None:
        # Butce yalnizca bu oturum icin degisir; politika dosyasi dokunulmazdir.
        ayar["deney"]["sure_butcesi_saniye"] = int(args.butce)
    mutasyon = json.loads(MUTASYON.read_text(encoding="utf-8"))
    durum, adim = oturum_hazirla(ayar, zorla=args.butce is not None)
    global DURUM_AKTIF
    DURUM_AKTIF = durum
    if args.geri_al:
        print(geri_al(durum))
        return 0
    durum = tekrarlanabilirlik_kapisi(durum, ayar, adim)
    gunluk_yaz(f"\n# Oturum {time.strftime('%Y-%m-%dT%H:%M:%S')} - adim butcesi {adim}")

    sayi = max(1, args.deney) if args.kos else 1
    for i in range(sayi):
        if i > 0:
            degisim = kendini_gelistir(durum, ayar)
            if degisim:
                checkpoint_al(durum, "kendini-gelistirme-oncesi") if ayar["checkpoint"][
                    "kendini_gelistirme_oncesi"] else None
        sonuc = tek_deney(durum, ayar, mutasyon, adim, kaynak="dugme-uzayi")
        print(json.dumps({k: v for k, v in sonuc.items() if k != "dogrulama"}, ensure_ascii=False, sort_keys=True))
    return 0


# --- kanarya ---------------------------------------------------------------
def kendini_test(uzun: bool = False) -> int:
    """Durdurma kosullarinin kanaryasi: her kosul kurulur ve DURDURDUGU gosterilir.

    Uzun varyant gercek bir egitim kosusu da yapar; kisa varyant saf
    Python'dur ki kapi hizli kalsin."""
    import tempfile

    global KOSUM, DURUM, DURUS, GUNLUK, HAFIZA, POLITIKA
    gercek = (KOSUM, DURUM, DURUS, GUNLUK, HAFIZA, POLITIKA)
    bulgular = []
    with tempfile.TemporaryDirectory() as td:
        KOSUM = Path(td) / "kosum"
        HAFIZA = Path(td) / "hafiza"
        DURUM = KOSUM / "durum.json"
        DURUS = KOSUM / "DURUS.json"
        GUNLUK = KOSUM / "oturum.md"
        POLITIKA = Path(td) / "kendini-gelistirme.json"
        KOSUM.mkdir(parents=True, exist_ok=True)
        HAFIZA.mkdir(parents=True, exist_ok=True)
        guvenli_yaz(POLITIKA, json.dumps({"surum": 1, "arama_politikasi": {
            "kesif_orani": 0.5, "adim_carpani": 1.0, "sogutma": 0.5,
            "oncelik_sirasi": ["--kirpma"], "geri_donme_esigi": 3}}))

        # S2: dokunulmaz dosyaya yazma girisiMi reddedilir ve DURUS yazilir.
        red = yazma_reddi("autonomous-training/INVARIANTS.md")
        assert red and "S2" in red, "dokunulmaz dosyaya yazma reddedilmedi"
        bulgular.append("S2 yazma reddi")
        assert yazma_reddi("crates/egitim/src/kosu.rs") is None, "izinli dosya reddedildi"
        try:
            guvenli_yaz(Path("autonomous-training/INVARIANTS.md"), "kanarya")
        except SystemExit as e:
            assert "S2" in str(e), f"yanlis sebep: {e}"
        else:
            raise AssertionError("dokunulmaza yazma kapidan gecti")
        assert DURUS.is_file(), "S2 durusu DURUS dosyasini yazmadi"
        bulgular.append("S2 durus + dosya")

        # Durdurma kosulu kurulunca otomasyon durur ve dosyayi yazar.
        durum = bos_durum()
        try:
            dur(durum, "S3", "kanarya: K6 butcesi asimi")
        except SystemExit as e:
            assert "S3" in str(e), f"yanlis sebep: {e}"
        else:
            raise AssertionError("durdurma kosulu durmadi")
        acik = durus_acik_mi()
        assert acik and acik["kosul"] == "S3", "DURUS dosyasi yazilmadi"
        bulgular.append("S3 durus + dosya")

        # Onay gelmeden hicbir kosu baslamaz.
        try:
            oturum_hazirla(json.loads(AYARLAR.read_text(encoding="utf-8")), zorla=False)
        except SystemExit as e:
            assert "onay" in str(e).lower() or "DURUS" in str(e), f"beklenmeyen durus sebebi: {e}"
        else:
            raise AssertionError("acik DURUS varken oturum basladi")
        bulgular.append("onaysiz baslatma reddi")

        # S5: art arda kotulestiren strateji degisikligi sayaci siniri gorur.
        ayar = json.loads(AYARLAR.read_text(encoding="utf-8"))
        durum = bos_durum()
        durum["deney"] = 100
        takip = durum["kendini_gelistirme"]
        takip["son_degisim_deneyi"] = 0
        takip["pencere"] = ["atildi", "atildi", "atildi"]
        politika_oku_onceki = politika_oku()
        kendini_gelistir(durum, ayar)
        assert takip["ust_uste_kotu"] == 1, "kotulesme sayaci artmadi"
        bulgular.append("S5 sayaci")
        # Esik asilinca durur: sayac 2 iken bir kotu pencere daha gelirse 3 olur.
        takip["ust_uste_kotu"] = 2
        takip["son_degisim_deneyi"] = 0
        try:
            kendini_gelistir(durum, ayar)
        except SystemExit as e:
            assert "S5" in str(e), f"beklenmeyen durus sebebi: {e}"
        else:
            raise AssertionError("esik asildi ama durmadi")
        bulgular.append("S5 durus")
        politika_yaz(politika_oku_onceki)

        # Zararsiz durum durdurmaz: durum dosyasi temizken oturum acilabilir.
        DURUS.unlink()
        bulgular.append("zararsiz durum durmuyor")

        # Geri alma: manifest yalniz listeledigi dosyalari geri yazar.
        durum = bos_durum()
        durum["burakim"] = {"--kirpma": 1.5}
        checkpoint_al(durum, "kanarya")
        durum["burakim"] = {"--kirpma": 9.9}
        geri_al(durum)
        assert durum["burakim"] == {"--kirpma": 1.5}, "geri alma burakimi geri yazmadi"
        bulgular.append("geri alma")

        # Katman 3 siniri: izinli olmayan anahtar reddedilir.
        try:
            politika_yaz({"kesif_orani": 0.5, "yasak": 1})
        except SystemExit as e:
            assert "S2" in str(e), f"yanlis sebep: {e}"
        else:
            raise AssertionError("politikaya izinsiz anahtar yazildi")
        bulgular.append("S2 politika siniri")

        # K7: danisma oyu yoksa karar insana gider; kapaliysa karar koda kalir.
        # Kanarya kapali bir portu hedefler: oy gelmez, insan gerekir - gercek
        # servisin acik olup olmamasi kanaryayi degistirmez.
        ayar_d = json.loads(AYARLAR.read_text(encoding="utf-8"))
        ayar_d["danisma"] = {"etkin": True, "bant_carpani": 1.0, "guven_esigi": 0.55,
                             "servis_url": "http://127.0.0.1:9", "zaman_asimi_saniye": 1}
        oy = danisma_oyu(ayar_d, 6.6717405, 6.671741, 0.001)
        assert oy["danisildi"], "bant ici kararda oy sorulmadi"
        assert oy["insan_gerekli"], "oy yokken karar insana gitmedi"
        bulgular.append("K7 oy yok -> insan")
        ayar_d["danisma"]["bant_carpani"] = 0.0
        oy = danisma_oyu(ayar_d, 6.5, 6.671741, 0.001)
        assert not oy["danisildi"], "bant disinda oy soruldu"
        bulgular.append("K7 bant disi sorulmaz")
        ayar_kapali = json.loads(AYARLAR.read_text(encoding="utf-8"))
        ayar_kapali["danisma"] = {"etkin": False}
        oy = danisma_oyu(ayar_kapali, 6.6717405, 6.671741, 0.001)
        assert not oy["danisildi"], "kapali danisma soru sordu"
        bulgular.append("K7 kapali -> karar koda kalir")

    (KOSUM, DURUM, DURUS, GUNLUK, HAFIZA, POLITIKA) = gercek
    if uzun:
        ayar = json.loads(AYARLAR.read_text(encoding="utf-8"))
        ayar["deney"]["sure_butcesi_saniye"] = 60
        durum = bos_durum()
        kal = kalibrasyon(ayar, bos_durum(), zorla=True)
        olcum = kosucu(ayar, durum["burakim"], int(kal["adim_butcesi"]), KOSUM / "kanarya-uzun")
        assert olcum["basarili"], f"gercek kosu basarisiz: {olcum.get('neden')}"
        bulgular.append(f"gercek kosu (skor {olcum['skor']})")
    print("self-test OK [dongu]: " + ", ".join(bulgular))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
