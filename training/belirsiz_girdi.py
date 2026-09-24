#!/usr/bin/env python3
"""Belirsiz girdi bataryasi: ayrıştırıcılar düşmanca girdiyle sınanır.

Madde 22 (Awesome Fuzzing). Yeni bağımlılık YOK: tohumlu ve tekrarlanabilir
bir mutasyon kümesi Python standart kütüphanesiyle üretilir, hedeflere
`lubot` ikilisi üzerinden verilir ve **panik/refus** sayılır.

Ölçüt (kayda geçen üç sayı):
* `panik`  — süreç panikledi (`panicked at` ya da sinyalle öldü). Beklenen: 0.
* `refus`  — süreç temiz bir *ret* ile çıktı (sıfırdan farklı kod, paniksiz).
* `kabul`  — girdi geçerli sayıldı (boş dosya gibi gerçekten geçerli durumlar).

Neden panik sayılıyor ve yanlış cevap sayılmıyor: bu batarya ayrıştırıcıların
*çökmemesi* gerektiğini ölçer. "Doğru ret" tarafı zaten `no-panic-path` ve
`provenance-fails-closed` kapılarının işidir; burada ölçülen, düşmanca girdinin
süreci düşürüp düşürmediğidir. Bir panik, fail-closed değil fail-silent'tir.

Tohum sabittir (`TOHUM`), mutasyonlar o tohumdan türetilir; aynı tohum aynı
vaka kümesini verir. Kullanım:

    python3 training/belirsiz_girdi.py --olc
    python3 training/belirsiz_girdi.py --kur      # kaydi yazar
    python3 training/belirsiz_girdi.py --self-test
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent
IKILI = KOK / "target" / "release" / "lubot"
KAYIT = KOK / "training" / "eval" / "sonuclar" / "belirsiz-girdi-2026-09-24.json"
TOHUM = 20260924
VAKA_SAYISI = 60
ZAMAN_ASIMI = 20

# --- tohumlu mutasyonlar -----------------------------------------------------
GECERLI_KAYIT = {
    "kind": "doc",
    "text": "Lubot okur, uretmez. Bu kayit bir korpus kaydidir.",
    "licence": "PolyForm-Shield-1.0.0",
    "attribution": "lubot (kendi eser)",
    "content_id": "0" * 64,
    "asset_id": "1" * 64,
}


def temel_girdiler() -> list[tuple[str, bytes]]:
    """Bataryanin temeli: gecerli bir kayit ve uc bozuk varyant."""
    gecerli = json.dumps(GECERLI_KAYIT, ensure_ascii=False).encode("utf-8")
    return [
        ("bos", b""),
        ("gecerli", gecerli + b"\n"),
        ("json-degil", b"bu bir json degil\n"),
        ("eksik-alan", json.dumps({"kind": "doc"}).encode("utf-8") + b"\n"),
        ("sozluk-kok", b"{}\n"),
        ("dizi-kok", b"[]\n"),
    ]


def mutasyonlar(tohum: int) -> list[tuple[str, bytes]]:
    """TOHUM'dan turetilen, tekrarlanabilir düşmanca vakalar."""
    rastgele = random.Random(tohum)
    temel = temel_girdiler()
    vakalar: list[tuple[str, bytes]] = list(temel)

    # 1) bit cirpmasi ve kesme
    for i in range(1, 13):
        ad, veri = temel[i % len(temel)]
        if not veri:
            veri = b'{"kind":"doc"}'
        bozuk = bytearray(veri)
        kac = rastgele.randrange(len(bozuk))
        bozuk[kac] ^= 1 << rastgele.randrange(8)
        vakalar.append((f"bit-cirpmasi-{i}", bytes(bozuk)))
        vakalar.append((f"kesme-{i}", veri[: rastgele.randrange(len(veri) + 1)]))

    # 2) gecersiz UTF-8 ve NUL: bayt duzeyi okuyan ayrıştırıcılar icin
    vakalar.append(("gecersiz-utf8", b'{"kind":"doc","text":"\xff\xfe"}\n'))
    vakalar.append(("nul-bayt", b'{"kind":"doc","text":"a\x00b"}\n'))
    vakalar.append(("bom", b"\xef\xbb\xbf" + temel[1][1]))

    # 3) derin ic ice yapi: ozyineleme tavanini yoklar
    vakalar.append(("derin-dizi", b"[" * 400 + b"]" * 400))
    vakalar.append(("derin-sozluk", b'{"a":' * 200 + b"1" + b"}" * 200))
    vakalar.append(("kendine-referans", b'{"kind":"doc","text":"' + b"\\" * 200 + b'"}'))

    # 4) sinir uzunluklari: bosluk ve cok buyuk sayi
    vakalar.append(("uzun-bosluk", b'{"kind":"doc","text":"' + b" " * 5000 + b'"}\n'))
    vakalar.append(("buyuk-sayi", b'{"kind":"doc","text":"1e999999"}\n'))
    vakalar.append(("negatif-uzunluk", b'{"kind":"doc","text":"-1"}\n'))

    # 5) gzip tarafi: bozuk baslik, kesik akis, yanlis sihirli bayt
    gecerli_gz = gzip.compress(temel[1][1])
    vakalar.append(("gz-bozuk-baslik", b"\x1f\x8b" + gecerli_gz[2:]))
    vakalar.append(("gz-kesik", gecerli_gz[: len(gecerli_gz) // 2]))
    vakalar.append(("gz-degil", temel[1][1]))
    vakalar.append(("gz-bombasi-basligi", gecerli_gz[:10] + b"\x00" * 50))

    # 6) alan tipi karisikligi: sema dogrulayiciyi yoklar
    for alan, deger in (("kind", 1), ("licence", []), ("text", {"x": 1}), ("content_id", None)):
        kayit = dict(GECERLI_KAYIT, **{alan: deger})
        vakalar.append((f"tip-{alan}", json.dumps(kayit, ensure_ascii=False).encode("utf-8") + b"\n"))

    return vakalar[:VAKA_SAYISI]


def hedefler(dosya: Path) -> list[list[str]]:
    """Vakayi tuketen uc ayristirici yolu (hepsi salt-okunur)."""
    return [
        ["corpus", str(dosya)],          # korpus/knowledge yukleyici
        ["guvenlik", "--path", str(dosya)],   # kimlik tarayicisi: bayt duzeyi
        ["dosya", "--path", str(dosya)],      # sihirli-bayt yonlendiricisi
    ]


def kos(binary: Path, argumanlar: list[str]) -> dict:
    basla = time.monotonic()
    try:
        kosu = subprocess.run([str(binary), *argumanlar], capture_output=True,
                              text=True, errors="replace", timeout=ZAMAN_ASIMI, check=False)
    except subprocess.TimeoutExpired:
        return {"cikis": None, "panik": False, "asildi": True, "cikti": ""}
    cikti = (kosu.stdout or "") + (kosu.stderr or "")
    return {"cikis": kosu.returncode, "panik": "panicked at" in cikti,
            "asildi": False, "sinyal": kosu.returncode < 0,
            "cikti": cikti.strip().splitlines()[-1][:160] if cikti.strip() else ""}


def olc(binary: Path = IKILI, tohum: int = TOHUM) -> dict:
    basla_zamani = time.monotonic()
    if not binary.is_file():
        raise SystemExit(f"ikili yok: {binary} (once: cargo build --release -p lubot)")
    vakalar = mutasyonlar(tohum)
    sonuclar, panik, asildi, ret, kabul = [], 0, 0, 0, 0
    with tempfile.TemporaryDirectory() as gecici:
        for ad, veri in vakalar:
            for i, hedef in enumerate(hedefler(Path(gecici) / f"vaka-{ad}.jsonl")):
                yol = Path(gecici) / f"vaka-{ad}-{i}.jsonl"
                yol.write_bytes(veri)
                sonuc = kos(binary, [hedef[0], *[a if a != str(Path(gecici) / f"vaka-{ad}.jsonl") else str(yol)
                                               for a in hedef[1:]]])
                panik += int(sonuc["panik"] or sonuc.get("sinyal", False))
                asildi += int(sonuc["asildi"])
                if sonuc["cikis"] == 0 and not sonuc["panik"]:
                    kabul += 1
                elif sonuc["cikis"] not in (None, 0):
                    ret += 1
                sonuclar.append({"vaka": ad, "hedef": hedef[0], **{k: v for k, v in sonuc.items()
                                                                   if k != "cikti"}, "ozet": sonuc["cikti"]})
    # Kayit bicimi deponun standart olcum kaydidir (`eval-runs-are-mechanical`
    # kapisi): tek makine-kontrol edilebilir boolean olcut + kaynak muhasebesi.
    # Olcut adi bir yargi kelimesi tasimaz; ne olctugu yazili.
    return {
        "surum": 1,
        "tarih": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "kosucu": "betik",
        "is": ("Belirsiz girdi bataryasi: tohumlu dusmanca vakalar uc ayristiriciya "
               "(korpus yukleyici, kimlik tarayicisi, sihirli-bayt yonlendiricisi) verildi; "
               "panik ve asilma sayildi."),
        "olcut": {"ad": "panik_ve_asilma_sayisinin_sifir_olmasi", "sonuc": panik == 0 and asildi == 0},
        "kaynaklar": {"sure_saniye": round(time.monotonic() - basla_zamani, 3), "girdi_jetonlari": 0,
                      "onbellekli_jetonlari": 0, "cikti_jetonlari": 0, "maliyet": 0.0},
        "tohum": tohum,
        "vaka_sayisi": len(vakalar),
        "kosu_sayisi": len(sonuclar),
        "panik": panik,
        "asildi": asildi,
        "ret": ret,
        "kabul": kabul,
        "kanit": {"vakalar": sonuclar},
    }


def ozet(rapor: dict) -> str:
    return (f"belirsiz girdi: {rapor['vaka_sayisi']} vaka / {rapor['kosu_sayisi']} kosu, "
            f"panik {rapor['panik']}, asildi {rapor['asildi']}, ret {rapor['ret']}, "
            f"kabul {rapor['kabul']} (tohum {rapor['tohum']})")


def kendini_test() -> list[str]:
    """Kanarya: batarya kendi panigini uretebilmeli, yoksa panik saymaz."""
    bulgular: list[str] = []
    vakalar = mutasyonlar(TOHUM)
    adlar = [ad for ad, _ in vakalar]
    assert len(adlar) == len(set(adlar)), "vaka adlari tekrarli"
    assert mutasyonlar(TOHUM) == vakalar, "batarya tohumla tekrarlanabilir degil"
    assert mutasyonlar(TOHUM + 1) != vakalar, "farkli tohum ayni bataryayi verdi"
    bulgular.append(f"{len(vakalar)} vaka, tohum tekrarlanabilir")
    # Panik tespiti: kasitli panikleyen bir sahte ikili ile olcum aracini sina.
    with tempfile.TemporaryDirectory() as gecici:
        sahte = Path(gecici) / "sahte.sh"
        sahte.write_text("#!/bin/sh\necho 'thread main panicked at x.rs:1:1'\nexit 101\n", encoding="utf-8")
        sahte.chmod(0o755)
        sonuc = kos(sahte, ["corpus", "yok.jsonl"])
        assert sonuc["panik"], "kasitli panik tespit edilmedi"
        bulgular.append("panik tespiti")
        # Sinyalle olmek de panik sayilir (fail-silent), ama temiz ret sayilmaz.
        sinyalli = Path(gecici) / "sinyalli.sh"
        sinyalli.write_text("#!/bin/sh\nkill -SEGV $$\n", encoding="utf-8")
        sinyalli.chmod(0o755)
        sonuc = kos(sinyalli, ["corpus", "yok.jsonl"])
        assert sonuc["sinyal"] is True, "sinyalle olum tespit edilmedi"
        bulgular.append("sinyalle olum (cokus)")
        # Temiz ret panik sayilmaz: ayrimin kendisi kanaryalanir.
        ret = Path(gecici) / "ret.sh"
        ret.write_text("#!/bin/sh\necho 'lubot: refused' >&2\nexit 1\n", encoding="utf-8")
        ret.chmod(0o755)
        sonuc = kos(ret, ["corpus", "yok.jsonl"])
        assert not sonuc["panik"] and not sonuc["sinyal"], "temiz ret panik sayildi"
        bulgular.append("temiz ret panik sayilmaz")
    return bulgular


def main(argv: list[str]) -> int:
    ayristirici = argparse.ArgumentParser(description=__doc__)
    ayristirici.add_argument("--olc", action="store_true")
    ayristirici.add_argument("--kur", action="store_true")
    ayristirici.add_argument("--tohum", type=int, default=TOHUM)
    ayristirici.add_argument("--ikili", type=Path, default=IKILI)
    ayristirici.add_argument("--self-test", action="store_true")
    args = ayristirici.parse_args(argv)
    if args.self_test:
        print("self-test OK [belirsiz-girdi]: " + ", ".join(kendini_test()))
        return 0
    if not (args.olc or args.kur):
        ayristirici.error("--olc, --kur ya da --self-test gerekli")
    rapor = olc(args.ikili, args.tohum)
    if args.kur:
        KAYIT.parent.mkdir(parents=True, exist_ok=True)
        KAYIT.write_text(json.dumps(rapor, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"kayit yazildi: {KAYIT.relative_to(KOK)}", file=sys.stderr)
    # Sozlesme: stdout JSON (kapi onu okur), insan ozeti stderr'de.
    print(json.dumps(rapor, ensure_ascii=False))
    print(ozet(rapor), file=sys.stderr)
    if rapor["panik"] or rapor["asildi"]:
        print(f"RED: panik {rapor['panik']}, asildi {rapor['asildi']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
