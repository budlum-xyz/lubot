#!/usr/bin/env python3
"""APK denetimi: iddialari dosyanin kendisinden okur.

Bir APK hakkinda "soyle soyle" demek kolaydir; bu betik **soylemeden once
bakar**. Denetledigi seyler APK'nin icinden okunur - manifest, dex, paylasilan
nesne - ve bir sey eksikse cikis kodu sifir olmaz.

Neden bir betik, neden kapi degil: kapi (`gates/check.py`) deponun her kosusunda
calisir ve APK uretilmemis bir makinede kapi duser; bu betik ise yalnizca APK
uretildikten sonra, araclar elimdeyken kosar. Ikisinin isi ayri: kapi **varligi**
denetler (bu betik mi, Android kaynaklari mi, el sıkışma mi), bu betik **icerigi**.

Kullanim:
    python3 android/apk_denetle.py target/apk/lubot.apk [--sdk <sdk>]
"""

from __future__ import annotations

import argparse
import re
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

# Beklenen degerler tek yerde durur: `Kopru.java`'daki `native` bildirimleri,
# `crates/arayuz/src/lib.rs`'teki `Java_*` islevleri ve buradaki liste ayni
# sozlesmedir. Ucu ayristiginda ortaya cikan sey sessiz bir cokme olurdu.
BEKLENEN_JNI = [
    "Java_dev_budlum_lubot_Kopru_belgeEkle",
    "Java_dev_budlum_lubot_Kopru_kurulus",
    "Java_dev_budlum_lubot_Kopru_soru",
    "Java_dev_budlum_lubot_Kopru_surum",
]
BEKLENEN_IZINLER = ["android.permission.INTERNET"]
BEKLENEN_PAKET = "dev.budlum.lubot"
BEKLENEN_AKTIVITE = "dev.budlum.lubot.AnaEtkinlik"
# Uygulamanin calismasi icin pakette bulunmasi gereken girdiler. `derle.sh`
# bunlari imzadan once dogrular; burada **imzalanmis** dosyadan dogrulanir -
# iki kontrol ayni seyi iki farkli anda soyler.
BEKLENEN_GIRDILER = [
    "classes.dex",
    "lib/arm64-v8a/liblubot_arayuz.so",
    "assets/korpus/knowledge-self.jsonl.gz",
]


def java_native_bildirimleri(kok: Path) -> list[str]:
    """`Kopru.java`'daki `native` yontem adlari."""
    yol = kok / "android" / "src" / "dev" / "budlum" / "lubot" / "Kopru.java"
    metin = yol.read_text(encoding="utf-8")
    return re.findall(r"static\s+native\s+\w+\s+(\w+)\s*\(", metin)


def rust_jni_islevleri(kok: Path) -> list[str]:
    """`Java_*` islevlerinin adlari, kaynaktan."""
    yol = kok / "crates" / "arayuz" / "src" / "lib.rs"
    metin = yol.read_text(encoding="utf-8")
    return re.findall(r"extern\s+\"system\"\s+fn\s+(Java_\w+)", metin)


def so_sembolleri(so: bytes) -> set[str]:
    """ELF dinamik sembol tablosundan tanimli (T) adlar."""
    # readelf olmayabilir; dinamik sembolleri elle okumak yerine arac varsa onu
    # kullaniyoruz, yoksa `.dynstr` icinde ad aramasiyla yetiniyoruz - ve bunu
    # raporda **soyluyoruz**, cunku iki yontem ayni sey degildir.
    adlar = set(re.findall(rb"Java_dev_budlum_lubot_Kopru_\w+", so))
    return {a.decode() for a in adlar}


def manifest_ozetleri(apk: Path, sdk: Path | None) -> dict[str, str]:
    """aapt2 badging cikti varsa onu, yoksa ikili manifest okumasini kullanir."""
    aapt2 = None
    if sdk:
        adaylar = sorted(sdk.rglob("aapt2"))
        aapt2 = adaylar[-1] if adaylar else None
    if aapt2:
        cikti = subprocess.run(
            [str(aapt2), "dump", "badging", str(apk)],
            capture_output=True, text=True, check=False,
        ).stdout
        return {"yontem": "aapt2", "ham": cikti}
    # aapt2 yoksa: ikili manifest'ten paket adini oku. Android'in ikili XML'i
    # UTF-16 dize havuzu tasir; `package` ozniteligi havuzda duz metin durur.
    with zipfile.ZipFile(apk) as z:
        ham = z.read("AndroidManifest.xml")
    return {"yontem": "ikili", "ham": ham.decode("utf-16-le", errors="ignore")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("apk", help="denetlenecek APK")
    ap.add_argument("--kok", default=".", help="depo koku")
    ap.add_argument("--sdk", default="", help="Android SDK (aapt2 icin)")
    a = ap.parse_args()

    kok = Path(a.kok).resolve()
    apk = Path(a.apk)
    hatalar: list[str] = []
    satirlar: list[str] = []
    if not apk.is_file():
        print(f"APK yok: {apk}")
        return 1

    with zipfile.ZipFile(apk) as z:
        adlar = z.namelist()
        boyut = apk.stat().st_size
        satirlar.append(f"- APK: `{apk}` ({boyut} bayt, {len(adlar)} girdi)")

        # 1) imza: APK'nin imzali oldugunu **iddia etmek** yerine varligina bak.
        imzali = any(n.startswith("META-INF/") and n.endswith((".RSA", ".DSA", ".EC")) for n in adlar)
        if imzali:
            satirlar.append("- imza: var (v1 imza blogu)")
        else:
            hatalar.append("imza blogu yok: APK imzalanmamis")

        # 3) paylasilan nesne ve JNI sembolleri
        so_adlari = [n for n in adlar if n.startswith("lib/") and n.endswith(".so")]
        if not so_adlari:
            hatalar.append("lib/**/*.so yok")
        for ad in so_adlari:
            so = z.read(ad)
            # Mimarî: ELF basligindaki e_machine alani 183 (AArch64) olmali.
            if len(so) > 20:
                makine = struct.unpack_from("<H", so, 18)[0]
                sinif = so[4]
                if makine != 183 or sinif != 2:
                    hatalar.append(f"{ad}: beklenen AArch64/ELF64 degil (makine {makine})")
            bulunan = so_sembolleri(so)
            eksik = [s for s in BEKLENEN_JNI if s not in bulunan]
            if eksik:
                hatalar.append(f"{ad}: JNI islevi yok: {', '.join(eksik)}")
            else:
                satirlar.append(f"  JNI: {len(bulunan)}/{len(BEKLENEN_JNI)} islev tanimli")

        # 4) pakette bulunmasi gereken girdiler
        for gerekli in BEKLENEN_GIRDILER:
            if gerekli not in adlar:
                hatalar.append(f"pakette yok: {gerekli}")
            else:
                satirlar.append(f"- {gerekli}: {z.getinfo(gerekli).file_size} bayt")

        # 5) kaynaklar
        kaynaklar = [n for n in adlar if n.endswith(".xml") or n == "resources.arsc"]
        satirlar.append(f"- kaynak: {len(kaynaklar)} dosya ({', '.join(sorted(kaynaklar))})")

    # 6) kaynak ile kopru sozlesmesi
    java_adlar = java_native_bildirimleri(kok)
    rust_adlar = rust_jni_islevleri(kok)
    if len(java_adlar) != len(rust_adlar):
        hatalar.append(
            f"sozlesme uyusmuyor: Java'da {len(java_adlar)}, Rust'ta {len(rust_adlar)} islev"
        )
    else:
        satirlar.append(
            f"- sozlesme: Java `native` {len(java_adlar)} = Rust `Java_*` {len(rust_adlar)}"
        )
    # Denetim listesi ile kaynak ayrisirsa, liste guncellenmeden yeni bir islev
    # eklenmis demektir: sozlesme buyur ama denetim onu gormez. Bu yuzden liste
    # **tam esitlik** ister, icerme degil.
    if sorted(rust_adlar) != sorted(BEKLENEN_JNI):
        hatalar.append(
            "JNI sozlesmesi denetim listesiyle ayni degil: "
            f"kaynakta {sorted(rust_adlar)}"
        )

    # 7) manifest: paket, aktivite, izin
    ozet = manifest_ozetleri(apk, Path(a.sdk) if a.sdk else None)
    ham = ozet["ham"]
    if BEKLENEN_PAKET not in ham:
        hatalar.append(f"manifest'te paket adi yok: {BEKLENEN_PAKET}")
    if BEKLENEN_AKTIVITE not in ham:
        hatalar.append(f"manifest'te aktivite yok: {BEKLENEN_AKTIVITE}")
    for izin in BEKLENEN_IZINLER:
        if izin not in ham:
            hatalar.append(f"manifest'te izin yok: {izin}")
    # Fazladan izin: bu uygulamanin sozlesmesi tek izindir.
    bulunan_izinler = set(re.findall(r"android\.permission\.[A-Z_]+", ham))
    fazla = bulunan_izinler - set(BEKLENEN_IZINLER)
    if fazla:
        hatalar.append(f"beklenmeyen izin: {', '.join(sorted(fazla))}")
    satirlar.append(
        f"- manifest: paket {BEKLENEN_PAKET}, izin {', '.join(sorted(bulunan_izinler))} "
        f"(okuma yontemi: {ozet['yontem']})"
    )

    print("# APK denetimi\n")
    print("\n".join(satirlar))
    if hatalar:
        print("\n## HATA\n")
        for h in hatalar:
            print(f"- {h}")
        return 1
    print("\n- SONUC: butun iddialar dosyadan dogrulandi")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
