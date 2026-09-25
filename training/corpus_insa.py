#!/usr/bin/env python3
"""Korpusu tek komutta yeniden inşa eder: kendi ağaç + kamu malı kayıtlar.

Üç adım, tek çıktı: `corpus/knowledge-self.jsonl.gz` (bu dosya `.gitignore`
içindedir; korpus **türevdir**, depoda tutulmaz — depoda tutulan şey türevin
*kaynağıdır*):

1. `training/build_corpus.py` — depo ağacından kayıtlar (kendi eser).
2. `veri/kamu-mali.jsonl.gz` — daha önce alınmış kamu malı kayıtlar. Bu dosya
   depoda **izlenir**: CI'ın ağ erişimi olmadan aynı korpusu kurabilmesi için.
   İçeriği `training/kamu_verisi.py --veri-ekle` ile yenilenir (operatör koşusu;
   lisans indirmeden önce kaynağın kendi kaydından doğrulanır, kaynak adı ve
   lisans adı yazılmaz).
3. İkisi tek dosyada birleşir: depo ağacında tek korpus, tek sayım.

Ağ erişimi gerekmez. Tekrar üretilebilirlik: aynı depo ağacı + aynı veri dosyası
→ birebir aynı korpus.

    python3 training/corpus_insa.py [--veri veri/kamu-mali.jsonl.gz]
"""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent
ANA = KOK / "corpus" / "knowledge-self.jsonl.gz"
VARSAYILAN_VERI = KOK / "veri" / "kamu-mali.jsonl.gz"


def calistir(komut: list[str]) -> dict:
    sonuc = subprocess.run(komut, cwd=KOK, capture_output=True, text=True, check=False)
    if sonuc.returncode != 0:
        raise SystemExit(
            f"komut dustu ({' '.join(komut)}):\n{sonuc.stdout[-500:]}\n{sonuc.stderr[-500:]}"
        )
    return json.loads(sonuc.stdout)


def satir_sayisi(yol: Path) -> int:
    with gzip.open(yol, "rt", encoding="utf-8") as dosya:
        return sum(1 for satir in dosya if satir.strip())


def birlestir(parcalar: list[Path], hedef: Path) -> int:
    """Parçaları tek korpus dosyasında birleştirir.

    Geçici dosyada kurulur, sonra yerine konur: yarı yazılmış bir korpus ölçümü
    sessizce eksik bırakırdı.
    """
    gecici = hedef.with_suffix(".gecici")
    sayi = 0
    with gzip.open(gecici, "wb", compresslevel=9) as cikti:
        for parca in parcalar:
            if not parca.is_file():
                raise SystemExit(f"korpus parcasi yok: {parca}")
            with gzip.open(parca, "rt", encoding="utf-8") as kaynak:
                for satir in kaynak:
                    if satir.strip():
                        cikti.write(satir.encode("utf-8"))
                        sayi += 1
    gecici.replace(hedef)
    return sayi


def main(argv: list[str]) -> int:
    ayristirici = argparse.ArgumentParser(description=__doc__)
    ayristirici.add_argument("--veri", type=Path, default=VARSAYILAN_VERI)
    ayristirici.add_argument("--kamu-mali-yok", action="store_true",
                             help="yalniz kendi agac korpusunu kur (kamu mali dosyasi olmadan)")
    args = ayristirici.parse_args(argv)

    yerel = calistir([sys.executable, "training/build_corpus.py", "--repo", ".",
                      "--out", str(ANA.relative_to(KOK))])
    parcalar = [ANA]
    kamu = 0
    if not args.kamu_mali_yok:
        kamu = satir_sayisi(args.veri)
        parcalar.append(args.veri)
    toplam = birlestir(parcalar, ANA)
    print(json.dumps({
        "kendi_agac": yerel["records"],
        "kamu_mali": kamu,
        "korpus": toplam,
        "jeton_hesabi": "training/egitim_butcesi.py --olc",
        "cikti": str(ANA.relative_to(KOK)),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
