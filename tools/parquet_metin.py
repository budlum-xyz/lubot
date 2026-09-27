#!/usr/bin/env python3
"""Parquet'i satır satır metne çeviren ayrı araç.

Eğitim koşucusu (`training/*.py`) yalnız stdlib kullanır: kapı
`training-runner-engineering-vs-data` bunu böyle tutuyor ve "sıfırdan"
iddiası bu yüzden ayakta kalıyor. Kaynakların dağıtım biçimi parquet
olduğu için çözme işi buraya, koşucunun **dışına** konuldu: bağımlılık
tek dosyada, beyan edilmiş ve gözlemlenebilir durur.

Ne yapar: metin sütunlarını blok blok okur ve her satırı tek satırlık
JSON olarak basar (`{"sütun": "değer", ...}`). Korpus kuralları (alan
beyaz listesi, ad/adres silme, kırpma) burada **uygulanmaz**; onlar
koşucunun işidir, çünkü ölçülen şey koşucunun ürettiği kayıttır.

Kullanım:

    python3 tools/parquet_metin.py --dosya /yol/dosya.parquet
"""

from __future__ import annotations

import argparse
import json
import sys

BLOK = 2048


def satirlar(dosya_yolu: str):
    import pyarrow as pa
    import pyarrow.parquet as pq

    dosya = pq.ParquetFile(dosya_yolu)
    metin_sutunlari = [
        alan.name
        for alan in dosya.schema_arrow
        if pa.types.is_string(alan.type) or pa.types.is_large_string(alan.type)
    ]
    for blok in dosya.iter_batches(batch_size=BLOK, columns=metin_sutunlari or None):
        for satir in blok.to_pylist():
            temiz = {
                ad: deger
                for ad, deger in satir.items()
                if isinstance(deger, str) and deger.strip()
            }
            if temiz:
                yield temiz


def main(argv: list[str]) -> int:
    ayristirici = argparse.ArgumentParser(description=__doc__)
    ayristirici.add_argument("--dosya", required=True)
    args = ayristirici.parse_args(argv)
    for satir in satirlar(args.dosya):
        sys.stdout.write(json.dumps(satir, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
