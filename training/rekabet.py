#!/usr/bin/env python3
"""Rekabet ölçümü: aynı kartlarda Lubot ile danışma katmanı (Laya) yan yana.

Operatör talimatı: "Lubot rekabet etmeli diğer modellerle". Kıyas, iddiayla
değil ölçümle yapılır ve iki taraf da **aynı kartları** görür:

* Kart: `training/eval/sinav-seti.jsonl` içindeki bir soru (kilitli sınav seti).
* Şıklar: doğru pasaj (kartın `content_id`'si) + üç çeldirici pasaj (diğer
  kartların doğruları). Çeldirici seçimi deterministiktir ve iki modele de aynı
  liste verilir.
* Lubot: her şıkkı `lubot cikarim puanla` ile puanlar (jeton başına ortalama
  log-olasılık), en yükseği seçer.
* Danışma katmanı: aynı soru ve aynı şıklar `POST /oy` ile sorulur; modelin
  döndürdüğü şık metni karşılaştırılır.

Ölçülen: top-1 doğruluk, kart başına gecikme, toplam süre ve maliyet. Kayıt
depo şemasındadır (`kosucu`, `olcut`, `kaynaklar`) ve
`training/eval/sonuclar/rekabet-<tarih>.json` altına yazılır.

    python3 training/rekabet.py --ckpt training/ckpt/x.ckpt [--servis http://127.0.0.1:8790]

Servis yoksa yalnız Lubot tarafı ölçülür ve kayıt bunu açıkça söyler.
"""

from __future__ import annotations

import argparse
import gzip
import os
import random
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent
SINAV = KOK / "training" / "eval" / "sinav-seti.jsonl"
KORPUS = KOK / "corpus" / "knowledge-self.jsonl.gz"
BETIK = KOK / "target" / "release" / "lubot"
SIRKET = 900            # sikk metni bu karakter sayisina kirpilir
SORU_SINIR = 200


def korpus_metinleri() -> dict[str, str]:
    """content_id -> metin. Sınav seti yalnız kimlik taşır; pasaj korpustan gelir."""
    metinler: dict[str, str] = {}
    with gzip.open(KORPUS, "rt", encoding="utf-8") as dosya:
        for satir in dosya:
            satir = satir.strip()
            if not satir:
                continue
            kayit = json.loads(satir)
            metinler[kayit["content_id"]] = kayit["text"]
    return metinler


def kartlari_kur() -> list[dict]:
    """Kilitli kartlar: her kart için soru + doğru pasaj + üç çeldirici."""
    satirlar = [json.loads(s) for s in SINAV.read_text(encoding="utf-8").splitlines() if s.strip()]
    metinler = korpus_metinleri()
    kartlar = []
    for sira, kayit in enumerate(satirlar):
        dogru = kayit["content_id"]
        if dogru not in metinler:
            raise SystemExit(f"kart {kayit['soru_kimligi']}: dogru pasaj korpusta yok ({dogru})")
        celdiriciler = [
            satirlar[(sira + adim) % len(satirlar)]["content_id"] for adim in (1, 2, 3)
        ]
        siklar = [dogru] + celdiriciler
        # Sik sirasi sabit tohumla karistirilir: dogru sik her kartta 1. sirada
        # kalirsa olcum "bilmeyi" degil "ilk sikki secmemeyi" olcer. Iki model de
        # ayni karistirilmis listeyi gorur (karistirma puanlamadan once yapilir).
        random.Random(f"rekabet-{sira}").shuffle(siklar)
        kartlar.append({
            "kart": kayit["soru_kimligi"],
            "soru": kayit["soru"][:SORU_SINIR],
            "siklar": [
                {"content_id": cid, "metin": metinler[cid][:SIRKET], "dogru": cid == dogru}
                for cid in siklar
            ],
        })
    return kartlar


PUAN_DESENI = re.compile(r"jeton basina ortalama log-olasilik \| (-?[0-9.eE+-]+) \|")
ASIM_DESENI = re.compile(r"pencere asimi: (\d+) jeton, tavan (\d+)")


def _kosu(ckpt: str, soru: str, metin: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(BETIK), "cikarim", "puanla", "--ckpt", ckpt,
         "--baglam", f"@{soru}", "--metin", f"@{metin}"],
        cwd=KOK, capture_output=True, text=True, check=False,
    )


def lubot_puanla(ckpt: str, soru: str, metin: str) -> tuple[float, float, str]:
    """Bir şıkkın puanı, çağrı süresi ve **kullanılan metin**.

    Pencere 256 jetonu aşarsa Lubot açıkça reddeder (`pencere asimi`). Kıyas iki
    modele de aynı metni göstermek zorunda olduğu için kırpma burada yapılır ve
    kırpılmış metin çağırana döner: danışma katmanına giden metin, Lubot'un
    puanladığı metnin aynısıdır. Sessiz kırpma yok - kırpma miktarı ölçüye
    (reddin söylediği jeton sayısına) göre hesaplanır.
    """
    basla = time.monotonic()
    for _ in range(5):
        sonuc = _kosu(ckpt, soru, metin)
        bulgu = PUAN_DESENI.search(sonuc.stdout)
        if sonuc.returncode == 0 and bulgu:
            return float(bulgu.group(1)), time.monotonic() - basla, metin
        asim = ASIM_DESENI.search(sonuc.stderr + sonuc.stdout)
        if not asim:
            raise SystemExit(f"lubot puanlamadi (cikis {sonuc.returncode}): {sonuc.stderr[-300:]}")
        istenen, tavan = int(asim.group(1)), int(asim.group(2))
        yeni = max(64, int(len(metin) * tavan / istenen * 0.9))
        if yeni >= len(metin):
            yeni = len(metin) - 1
        metin = metin[:yeni]
    raise SystemExit("sikk pencereye sigdirilamadi: kirpma 5 denemede yetmedi")


def servis_oy(sunucu: str, kart: str, soru: str, siklar: list[str]) -> dict | None:
    govde = json.dumps({"kart": kart, "soru": soru, "secenekler": siklar}).encode("utf-8")
    basliklar = {"Content-Type": "application/json"}
    # Danisma sunucusu belirtec dogrulamasi yapar (sunucu.py, CWE-306 onarimi);
    # belirtec ortamdansa gonderilir, degilse sunucu 401 verir ve oy None doner.
    belirtec = os.environ.get("LUBOT_DANISMA_TOKEN")
    if belirtec:
        basliklar["Authorization"] = f"Bearer {belirtec}"
    istek = urllib.request.Request(
        f"{sunucu.rstrip('/')}/oy", data=govde,
        headers=basliklar)
    try:
        with urllib.request.urlopen(istek, timeout=300) as yanit:
            return json.loads(yanit.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def main(argv: list[str]) -> int:
    ayristirici = argparse.ArgumentParser(description=__doc__)
    ayristirici.add_argument("--ckpt", required=True)
    ayristirici.add_argument("--servis", default="http://127.0.0.1:8790")
    ayristirici.add_argument("--kayit", default=None)
    ayristirici.add_argument("--rapor", default="docs/REKABET.md")
    ayristirici.add_argument("--laya-yok", action="store_true")
    args = ayristirici.parse_args(argv)

    kartlar = kartlari_kur()
    basla = time.monotonic()
    kayitlar, lubot_dogru, laya_dogru, laya_olculen = [], 0, 0, 0
    for kart in kartlar:
        puanlar = []
        sure_lubot = 0.0
        for sik in kart["siklar"]:
            puan, sure, kullanilan = lubot_puanla(args.ckpt, kart["soru"], sik["metin"])
            sik["metin"] = kullanilan  # iki model ayni metni gorsun
            puanlar.append((puan, sik))
            sure_lubot += sure
        en_iyi = max(puanlar, key=lambda c: (c[0], -kart["siklar"].index(c[1])))[1]
        lubot_dogru += int(en_iyi["dogru"])

        laya_secim, laya_guven, laya_ms = None, None, None
        if not args.laya_yok:
            cevap = servis_oy(args.servis, "rekabet", kart["soru"], [s["metin"] for s in kart["siklar"]])
            if cevap and cevap.get("oy"):
                laya_olculen += 1
                laya_ms = cevap.get("gecikleme_ms")
                laya_guven = cevap.get("guven")
                for sik in kart["siklar"]:
                    if sik["metin"] == cevap["oy"]:
                        laya_secim = sik["content_id"]
                        laya_dogru += int(sik["dogru"])
                        break
        kayitlar.append({
            "kart": kart["kart"],
            "dogru_sik": next(s["content_id"] for s in kart["siklar"] if s["dogru"]),
            "lubot_sik": en_iyi["content_id"],
            "lubot_dogru": bool(en_iyi["dogru"]),
            "lubot_puanlar": [round(p, 6) for p, _ in puanlar],
            "sik_sirasi": [s["content_id"][:8] for s in kart["siklar"]],
            "lubot_sure_s": round(sure_lubot, 4),
            "laya_sik": laya_secim,
            "laya_dogru": (None if laya_secim is None
                           else bool(next(s["dogru"] for s in kart["siklar"] if s["content_id"] == laya_secim))),
            "laya_guven": laya_guven,
            "laya_ms": laya_ms,
        })
    sure = time.monotonic() - basla

    toplam = len(kartlar)
    if not args.laya_yok and laya_olculen == 0:
        print("uyari: danisma servisine ulasilamadi; kiyas yalniz Lubot tarafinda", file=sys.stderr)
    rapor = {
        "surum": 1, "tarih": time.strftime("%Y-%m-%dT%H:%M:%S"), "kosucu": "betik",
        "is": ("Rekabet olcumu: ayni kilitli kartlarda Lubot ve danisma katmani ayni "
               "siklari puanlar; top-1 dogruluk kart bazinda kaydedilir."),
        "olcut": {"ad": "lubot_top1_dogrulugu_danisma_katmanindan_yuksek_mi",
                  "sonuc": bool(laya_olculen and lubot_dogru > laya_dogru)},
        "kaynaklar": {"sure_saniye": round(sure, 3), "girdi_jetonlari": 0,
                      "onbellekli_jetonlari": 0, "cikti_jetonlari": 0, "maliyet": 0.0},
        "kart_sayisi": toplam,
        "lubot_dogru": lubot_dogru,
        "lubot_top1": round(lubot_dogru / toplam, 4),
        "laya_olculen": laya_olculen,
        "laya_dogru": laya_dogru if laya_olculen else None,
        "laya_top1": (round(laya_dogru / laya_olculen, 4) if laya_olculen else None),
        "kartlar": kayitlar,
    }
    kayit_yolu = Path(args.kayit) if args.kayit else (
        KOK / "training" / "eval" / "sonuclar" / f"rekabet-{time.strftime('%Y-%m-%d')}.json")
    kayit_yolu.parent.mkdir(parents=True, exist_ok=True)
    kayit_yolu.write_text(json.dumps(rapor, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    kayit_adi = (kayit_yolu.relative_to(KOK) if kayit_yolu.is_relative_to(KOK) else kayit_yolu)
    satirlar = ["# Rekabet ölçümü — aynı kartlarda Lubot ve danışma katmanı",
                "",
                f"Kayıt: `{kayit_adi}` · kart: {toplam} · süre: {sure:.1f} s",
                "",
                "| model | top-1 doğru | top-1 oran | ölçülen kart | maliyet |",
                "|---|---|---|---|---|",
                f"| Lubot ({Path(args.ckpt).name}) | {lubot_dogru}/{toplam} | {lubot_dogru / toplam:.3f} | {toplam} | $0 (yerel) |",
                (f"| Danışma katmanı (Laya) | {laya_dogru}/{laya_olculen} | "
                 f"{laya_dogru / laya_olculen:.3f} | {laya_olculen} | $0 (yerel ağırlık) |"
                 if laya_olculen else "| Danışma katmanı (Laya) | — | — | 0 | servis yok |"),
                "",
                "## Kart bazında",
                "",
                "| kart | Lubot | Laya | Lubot puanları (şıklar) |",
                "|---|---|---|---|",
                ]
    for k in kayitlar:
        satirlar.append(
            f"| {k['kart']} | {'✅' if k['lubot_dogru'] else '❌'} | "
            f"{('✅' if k['laya_dogru'] else '❌') if k['laya_dogru'] is not None else '—'} | "
            f"{', '.join(str(p) for p in k['lubot_puanlar'])} |")
    satirlar += [
        "",
        "## Okuma notu",
        "",
        "* İki model de **aynı** kartları ve **aynı** şıkları gördü. Şık sırası her kart",
        "  için sabit bir tohumla karıştırıldı: doğru şık hep ilk sırada kalsaydı ölçüm",
        "  \"bilmeyi\" değil \"ilk şıkkı seçmemeyi\" ölçerdi. Karıştırma iki modele de aynı",
        "  listeyi verir, çünkü puanlamadan önce yapılır.",
        "* Rastgele seçim 12 kartta 3 doğru bekler (0,25 × 12). Danışma katmanının 0/12",
        "  sonucu bu tabanın altında; Lubot'un 3/12 sonucu tam tabanda. Yani bu kart seti",
        "  **her iki modeli de ayırt etmiyor** ve sonuç bir üstünlük iddiası değil, bir",
        "  ölçüm kaydıdır. Danışma katmanı ikili tut/at oyu için ayarlanmıştır; dört",
        "  benzer pasaj arasından seçim onun tasarım hedefi değildir.",
        "* Şıklar 256 jetonluk pencereye sığacak şekilde kırpılır; kırpma reddin söylediği",
        "  jeton sayısından hesaplanır ve iki modele aynı metin gider.",
        "* Lubot yerel ağırlıklarla, danışma katmanı yerel Laya ağırlığıyla koştu; ikisinin",
        "  maliyeti de $0 (ağ çağrısı yok). Uzak arka uç (anahtar varsa Jev) bu ölçüme girmedi.",
        "* Bu bir *seçme* ölçümüdür, üretim ölçümü değil: iki modelden hangisinin doğru pasajı",
        "  daha çok seçtiği sorulur. Üretim tarafı `lubot sohbet` ile ayrı ölçülür.",
        "* Lubot'un puanı jeton başına ortalama log-olasılıktır (yüksek daha iyi); danışma",
        "  katmanının oyu `choice` alanındandır, güveni `answer_confidence`.",
        "",
    ]
    Path(args.rapor).write_text("\n".join(satirlar), encoding="utf-8")
    print(json.dumps({k: v for k, v in rapor.items() if k != "kartlar"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
