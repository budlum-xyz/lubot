#!/usr/bin/env python3
"""Rakip model kosucusu: beyan edilmis sinifa karsi olcum araci (AA protokolu).

Bu betik Lubot'un urun kodu degildir; bir OLCEK aracidir. GG'nin beyan
ettigi kucuk acik model sinifindan bir model, `training/rekabet.py` ile
birebir ayni kilitli kartlari ve ayni siklari gorur ve her kartta tek bir
kapali secim yapar. Amac "Lubot bu sinifa karsi gorev ekseninde nerede"
sorusuna yayinlanmis sayi degil, bu makinede olculmus sayi vermektir.

K2/AA siniri fizikseldir:
- Bu betik dosya YAZMAZ; cikti yalniz stdout'a JSON olarak gider ve onu
  cagiran stdlib betik yalnizca `training/eval/sonuclar/` altina kaydeder.
- Rakip modelin urettigi hicbir bayt corpus/'a, training/curriculum/'a veya
  baska bir egitim girdisine giremez; kiyas raporu tek yasama alanidir.
- Agirliklar yerel bir dizinden yuklenir; kosu sirasinda agdan bir sey
  cekilmez, disariya istek atilmaz.

Kullanim:
    python3 tools/rakip_kosucu.py --model /yol/model --kartlar /tmp/kartlar.json

Girdi kartlar dosyasi `training/rekabet.py::kartlari_kur` ciktisidir
(soru + dort sik, sik metinleri ayni kirpmayla). Cikti: kart basina secim,
dogruluk, jeton sayilari ve sure; toplam kaynak muhasebesiyle.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

SECIM_DESENI = re.compile(r"[1-4]")

# Talimat rakibin en guclu dilinde (model agirlikli olarak Ingilizce egitildi);
# soru ve pasajlarin kendisi degistirilmeden Turkce kaliyor. Bu, rakibe icerik
# ayni kalmak uzere en iyi sansini vermek icin bilincli bir yontem secimi.
SISTEM = (
    "You are a reading assistant. You will be given a question in Turkish and "
    "four numbered passages in Turkish. Choose the single passage that answers "
    "the question. Write only the digit of that passage: 1, 2, 3 or 4. Write "
    "nothing else."
)


def kart_metni(kart: dict) -> str:
    """The user message: the question and the four numbered passages, exactly
    as the caller prepared them (same truncation both sides saw)."""
    parcalar = [f"Soru: {kart['soru']}", "", "Pasajlar:"]
    for i, sik in enumerate(kart["siklar"], 1):
        parcalar.append(f"[{i}] {sik['metin']}")
    parcalar.append("")
    parcalar.append("Cevap (yalniz rakam):")
    return "\n".join(parcalar)


def main(argv: list[str]) -> int:
    ayristirici = argparse.ArgumentParser(description=__doc__)
    ayristirici.add_argument("--model", required=True, help="yerel model dizini")
    ayristirici.add_argument("--kartlar", required=True, help="kartlar JSON dosyasi")
    ayristirici.add_argument("--en-cok-jeton", type=int, default=32)
    args = ayristirici.parse_args(argv)

    model_dir = Path(args.model)
    if not model_dir.is_dir():
        print(f"model dizini yok: {model_dir}", file=sys.stderr)
        return 2
    kartlar = json.loads(Path(args.kartlar).read_text(encoding="utf-8"))

    # Import here, not at module top: the measurement tool loads its heavy
    # dependencies only when it is actually run, and a missing torch is a
    # clean refusal, not a crash at import time.
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"rakip kosucu icin torch/transformers kurulu degil: {e}", file=sys.stderr)
        return 3

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    # bf16 on CPU: the weights are stored bf16, this halves the footprint on
    # a small machine, and the comparison is recorded with the dtype stated.
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype=torch.bfloat16)
    model.eval()

    parametre = 0
    for p in model.parameters():
        parametre += p.numel()

    sonuc_kartlar = []
    toplam_girdi = 0
    toplam_cikti = 0
    baslangic = time.monotonic()
    for kart in kartlar:
        mesajlar = [
            {"role": "system", "content": SISTEM},
            {"role": "user", "content": kart_metni(kart)},
        ]
        prompt = tokenizer.apply_chat_template(
            mesajlar, tokenize=False, add_generation_prompt=True
        )
        giris = tokenizer(prompt, return_tensors="pt")
        girdi_jeton = int(giris["input_ids"].shape[1])
        kart_basladi = time.monotonic()
        with torch.no_grad():
            uretim = model.generate(
                **giris,
                max_new_tokens=args.en_cok_jeton,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        yeni = uretim[0][girdi_jeton:]
        metin = tokenizer.decode(yeni, skip_special_tokens=True)
        cikti_jeton = int(len(yeni))
        ilk = SECIM_DESENI.search(metin)
        secim = int(ilk.group(0)) - 1 if ilk else None
        dogru = secim is not None and kart["siklar"][secim]["dogru"]
        toplam_girdi += girdi_jeton
        toplam_cikti += cikti_jeton
        sonuc_kartlar.append({
            "kart": kart["kart"],
            "secim": secim,
            "dogru": dogru,
            "uretme_metni": metin[:120],
            "girdi_jetonlari": girdi_jeton,
            "cikti_jetonlari": cikti_jeton,
            "sure_saniye": round(time.monotonic() - kart_basladi, 3),
        })

    dogrular = sum(1 for k in sonuc_kartlar if k["dogru"])
    print(json.dumps({
        "model": {
            "dizin": str(model_dir),
            "parametre": parametre,
            "uretim": "greedy (do_sample=False)",
            "hassasiyet": "bfloat16",
            "en_cok_jeton": args.en_cok_jeton,
        },
        "kart_sayisi": len(kartlar),
        "dogru": dogrular,
        "top1": dogrular / len(kartlar) if kartlar else 0.0,
        "kartlar": sonuc_kartlar,
        "toplam": {
            "girdi_jetonlari": toplam_girdi,
            "cikti_jetonlari": toplam_cikti,
            "sure_saniye": round(time.monotonic() - baslangic, 3),
        },
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
