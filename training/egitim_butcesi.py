#!/usr/bin/env python3
"""Measure the training budget the regularization policy is written against.

NN §8 step 5 asks for a regularization policy for the data-limited regime and
for the repetition question to become a metric rather than an assumption. The
policy lives in `training/duzenlilestirme.json`; this script measures the
input-side facts that policy is written against, with the frozen vocabulary:

* how many BPE tokens the corpus this machine builds actually holds;
* how many token passes the declared epoch count turns that into;
* the effective-source ratio (`unique / passes`), which is what "more epochs is
  better" forgets to state - at four epochs it is 0.25, and a run that reports
  improvement without it does not say where the improvement came from;
* tokens per parameter, against the committed spec.

What is **not** measured here, and is labelled as such in the policy: the
marginal value of the k-th pass over a token. That needs a training run. This
script measures the budget, not the returns, and says so.

Usage:
    python3 training/egitim_butcesi.py --olc       # measure (JSON)
    python3 training/egitim_butcesi.py --kur       # measure + write the record
    python3 training/egitim_butcesi.py --dogrula   # re-measure and compare
    python3 training/egitim_butcesi.py --self-test
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POLITIKA = ROOT / "training" / "duzenlilestirme.json"
KAYIT = ROOT / "training" / "eval" / "sonuclar" / "egitim-butcesi-2026-09-23.json"
KORPUS = ROOT / "corpus" / "knowledge-self.jsonl.gz"
GRANT_KAYNAGI = ROOT / "crates" / "grant" / "src" / "training.rs"


def _modul(ad: str, dosya: str):
    spec = importlib.util.spec_from_file_location(ad, str(ROOT / "training" / dosya))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def politika_oku() -> dict:
    """Read the policy, fail-closed: a number nobody wrote down is not a policy."""
    if not POLITIKA.is_file():
        raise SystemExit(f"{POLITIKA.relative_to(ROOT)} yok: beyansiz politika olculemez")
    raw = json.loads(POLITIKA.read_text(encoding="utf-8"))
    for field in ("surum", "weight_decay", "max_epochs", "epoch_basina_kalite_olcutu"):
        if field not in raw:
            raise SystemExit(f"{POLITIKA.name}: zorunlu alan yok: {field}")
    for field in ("weight_decay", "max_epochs"):
        val = raw[field]
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise SystemExit(f"{POLITIKA.name}: {field} sayi degil")
    if not 0.0 < raw["weight_decay"] <= 1.0:
        raise SystemExit(
            f"{POLITIKA.name}: weight_decay {raw['weight_decay']} (0, 1] araliginda degil"
        )
    if not isinstance(raw["max_epochs"], int) or raw["max_epochs"] < 1:
        raise SystemExit(f"{POLITIKA.name}: max_epochs en az 1 olmali")
    return raw


def protokol_tavani() -> int:
    """The epoch ceiling from the grant crate itself.

    Read rather than repeated: a second literal for the same protocol constant
    is a disagreement waiting to happen, and the crate is the authority.
    """
    kaynak = GRANT_KAYNAGI.read_text(encoding="utf-8")
    m = re.search(r"MAX_TRAINING_GRANT_EPOCHS:\s*u32\s*=\s*(\d+)", kaynak)
    if not m:
        raise SystemExit(f"{GRANT_KAYNAGI.name}: MAX_TRAINING_GRANT_EPOCHS bulunamadi")
    return int(m.group(1))


def jeton_say() -> tuple[int, int, str]:
    """Tokenize the corpus this machine builds, with the frozen vocabulary.

    Returns (token count, record count, vocab family). The vocabulary is loaded
    through the trainer's own fail-closed loader, so a vocab that has drifted
    structurally is refused here rather than quietly mis-encoding.
    """
    spec = json.loads((ROOT / "training" / "model_spec.json").read_text(encoding="utf-8"))
    aile = spec["vocab_family"]
    vocab_path = ROOT / "training" / "tokenizer" / f"{aile}.json"
    tt = _modul("train_tokenizer", "train_tokenizer.py")
    vocab = tt.load_vocab(str(vocab_path))
    if not KORPUS.is_file():
        raise SystemExit(
            f"{KORPUS.relative_to(ROOT)} yok: CI korpusu kapilardan once kurar"
        )
    toplam = 0
    kayit_sayisi = 0
    with gzip.open(KORPUS, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            kayit_sayisi += 1
            toplam += len(tt.encode(json.loads(line)["text"], vocab))
    return toplam, kayit_sayisi, aile


def politika_ihlalleri(politika: dict, tavan: int) -> list[str]:
    """Breaches that need no corpus.

    CI runs the gate self-tests BEFORE the corpus exists, so a canary that had
    to tokenize could never run there. Kept pure so the canary is really
    executed in CI, and used by the measurement so there is only one rule.
    """
    ihlaller: list[str] = []
    epoch = politika["max_epochs"]
    if epoch > tavan:
        ihlaller.append(
            f"max_epochs {epoch} protokol tavanini asiyor ({tavan}, "
            f"{GRANT_KAYNAGI.relative_to(ROOT)})"
        )
    return ihlaller


def butce_olc() -> dict:
    """Measure the budget against the policy. Returns the numbers and any breach."""
    politika = politika_oku()
    tavan = protokol_tavani()
    jeton, kayit_sayisi, aile = jeton_say()
    spec = json.loads((ROOT / "training" / "model_spec.json").read_text(encoding="utf-8"))
    param = spec["params"]["toplam"]
    epoch = politika["max_epochs"]
    gecis = jeton * epoch
    ihlaller = politika_ihlalleri(politika, tavan)
    if jeton <= 0:
        ihlaller.append("korpus hic jeton uretmedi: olculecek bir butce yok")
    return {
        "politika_surumu": politika["surum"],
        "sozluk_ailesi": aile,
        "korpus_kayit_sayisi": kayit_sayisi,
        "benzersiz_jeton": jeton,
        "parametre": param,
        "weight_decay": politika["weight_decay"],
        "max_epochs": epoch,
        "protokol_epoch_tavani": tavan,
        "toplam_gecis": gecis,
        "tekrar_carpani": epoch,
        "etkin_kaynak_orani": round(jeton / gecis, 6) if gecis else 0.0,
        "jeton_basina_param_bir_epoch": round(jeton / param, 6) if param else 0.0,
        "jeton_basina_param_tam_butce": round(gecis / param, 6) if param else 0.0,
        "param_basina_benzersiz_jeton_tersi": round(param / jeton, 4) if jeton else 0.0,
        "spec_jeton_basina_param_beyani": spec.get("tokens_per_param_not"),
        # BULGU, duzeltme degil: the spec's data budget is quoted against the
        # surface corpus; the corpus CI builds is the self corpus, and the two
        # differ by an order of magnitude. Recorded, flagged, not resolved -
        # which corpus trains the model is an operator decision.
        "bulgu_veri_butcesi": {
            "olculen": (
                f"self korpusu (corpus/knowledge-self.jsonl.gz) {jeton} benzersiz "
                f"jeton uretiyor; {param} parametreye karsi "
                f"{round(jeton / param, 6) if param else 0.0} jeton/param. "
                "Spec'in 1.94 jeton/param beyani YUZEY korpusuna (1.791.712 jeton, "
                f"spec notu) ait; fark {round(1.94 / (jeton / param), 2) if jeton and param else 0.0} kat."
            ),
            "hukum": (
                f"spec'in veri butcesi beyani self korpusu icin tutmuyor: {param} "
                f"parametre {jeton} benzersiz jetona karsi geliyor (jeton basina "
                "~10 parametre). Bu bir BULGUDUR; hangi korpusun egitilecegi "
                "operator karari."
            ),
            "yapilmayan": (
                "model_spec.json degistirilmedi, spec beyani silinmedi, korpus "
                "degistirilmedi. Kayit bulguyu tasir, duzeltme yapmaz."
            ),
            "olculen_korpus": "self (corpus/knowledge-self.jsonl.gz)",
            "olculen_benzersiz_jeton": jeton,
            "spec_beyaninin_dayandigi_korpus": "yuzey (1.791.712 jeton, spec notu)",
            "olculen_jeton_basina_param": round(jeton / param, 6) if param else 0.0,
            "spec_beyani_jeton_basina_param": 1.94,
            "fark_kati": round(1.94 / (jeton / param), 2) if jeton and param else 0.0,
        },
        "ihlaller": ihlaller,
        "olculmeyen": [
            "k. gecisin marjinal degeri (egitim kosusu gerektirir)",
            "weight_decay'in bu verideki etkisi (egitim kosusu gerektirir)",
            "epoch_basina_kalite_olcutu degerleri (henüz epoch kosulmadi)",
        ],
    }


def kayit_yaz(butce: dict, sure: float) -> dict:
    return {
        "kosucu": "betik",
        "tarih": "2026-09-23",
        "is": "egitim-butcesi-olcumu",
        "olcut": {
            "ad": "beyan_edilen_epoch_sayisinin_protokol_tavanini_asmamasi_ve_butcenin_tekrar_uretilmesi",
            "sonuc": not butce["ihlaller"],
        },
        "kaynaklar": {
            "sure_saniye": round(sure, 3),
            "girdi_jetonlari": 0,
            "onbellekli_jetonlari": 0,
            "cikti_jetonlari": 0,
            "maliyet": 0.0,
        },
        "kanit": (
            f"training/egitim_butcesi.py --dogrula butceyi donmus {butce['sozluk_ailesi']} "
            f"sozluguyle yeniden sayar: {butce['korpus_kayit_sayisi']} kayit -> "
            f"{butce['benzersiz_jeton']} benzersiz jeton; beyan edilen {butce['max_epochs']} epoch "
            f"{butce['toplam_gecis']} gecis eder, etkin kaynak orani "
            f"{butce['etkin_kaynak_orani']}; protokol tavani {butce['protokol_epoch_tavani']}"
            f" ({GRANT_KAYNAGI.relative_to(ROOT)} icinden okundu)."
        ),
        "not": (
            "Bu kayit BUTCEYI olcer, getiriyi degil: ayni verinin k. gecisinin marjinal "
            "degeri yalniz bir egitim kosusunda olculur ve 'olculmeyen' alaninda "
            "listelenir. Etkin kaynak orani 1/epoch'tur; 'daha cok epoch = daha iyi' "
            "varsayimi bu oran raporlanmadan korlemesine yapilir."
        ),
        "butce": butce,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--olc", action="store_true")
    parser.add_argument("--kur", action="store_true")
    parser.add_argument("--dogrula", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    baslangic = time.time()
    butce = butce_olc()
    sure = time.time() - baslangic

    if args.dogrula:
        if not KAYIT.is_file():
            print(f"FINDING: {KAYIT.relative_to(ROOT)} yok: kayit yoksa butce de yoktur")
            return 1
        kayit = json.loads(KAYIT.read_text(encoding="utf-8"))
        kayitli = kayit["butce"]
        bulgular = list(butce["ihlaller"])
        if kayit.get("olcut", {}).get("sonuc") is not True:
            bulgular.append("kayit, ihlalsiz bir butce oldugunu soylemiyor")
        if kayitli.get("politika_surumu") != butce["politika_surumu"]:
            bulgular.append(
                f"politika surumu kayittan farkli: {kayitli.get('politika_surumu')} "
                f"-> {butce['politika_surumu']}"
            )
        # The budget is measured from the corpus, and the corpus grows with the
        # tree; what must hold is that the count did not *shrink* and that the
        # derived arithmetic still follows from it.
        if kayitli.get("benzersiz_jeton", 0) > butce["benzersiz_jeton"]:
            bulgular.append(
                f"benzersiz jeton kayittan az: {kayitli.get('benzersiz_jeton')} "
                f"-> {butce['benzersiz_jeton']}"
            )
        if butce["toplam_gecis"] != butce["benzersiz_jeton"] * butce["max_epochs"]:
            bulgular.append("toplam gecis benzersiz jeton x epoch'tan turetilmiyor")
        if bulgular:
            for b in bulgular:
                print(f"FINDING: {b}")
            return 1
        print(
            "egitim butcesi yeniden olculdu: "
            f"{butce['benzersiz_jeton']} benzersiz jeton ({butce['korpus_kayit_sayisi']} kayit), "
            f"{butce['max_epochs']} epoch -> {butce['toplam_gecis']} gecis, "
            f"etkin kaynak orani {butce['etkin_kaynak_orani']}, "
            f"jeton/param {butce['jeton_basina_param_tam_butce']}"
        )
        return 0

    if args.kur:
        KAYIT.parent.mkdir(parents=True, exist_ok=True)
        KAYIT.write_text(
            json.dumps(kayit_yaz(butce, sure), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "yazildi": str(KAYIT.relative_to(ROOT)),
            **{k: v for k, v in butce.items() if k != "olculmeyen"},
            "ihlaller": butce["ihlaller"],
        }, ensure_ascii=False, indent=2, sort_keys=True))
        return 1 if butce["ihlaller"] else 0

    print(json.dumps(butce, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if butce["ihlaller"] else 0


def self_test() -> int:
    """Canaries: an epoch count over the protocol ceiling, a weight decay out of
    range and a malformed policy must each be refused."""
    gercek = POLITIKA.read_text(encoding="utf-8")
    politika = json.loads(gercek)
    tavan = protokol_tavani()
    try:
        # Kanarya 1: protokol tavanini asan epoch reddedilmeli. Korpus
        # gerektirmeyen saf kural uzerinden: CI'da self-test adimi korpus
        # kurulmadan once kosar.
        asiri = json.loads(json.dumps(politika))
        asiri["max_epochs"] = tavan + 1
        if not politika_ihlalleri(asiri, tavan):
            raise SystemExit("self-test: protokol tavanini asan epoch yakalanmadi")
        # Kanarya 2: beyan edilen politika kendi tavanini asmamali.
        if politika_ihlalleri(politika, tavan):
            raise SystemExit("self-test: beyan edilen politika kendi tavanini asiyor")

        bozuk = json.loads(json.dumps(politika))
        bozuk["weight_decay"] = 1.5
        POLITIKA.write_text(json.dumps(bozuk, ensure_ascii=False), encoding="utf-8")
        try:
            politika_oku()
        except SystemExit:
            pass
        else:
            raise SystemExit("self-test: aralik disi weight_decay kabul edildi")

        eksik = json.loads(json.dumps(politika))
        del eksik["max_epochs"]
        POLITIKA.write_text(json.dumps(eksik, ensure_ascii=False), encoding="utf-8")
        try:
            politika_oku()
        except SystemExit:
            pass
        else:
            raise SystemExit("self-test: eksik alanli politika kabul edildi")
    finally:
        POLITIKA.write_text(gercek, encoding="utf-8")

    if not KORPUS.is_file():
        # CI sirasinda dogru yol: politika denetimi kosuldu, jeton olcumu
        # korpusdan sonra kapinin kendisinde kosuyor.
        print("self-test OK (korpus yok: politika kanaryalari kosuldu)")
        return 0
    if butce_olc()["ihlaller"]:
        raise SystemExit("self-test: beyan edilen politika kendi olcumunde ihlal uretiyor")
    if butce_olc()["benzersiz_jeton"] <= 0:
        raise SystemExit("self-test: olcum hic jeton saymadi")
    print("self-test OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
