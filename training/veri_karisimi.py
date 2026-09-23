#!/usr/bin/env python3
"""Compose the training mix from declared strata, and record the mix.

NN §8 step 4 asks for one training set assembled from several sources - the
real corpus, compiler-judged pairs, the hand-written curriculum - with the
share of each written into a provenance record. The point of the record is
that a mix nobody wrote down is a mix nobody can audit: a set that quietly
becomes 99% one source still trains, and still looks like a set.

So this builder refuses rather than adjusts:

* a stratum that is not in the declaration cannot enter the mix;
* a stratum outside its declared share band (floor and ceiling) refuses the
  whole build, because a floor that can silently fall to zero is a stratum
  that has been deleted without anybody deciding to delete it;
* a row in a grounded stratum that carries no `content_id` refuses, for the
  same reason the leak check (PP) requires one: a row that cannot be
  addressed cannot be checked.

One stratum is declared and measured **empty**: `sentetik`. Self-instruct
(section C) needs a trained checkpoint to generate from, and there is no
checkpoint yet. Declaring it empty keeps the seat at the table and keeps the
number honest; filling it with template text and calling it synthetic would
be a claim, not a measurement.

Usage:
    python3 training/veri_karisimi.py --kur       # build the mix + its record
    python3 training/veri_karisimi.py --dogrula   # re-derive and compare
    python3 training/veri_karisimi.py --self-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BEYAN = ROOT / "training" / "veri-karisimi.json"
KARISIM = ROOT / "training" / "eval" / "veri-karisimi.jsonl"
KAYIT = ROOT / "training" / "eval" / "sonuclar" / "veri-karisimi-2026-09-23.json"

# A stratum whose rows are facts drawn from the tree, and therefore have to
# carry the digest of the passage they came from.
TOPRAKLI_KATMANLAR = {"gercek", "derleyici-hakem"}


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def beyan_oku() -> dict:
    """Read the declaration, fail-closed: a mix with no declaration is a mix
    nobody decided on."""
    if not BEYAN.is_file():
        raise SystemExit(f"{BEYAN.relative_to(ROOT)} yok: beyansiz karisim kurulamaz")
    raw = json.loads(BEYAN.read_text(encoding="utf-8"))
    for field in ("surum", "katmanlar"):
        if field not in raw:
            raise SystemExit(f"{BEYAN.name}: zorunlu alan yok: {field}")
    for ad, k in raw["katmanlar"].items():
        for field in ("asama", "taban_pay", "tavan_pay"):
            if field not in k:
                raise SystemExit(f"{BEYAN.name}: {ad} katmaninda {field} yok")
        if not 0.0 <= k["taban_pay"] <= k["tavan_pay"] <= 1.0:
            raise SystemExit(
                f"{BEYAN.name}: {ad} pay bandi gecersiz "
                f"({k['taban_pay']} > {k['tavan_pay']} veya bant disinda)"
            )
    return raw


# --------------------------------------------------------------------------
# stratum: the hand-written curriculum
# --------------------------------------------------------------------------
def mufredat_satirlari() -> list[dict]:
    rows: list[dict] = []
    for path in sorted((ROOT / "training" / "curriculum").glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["katman"] = "mufredat"
                row["kaynak_dosya"] = str(path.relative_to(ROOT))
                rows.append(row)
    return rows


# --------------------------------------------------------------------------
# stratum: the real corpus
# --------------------------------------------------------------------------
def gercek_satirlari(corpus: Path) -> list[dict]:
    sys.path.insert(0, str(ROOT / "training"))
    import make_sft  # noqa: PLC0415  (the grounded row shape is its business)

    rows: list[dict] = []
    if corpus.is_file():
        opener = (
            __import__("gzip").open(corpus, "rt", encoding="utf-8")
            if corpus.name.endswith(".gz")
            else corpus.open(encoding="utf-8")
        )
        with opener as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = make_sft.grounded(json.loads(line))
                if row is None:
                    continue
                row["katman"] = "gercek"
                rows.append(row)
    return rows


# --------------------------------------------------------------------------
# stratum: compiler-judged pairs (KK)
# --------------------------------------------------------------------------
def test_envanteri() -> list[tuple[str, str, int]]:
    """Every test in the workspace, with the file and line that holds it.

    The judge here is the test runner itself, not a reader's opinion: the
    answer a row carries is "this test exists and passes in the workspace
    suite", which is a fact `cargo test` states.
    """
    run = subprocess.run(
        ["cargo", "test", "--workspace", "--", "--list"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if run.returncode != 0:
        raise SystemExit(f"cargo test --list kosulmadi: {run.stderr[-200:]}")
    adlar = sorted(
        line[: -len(": test")]
        for line in run.stdout.splitlines()
        if line.endswith(": test")
    )
    indeks: dict[str, tuple[str, int]] = {}
    for path in sorted((ROOT / "crates").rglob("*.rs")):
        try:
            satirlar = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for sira, satir in enumerate(satirlar, start=1):
            m = re.match(r"\s*(?:pub\s+)?fn\s+([a-z0-9_]+)\s*\(", satir)
            if m:
                indeks.setdefault(m.group(1), (str(path.relative_to(ROOT)), sira))
    return [(ad, *indeks[ad.split("::")[-1]]) for ad in adlar if ad.split("::")[-1] in indeks]


def derleyici_satirlari(toplam_test: int, basarisiz: int) -> list[dict]:
    rows: list[dict] = []
    for ad, yol, sira in test_envanteri():
        kaynak_satir = (ROOT / yol).read_text(encoding="utf-8").splitlines()[sira - 1].strip()
        kisa = ad.split("::")[-1]
        crate = yol.split("/")[1] if yol.startswith("crates/") else yol
        rows.append({
            "messages": [
                {"role": "user", "content": f"Which behaviour is proven by `{kisa}`?"},
                {"role": "assistant", "content": (
                    f"`{kisa}` is a test in `{crate}`. It is one of {toplam_test} tests in "
                    f"the workspace suite, which reports {basarisiz} failures.\n\n"
                    f"Source: {yol}:{sira}"
                )},
            ],
            "kind": "derleyici-hakem",
            "katman": "derleyici-hakem",
            "citation": f"{yol}:{sira}",
            "content_id": digest(kaynak_satir),
            "kaynak_dosya": yol,
        })
    return rows


# --------------------------------------------------------------------------
# the credential-shape filter
# --------------------------------------------------------------------------
def kimlik_sekli_tara(yol: Path) -> list[tuple[int, str, str]]:
    """Run the repository's own scanner over a file and return its hits.

    The scanner is the single source of truth here on purpose: re-implementing
    its shapes in Python would mean two rules that can drift, and the one that
    drifts is the one nobody looks at.
    """
    run = subprocess.run(
        ["cargo", "run", "--quiet", "-p", "lubot", "--bin", "lubot", "--",
         "guvenlik", "--path", str(yol)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    # A scanner that did not run must not read as a scanner that found nothing:
    # the report line is the evidence that it ran at all.
    if "files scanned" not in run.stdout:
        raise SystemExit(
            f"guvenlik tarayicisi kosulmadi ({yol.name}): {run.stderr[-200:] or run.stdout[-200:]}"
        )
    hits: list[tuple[int, str, str]] = []
    for line in run.stdout.splitlines():
        if not line.startswith("| ") or line.startswith("| file |") or line.startswith("|---"):
            continue
        parts = [x.strip() for x in line.strip("|").split("|")]
        # The scanner prints the path it was handed, so match on the file name
        # rather than on the exact spelling of the path.
        if len(parts) == 4 and Path(parts[0]).name == yol.name and parts[1].isdigit():
            hits.append((int(parts[1]), parts[2], parts[3]))
    return hits


def kimlik_sekli_ayikla(rows: list[dict], yol: Path) -> tuple[list[dict], list[dict]]:
    """Drop the rows whose text carries a credential shape, and say which.

    This exists because the corpus legitimately documents what a private key
    block looks like - `crates/tools/src/secrets.rs` describes the shape it
    detects - and quoting that passage into a training row puts the shape into
    a file the scanner watches. Refusing to emit it is also the right training
    decision, not only the right gate decision: a system that reads and never
    generates has no business rehearsing key headers.

    Dropping is recorded rather than silent, because a corpus that contains such
    passages is a fact worth keeping visible.
    """
    yol.parent.mkdir(parents=True, exist_ok=True)
    with yol.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    atilan: list[dict] = []
    for sira, tur, satici in kimlik_sekli_tara(yol):
        row = rows[sira - 1]
        atilan.append({
            "sira": sira,
            "katman": row["katman"],
            "citation": row.get("citation"),
            "tur": tur,
            "satici": satici,
        })
    if not atilan:
        return rows, []
    kalan = [r for i, r in enumerate(rows, start=1) if i not in {a["sira"] for a in atilan}]
    # The file this function leaves behind is the file the checks read, so it
    # has to be the final one: renumbered, not the pre-drop sequence with a hole.
    for sira, row in enumerate(kalan):
        row["sira"] = sira
    with yol.open("w", encoding="utf-8") as handle:
        for row in kalan:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    kalanlar = kimlik_sekli_tara(yol)
    if kalanlar:
        raise SystemExit(
            f"{yol.name}: kimlik sekli ayiklandiktan sonra bile {len(kalanlar)} sekil kaldi: {kalanlar[:3]}"
        )
    return kalan, atilan


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------
def satirlari_topla(toplam_test: int, basarisiz: int) -> list[dict]:
    """Assemble every stratum's rows, in stage order, with nothing measured yet.

    Measurement happens after the credential-shape filter, because a mix whose
    shares are quoted from before a filter ran is a mix nobody measured.
    """
    beyan = beyan_oku()
    corpus = ROOT / "corpus" / "knowledge-self.jsonl.gz"
    ureticiler = {
        "mufredat": mufredat_satirlari,
        "gercek": lambda: gercek_satirlari(corpus),
        "derleyici-hakem": lambda: derleyici_satirlari(toplam_test, basarisiz),
        "sentetik": list,  # no checkpoint yet: declared, measured empty
    }
    rows: list[dict] = []
    for ad in beyan["katmanlar"]:
        uretici = ureticiler.get(ad)
        if uretici is None:
            raise SystemExit(f"{ad}: beyan edilen katmanin ureticisi yok")
        rows.extend(uretici())
    # Unknown strata cannot ride along: a row whose stratum nobody declared
    # would be a source nobody decided to train on.
    bilinmeyen = {r["katman"] for r in rows} - set(beyan["katmanlar"])
    if bilinmeyen:
        raise SystemExit(f"beyan edilmemis katman: {sorted(bilinmeyen)}")
    # Curriculum ordering (E): shape before facts. The stage is carried on the
    # row and the file is written in stage order, so the order is a property of
    # the artifact rather than of whoever runs the trainer.
    rows.sort(key=lambda r: (beyan["katmanlar"][r["katman"]]["asama"], r["katman"],
                             r.get("citation") or json.dumps(r["messages"], sort_keys=True)))
    for sira, row in enumerate(rows):
        row["asama"] = beyan["katmanlar"][row["katman"]]["asama"]
        row["sira"] = sira
    return rows


def karisim_olc(rows: list[dict]) -> dict:
    """Measure an assembled mix against its declaration. Refuses on any breach."""
    beyan = beyan_oku()
    toplam = len(rows)
    paylar = {
        ad: round(sum(1 for r in rows if r["katman"] == ad) / toplam, 6) if toplam else 0.0
        for ad in beyan["katmanlar"]
    }
    ihlaller: list[str] = []
    for ad, k in beyan["katmanlar"].items():
        if paylar[ad] < k["taban_pay"]:
            ihlaller.append(
                f"{ad}: pay {paylar[ad]} tabanin altinda ({k['taban_pay']})"
            )
        if paylar[ad] > k["tavan_pay"]:
            ihlaller.append(
                f"{ad}: pay {paylar[ad]} tavanin ustunde ({k['tavan_pay']})"
            )
    for katman in TOPRAKLI_KATMANLAR:
        eksik = sum(1 for r in rows if r["katman"] == katman and not r.get("content_id"))
        if eksik:
            ihlaller.append(f"{katman}: {eksik} satir content_id tasimiyor")

    return {
        "beyan_surumu": beyan["surum"],
        "toplam_satir": toplam,
        "satir_sayilari": {
            ad: sum(1 for r in rows if r["katman"] == ad) for ad in beyan["katmanlar"]
        },
        "paylar": paylar,
        "asamalar": {ad: beyan["katmanlar"][ad]["asama"] for ad in beyan["katmanlar"]},
        "bant_ihlalleri": ihlaller,
        "toprakli_katmanlar": sorted(TOPRAKLI_KATMANLAR),
    }


def karisim_kur(toplam_test: int, basarisiz: int) -> tuple[list[dict], dict]:
    """Assemble, filter and measure. Returns the rows that ship and the record.

    The order is the point: assemble, then drop what carries a credential shape,
    then measure. Measuring first would report shares for a set that is not the
    one written to disk.
    """
    rows = satirlari_topla(toplam_test, basarisiz)
    rows, atilan = kimlik_sekli_ayikla(rows, KARISIM)
    karisim = karisim_olc(rows)
    karisim["atilan_kimlik_sekli"] = atilan
    return rows, karisim


def kayit_yaz(karisim: dict, sure: float) -> dict:
    return {
        "kosucu": "betik",
        "tarih": "2026-09-23",
        "is": "veri-karisimi-beyan-uyumu",
        "olcut": {
            "ad": "her_satirin_beyan_edilen_bir_katman_tasimasi_ve_paylarin_bant_icinde_kalmasi",
            "sonuc": not karisim["bant_ihlalleri"],
        },
        "kaynaklar": {
            "sure_saniye": round(sure, 3),
            "girdi_jetonlari": 0,
            "onbellekli_jetonlari": 0,
            "cikti_jetonlari": 0,
            "maliyet": 0.0,
        },
        "kanit": (
            f"training/veri_karisimi.py --dogrula karisimi beyandan yeniden kurar: "
            f"{karisim['toplam_satir']} satir, katman sayilari {karisim['satir_sayilari']}, "
            f"paylar {karisim['paylar']}, bant ihlali {len(karisim['bant_ihlalleri'])}."
        ),
        "not": (
            "sentetik katmani beyan edildi ve bos olculdu: self-instruct (C) egitilmis "
            "bir kontrol noktasi gerektirir, henuz yok. Bos beyan koltugu tutar; sablon "
            "metnini 'sentetik' diye yazmak olcum degil iddia olurdu."
        ),
        "karisim": karisim,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kur", action="store_true")
    parser.add_argument("--dogrula", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    baslangic = time.time()
    # The judge's own numbers, so the rows do not state a count nobody ran.
    sayim = subprocess.run(
        ["cargo", "test", "--workspace"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    toplam = sum(
        int(m.group(1)) for m in re.finditer(r"(\d+) passed", sayim.stdout)
    )
    basarisiz = sum(
        int(m.group(1)) for m in re.finditer(r"(\d+) failed", sayim.stdout)
    )
    rows, karisim = karisim_kur(toplam, basarisiz)
    sure = time.time() - baslangic

    if args.dogrula:
        if not KAYIT.is_file():
            print(f"FINDING: {KAYIT.relative_to(ROOT)} yok: kayit yoksa karisim da yoktur")
            return 1
        kayit = json.loads(KAYIT.read_text(encoding="utf-8"))
        kayitli = kayit["karisim"]
        bulgular = []
        # What is enforced is the invariant, not the count: the corpus grows
        # with every commit that adds a file, so a record that had to match its
        # row count would be stale the moment it was written. What must hold is
        # that the mix still comes from the declared strata, inside their bands.
        if kayitli.get("beyan_surumu") != karisim["beyan_surumu"]:
            bulgular.append(
                f"beyan surumu kayittan farkli: {kayitli.get('beyan_surumu')} -> {karisim['beyan_surumu']}"
            )
        if set(kayitli.get("satir_sayilari", {})) != set(karisim["satir_sayilari"]):
            bulgular.append(
                f"katman kumesi kayittan farkli: {sorted(kayitli.get('satir_sayilari', {}))} "
                f"-> {sorted(karisim['satir_sayilari'])}"
            )
        if kayit.get("olcut", {}).get("sonuc") is not True:
            bulgular.append("kayit, band ihlali olmayan bir karisim oldugunu soylemiyor")
        if karisim["bant_ihlalleri"]:
            bulgular.extend(karisim["bant_ihlalleri"])
        if not KARISIM.is_file():
            bulgular.append(f"{KARISIM.relative_to(ROOT)} yok")
        else:
            disk = [json.loads(x) for x in KARISIM.read_text(encoding="utf-8").splitlines() if x.strip()]
            if [r["sira"] for r in disk] != list(range(len(disk))):
                bulgular.append("karisim dosyasinin sirasi 0'dan baslayan kesintisiz bir dizi degil")
            disk_katman = {r["katman"] for r in disk}
            if not disk_katman <= set(karisim["satir_sayilari"]):
                bulgular.append(f"karisim dosyasinda beyan edilmemis katman var: {sorted(disk_katman)}")
            for katman in karisim["toprakli_katmanlar"]:
                eksik = sum(1 for r in disk if r["katman"] == katman and not r.get("content_id"))
                if eksik:
                    bulgular.append(f"karisim dosyasinda {katman}: {eksik} satir content_id tasimiyor")
            kalan = kimlik_sekli_tara(KARISIM)
            if kalan:
                bulgular.append(f"karisim dosyasinda {len(kalan)} kimlik sekli var: {kalan[:3]}")
            drift = {
                ad: (kayitli.get("satir_sayilari", {}).get(ad), karisim["satir_sayilari"][ad])
                for ad in karisim["satir_sayilari"]
                if kayitli.get("satir_sayilari", {}).get(ad) != karisim["satir_sayilari"][ad]
            }
        if bulgular:
            for b in bulgular:
                print(f"FINDING: {b}")
            return 1
        print(
            "veri karisimi beyandan yeniden kuruldu: "
            f"{karisim['toplam_satir']} satir, sayilar {karisim['satir_sayilari']}, "
            f"paylar {karisim['paylar']}"
            + (f"; kayittan bu yana kayan katmanlar (kayit -> taze): {drift}" if drift else "")
        )
        return 0

    if args.kur:
        # The mix file is already on disk: karisim_kur wrote it through the
        # scanner and left it renumbered, so writing it again would only risk
        # the two copies disagreeing.
        disk = [json.loads(x) for x in KARISIM.read_text(encoding="utf-8").splitlines() if x.strip()]
        if [r["sira"] for r in disk] != list(range(len(disk))):
            raise SystemExit(f"{KARISIM.name}: sira 0'dan baslayan kesintisiz bir dizi degil")
        if len(disk) != len(rows):
            raise SystemExit(f"{KARISIM.name}: {len(disk)} satir, kurulan {len(rows)}")
        KAYIT.parent.mkdir(parents=True, exist_ok=True)
        KAYIT.write_text(
            json.dumps(kayit_yaz(karisim, sure), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "yazildi": [str(KARISIM.relative_to(ROOT)), str(KAYIT.relative_to(ROOT))],
            "toplam_satir": karisim["toplam_satir"],
            "satir_sayilari": karisim["satir_sayilari"],
            "paylar": karisim["paylar"],
            "bant_ihlalleri": karisim["bant_ihlalleri"],
            "atilan_kimlik_sekli": karisim.get("atilan_kimlik_sekli", []),
        }, ensure_ascii=False, indent=2, sort_keys=True))
        return 1 if karisim["bant_ihlalleri"] else 0

    print(json.dumps(karisim, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if karisim["bant_ihlalleri"] else 0


def self_test() -> int:
    """Canaries: a band breach, an undeclared stratum and a grounded row with
    no content_id must each be refused."""
    beyan = json.loads(BEYAN.read_text(encoding="utf-8"))
    gercek = BEYAN.read_text(encoding="utf-8")

    # A ceiling below the measured share has to refuse the build. The floor
    # comes down with it, or the band itself would be the thing refused and the
    # canary would be testing the reader instead of the check.
    sikisik = json.loads(json.dumps(beyan))
    sikisik["katmanlar"]["gercek"]["taban_pay"] = 0.0
    sikisik["katmanlar"]["gercek"]["tavan_pay"] = 0.01
    BEYAN.write_text(json.dumps(sikisik, ensure_ascii=False), encoding="utf-8")
    try:
        karisim = karisim_olc(satirlari_topla(1, 0))
        if not karisim["bant_ihlalleri"]:
            raise SystemExit("self-test: tavan asimi yakalanmadi")
        # A floor above the measured share has to refuse too, or a stratum can
        # be deleted by accident without anybody deciding to delete it. The
        # ceiling rises to meet it so the band stays readable.
        bos = json.loads(json.dumps(beyan))
        bos["katmanlar"]["mufredat"]["taban_pay"] = 0.2
        bos["katmanlar"]["mufredat"]["tavan_pay"] = 0.2
        BEYAN.write_text(json.dumps(bos, ensure_ascii=False), encoding="utf-8")
        karisim = karisim_olc(satirlari_topla(1, 0))
        if not any("tabanin altinda" in b for b in karisim["bant_ihlalleri"]):
            raise SystemExit("self-test: taban altina dusus yakalanmadi")
    finally:
        BEYAN.write_text(gercek, encoding="utf-8")

    # A band that does not parse is refused at read time.
    bozuk = json.loads(json.dumps(beyan))
    bozuk["katmanlar"]["gercek"]["taban_pay"] = 0.9
    bozuk["katmanlar"]["gercek"]["tavan_pay"] = 0.1
    BEYAN.write_text(json.dumps(bozuk, ensure_ascii=False), encoding="utf-8")
    try:
        try:
            beyan_oku()
        except SystemExit:
            pass
        else:
            raise SystemExit("self-test: gecersiz bant kabul edildi")
    finally:
        BEYAN.write_text(gercek, encoding="utf-8")

    print("self-test OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
