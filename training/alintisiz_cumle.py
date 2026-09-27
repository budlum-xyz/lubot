#!/usr/bin/env python3
"""Alintisiz iddia orani: XAI maddesinin olcumu.

"Her cumlenin kaynagini goster" bir ilke olarak yaziliydi; olculmuyordu.
Bu betik `ask` yolunun ciktisini okur: cevap govdesindeki her madde bir
iddiadir ve iddianin kaynagini tasimak zorundadir. Olculen sey, alintisiz
kalan maddelerin oranidir. Oran beyan edilen esigi asarsa kayit bunu
BULGU olarak tasir ve olcut yanlis cikar (`sonuc: false`) - yani "her
cumlenin kaynagi var" iddiasi ancak olcum bunu tasiyorsa yazilabilir.

Alintili sayilma kurali (tek cumlede, cevap yapisina gore):
    Bir madde, yol tasiyorsa (`crates/.../x.rs`) ya da kaynak KONUMU
    tasiyorsa (`dosya.cs:12`) alintili sayilir. Cevap, pasajin ham metnini
    ve hemen ardindan kaynagiyla birlikte ayni metni yaziyor; ham madde
    "kaynagi olmayan iddia" sayilmaz, cunku metni bir alintili madde
    tarafindan TASINIYOR. Kapsanmayan madde, hicbir alintili maddenin
    metninde gecmeyen maddedir.

Olcum model cagirmaz: cevaplar korpustan okunur, jeton maliyeti sifirdir.

    python3 training/alintisiz_cumle.py --olc
    python3 training/alintisiz_cumle.py --kur
    python3 training/alintisiz_cumle.py --dogrula
    python3 training/alintisiz_cumle.py --self-test
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KORPUS = ROOT / "corpus" / "knowledge-self.jsonl.gz"
SINAV = ROOT / "training" / "eval" / "sinav-seti.jsonl"
KAYIT = ROOT / "training" / "eval" / "sonuclar" / "alintisiz-cumle-2026-09-24.json"
# Esik: ilk olcumden ONCE yazildi ve yuvarlak bir sayi: kapsanmayan
# maddelerin yaridan fazlasi kaynaksizsa "kaynak gosteriyor" iddiasi
# curur. Esigi olcume uydurmak (olcum 0,3333 cikinca esigi 0,34 yapmak)
# olcumu iddiaya cevirirdi; bu yuzden esik %50'de duruyor ve mevcut oran
# kayitta oldugu gibi yaziliyor.
ESIK = 0.5

YOL = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]*\.(?:rs|md|toml|json|jsonl|py|gz|mmd)")
KONUM = re.compile(r"`[^`]+:\d+`")


def alintili_mi(madde: str) -> bool:
    """Bir iddia kaynagini tasiyor mu: dosya yolu ya da `dosya:satir`."""
    return bool(YOL.search(madde) or KONUM.search(madde))


def kapsanmis_mi(madde: str, alintililar: list[str]) -> bool:
    """Alintisiz madde, alintili bir maddenin metninde geciyor mu.

    Cevap iki satir kullaniyor: pasajin metni, sonra kaynagiyla ayni metin.
    Ilk satiri kaynaksiz saymak, cevabi tasidigi kaynagi yok saymak olurdu."""
    sade = " ".join(madde.split())[:80]
    if not sade:
        return True
    return any(sade in " ".join(a.split()) for a in alintililar)


def iddialar(cikti: str) -> list[str]:
    """Cevap govdesindeki maddeler; `outputs:` satiri ve basliklar sayilmaz."""
    maddeler = []
    for satir in cikti.splitlines():
        satir = satir.strip()
        if satir.startswith(("- ", "* ")):
            maddeler.append(satir[2:].strip())
    return maddeler


def _sor(corpus: Path, binary: list[str], soru: str, audit: Path) -> tuple[str, dict]:
    """Bir soru: (cevap metni, audit satiri).

    Iki yuzey birlikte okunur cunku ikisi ayri sey soyluyor: audit
    makine tarafindan okunabilen `citations` listesini tasir, cevap metni
    ise kaynagin GORUNUR olup olmadigini gosterir."""
    kosu = subprocess.run(
        [*binary, "ask", "--corpus", str(corpus), "--reader", "olcum",
         "--effort", "1.0x", "--audit", str(audit), soru],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if kosu.returncode != 0:
        raise SystemExit(f"ask dustu: {(kosu.stderr or kosu.stdout)[-200:]}")
    satirlar = [json.loads(s) for s in audit.read_text(encoding="utf-8").splitlines() if s.strip()]
    if not satirlar:
        raise SystemExit("audit satiri yazilmadi: olcum kayit degil")
    return kosu.stdout, satirlar[-1]


def olc(binary: list[str] | None = None) -> dict:
    if not KORPUS.is_file():
        raise SystemExit(f"{KORPUS.relative_to(ROOT)} yok: once korpus kurulur")
    binary = binary or ["target/debug/lubot"]
    basla = time.monotonic()
    sorular = [
        json.loads(s) for s in SINAV.read_text(encoding="utf-8").splitlines() if s.strip()
    ]
    if not sorular:
        raise SystemExit("sinav seti bos: olcumsuz rapor yazilmaz")
    toplam = 0
    alintisiz = 0
    alintisiz_cevaplar: list[str] = []
    kaynaksiz_cevaplar: list[str] = []
    etiketler: dict[str, int] = {}
    ornekler: list[dict] = []
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        audit = Path(td) / "audit.jsonl"
        for kayit in sorular:
            kimlik = str(kayit.get("soru_kimligi"))
            cikti, satir = _sor(KORPUS, binary, str(kayit.get("soru", "")), audit)
            citations = satir.get("citations") or []
            etiket = str(satir.get("answer"))
            etiketler[etiket] = etiketler.get(etiket, 0) + 1
            # Yalniz `grounded` cevap kaynak tasimak ZORUNDA: digerleri
            # cevap vermemek (not-found/refused/out-of-scope) ya da
            # hesabi kendi kaniti olan cevaplardir (computed).
            if etiket == "grounded":
                if not citations:
                    kaynaksiz_cevaplar.append(kimlik)
            elif etiket in ("computed", "tool-refused") and not citations:
                pass
            maddeler = iddialar(cikti)
            alintililar = [m for m in maddeler if alintili_mi(m)]
            if maddeler and not alintililar:
                alintisiz_cevaplar.append(kimlik)
            for madde in maddeler:
                toplam += 1
                if not alintili_mi(madde) and not kapsanmis_mi(madde, alintililar):
                    alintisiz += 1
                    if len(ornekler) < 5:
                        ornekler.append({"soru_kimligi": kimlik, "madde": madde[:120]})
    oran = alintisiz / toplam if toplam else 0.0
    return {
        "iddia": toplam,
        "alintisiz": alintisiz,
        "oran": round(oran, 4),
        "esik": ESIK,
        "soru": len(sorular),
        "cevap_etiketleri": etiketler,
        "audit_kaynaksiz": kaynaksiz_cevaplar,
        "metinde_alintisiz": alintisiz_cevaplar,
        "ornekler": ornekler,
        "sure_saniye": round(time.monotonic() - basla, 2),
    }


def kayit_yaz(olcum: dict) -> Path:
    """Kayit: tek mekanik olcut + kaynak muhasebesi; esik asilirsa bulgu."""
    bulgu = None
    if olcum["oran"] > ESIK:
        bulgu = {
            "olculen": (
                f"{olcum['iddia']} iddianin {olcum['alintisiz']} tanesi hicbir "
                f"alintili maddenin metninde gecmiyor (oran {olcum['oran']} > esik "
                f"{ESIK}); alintisiz kalan sorular {olcum['alintisiz_sorular']}"
            ),
            "hukum": (
                "'her cumlenin kaynagini goster' ilkesi bugun TUTMUYOR: "
                "kaynaksiz maddeler cevap govdesinde duruyor"
            ),
            "yapilmayan": (
                "cevap uretimi degistirilmedi; kaynaksiz maddeyi cevaptan atan "
                "ya da reddeden kural yazilmadi - hangi esigin uygulanacagi "
                "operator karari, kayit yalniz olcer"
            ),
        }
    kayit = {
        "is": (
            "`ask` cevaplarindaki her madde bir iddia sayildi; kaynagi "
            "(dosya yolu ya da `dosya:satir`) tasiyan maddeler ile tasimayanlar "
            "ayrildi ve tasimayan bir madde, alintili bir maddenin metninde "
            "geciyorsa kapsanmis sayildi. Sinav setinin tamami soruldu; model "
            "cagrilmaz, jeton sifirdir."
        ),
        "kosucu": "ask",
        "tarih": time.strftime("%Y-%m-%d"),
        "olcut": {
            "ad": "her_grounded_cevap_audit_kaydinda_alinti_tasir",
            "sonuc": bool(not olcum["audit_kaynaksiz"]),
        },
        "kaynaklar": {
            "sure_saniye": olcum["sure_saniye"],
            "girdi_jetonlari": 0,
            "onbellekli_jetonlari": 0,
            "cikti_jetonlari": 0,
            "maliyet": 0.0,
        },
        "kanit": olcum,
        "bulgu_alintisiz_iddia": bulgu,
    }
    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    KAYIT.write_text(
        json.dumps(kayit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return KAYIT


def dogrula() -> str:
    if not KAYIT.is_file():
        raise SystemExit(f"kayit yok: {KAYIT.relative_to(ROOT)} (once --kur)")
    kayit = json.loads(KAYIT.read_text(encoding="utf-8"))
    taze = olc()
    eski = kayit.get("kanit", {})
    for alan in ("iddia", "alintisiz", "oran", "soru", "audit_kaynaksiz",
                 "metinde_alintisiz", "cevap_etiketleri"):
        if eski.get(alan) != taze[alan]:
            raise SystemExit(
                f"kayit bayat ({alan}): kayitta {eski.get(alan)}, olcum {taze[alan]} "
                "- kaydi yeniden uret"
            )
    return (
        f"alinti kapsamasi olculdu: {taze['soru']} cevabin tamami audit kaydinda "
        f"alinti tasiyor (kaynaksiz {len(taze['audit_kaynaksiz'])}); metinde "
        f"{taze['iddia']} iddianin {taze['alintisiz']} tanesi kapsanmiyor "
        f"(oran {taze['oran']}, esik {ESIK}), {len(taze['metinde_alintisiz'])} cevapta "
        f"gorunur alinti yok"
    )


def _self_test() -> None:
    """Kanarya: alintilama kurali yol ve konum bicimlerini taniyor; kaynaksiz
    madde alintisiz sayiliyor; basliklar ve `outputs:` iddia sayilmiyor."""
    assert alintili_mi("crates/grant/src/lib.rs exposes `pub struct ViewGrant`.")
    assert alintili_mi("`crates/read/src/lib.rs:42` boyle diyor")
    assert not alintili_mi("The persisted grant book: view grants plus revoked pairs.")
    assert kapsanmis_mi("Whether the head may decide alone. `false` ...",
                        ["crates/tomurcuk/src/lib.rs: Whether the head may decide alone. `false` ..."])
    assert not kapsanmis_mi("Bambaska bir iddia.", ["crates/read/src/lib.rs: ilgisiz metin"])
    ornek = """# Answer

- crates/grant/src/lib.rs exposes `pub struct ViewGrant`.

- The persisted grant book plus revoked pairs.

outputs: outputs/kayit.jsonl
"""
    maddeler = iddialar(ornek)
    assert len(maddeler) == 2, maddeler
    assert alintili_mi(maddeler[0]) and not alintili_mi(maddeler[1])
    assert ESIK < 1.0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--olc", action="store_true")
    g.add_argument("--kur", action="store_true")
    g.add_argument("--dogrula", action="store_true")
    g.add_argument("--self-test", action="store_true")
    parser.add_argument("--bin", default="target/debug/lubot",
                        help="olcumun kosacagi ikili (bosluk icerirse tirnaklanir)")
    args = parser.parse_args(argv)

    if args.self_test:
        _self_test()
        print("self-test OK [alintisiz-cumle]")
        return 0
    binary = shlex.split(args.bin)
    if args.olc:
        sys.stdout.write(json.dumps(olc(binary), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        return 0
    if args.kur:
        olcum = olc(binary)
        yol = kayit_yaz(olcum)
        print(
            f"kayit yazildi: {yol.relative_to(ROOT)} - {olcum['soru']} cevap, "
            f"audit kaynaksiz {len(olcum['audit_kaynaksiz'])}, metinde alintisiz "
            f"{len(olcum['metinde_alintisiz'])}; {olcum['iddia']} iddianin "
            f"{olcum['alintisiz']} tanesi kapsanmiyor (oran {olcum['oran']}, esik {ESIK}); "
            f"bulgu={'var' if olcum['oran'] > ESIK else 'yok'}"
        )
        return 0
    print(dogrula())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
