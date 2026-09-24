#!/usr/bin/env python3
"""Kimlik bilgisi bicimlerinin yakalama orani: iki katalog maddesinin olcumu.

Iki madde ayni soruyu soruyordu ve ikisi de "olculmuyor" diyordu:

* kimlik bilgisi bicimleri: `lubot guvenlik` tarayicisi bilinen bicimlere
  karsi ne kadar yakaliyor (yakalama orani);
* duzenli ifadeler: ayni desenlerin yanlis-pozitif/yanlis-negatif denetimi.

Fikstur CALISMA ANINDA uretilir ve gecici dizinde durur: agac, tam bir
kimlik bilgisi bicimi tasimaz. Tasimasaydi hem `no-secret-material` kapisi
hakli olarak kirmizi yanardi hem de tarayici kendini taramis olurdu -
tarayicinin kaynagini atlamasinin sebebi tam olarak bu.

Olcut tek cumlede:
    bilinen bicimlerin TAMAMI yakalanir (yakalama orani 1,0) ve
    yakalanmamasi gereken metinlerin HICBIRI yakalanmaz (yanlis-pozitif 0).

Ikisi birden saglanmazsa kayit bulgu tasir ve olcut yanlis cikar.

    python3 training/kimlik_bicimleri.py --olc
    python3 training/kimlik_bicimleri.py --kur
    python3 training/kimlik_bicimleri.py --dogrula
    python3 training/kimlik_bicimleri.py --self-test
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KAYIT = ROOT / "training" / "eval" / "sonuclar" / "kimlik-bicimleri-2026-09-24.json"

# Fikstur: (ad, satir, yakalanmali_mi). Bicimler parcalardan kurulur; bu
# dosyanin kendisi tam bir kimlik bilgisi tasimaz.
# Ozel anahtar basligi parcalardan kurulur: boylece dosyada tam bir anahtar
# blogu literal'i bulunmaz (token ornekleri de ayni sekilde uretilir). Aksi
# hâlde `no-secret-material` kapisi haklı olarak kirmizi yanardi: fikstur
# gercek bir bicim tasimali, ama kaynak dosya bir anahtar blogu tasimamali.
PEM_BASI = "-----BEGIN "
PEM_SONU = " PRIVATE KEY-----"


def fikstur() -> list[tuple[str, str, bool]]:
    A36 = "A" * 36
    B25 = "b" * 25
    C25 = "C" * 25
    D16 = "D" * 16
    E12 = "e" * 12
    return [
        # --- yakalanmasi gerekenler (bilinen bicimler) -------------------
        ("github-klasik", f"token = ghp_{A36}", True),
        ("github-klasik-oauth", f"oauth = gho_{A36}", True),
        ("github-ince-taneli", f"pat = github_pat_{B25}", True),
        ("model-api", f"key = sk-{C25}", True),
        ("aws-erisim", f"aws = AKIA{D16}", True),
        ("slack", f"slack = xoxb-{E12}", True),
        ("pem-rsa", PEM_BASI + "RSA" + PEM_SONU, True),
        ("pem-openssh", PEM_BASI + "OPENSSH" + PEM_SONU, True),
        ("gizli-satir-ici", f"Authorization: Bearer ghp_{A36} # yorum", True),
        ("kod-citi-icinde", f"```\\napi = \"sk-{C25}\"\\n```", True),
        # --- yakalanmamasi gerekenler (mention / ornek / yakin kacirma) --
        ("kisa-ghp", f"ghp_{'A' * 35}", False),
        ("kisa-sk", "sk-abc", False),
        ("kisa-akia", "AKIA1234", False),
        ("kisa-xoxb", "xoxb-abc", False),
        ("duz-metin", "the key is in the vault", False),
        ("onek-anlatimi", "the ghp_ prefix marks a GitHub classic token", False),
        ("dokuman-ornegi", "`ghp_` with nine characters is a documentation example", False),
        ("kisa-model-api", f"sk-{'a' * 19}", False),
        ("buyuk-harf-onek", f"GHP_{A36}", False),
        ("ozel-anahtar-degil", "-----BEGIN PUBLIC KEY-----", False),
        ("yalniz-hex", "digest = " + "9f" * 32, False),
        ("mektup-eki", "-----BEGIN CERTIFICATE-----", False),
    ]


def _tara(hedef: Path, binary: list[str]) -> tuple[bool, str]:
    """`guvenlik --path` kosar: (temiz_mi, cikti)."""
    kosu = subprocess.run(
        [*binary, "guvenlik", "--path", str(hedef)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if kosu.returncode not in (0, 1):
        raise SystemExit(f"guvenlik dustu ({kosu.returncode}): {(kosu.stderr or kosu.stdout)[-200:]}")
    return kosu.returncode == 0, kosu.stdout + kosu.stderr


def olc(binary: list[str] | None = None) -> dict:
    binary = binary or ["target/debug/lubot"]
    basla = time.monotonic()
    kayitlar = fikstur()
    yakalanan = 0
    kacirilan: list[str] = []
    yanlis_pozitif: list[str] = []
    temiz_gecen = 0
    with tempfile.TemporaryDirectory() as td:
        kok = Path(td)
        for ad, satir, bekleniyor in kayitlar:
            dosya = kok / f"{ad}.txt"
            dosya.write_text(satir + "\n", encoding="utf-8")
            temiz, _ = _tara(dosya, binary)
            yakalandi = not temiz
            if bekleniyor:
                if yakalandi:
                    yakalanan += 1
                else:
                    kacirilan.append(ad)
            elif yakalandi:
                yanlis_pozitif.append(ad)
            else:
                temiz_gecen += 1
    pozitif = sum(1 for _, _, b in kayitlar if b)
    negatif = len(kayitlar) - pozitif
    yakalama_orani = yakalanan / pozitif if pozitif else 0.0
    yanlis_pozitif_orani = len(yanlis_pozitif) / negatif if negatif else 0.0
    return {
        "bicim": pozitif,
        "yakalanan": yakalanan,
        "yakalama_orani": round(yakalama_orani, 4),
        "kacirilan": kacirilan,
        "temiz_metin": negatif,
        "temiz_gecen": temiz_gecen,
        "yanlis_pozitif": yanlis_pozitif,
        "yanlis_pozitif_orani": round(yanlis_pozitif_orani, 4),
        "sure_saniye": round(time.monotonic() - basla, 2),
        "fikstur_dosyada_yok": True,
    }


def kayit_yaz(olcum: dict) -> Path:
    tam = olcum["yakalanan"] == olcum["bicim"] and not olcum["yanlis_pozitif"]
    bulgu = None
    if olcum["kacirilan"]:
        bulgu = {
            "olculen": (
                f"{olcum['bicim']} bilinen bicimden {len(olcum['kacirilan'])} tanesi "
                f"yakalanmadi: {', '.join(olcum['kacirilan'])}"
            ),
            "hukum": "tarayici bilinen bicimlerin tamamini yakalayamiyor",
            "yapilmayan": "desenler degistirilmedi; kayit yalniz olcer",
        }
    elif olcum["yanlis_pozitif"]:
        bulgu = {
            "olculen": (
                f"{olcum['temiz_metin']} temiz metinden {len(olcum['yanlis_pozitif'])} "
                f"tanesi yanlis yakalandi: {', '.join(olcum['yanlis_pozitif'])}"
            ),
            "hukum": "tarayici temiz metni isaretliyor: gurultu, gercek alarmi gomuyor",
            "yapilmayan": "desenler degistirilmedi; kayit yalniz olcer",
        }
    kayit = {
        "is": (
            "Kimlik bilgisi tarayicisi, calisma aninda uretilen fikstur uzerinde "
            "olculdu: bilinen bicimler yakalanmali, mention ve yakin kacirmalar "
            "yakalanmamali. Fikstur agacta durmaz (aks halde kapi hakli olarak "
            "kirmizi yanardi); dosyalar gecici dizinde uretilir ve silinir."
        ),
        "kosucu": "guvenlik",
        "tarih": time.strftime("%Y-%m-%d"),
        "olcut": {
            "ad": "bilinen_bicimlerin_tamami_yakalandi_ve_yanlis_pozitif_sifir",
            "sonuc": bool(tam),
        },
        "kaynaklar": {
            "sure_saniye": olcum["sure_saniye"],
            "girdi_jetonlari": 0,
            "onbellekli_jetonlari": 0,
            "cikti_jetonlari": 0,
            "maliyet": 0.0,
        },
        "kanit": olcum,
        "bulgu": bulgu,
    }
    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    KAYIT.write_text(
        json.dumps(kayit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return KAYIT


def _bulgu(kayit: dict, taze: dict) -> str | None:
    """Kaydin semasi ve tazeligi (saf fonksiyon; kapi da bunu kullanir)."""
    kanit = kayit.get("kanit")
    if not isinstance(kanit, dict):
        return "kayit kanit tasimiyor"
    for alan in ("bicim", "yakalanan", "temiz_metin", "temiz_gecen"):
        if not isinstance(kanit.get(alan), int) or kanit[alan] < 0:
            return f"{alan} sayi degil"
    if kanit["bicim"] <= 0 or kanit["temiz_metin"] <= 0:
        return "fikstur iki yonden de dolu olmali: yalniz pozitif ya da yalniz negatif olcum degil"
    if kanit.get("yakalama_orani") != round(kanit["yakalanan"] / kanit["bicim"], 4):
        return "yakalama_orani sayilarla tutusmuyor"
    if kanit.get("yanlis_pozitif_orani") != round(
        len(kanit.get("yanlis_pozitif", [])) / kanit["temiz_metin"], 4
    ):
        return "yanlis_pozitif_orani sayilarla tutusmuyor"
    if kanit.get("kacirilan") and kanit["yakalanan"] + len(kanit["kacirilan"]) != kanit["bicim"]:
        return "kacirilan listesi sayilarla tutusmuyor"
    sonuc = kayit.get("olcut", {}).get("sonuc")
    beklenen = not kanit.get("kacirilan") and not kanit.get("yanlis_pozitif")
    if sonuc is not bool(beklenen):
        return f"olcut tutarsiz: kacirilan {len(kanit.get('kacirilan', []))}, yanlis pozitif {len(kanit.get('yanlis_pozitif', []))}"
    for alan in ("bicim", "yakalanan", "yakalama_orani", "kacirilan", "temiz_metin",
                 "temiz_gecen", "yanlis_pozitif", "yanlis_pozitif_orani"):
        if kanit.get(alan) != taze.get(alan):
            return (
                f"kayit bayat ({alan}): kayitta {kanit.get(alan)}, olcum {taze.get(alan)} "
                "- kaydi yeniden uret"
            )
    return None


def dogrula() -> str:
    if not KAYIT.is_file():
        raise SystemExit(f"kayit yok: {KAYIT.relative_to(ROOT)} (once --kur)")
    kayit = json.loads(KAYIT.read_text(encoding="utf-8"))
    taze = olc()
    bulgu = _bulgu(kayit, taze)
    if bulgu:
        raise SystemExit(bulgu)
    return (
        f"kimlik bicimleri olculdu: {taze['yakalanan']}/{taze['bicim']} bilinen bicim "
        f"yakalandi (oran {taze['yakalama_orani']}), {taze['temiz_gecen']}/{taze['temiz_metin']} "
        f"temiz metin dokunulmadan gecti (yanlis pozitif {len(taze['yanlis_pozitif'])})"
    )


def _self_test() -> None:
    """Kanarya: sayim fonksiyonlari bir kacirmayi ve bir yanlis-pozitifi
    ayri ayri gorur; kayit semasi tutarsizsa dogrulama duser."""
    kacan = {"bicim": 4, "yakalanan": 3, "yakalama_orani": 0.75, "kacirilan": ["x"],
             "temiz_metin": 3, "temiz_gecen": 3, "yanlis_pozitif": [],
             "yanlis_pozitif_orani": 0.0}
    kayit = {"olcut": {"sonuc": False}, "kanit": dict(kacan)}
    assert _bulgu(kayit, kacan) is None, "gecerli kacirma kaydi reddedildi"
    yanlis_bayrak = {"olcut": {"sonuc": True}, "kanit": dict(kacan)}
    assert "olcut tutarsiz" in (_bulgu(yanlis_bayrak, kacan) or ""), "yanlis bayrak gecti"
    fp = {"bicim": 4, "yakalanan": 4, "yakalama_orani": 1.0, "kacirilan": [],
          "temiz_metin": 3, "temiz_gecen": 2, "yanlis_pozitif": ["y"],
          "yanlis_pozitif_orani": 0.3333}
    assert _bulgu({"olcut": {"sonuc": False}, "kanit": fp}, fp) is None, "fp kaydi reddedildi"
    # Kendi icinde tutarli, ama taze olcumden farkli: bayatlik boyle gorunur.
    bayat_kanit = {"bicim": 4, "yakalanan": 2, "yakalama_orani": 0.5,
                   "kacirilan": ["x", "y"], "temiz_metin": 3, "temiz_gecen": 3,
                   "yanlis_pozitif": [], "yanlis_pozitif_orani": 0.0}
    bayat = {"olcut": {"sonuc": False}, "kanit": bayat_kanit}
    assert "bayat" in (_bulgu(bayat, kacan) or ""), "bayat sayi gecti"
    assert any(bekleniyor for _, _, bekleniyor in fikstur()), "fiksturde pozitif yok"
    assert any(not b for _, _, b in fikstur()), "fiksturde negatif yok"


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
        print("self-test OK [kimlik-bicimleri]")
        return 0
    binary = shlex.split(args.bin)
    if not binary:
        raise SystemExit("--bin bos")
    if args.olc:
        sys.stdout.write(json.dumps(olc(binary), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        return 0
    if args.kur:
        olcum = olc(binary)
        yol = kayit_yaz(olcum)
        print(
            f"kayit yazildi: {yol.relative_to(ROOT)} - {olcum['yakalanan']}/{olcum['bicim']} "
            f"yakalandi (oran {olcum['yakalama_orani']}), yanlis pozitif "
            f"{len(olcum['yanlis_pozitif'])}/{olcum['temiz_metin']}; kacirilan "
            f"{olcum['kacirilan']}"
        )
        return 0
    print(dogrula())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
