#!/usr/bin/env python3
"""System One arka ucu: yerel Laya (Apache-2.0) ve uzak Jev (anahtarli) tek arayuzde.

Neden servis: karar basina model yuklemek (bu makinede ~4,3 s) karari
yavaslatir. Sunucu modeli **bir kez** yukler ve bellekte tutar; her `/oy`
istegi yalniz bir ileri gecis yapar. Olcum (2 vCPU, bf16, multilingual):
yukleme ~4,3 s | karar 1,5-3,8 s (Ingilizce ~10 s) | tepe RSS ~1,5 GB.

Bellek dersi (olculdu): kita Laya'yi fp32 yukler; 421M parametre ~1,7 GB eder
ve 1,94 GiB RAM'li bu makinede OOM ile olur. Agirliklar f16 saklanir, model
bf16'da kurulur. fp32 secenegi bilerek yoktur: bu makinede calismaz.

Arayuz:

    GET  /saglik          -> {"durum": "hazir", "arka_uc": ..., "yukleme_ms": ...}
    POST /oy              -> {"oy": "tut"|"at", "guven": 0.0-1.0, "arka_uc": ..., "gecikme_ms": ...}
      govde: {"kart": "tut_at", "soru": "...", "secenekler": ["tut", "at"]}

Sunucu bir **oy** dondurur; esik ve karar kurali `autonomous-training/danisma.py`
ile `dongu.py`'de yazilidir. Anahtar yalnizca ortam degiskeninden okunur
(`TYPESAFE_API_KEY`); hicbir zaman dosyaya yazilmaz ve kayda gecmez.

Kullanim:

    python3 training/danisma/sunucu.py --port 8790 --arka-uc otomatik
    python3 training/danisma/sunucu.py --kendini-test      # model yuklemeden
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BURASI = Path(__file__).resolve().parent
KARTLAR = BURASI / "kararlar.json"
JEK_UC = "https://api.typesafe.ai/v1/systemone"
JEK_MODEL = "jev-1.13.0"
CHECKPOINT = "convaiinnovations/laya"
ALT_KLASOR = "multilingual"
KOSU_DTYPE = "bfloat16"

_AGENT = None
_KILIT = threading.Lock()


def kart_ayari(kart: str) -> dict:
    return json.loads(KARTLAR.read_text(encoding="utf-8"))["kartlar"][kart]


def anahtar() -> str | None:
    """Anahtar yalnizca ortamdan ya da (varsa) ev dizinindeki tek satirlik dosyadan."""
    anahtar_degeri = os.environ.get("TYPESAFE_API_KEY")
    if anahtar_degeri:
        return anahtar_degeri.strip()
    dosya = Path.home() / ".jev-anahtar"
    if dosya.is_file():
        return dosya.read_text(encoding="utf-8").strip()
    return None


def arka_uc_sec(istenen: str) -> str:
    if istenen != "otomatik":
        return istenen
    return "jev" if anahtar() else "laya"


# ---------------------------------------------------------------- yerel Laya
def model_yukle(alt_klasor: str = ALT_KLASOR, dtype_adi: str = KOSU_DTYPE):
    """Laya'yi bellek-dostu sekilde yukler (tek sefer; sonra onbellekten doner)."""
    global _AGENT
    if _AGENT is not None:
        return _AGENT
    import torch

    torch.set_num_threads(int(os.environ.get("LUBOT_DANISMA_IS_PARCA", "2")))
    import laya.common as C
    from transformers import AutoConfig, AutoModel

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_adi]

    def build_model_hafif(cfg, encoder_dir=None, pretrained=True):
        if not pretrained or (encoder_dir and os.path.exists(encoder_dir)):
            ecfg = AutoConfig.from_pretrained(encoder_dir or cfg["encoder"])
            C._apply_rope_config(ecfg)
            enc = AutoModel.from_config(ecfg, attn_implementation="sdpa", dtype=dtype)
        else:
            enc = AutoModel.from_pretrained(cfg["encoder"], attn_implementation="sdpa", dtype=dtype)
        return C.DecisionModel(enc, cfg.get("head_layers", 2), len(cfg.get("act_costs", {})) + 1)

    C.build_model = build_model_hafif
    import laya.agent as A

    A.build_model = build_model_hafif
    import laya as L

    _AGENT = L.load(CHECKPOINT, subfolder=alt_klasor, device="cpu")
    return _AGENT


def criteria_metne_cevir(secim, secenekler: list[str], anahtarlar: list[str] | None = None) -> str | None:
    """Modelin dondurdugu etiketi secenek metnine cevirir - tahmin etmeden.

    Iki bicim kabul edilir ve ikisi de *kesin*: dogrudan metin (`tut`) ya da
    **bizim gonderdigimiz** kriter anahtari (`o0`). Sayidan taban tahmini
    yapilmaz: 0/1 tabanli bir sezgi ilk denemede "o1"i yanlis esledi - oyu
    tahmin etmek karari tahmin etmek olurdu. Bilinmeyen etiket `None` doner ve
    oy gecersiz sayilir, yani karar insana gider.
    """
    if not isinstance(secim, str):
        return None
    if secim in secenekler:
        return secim
    if anahtarlar:
        for anahtar, metin in zip(anahtarlar, secenekler):
            if secim == anahtar:
                return metin
    return None


def yerel_oy(kart: str, soru: str, secenekler: list[str]) -> dict:
    agent = model_yukle()
    anahtarlar = [f"o{i}" for i in range(len(secenekler))]
    sorular = {kart: {"type": "choice", "instructions": soru,
                      "criteria": dict(zip(anahtarlar, secenekler))}}
    basla = time.monotonic()
    with _KILIT:  # tek model, tek is parcasi: istekler siraya girer
        ham = agent.system_one(state={"karar": kart}, questions=sorular)
    gecikme = int((time.monotonic() - basla) * 1000)
    cevap = (ham.get("answers") or {}).get(kart) or {}
    # Laya `choice` alaninda **kriter anahtarini** dondurur (o0/o1), metni degil.
    # Esleme olmadan oy "None" duser: kanaryada tam olarak bu yakalandi.
    secim = criteria_metne_cevir(cevap.get("choice"), secenekler, anahtarlar)
    # `confidence` entropi tabanlidir (kalibre degil); kalibre sayi `answer_confidence`.
    guven = cevap.get("answer_confidence")
    if not isinstance(guven, (int, float)):
        guven = cevap.get("confidence") or 0.0
    return {"oy": secim if secim in secenekler else None, "guven": float(guven),
            "arka_uc": f"laya:{ALT_KLASOR}:{KOSU_DTYPE}", "gecikme_ms": gecikme,
            "dagilim": cevap.get("probabilities")}


# ---------------------------------------------------------------- uzak Jev
def jev_oy(kart: str, soru: str, secenekler: list[str]) -> dict:
    anahtar_degeri = anahtar()
    if not anahtar_degeri:
        return {"oy": None, "guven": 0.0, "arka_uc": "jev-yok", "gecikme_ms": 0,
                "gerekce": "TYPESAFE_API_KEY yok"}
    anahtarlar = [f"o{i}" for i in range(len(secenekler))]
    govde = json.dumps({
        "model": JEK_MODEL,
        "state": {"karar": kart},
        "questions": {kart: {"type": "choice", "instructions": soru,
                             "criteria": dict(zip(anahtarlar, secenekler))}},
    }).encode("utf-8")
    istek = urllib.request.Request(
        JEK_UC, data=govde, method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {anahtar_degeri}"})
    basla = time.monotonic()
    try:
        with urllib.request.urlopen(istek, timeout=30) as yanit:
            veri = json.loads(yanit.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as hata:
        return {"oy": None, "guven": 0.0, "arka_uc": "jev", "gecikme_ms": 0,
                "gerekce": f"jev hatasi: {type(hata).__name__}"}
    cevap = (veri.get("answers") or {}).get(kart) or {}
    secim = criteria_metne_cevir(cevap.get("choice"), secenekler, anahtarlar)
    guven = cevap.get("confidence")
    return {"oy": secim if secim in secenekler else None,
            "guven": float(guven) if isinstance(guven, (int, float)) else 0.0,
            "arka_uc": f"jev:{veri.get('model', JEK_MODEL)}",
            "gecikme_ms": int((time.monotonic() - basla) * 1000),
            "girdi_jetonu": (veri.get("usage") or {}).get("input_tokens"),
            "dagilim": cevap.get("probabilities")}


def oy_uret(arka_uc: str, kart: str, soru: str, secenekler: list[str]) -> dict:
    if arka_uc not in ("laya", "jev"):
        raise ValueError(f"bilinmeyen arka uc: {arka_uc}")
    return yerel_oy(kart, soru, secenekler) if arka_uc == "laya" else jev_oy(kart, soru, secenekler)


# ---------------------------------------------------------------- HTTP
class Isleyici(BaseHTTPRequestHandler):
    protokol_surumu = "HTTP/1.1"
    arka_uc = "laya"
    yukleme_ms = 0

    def _yaz(self, kod: int, govde: dict) -> None:
        veri = json.dumps(govde, ensure_ascii=False).encode("utf-8")
        self.send_response(kod)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(veri)))
        self.end_headers()
        self.wfile.write(veri)

    def do_GET(self) -> None:  # noqa: N802 - http.server sozlesmesi
        if self.path.startswith("/saglik"):
            self._yaz(200, {"durum": "hazir", "arka_uc": self.arka_uc,
                            "yukleme_ms": self.yukleme_ms, "model_bellekte": _AGENT is not None})
        else:
            self._yaz(404, {"hata": "bilinmeyen yol"})

    def do_POST(self) -> None:  # noqa: N802 - http.server sozlesmesi
        if not self.path.startswith("/oy"):
            self._yaz(404, {"hata": "bilinmeyen yol"})
            return
        uzunluk = int(self.headers.get("Content-Length") or 0)
        try:
            istek = json.loads(self.rfile.read(uzunluk).decode("utf-8"))
        except ValueError:
            self._yaz(400, {"hata": "govde JSON degil"})
            return
        kart = istek.get("kart") or "tut_at"
        soru = istek.get("soru") or ""
        secenekler = istek.get("secenekler") or ["tut", "at"]
        try:
            self._yaz(200, oy_uret(self.arka_uc, kart, soru, secenekler))
        except Exception as hata:  # noqa: BLE001 - oy yoklugu olarak raporlanir
            self._yaz(503, {"oy": None, "guven": 0.0, "arka_uc": self.arka_uc,
                            "gerekce": f"{type(hata).__name__}: {hata}"})

    def log_message(self, bicim: str, *args: object) -> None:  # sessiz gunluk
        return


def sunucu_kur(port: int, arka_uc: str, on_yukle: bool = True) -> ThreadingHTTPServer:
    basla = time.monotonic()
    if arka_uc == "laya" and on_yukle:
        model_yukle()
    Isleyici.arka_uc = arka_uc
    Isleyici.yukleme_ms = int((time.monotonic() - basla) * 1000)
    return ThreadingHTTPServer(("0.0.0.0", port), Isleyici)


def kendini_test() -> list[str]:
    bulgular: list[str] = []
    assert arka_uc_sec("otomatik") in ("jev", "laya"), "otomatik secim tanimsiz"
    assert arka_uc_sec("laya") == "laya", "acik secim ezildi"
    secenekler = ["tut", "at"]
    anahtarlar = ["o0", "o1"]
    assert criteria_metne_cevir("tut", secenekler, anahtarlar) == "tut", "dogrudan metin eslenmedi"
    assert criteria_metne_cevir("o1", secenekler, anahtarlar) == "at", "kriter anahtari eslenmedi"
    assert criteria_metne_cevir("o0", secenekler, anahtarlar) == "tut", "ilk anahtar eslenmedi"
    assert criteria_metne_cevir("madde-01", secenekler, anahtarlar) is None, (
        "gondermedigimiz anahtar bicimi kabul edildi (tahmin)"
    )
    assert criteria_metne_cevir("sacma", secenekler, anahtarlar) is None, "bilinmeyen etiket kabul edildi"
    assert criteria_metne_cevir(None, secenekler, anahtarlar) is None, "bos etiket kabul edildi"
    assert criteria_metne_cevir("2", secenekler, anahtarlar) is None, "sayidan taban tahmini yapildi"
    bulgular.append("kriter->metin eslemesi")
    k = kart_ayari("tut_at")
    assert k["secenekler"] == ["tut", "at"], "tut_at kartinin secenekleri degismis"
    bulgular.append("arka uc secimi + kart tanimi")
    # Anahtar yoklugunda uzak uc istisna atmaz, oy yoklugu doner.
    eski = os.environ.pop("TYPESAFE_API_KEY", None)
    try:
        oy = jev_oy("tut_at", "kanarya", ["tut", "at"])
        assert oy["oy"] is None, "anahtarsiz jev oy dondurdu"
        bulgular.append("anahtarsiz uzak uc oy yoklugu")
    finally:
        if eski:
            os.environ["TYPESAFE_API_KEY"] = eski
    # Sunucu tarafi: bilinmeyen arka uc reddedilir.
    try:
        oy_uret("yok-boyle-uc", "tut_at", "x", ["tut", "at"])
    except ValueError:
        bulgular.append("bilinmeyen arka uc reddi")
    else:
        raise AssertionError("bilinmeyen arka uc kabul edildi")
    return bulgular


def main(argv: list[str]) -> int:
    ayristirici = argparse.ArgumentParser(description=__doc__)
    ayristirici.add_argument("--port", type=int, default=8790)
    ayristirici.add_argument("--arka-uc", default="otomatik", choices=["laya", "jev", "otomatik"])
    ayristirici.add_argument("--on-yukleme-yok", action="store_true")
    ayristirici.add_argument("--kendini-test", action="store_true")
    args = ayristirici.parse_args(argv)
    if args.kendini_test:
        print("self-test OK [danisma-sunucu]: " + ", ".join(kendini_test()))
        return 0
    secilen = arka_uc_sec(args.arka_uc)
    sunucu = sunucu_kur(args.port, secilen, on_yukle=not args.on_yukleme_yok)
    print(json.dumps({"durum": "dinliyor", "port": args.port, "arka_uc": secilen,
                      "yukleme_ms": Isleyici.yukleme_ms,
                      "url": f"http://127.0.0.1:{args.port}"}, ensure_ascii=False), flush=True)
    try:
        sunucu.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        sunucu.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
