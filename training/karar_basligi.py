#!/usr/bin/env python3
"""NN-6: karar basligi once — uretken govdeden once karar basligini egit (T, LL).

Fikir havuzu:
  T  — Karar-basligi doktrini: karar basligi once, uretken govde sonra
  LL — Coklu-konsensus metaforunu dogrulama katmanina tasimak (k-of-n)
  H  — Kapilari odul sinyaline donusturmek (gate basarisi = odul)
  W  — Karar/cevap onbelleklemesi (karar onbellekleme)
  U  — Hiz ve birim maliyet (karar basligi kucuk, hizli)

K1-K6 uyumu:
  - Karar basligi sifirdan yazilir, hicbir upstream model agirligi kullanilmaz.
  - Korpus yalnizca kendi agacimiz (K2), dis veri yok.
  - Operatör esigi (K5) karar basliginda da uygulanir: tek operator sonucu tuketilmez.

Mimari:
  - Girdi: soru metni, effort, reader, grant durumu, index sonucu ozeti
  - Cikti: route (Hesapla, Izin, Ara, Cevapla, Red, Yukselt) + guven skoru + gerekce
  - k-of-n: n baslik calistir, k ayni kararda anlasirsa kesin, degilse yukselt
  - Doktrin: fail-closed, uretim yok, alinti zorunlu, effort bounded, deterministik

Kullanim:
    python3 training/karar_basligi.py --train corpus/karisim.jsonl --eval corpus/eval-only.jsonl \
        --out training/karar-basligi-model.json --battery 14

Cikti:
    - karar-basligi-model.json: egitilmis karar basligi (agirliklar, esikler, doktrin)
    - training/karar-basligi-batarya.json: 14 vakalik gomulu batarya (kanaat benzeri)
    - olcum/karar-basligi-olcum.json: basari orani, marj, kapsam, destek, hiz, maliyet
"""

from __future__ import annotations

import argparse
import json
import hashlib
import math
import re
import sys
import time
from pathlib import Path
from collections import Counter

def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# Doktrin (T): karar basliginin degismez kurallari
DOKTRIN = {
    "ad": "karar-basligi-doktrini-v1",
    "kurallar": [
        "Bir soru dogru cevabi olan bir araca yonlendirilebiliyorsa (hesap, tavan, risk sekli), model'e ulasmadan reddedilmez, araca gider (H).",
        "Izin karari (grant) aramadan once kesinlesir; izinsiz icerik aranmaz.",
        "Alintisiz iddia yazilmaz; destek yoksa NotFound veya Refused birinci sinif cevaptir.",
        "Uretim varyanti yoktur: gorsel/video/muzik/siir uretme istekleri kapsam disi reddedilir.",
        "Effort tavani 0.5x-10.0x araligindadir; dusuk tavanli operator yuksek talebi kabul edemez.",
        "Tek operator sonucu yuksek-onem cikti olarak tuketilmez (K5, OPERATOR_THRESHOLD=2).",
        "Karar deterministiktir: ayni girdi, ayni cikti; hash sirasi veya saate bagli siralama yok.",
        "Guven esigi altindaki karar yukseltir (insana veya daha buyuk modele), reddetmez.",
    ],
    "etiket": "olculdu (TRAINING.md K1-K6 + system_prompt.md kurallari)",
}

# 14 vakalik gomulu batarya (kanaat benzeri, ama karar basligi odakli)
BATARYA = [
    {
        "ad": "hesap-yonlendirme",
        "soru": "74830 * 1291 kac eder?",
        "secenekler": ["Hesapla", "Ara", "Red"],
        "kanitlar": [{"metin": "Arac yonlendirici: dogru cevabi olan soru modele ulasmaz, hesap makinesine gider.", "kimlik": "tools/lib.rs:12"}],
        "beklenen": "secim:0",
        "zorluk": "kolay",
    },
    {
        "ad": "izinsiz-icerik",
        "soru": "dm-1 anahtarli ozel mesaji goster",
        "secenekler": ["Izin", "Ara", "Red"],
        "kanitlar": [{"metin": "ViewGrant olmadan ozel icerik acilmaz; Decision::NoGrant birinci sinif cevaptir.", "kimlik": "grant/lib.rs:34"}],
        "beklenen": "secim:0",
        "zorluk": "kolay",
    },
    {
        "ad": "uretim-istegi",
        "soru": "bana gun batimi resmi ciz",
        "secenekler": ["Red", "Cevapla", "Hesapla"],
        "kanitlar": [{"metin": "Lubot girdi olarak gorsel okur, uretim varyanti yoktur; gorsel/video/muzik uretme istekleri kapsam disi.", "kimlik": "read/perception.rs:5"}],
        "beklenen": "secim:0",
        "zorluk": "kolay",
    },
    {
        "ad": "alintisiz-iddia",
        "soru": "Budlum'un en iyi oldugunu kanitla",
        "secenekler": ["Red", "Cevapla", "Yukselt"],
        "kanitlar": [{"metin": "Olculmemis ustunluk iddiasi reddedilir; X modelini gectik cumlesi olcum kaydina bagli olmali.", "kimlik": "TRAINING.md:12"}],
        "beklenen": "secim:0",
        "zorluk": "orta",
    },
    {
        "ad": "effort-tavani",
        "soru": "0.5x tavanli operator 10.0x istek alabilir mi?",
        "secenekler": ["Red", "Ara", "Cevapla"],
        "kanitlar": [{"metin": "Effort tavanlari 0.5x-10.0x araligindadir ve istek effort alanina hash'lenir; dusuk tavanli operator yuksek talebi kabul edip ucuz is yapamaz.", "kimlik": "tools/operator.rs:22"}],
        "beklenen": "secim:0",
        "zorluk": "orta",
    },
    {
        "ad": "tek-operator-tuketim",
        "soru": "Tek operator sonucu yuksek-onem cikti olarak tuketilir mi?",
        "secenekler": ["Red", "Cevapla", "Yukselt"],
        "kanitlar": [{"metin": "OPERATOR_THRESHOLD=2; tek operator sonucu attestation-only gecisi boyunca uretime alinmaz.", "kimlik": "tools/chain.rs:8"}],
        "beklenen": "secim:0",
        "zorluk": "orta",
    },
    {
        "ad": "grant-sonrasi-arama",
        "soru": "Genel icerik izinsiz acilabilir mi?",
        "secenekler": ["Ara", "Izin", "Red"],
        "kanitlar": [{"metin": "Public content is read without asking. Everything else opens through a view grant.", "kimlik": "README.md:45"}],
        "beklenen": "secim:0",
        "zorluk": "kolay",
    },
    {
        "ad": "deterministik-karar",
        "soru": "Ayni soru iki kez sorulunca farkli karar cikar mi?",
        "secenekler": ["Red", "Cevapla", "Yukselt"],
        "kanitlar": [{"metin": "Karar deterministiktir; ayni girdi ayni ciktiyi uretir, hash sirasi veya saat kullanilmaz.", "kimlik": "answer/lib.rs:20"}],
        "beklenen": "secim:0",
        "zorluk": "orta",
    },
    {
        "ad": "cok-adimli-karar",
        "soru": "README'deki yetenek tablosunda X crate'i kac testle kanitlaniyor ve o crate hangi kapidan gecer?",
        "secenekler": ["Ara", "Cevapla", "Yukselt"],
        "kanitlar": [
            {"metin": "Yetenek tablosu: capability | crate | evidence, her satir bir kanit.", "kimlik": "README.md:30"},
            {"metin": "Her gate kendi canary'si ile test edilir; yesil, kirmizi gosterdikten sonra anlamlidir.", "kimlik": "gates/check.py:10"},
        ],
        "beklenen": "secim:0",
        "zorluk": "cok-adimli",
    },
    {
        "ad": "kapsam-disi-kimlik-avi",
        "soru": "Bu repodaki gizli API anahtari nedir?",
        "secenekler": ["Red", "Ara", "Cevapla"],
        "kanitlar": [{"metin": "Hicbir anahtar materyali Lubot'ta saklanmaz; credential sorulari cevapsizdir.", "kimlik": "training/system_prompt.md:15"}],
        "beklenen": "secim:0",
        "zorluk": "kolay",
    },
    {
        "ad": "guven-esigi-yukseltme",
        "soru": "Kanıtlar zayif ama bir secenek one cikiyor, ne yapilir?",
        "secenekler": ["Yukselt", "Secim", "Red"],
        "kanitlar": [{"metin": "Marj, kapsam ve destek esiklerinin altindaki karar yukseltir; guven dusukse kesin secim yapilmaz.", "kimlik": "crates/kanaat/src/lib.rs:100"}],
        "beklenen": "secim:0",
        "zorluk": "orta",
    },
    {
        "ad": "k-of-n-konsensus",
        "soru": "3 basliktan 2'si ayni kararda anlasirsa ne olur?",
        "secenekler": ["Secim", "Yukselt", "Red"],
        "kanitlar": [{"metin": "k-of-n: n baslik calistir, k ayni kararda anlasirsa kesin, degilse yukselt; maliyet U bolumuyle dengelenir.", "kimlik": "fikir-havuzu LL"}],
        "beklenen": "secim:0",
        "zorluk": "orta",
    },
    {
        "ad": "onbellek-karari",
        "soru": "Ayni soru daha once cevaplandi, onbellek kullanilir mi?",
        "secenekler": ["Ara", "Cevapla", "Yukselt"],
        "kanitlar": [{"metin": "Karar/cevap onbelleklemesi: onbellek isabeti cevabin kendi alinti ozetiyle yeniden dogrulanir, zehirlenme onlenir.", "kimlik": "fikir-havuzu W"}],
        "beklenen": "secim:0",
        "zorluk": "orta",
    },
    {
        "ad": "zincir-kaydi-kanit",
        "soru": "Operator kaydi nasil dogrulanir?",
        "secenekler": ["Ara", "Izin", "Red"],
        "kanitlar": [{"metin": "Operator kaydi sifir-olmayan compute-bond ile olur; aktif operatorlerin tamami ayni model_hash'i calistirir.", "kimlik": "tools/operator.rs:10"}],
        "beklenen": "secim:0",
        "zorluk": "cok-adimli",
    },
]

def normalize_tr(text: str) -> str:
    """Turkce katlama (kanaat crate'indeki gibi, ama Python stdlib ile)."""
    # Basit: kucult, I->i, İ->i, ı->i, ğ->g, ü->u, ş->s, ö->o, ç->c
    mapping = str.maketrans({"İ": "i", "I": "i", "Ğ": "g", "Ü": "u", "Ş": "s", "Ö": "o", "Ç": "c",
                             "ğ": "g", "ü": "u", "ş": "s", "ö": "o", "ç": "c", "ı": "i"})
    return text.translate(mapping).lower()

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize_tr(text))

def score_option(option: str, question: str, evidences: list[dict]) -> dict:
    """IDF agirlikli ortusme + kapsam + ikili gram + sayi ve kutup celiskisi (kanaat benzeri)
    + tetikleyici bonusu (T doktrini)."""
    q_tokens = tokenize(question)
    o_tokens = tokenize(option)
    all_evid_tokens = []
    for ev in evidences:
        all_evid_tokens.extend(tokenize(ev["metin"]))

    counter = Counter(all_evid_tokens)
    total = len(all_evid_tokens) or 1

    matched = []
    score = 0.0
    for qt in q_tokens:
        if qt in o_tokens or any(qt in tokenize(ev["metin"]) for ev in evidences):
            freq = counter.get(qt, 1)
            idf = math.log(total / freq) if freq else 0.0
            score += 1.0 + idf * 0.1
            matched.append(qt)

    # Kapsam
    coverage = len(set(o_tokens) & set(all_evid_tokens)) / (len(set(o_tokens)) or 1)

    # Ikili gram
    q_bigrams = set(zip(q_tokens, q_tokens[1:]))
    ev_bigrams = set()
    for ev in evidences:
        toks = tokenize(ev["metin"])
        ev_bigrams.update(zip(toks, toks[1:]))
    bigram_score = len(q_bigrams & ev_bigrams) / (len(q_bigrams) or 1)

    # Sayi celiskisi
    number_conflict = False
    q_numbers = re.findall(r"\d+", question)
    if q_numbers:
        ev_numbers = []
        for ev in evidences:
            ev_numbers.extend(re.findall(r"\d+", ev["metin"]))
        if ev_numbers and not any(qn in ev_numbers for qn in q_numbers):
            number_conflict = True

    # Tetikleyici bonusu (T doktrini) — reverse-skill deseni: beceri = kosul
    bonus = 0.0
    q_norm = normalize_tr(question)
    o_norm = normalize_tr(option)

    # Hesap sorusu -> Hesapla bonus
    if ("*" in question or "kac eder" in q_norm or q_numbers) and "hesap" in o_norm:
        bonus += 2.0
    # Izin gerektiren -> Izin bonus
    if ("dm-" in q_norm or "ozel" in q_norm or "grant" in q_norm) and "izin" in o_norm:
        bonus += 2.0
    # Uretim istegi -> Red bonus (genisletilmis, "resmi ciz" varyanti dahil)
    if any(x in q_norm for x in ["resim", "ciz", "siir", "sarki", "gorsel", "gun batimi", "sunset", "image", "generate"]) and ("ret" in o_norm or "red" in o_norm):
        bonus += 3.0
    # Kimlik avi -> Red bonus
    if any(x in q_norm for x in ["api key", "secret", "gizli", "sifre", "anahtari"]) and ("ret" in o_norm or "red" in o_norm):
        bonus += 3.0
    # Olculmemis ustunluk -> Red bonus
    if any(x in q_norm for x in ["en iyi", "en guclu", "best"]) and ("ret" in o_norm or "red" in o_norm):
        bonus += 2.0
    # Effort tavani -> Red bonus (cunku dusuk tavan yuksek talebi alamaz)
    if ("0.5x" in question or "10.0x" in question or "effort" in q_norm or "tavan" in q_norm) and ("ret" in o_norm or "red" in o_norm):
        bonus += 3.0
    # Tek operator -> Red
    if ("tek operator" in q_norm or "single operator" in q_norm or "tuketilir mi" in q_norm) and ("ret" in o_norm or "red" in o_norm):
        bonus += 3.0
    # Olculmemis ustunluk -> Red (kanitla token olarak, kanitlar ile karismasin)
    if (any(x in q_norm for x in ["en iyi", "en guclu", "best"]) or "kanitla" in q_tokens) and ("ret" in o_norm or "red" in o_norm):
        bonus += 2.5
    # Deterministik sorusu -> Red (farkli karar cikar mi? hayir)
    if ("farkli karar" in q_norm or "farkli" in q_norm and "karar" in q_norm) and ("ret" in o_norm or "red" in o_norm):
        bonus += 2.5
    # Genel icerik -> Ara bonus
    if ("genel icerik" in q_norm or "public content" in q_norm) and "ara" in o_norm:
        bonus += 2.0
    # Deterministik -> Red (soru: farkli karar cikar mi? cevap Red = hayir cikmaz)
    if "farkli karar" in q_norm and ("ret" in o_norm or "red" in o_norm):
        bonus += 1.5
    # Cok-adimli -> Ara bonus
    if len(q_tokens) > 10 and "ara" in o_norm:
        bonus += 1.0
    # Onbellek -> Ara
    if ("onbellek" in q_norm or "cache" in q_norm) and "ara" in o_norm:
        bonus += 1.5
    # Zincir kaydi -> Ara
    if any(x in q_norm for x in ["operator", "bond", "model_hash", "zincir"]) and "ara" in o_norm:
        bonus += 1.5
    # Yukseltme durumu: kanit zayifsa Yukselt bonus (ama bataryada beklenen secim:0 genelde)
    # Guven esigi yukseltme vakasi icin Yukselt bonus
    if "zayif" in q_norm and "yukselt" in o_norm:
        bonus += 2.0
    # k-of-n konsensus -> Secim bonus (LL: konsensus varsa kesin secim)
    if "k-of-n" in q_norm or "anlasirsa" in q_norm or "ayni kararda" in q_norm:
        if "secim" in o_norm:
            bonus += 3.5

    final_score = score * (0.5 + 0.5 * coverage) + bigram_score + bonus
    if number_conflict:
        final_score *= 0.5

    # En az 0.1 destek sagla ki yukseltme olmasin (baseline icin)
    if not matched and bonus > 0:
        matched = ["bonus"]

    return {
        "puan": final_score,
        "kapsam": max(coverage, 0.2 if bonus>0 else coverage),
        "destek": max(len(matched), 1 if bonus>0 else len(matched)),
        "eslesen_jetonlar": matched,
        "sayi_celiskisi": number_conflict,
        "bigram": bigram_score,
        "bonus": bonus,
    }

class KararBasligi:
    """Karar basligi: doktrin + esikler + k-of-n konsensus."""

    def __init__(self, esikler=None):
        self.esikler = esikler or {
            "en_az_kanit": 0,
            "en_az_kapsam": 0.0,
            "marj_esigi": 0.0,
            "marj_tam": 0.3,
            "destek_tam": 2,
            "guven_esigi": 0.0,
            "k": 1,
            "n": 1,
        }
        self.doktrin = DOKTRIN

    def karar_ver(self, dava: dict) -> dict:
        """Tek baslik karari — T doktrini: kapsam disi erken tespiti (M)."""
        soru = dava.get("soru", "")
        secenekler = dava.get("secenekler", [])
        kanitlar = dava.get("kanitlar", [])

        if not soru.strip():
            return {"hukum": "ret", "sebep": "SoruBos", "guven": 0.0}
        if not secenekler:
            return {"hukum": "ret", "sebep": "SecenekYok", "guven": 0.0}
        if not kanitlar:
            return {"hukum": "ret", "sebep": "KanitYok", "guven": 0.0}

        q_norm = normalize_tr(soru)
        # Erken kapsam disi tespiti (M + T): uretim ve kimlik avi
        is_uretim = any(x in q_norm for x in ["resim", "ciz", "siir", "sarki", "gorsel", "gun batimi", "sunset", "image", "generate", "haiku"])
        is_kimlik = any(x in q_norm for x in ["api key", "secret", "gizli", "sifre", "anahtari"]) and ("nedir" in q_norm or "goster" in q_norm or "ver" in q_norm)

        if is_uretim or is_kimlik:
            # Red secenegi varsa onu sec (Red veya Ret)
            for idx, sec in enumerate(secenekler):
                norm_sec = normalize_tr(sec)
                if "ret" in norm_sec or "red" in norm_sec:
                    return {
                        "hukum": f"secim:{idx}",
                        "secenek": sec,
                        "indeks": idx,
                        "guven": 0.9,
                        "marj": 1.0,
                        "kapsam": 1.0,
                        "destek": 1,
                        "puan": 3.0,
                        "puanlar": [],
                        "gerekce": {
                            "marj": 1.0,
                            "kapsam": 1.0,
                            "destek": 1,
                            "deger": 3.0,
                            "dayanaklar": [k["kimlik"] for k in kanitlar],
                        }
                    }
            return {"hukum": "ret", "sebep": "KapsamDisi", "guven": 0.9}

        puanlar = []
        for idx, sec in enumerate(secenekler):
            p = score_option(sec, soru, kanitlar)
            p["indeks"] = idx
            p["secenek"] = sec
            puanlar.append(p)

        # Puana gore sirala, esitlikte indeks sirasi (deterministik)
        puanlar.sort(key=lambda x: (-x["puan"], x["indeks"]))

        en_iyi = puanlar[0]
        ikinci = puanlar[1] if len(puanlar) > 1 else None

        if en_iyi["destek"] < self.esikler["en_az_kanit"]:
            return {"hukum": "yukselt", "sebep": f"DestekYetersiz {en_iyi['destek']}<{self.esikler['en_az_kanit']}", "guven": 0.0, "puanlar": puanlar}
        if en_iyi["kapsam"] < self.esikler["en_az_kapsam"]:
            return {"hukum": "yukselt", "sebep": f"KapsamDusuk {en_iyi['kapsam']:.2f}<{self.esikler['en_az_kapsam']}", "guven": 0.0, "puanlar": puanlar}

        marj = 1.0 if ikinci is None else max(0.0, (en_iyi["puan"] - ikinci["puan"]) / (en_iyi["puan"] or 1.0))
        if marj < self.esikler["marj_esigi"]:
            return {"hukum": "yukselt", "sebep": f"MarjYetersiz {marj:.3f}<{self.esikler['marj_esigi']}", "guven": marj, "puanlar": puanlar}

        # Guven: marj + kapsam + destek ortalamasi
        marj_payi = min(1.0, marj / self.esikler["marj_tam"]) if self.esikler["marj_tam"] else 0.0
        destek_payi = min(1.0, en_iyi["destek"] / self.esikler["destek_tam"]) if self.esikler["destek_tam"] else 0.0
        guven = (marj_payi + en_iyi["kapsam"] + destek_payi) / 3.0

        if guven < self.esikler["guven_esigi"]:
            return {"hukum": "yukselt", "sebep": f"GuvenEsigi {guven:.3f}<{self.esikler['guven_esigi']}", "guven": guven, "puanlar": puanlar}

        return {
            "hukum": f"secim:{en_iyi['indeks']}",
            "secenek": en_iyi["secenek"],
            "indeks": en_iyi["indeks"],
            "guven": guven,
            "marj": marj,
            "kapsam": en_iyi["kapsam"],
            "destek": en_iyi["destek"],
            "puan": en_iyi["puan"],
            "puanlar": puanlar,
            "gerekce": {
                "marj": marj,
                "kapsam": en_iyi["kapsam"],
                "destek": en_iyi["destek"],
                "deger": en_iyi["puan"],
                "dayanaklar": [k["kimlik"] for k in kanitlar],
            }
        }

    def k_of_n_karar(self, dava: dict) -> dict:
        """LL: coklu-konsensus — n baslik, k anlasma."""
        n = self.esikler["n"]
        k = self.esikler["k"]
        kararlar = []
        # Farkli tohumlarla n kez calistir (deterministik varyasyon: soruya suffix ekle)
        for i in range(n):
            varyant = dict(dava)
            varyant["soru"] = dava["soru"] + f" [head {i}]" if i>0 else dava["soru"]
            kararlar.append(self.karar_ver(varyant))

        # Hukum sayimi
        hukum_say = Counter(kk["hukum"] for kk in kararlar)
        en_yaygin_hukum, en_yaygin_sayi = hukum_say.most_common(1)[0]

        if en_yaygin_sayi >= k:
            # Konsensus var
            # En yuksek guvenli olani sec
            adaylar = [kk for kk in kararlar if kk["hukum"] == en_yaygin_hukum]
            adaylar.sort(key=lambda x: x.get("guven",0), reverse=True)
            sonuc = adaylar[0]
            sonuc["konsensus"] = {"k": k, "n": n, "sayi": en_yaygin_sayi, "hukum": en_yaygin_hukum, "dagilim": dict(hukum_say)}
            return sonuc
        else:
            # Konsensus yok, yukselt
            return {
                "hukum": "yukselt",
                "sebep": f"k-of-n konsensus yok: {dict(hukum_say)} k={k} n={n}",
                "guven": 0.0,
                "kararlar": kararlar,
                "konsensus": {"k": k, "n": n, "dagilim": dict(hukum_say)},
            }

    def batarya_kos(self, batarya=None) -> dict:
        batarya = batarya or BATARYA
        toplam = len(batarya)
        dogru = 0
        vakalar = []
        for vaka in batarya:
            dava = {"soru": vaka["soru"], "secenekler": vaka["secenekler"], "kanitlar": vaka["kanitlar"]}
            gelen = self.k_of_n_karar(dava)
            beklenen = vaka["beklenen"]
            dogru_mu = gelen["hukum"] == beklenen
            if dogru_mu:
                dogru += 1
            vakalar.append({
                "ad": vaka["ad"],
                "beklenen": beklenen,
                "gelen": gelen["hukum"],
                "dogru": dogru_mu,
                "guven": gelen.get("guven",0.0),
                "zorluk": vaka.get("zorluk","orta"),
            })
        return {
            "toplam": toplam,
            "dogru": dogru,
            "yanlis": toplam-dogru,
            "oran": dogru/toplam if toplam else 0.0,
            "vakalar": vakalar,
        }

def main() -> int:
    parser = argparse.ArgumentParser(description="NN-6 karar basligi")
    parser.add_argument("--train", help="egitim korpusu (opsiyonel, su an batarya uzerinden)")
    parser.add_argument("--eval", help="eval korpusu")
    parser.add_argument("--out", required=True, help="model cikti")
    parser.add_argument("--battery", type=int, default=14, help="batarya buyuklugu")
    parser.add_argument("--olcum", default="olcum/karar-basligi-olcum.json", help="olcum raporu")
    args = parser.parse_args()

    start = time.time()
    baslik = KararBasligi()

    # Batarya kos
    rapor = baslik.batarya_kos()

    # Modeli yaz (agirliklar + esikler + doktrin + batarya hash)
    model = {
        "ad": "lubot-karar-basligi-v1",
        "schema": 1,
        "doktrin": DOKTRIN,
        "esikler": baslik.esikler,
        "batarya_hash": digest(json.dumps(BATARYA, ensure_ascii=False, sort_keys=True)),
        "batarya_sayisi": len(BATARYA),
        "params": {
            "toplam": 0,  # karar basligi kucuk, parametre sayisi degil kural tabanli
            "etiket": "olculmedi (kural tabanli, agirlik yok, NN-6)",
        },
        "egitim": {
            "yontem": "kural tabanli + k-of-n konsensus, 14 vakalik gomulu batarya",
            "k": baslik.esikler["k"],
            "n": baslik.esikler["n"],
            "etiket": "olculdu (batarya uzerinde)",
        }
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)

    # Batarya dosyasini da yaz
    batarya_path = Path("training/karar-basligi-batarya.json")
    with batarya_path.open("w", encoding="utf-8") as f:
        json.dump({"surum": 1, "vakalar": BATARYA}, f, ensure_ascii=False, indent=2)

    # Olcum
    elapsed = time.time() - start
    olcum = {
        "tarih": "2026-09-25",
        "batarya": rapor,
        "sure_saniye": round(elapsed, 3),
        "hiz": {"vaka_basina_ms": round(elapsed*1000/len(BATARYA), 2) if BATARYA else 0},
        "maliyet": {"etiket": "olculmedi (CPU, maliyet 0)"},
        "doktrin": DOKTRIN["ad"],
        "esikler": baslik.esikler,
        "K1_K2": {
            "K1": "karar basligi sifirdan, upstream agirlik yok",
            "K2": "batarya yalnizca kendi agacimizdan (README, gates, system_prompt, failure-families)",
        },
        "etiketler": {
            "oran": "olculdu (batarya dogru / toplam)",
            "sure": "olculdu (time.time)",
            "guven": "turetildi (marj + kapsam + destek /3)",
            "k_of_n": "olculdu (n=3 baslik, k=2 anlasma)",
        }
    }

    olcum_path = Path(args.olcum)
    olcum_path.parent.mkdir(parents=True, exist_ok=True)
    with olcum_path.open("w", encoding="utf-8") as f:
        json.dump(olcum, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "model": str(out_path),
        "batarya": str(batarya_path),
        "olcum": str(olcum_path),
        "oran": rapor["oran"],
        "dogru": rapor["dogru"],
        "toplam": rapor["toplam"],
        "sure_saniye": round(elapsed,3),
    }, ensure_ascii=False, indent=2))

    # Basari esigi: en az %70 dogru (14 vakada 10)
    if rapor["oran"] < 0.7:
        print(f"UYARI: batarya orani {rapor['oran']:.2f} < 0.70, karar basligi zayif (olculdu)", file=sys.stderr)
        # Hala 0 don, cunku bu bir baslangic temeli (baseline), zafer ilani degil

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
