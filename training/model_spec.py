#!/usr/bin/env python3
"""NN-3: mimari spesifikasyonu — derin-dar transformer + μP parametrizasyonu.

K1 (sıfırdan): yalnızca standart kütüphane. K6: model boyutu ölçülen donanım
tavanının altında kalır; tavan sayısı dışarıdan kabullenilmez, bench
raporundan gelir. HH/U: her sayı kaynağıyla etiketlenir.

μP tablosu (yöntem ilhamı: Tensor Programs V / μTransfer; bu makinede
ÖLÇÜLMEDİ — ölçüm, NN-4 eğitim koşusunun işi):
  parametre grubu        başlangıç std        öğrenme hızı (Adam)
  embedding              sabit (genişlikten   α (genişlikten bağımsız)
                         bağımsız, base HP)
  hidden ağırlıklar      sqrt(2/fan_in)       α (hedef genişlikte);
                                              proxy genişlik P'den transfer
                                              istenirse α·P/n
  readout ağırlıkları    sqrt(2)/fan_in       α/fan_in
  dikkat ölçeği          1/d_k (standart 1/sqrt(d_k) DEĞİL)
Kaynak: Yang et al., Tensor Programs V (arXiv 2203.03466); Lingle 2024
(arXiv 2404.05728) transformer uygulaması.

Ağırlık bağlama (weight tying) × μP gerilimi ve çözümü (işaretli karar):
  embedding ve readout matrisi aynı (bağlı) → paylaşılan matris EMBEDDING
  kurallarıyla yaşar (init std sabit, LR α); readout tarafının Θ(1/n²)
  etkisi ileri geçişte logit ölçeği 1/d_model ile sağlanır. Bu çözüm
  yöntem ilhamıdır, ölçülmedi; NN-4 koşusunda ilk denetlenecek karardır.

Kullanım:
    python3 model_spec.py --validate training/model_spec.json
    python3 model_spec.py --candidates --tavan tavan.json --out adaylar.json
    python3 model_spec.py --selftest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# μP tablosu: kod içinde veri olarak taşır; formüller kapalı sabit değil,
# her grupta açık yazar. Etiket: ölçülmedi (yöntem ilhamı).
MU_P_TABLOSU = {
    "embedding": {
        "init_std": "sabit",
        "init_std_formulu": "base_init_std (genişlikten bağımsız)",
        "lr": "α",
        "lr_formulu": "base_lr (genişlikten bağımsız)",
    },
    "hidden": {
        "init_std": "sqrt(2/fan_in)",
        "init_std_formulu": "(2/fan_in) ** 0.5",
        "lr": "α",
        "lr_formulu": "base_lr (hedef genişlikte); proxy P transferi: base_lr*P/n",
    },
    "readout": {
        "init_std": "sqrt(2)/fan_in",
        "init_std_formulu": "(2 ** 0.5)/fan_in",
        "lr": "α/fan_in",
        "lr_formulu": "base_lr/fan_in",
    },
    "dikkat_olcegi": {
        "olcek": "1/d_k",
        "not": "standart 1/sqrt(d_k) değil; μP transformer uygulaması (Lingle 2024)",
    },
}
MU_P_ETIKET = "yöntem ilhamı: Tensor Programs V; ölçülmedi"


def init_std(role: str, fan_in: int, base_init_std: float) -> float:
    """μP başlangıç std'si; role tablodaki gruplardan biri olmalı."""
    if role == "embedding":
        return base_init_std
    if role == "hidden":
        return (2.0 / fan_in) ** 0.5
    if role == "readout":
        return (2.0 ** 0.5) / fan_in
    raise SystemExit(f"bilinmeyen parametre grubu: {role}")


def lr_carpani(role: str, fan_in: int) -> str:
    """Öğrenme hızı çarpanı formülü (base_lr'e göre)."""
    if role == "embedding":
        return "1"
    if role == "hidden":
        return "1"
    if role == "readout":
        return f"1/fan_in (fan_in={fan_in})"
    raise SystemExit(f"bilinmeyen parametre grubu: {role}")


# --------------------------------------------------------------------------
# parametre muhasebesi: tensersiz sabit yok, her tensör açık sayılır.
# decoder-only, pre-norm, bağlı embedding (readout matrisi yok).
def say_params(d_model: int, n_layers: int, d_ff: int, vocab_size: int) -> dict:
    """Tensersiz parametre sayımı; her bileşen adıyla listelenir."""
    emb = vocab_size * d_model
    attn = n_layers * 4 * (d_model * d_model + d_model)      # wq wk wv wo + bias
    mlp = n_layers * (d_model * d_ff + d_ff + d_ff * d_model + d_model)  # w1 b1 w2 b2
    norm = n_layers * 2 * 2 * d_model + 2 * d_model          # katman başına 2 LN + final LN
    total = emb + attn + mlp + norm
    return {
        "embedding_bagli": emb,
        "dikkat": attn,
        "mlp": mlp,
        "layernorm": norm,
        "toplam": total,
    }


# --------------------------------------------------------------------------
def validate_spec(spec: dict) -> dict:
    """Yapısal + μP tutarlılık denetimi. Hata varsa SystemExit."""
    for alan in ("name", "schema", "vocab_family", "vocab_size", "d_model",
                 "n_layers", "n_heads", "d_ff", "max_seq_len", "param_groups",
                 "params", "ceiling_reference"):
        if alan not in spec:
            raise SystemExit(f"spec'te eksik alan: {alan}")
    d, L, H, dff = spec["d_model"], spec["n_layers"], spec["n_heads"], spec["d_ff"]
    if spec["schema"] != 1:
        raise SystemExit(f"bilinmeyen schema sürümü: {spec['schema']}")
    if d % H != 0:
        raise SystemExit(f"d_model {d}, n_heads {H}'e bölünmüyor")
    if dff % d != 0:
        raise SystemExit(f"d_ff {dff}, d_model {d}'in katı değil")
    if spec["weight_tying"].get("tied") is not True:
        raise SystemExit("bu spec ailesi bağlı embedding ister (K6: küçük model)")
    if spec["weight_tying"].get("logit_scale") != "1/d_model":
        raise SystemExit("bağlı readout'un μP çözümü logit ölçeği 1/d_model olmalı")
    if spec["attention_scale"] != "1/d_k":
        raise SystemExit("dikkat ölçeği μP biçimi 1/d_k olmalı")

    # μP parametre grupları: her tensör rolü tabloyla eşleşmeli.
    roller = set()
    for g in spec["param_groups"]:
        rol = g["role"]
        if rol not in ("embedding", "hidden", "readout", "layernorm"):
            raise SystemExit(f"parametre grubunda bilinmeyen rol: {rol}")
        roller.add(rol)
        if rol == "layernorm":
            if g.get("init") != "1 (bias 0)":
                raise SystemExit(f"{g['tensor']}: LN init 1 / bias 0 olmalı")
            continue
        if g.get("init_std_formulu") != MU_P_TABLOSU[rol]["init_std_formulu"]:
            raise SystemExit(f"{g['tensor']}: init formülü μP tablosuyla çelişiyor")
        if g.get("lr_formulu") != MU_P_TABLOSU[rol]["lr_formulu"]:
            raise SystemExit(f"{g['tensor']}: LR formülü μP tablosuyla çelişiyor")
    if "readout" in roller:
        raise SystemExit("bağlı spec'te ayrı readout grubu olmamalı (embedding yaşar)")
    if "embedding" not in roller or "hidden" not in roller:
        raise SystemExit("embedding ve hidden grupları zorunlu")

    # parametre muhasebesi: beyan edilen sayı tensersiz sayımla eşleşmeli.
    muhasebe = say_params(d, L, dff, spec["vocab_size"])
    if muhasebe["toplam"] != spec["params"]["toplam"]:
        raise SystemExit(
            f"param sayısı uyuşmuyor: beyan {spec['params']['toplam']}, "
            f"sayım {muhasebe['toplam']} ({muhasebe})")
    for k in ("embedding_bagli", "dikkat", "mlp", "layernorm"):
        if muhasebe[k] != spec["params"][k]:
            raise SystemExit(f"param bileşeni uyuşmuyor: {k}")

    # K6: ölçülen tavanın altında. Tavan sayısı bench zincirinden gelir.
    tav = spec["ceiling_reference"]
    if tav.get("label") != "olculdu (bench_hardware zinciri, turetildi tavan)":
        raise SystemExit("ceiling_reference etiketi bench zincirini göstermeli")
    if spec["params"]["toplam"] > tav["max_params_train_fp32_adamw"]:
        raise SystemExit(
            f"K6 ihlali: {spec['params']['toplam']} param > ölçülen tavan "
            f"{tav['max_params_train_fp32_adamw']}")

    d_k = d // H
    return {
        "name": spec["name"],
        "d_model": d, "n_layers": L, "n_heads": H, "d_k": d_k, "d_ff": dff,
        "params": muhasebe,
        "tokens_per_param": None,
        "dogrulama": "spec tutarlı: yapı, μP tablosu, bağlama çözümü, sayım, tavan",
    }


# --------------------------------------------------------------------------
def adaylar(ceiling: int, vocab_size: int) -> list[dict]:
    """Derin-dar aday ızgarası; hepsi ölçülen tavanın altında filtrelenir."""
    cikti = []
    for d_model in (32, 48, 64, 96, 128, 192, 256):
        for n_layers in (2, 4, 6, 8, 10, 12, 16, 20, 24):
            d_ff = 4 * d_model
            n_heads = max(2, d_model // 32)
            if d_model % n_heads != 0:
                continue
            m = say_params(d_model, n_layers, d_ff, vocab_size)
            if m["toplam"] > ceiling:
                continue
            cikti.append({
                "d_model": d_model, "n_layers": n_layers, "n_heads": n_heads,
                "d_ff": d_ff, "params": m["toplam"],
                "params_label": "turetildi (tensersiz sayım formülü)",
            })
    cikti.sort(key=lambda c: c["params"])
    return cikti


# --------------------------------------------------------------------------
def selftest() -> None:
    """İç tutarlılık: sayım formülü vs eleman sayısı; μP genişleme kuralları."""
    # 1) sayım: küçük spec'i eleman ele say.
    d, L, dff, V = 8, 2, 16, 32
    m = say_params(d, L, dff, V)
    eleman = V * d  # embedding
    eleman += L * 4 * (d * d + d)  # qkvo + bias
    eleman += L * (d * dff + dff + dff * d + d)  # mlp
    eleman += L * 2 * 2 * d + 2 * d  # normlar
    assert m["toplam"] == eleman, (m, eleman)
    # 2) μP genişleme: d_model ×2 → embedding std değişmez, hidden ×1/sqrt(2),
    #    readout ×1/2; LR'ler tablo gereği genişlikten bağımsız (embedding/hidden).
    s0 = init_std("embedding", 64, 0.02)
    s1 = init_std("embedding", 128, 0.02)
    assert s0 == s1 == 0.02
    assert abs(init_std("hidden", 128, 0.02) / init_std("hidden", 64, 0.02) - 2 ** -0.5) < 1e-12
    assert abs(init_std("readout", 128, 0.02) / init_std("readout", 64, 0.02) - 0.5) < 1e-12
    assert lr_carpani("embedding", 128) == lr_carpani("hidden", 128) == "1"
    # 3) aday filtresi: tavan altındakiler kalır, sıralıdır.
    a = adaylar(ceiling=100_000, vocab_size=256)
    assert a and all(c["params"] <= 100_000 for c in a)
    assert [c["params"] for c in a] == sorted(c["params"] for c in a)
    # 4) readout LR daralır (α/fan_in): fan_in ×2 → çarpan yarıya.
    assert lr_carpani("readout", 128) != lr_carpani("readout", 256)
    # 5) etiket disiplini: tablonun etiketi ölçülmedi taşır.
    assert "ölçülmedi" in MU_P_ETIKET


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--validate", metavar="SPEC", help="spec dosyasını doğrula")
    ap.add_argument("--candidates", action="store_true",
                    help="derin-dar aday ızgarasını çıkar (--tavan ile)")
    ap.add_argument("--tavan", metavar="JSON",
                    help="recommend_model_size.py tavan raporu (adaylar için)")
    ap.add_argument("--vocab", metavar="JSON",
                    help="donmuş sözlük dosyası (adaylar için vocab_size kaynağı)")
    ap.add_argument("--out", metavar="JSON", help="aday çıktı dosyası")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        selftest()
        print("model_spec self-test: 5/5 tamam")
        return 0
    if args.validate:
        spec = json.loads(Path(args.validate).read_text(encoding="utf-8"))
        rapor = validate_spec(spec)
        rapor["tokens_per_param"] = (
            spec["corpus_reference"]["bpe_tokens"] / spec["params"]["toplam"]
            if spec.get("corpus_reference") else None)
        print(json.dumps(rapor, ensure_ascii=False, indent=1))
        return 0
    if args.candidates:
        if not args.tavan or not args.vocab:
            raise SystemExit("--candidates için --tavan ve --vocab gerekir (K6: tavan ölçülmeden, sözlük olmadan aday çıkarmayız)")
        t = json.loads(Path(args.tavan).read_text(encoding="utf-8"))
        tav = t["derived"]["max_params"]["train_fp32_adamw"]["value"]
        V = json.loads(Path(args.vocab).read_text(encoding="utf-8"))["vocab_size"]
        cikti = {
            "purpose": "NN-3 derin-dar adayları; hepsi ölçülen donanım tavanının altında",
            "ceiling": tav,
            "ceiling_label": "olculdu (bench) -> turetildi (usable_bytes/16)",
            "vocab_size": V,
            "adaylar": adaylar(tav, V),
        }
        hedef = Path(args.out) if args.out else sys.stdout
        metin = json.dumps(cikti, ensure_ascii=False, indent=1)
        if args.out:
            hedef.write_text(metin + "\n", encoding="utf-8")
            print(f"{len(cikti['adaylar'])} aday -> {args.out}")
        else:
            print(metin)
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
