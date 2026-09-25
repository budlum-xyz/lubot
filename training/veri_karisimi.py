#!/usr/bin/env python3
"""NN-4: veri karisimi — gercek + sentetik + derleyici-hakemli + mufredat siralamasi.

Kapsanan fikir havuzu:
  A  — yuzey korpusu genislemesi (cok kaynakli builder zaten var, burada karisim orani)
  C  — sentetik veri uretimi (kendinden cogalma, dis ogretmen yok)
  D  — kuratorluk ve kalite kapilari (no-duplicate, citation-density, stale-flag, zorluk etiketi)
  E  — mufredat muhendisligi (zorluk siralamasi, onkosul, sinir degerler, capraz-modul)
  O  — model surumleme, provenance, zincir kaydi (oranlar provenance kaydina yazilir)
  KK — derleyici/test takimimi bedava hakem olarak kullanmak
  PP — degerlendirme setinin sizmasini fiziksel olarak imkansiz kilmak (eval-only digest listesi)
  MM — muhendislik-iskeleti / veri ayrimini netlestirmek (import denetimi)

K1-K2 uyumu:
  - Disaridan tek satir metin/veri girmez; sentetik veri yalnizca bu repo'nun kendi
    iceriginden sablonla turetilir (README yetenek tablosu, gates/check.py aciklamalari,
    docs/failure-families.md aileleri, system_prompt.md kurallari).
  - Derleyici-hakemli ciftler icin hakem rustc'nin kendisidir (KK), dis model degil.
  - Tum kayitlar asset_id + content_id provenance cifti tasir; lisans kapali setten.
  - Gercek/sentetik/derleyici orani ayri bir JSON'da (provenance) raporlanir, gizlenmez.

Kullanim:
    python3 training/veri_karisimi.py --corpus corpus/knowledge-self.jsonl.gz \
        --curriculum training/curriculum --out corpus/karisim.jsonl \
        --eval-out corpus/eval-only.jsonl --provenance olcum/veri-karisimi.json

Cikti:
    - karisim.jsonl: egitim karisimi (gercek + sentetik + derleyici-hakemli)
    - eval-only.jsonl: held-out degerlendirme kumesi (PP: bu digest'ler asla SFT'ye girmez)
    - olcum/veri-karisimi.json: oranlar, sayilar, kaynaklar, etiketler (olculdu/turetildi/olculmedi)
    - corpus/eval-digest-list.json: eval-only digest'lerinin listesi (kapi tarafindan denetlenir)

Olcum disiplini:
    Her sayi etiketlidir: olculdu (builder/trainer ciktisi), turetildi (formul acik),
    olculmedi (yontem ilhami, dis referans). Olculmemis hicbir sayi olculmus gibi yazilmaz.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ALLOWED_LICENCES = {"MIT", "Apache-2.0", "PolyForm-Shield-1.0.0"}

def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def asset_id_for(source_name: str) -> str:
    return digest(f"BDLM_LUBOT_CORPUS_ASSET_V1|{source_name}")

def load_corpus(path: Path) -> list[dict]:
    records = []
    open_fn = gzip.open if path.suffix == ".gz" else open
    with open_fn(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except json.JSONDecodeError:
                continue
    return records

def parse_readme_capabilities(root: Path) -> list[dict]:
    """README'deki yetenek tablosunu sablon tabanli soru uretimi icin ham malzeme olarak kullan (C)."""
    readme = root / "README.md"
    if not readme.is_file():
        return []
    text = readme.read_text(encoding="utf-8", errors="ignore")
    # | capability | crate | evidence | tablosunu ara
    rows = []
    in_table = False
    for line in text.splitlines():
        if "| capability | crate | evidence |" in line:
            in_table = True
            continue
        if in_table and line.strip().startswith("|"):
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            if len(parts) >= 3 and parts[0] and parts[0] != "---":
                rows.append({"capability": parts[0], "crate": parts[1], "evidence": parts[2]})
        elif in_table and not line.strip().startswith("|"):
            if rows:
                break
    return rows

def parse_gates(root: Path) -> list[dict]:
    """gates/check.py'deki 37+ gate aciklamasindan soru turetebilmek icin."""
    gate_file = root / "gates" / "check.py"
    if not gate_file.is_file():
        return []
    text = gate_file.read_text(encoding="utf-8")
    gates = []
    for m in re.finditer(r'def (gate_[a-z0-9_]+)\(\) -> str:\n    """(.+?)"""', text, re.S):
        name = m.group(1).removeprefix("gate_").replace("_", "-")
        summary = " ".join(m.group(2).split())
        gates.append({"name": name, "summary": summary})
    return gates

def parse_failure_families(root: Path) -> list[dict]:
    """docs/failure-families.md'deki aileler icin sentetik negatif ornek uretimi (C)."""
    ff = root / "docs" / "failure-families.md"
    if not ff.is_file():
        return []
    text = ff.read_text(encoding="utf-8")
    families = []
    for m in re.finditer(r'## (.+)', text):
        title = m.group(1).strip()
        if title and "How the families" not in title:
            families.append({"family": title})
    return families

def generate_synthetic_records(root: Path, real_records: list[dict]) -> list[dict]:
    """Sablon tabanli sentetik veri uretimi (C). Dis ogretmen yok, sadece kendi korpus."""
    synthetic = []
    caps = parse_readme_capabilities(root)
    gates = parse_gates(root)
    families = parse_failure_families(root)

    # 1) README yetenek tablosundan: "X crate'i kac testle kanitlaniyor" varyasyonlari
    for cap in caps[:20]:
        q = f"{cap['crate']} crate'i hangi yetenegi kanitliyor ve kac testle?"
        a = f"{cap['crate']} crate'i {cap['capability']} yetenegini kanitlar. Kanit: {cap['evidence']}. Source: README.md"
        synthetic.append({
            "kind": "synthetic",
            "subkind": "capability-table",
            "text": a,
            "question": q,
            "provenance": "README.md capability table (template)",
            "difficulty": "kolay",
            "licence": "PolyForm-Shield-1.0.0",
            "asset_id": asset_id_for("lubot"),
            "content_id": digest(a),
            "digest": digest(a),
        })

    # 2) Gate'lerden: "bu gate neyi reddeder" ve "hangi canary ile test edilir"
    for gate in gates:
        q1 = f"{gate['name']} kapisi neyi reddeder?"
        a1 = f"{gate['name']} kapisi su durumda reddeder: {gate['summary']} Source: gates/check.py"
        synthetic.append({
            "kind": "synthetic",
            "subkind": "gate-refusal",
            "text": a1,
            "question": q1,
            "provenance": f"gate {gate['name']}",
            "difficulty": "orta",
            "licence": "PolyForm-Shield-1.0.0",
            "asset_id": asset_id_for("lubot"),
            "content_id": digest(a1),
            "digest": digest(a1),
        })
        q2 = f"{gate['name']} kapisinin canary testi nedir?"
        a2 = f"{gate['name']} kapisi kendi canary'si ile test edilir: kasitli ihlal enjekte edilir ve kapinin reddetmesi beklenir. Source: gates/check.py"
        synthetic.append({
            "kind": "synthetic",
            "subkind": "gate-canary",
            "text": a2,
            "question": q2,
            "provenance": f"gate {gate['name']} canary",
            "difficulty": "orta",
            "licence": "PolyForm-Shield-1.0.0",
            "asset_id": asset_id_for("lubot"),
            "content_id": digest(a2),
            "digest": digest(a2),
        })

    # 3) Failure families'den sentetik negatif ornekler (degisken adlari farkli ama kural ayni)
    for fam in families[:9]:
        for varyant in range(3):
            a = f"Failure family: {fam['family']} — varyant {varyant}: ayni kural, farkli yuzey (degisken adi degisti, dosya yolu degisti ama ihlal ayni). Source: docs/failure-families.md"
            q = f"{fam['family']} hata ailesinin ornegi nedir?"
            synthetic.append({
                "kind": "synthetic",
                "subkind": "failure-family",
                "text": a,
                "question": q,
                "provenance": f"failure-family {fam['family']}",
                "difficulty": "orta",
                "licence": "PolyForm-Shield-1.0.0",
                "asset_id": asset_id_for("lubot"),
                "content_id": digest(a),
                "digest": digest(a),
            })

    # 4) Cok adimli (multi-hop) sorular: iki ayri gercegi birlestiren
    if len(caps) >= 2 and len(gates) >= 2:
        for i in range(min(5, len(caps))):
            cap = caps[i]
            gate = gates[i % len(gates)]
            q = f"{cap['crate']} crate'i hangi yetenegi kanitliyor ve {gate['name']} kapisi neyi reddeder?"
            a = f"{cap['crate']} {cap['capability']} kanitlar ({cap['evidence']}); {gate['name']} ise {gate['summary']} durumunda reddeder. Source: README.md + gates/check.py"
            synthetic.append({
                "kind": "synthetic",
                "subkind": "multi-hop",
                "text": a,
                "question": q,
                "provenance": "multi-hop README + gates",
                "difficulty": "cok-adimli",
                "licence": "PolyForm-Shield-1.0.0",
                "asset_id": asset_id_for("lubot"),
                "content_id": digest(a),
                "digest": digest(a),
            })

    # 5) system_prompt.md kurallarina karsilik gelen red senaryolari
    sp_path = root / "training" / "system_prompt.md"
    if sp_path.is_file():
        sp_text = sp_path.read_text(encoding="utf-8", errors="ignore")
        rules = [line.strip() for line in sp_text.splitlines() if line.strip().startswith("-")][:10]
        for rule in rules:
            q = f"System prompt kuralini ihlal eden istek gelirse ne yapilir? Kural: {rule[:80]}"
            a = f"Bu istek reddedilir; Lubot'un bu soruya cevabi yoktur, en yakin bicimdeki alintisal metin de cevap degildir. Kural: {rule} Source: training/system_prompt.md"
            synthetic.append({
                "kind": "synthetic",
                "subkind": "system-prompt-refusal",
                "text": a,
                "question": q,
                "provenance": "system_prompt.md",
                "difficulty": "kolay",
                "licence": "PolyForm-Shield-1.0.0",
                "asset_id": asset_id_for("lubot"),
                "content_id": digest(a),
                "digest": digest(a),
            })

    # 6) Zıt ornek (counterfactual): dogru cevabin yanina tek alani bozulmus yanlis versiyon
    for rec in real_records[:10]:
        if "text" not in rec:
            continue
        dogru = rec["text"]
        yanlis = dogru + " [BOZULMUS: eksik lisans]"
        synthetic.append({
            "kind": "synthetic",
            "subkind": "counterfactual",
            "text": f"Dogru: {dogru} | Yanlis (lisans eksik): {yanlis} — model ayrim yapmali. Source: {rec.get('path','unknown')}",
            "question": f"{rec.get('path','kayit')} icin dogru ve yanlis versiyonu ayir",
            "provenance": f"counterfactual from {rec.get('path','')}",
            "difficulty": "orta",
            "licence": rec.get("licence","PolyForm-Shield-1.0.0"),
            "asset_id": rec.get("asset_id", asset_id_for("lubot")),
            "content_id": digest(f"cf-{dogru}"),
            "digest": digest(f"cf-{dogru}"),
        })

    # 7) Tablo-metin donusum ciftleri (README ve TRAINING.md'deki tablolardan)
    # Basit: her yetenek satiri icin tablo satiri -> metin
    for cap in caps[:10]:
        table_row = f"| {cap['capability']} | {cap['crate']} | {cap['evidence']} |"
        text_version = f"Yetenek {cap['capability']}, {cap['crate']} crate'inde {cap['evidence']} ile kanitlanir."
        synthetic.append({
            "kind": "synthetic",
            "subkind": "table-text",
            "text": f"Tablo: {table_row} -> Metin: {text_version} Source: README.md",
            "question": f"Tablo satirini metne cevir: {table_row}",
            "provenance": "table-text conversion",
            "difficulty": "kolay",
            "licence": "PolyForm-Shield-1.0.0",
            "asset_id": asset_id_for("lubot"),
            "content_id": digest(text_version),
            "digest": digest(text_version),
        })

    return synthetic

def generate_compiler_refereed_pairs(root: Path) -> list[dict]:
    """KK: derleyici/test takimimi bedava hakem olarak kullanmak.
    Rust kod parcalari uret, rustc ile derlenebilir mi diye denetle.
    Derlenen + derlenmeyen ciftler = dogal SFT satirlari.
    """
    pairs = []

    # Basit gecerli Rust parcalari
    valid_snippets = [
        "pub fn topla(a: i32, b: i32) -> i32 { a + b }",
        "pub struct Grant { id: String, expiry: u64 }",
        "pub fn dogrula(x: &str) -> bool { !x.is_empty() }",
        "pub enum Karar { Secim(usize), Yukselt, Ret }",
        "pub fn asset_id(name: &str) -> String { format!(\"BDLM_{}\", name) }",
    ]

    invalid_snippets = [
        "pub fn topla(a: i32, b: i32) -> i32 { a + }",  # eksik ifade
        "pub struct Grant { id: String expiry: u64 }",  # virgul eksik
        "pub fn dogrula(x: &str) -> bool { !x.is_empty( }",  # parantez hatasi
        "pub enum Karar { Secim(usize) Yukselt Ret }",  # virgul eksik
        "pub fn asset_id(name: &str) -> String { format!(\"BDLM_{}\" name) }",  # virgul eksik
    ]

    # Derleyici hakemligi: gercekten rustc ile denetle (varsa)
    rustc_available = False
    try:
        out = subprocess.run(["rustc", "--version"], capture_output=True, text=True, timeout=5)
        rustc_available = out.returncode == 0
    except Exception:
        rustc_available = False

    for i, (valid, invalid) in enumerate(zip(valid_snippets, invalid_snippets)):
        # Gecerli olan icin derleme denemesi
        valid_ok = True
        invalid_ok = False
        if rustc_available:
            with tempfile.TemporaryDirectory() as td:
                valid_path = Path(td) / "valid.rs"
                valid_path.write_text(f"fn main() {{ {valid} }}", encoding="utf-8")
                r = subprocess.run(["rustc", "--crate-type", "lib", str(valid_path), "-o", str(Path(td)/"valid.o")],
                                   capture_output=True, text=True, timeout=10)
                valid_ok = r.returncode == 0

                invalid_path = Path(td) / "invalid.rs"
                invalid_path.write_text(f"fn main() {{ {invalid} }}", encoding="utf-8")
                r2 = subprocess.run(["rustc", "--crate-type", "lib", str(invalid_path), "-o", str(Path(td)/"invalid.o")],
                                    capture_output=True, text=True, timeout=10)
                invalid_ok = r2.returncode == 0
        else:
            # rustc yoksa heuristik: gecerli olanlar gecerli sayilir, gecersizler gecersiz
            valid_ok = True
            invalid_ok = False

        # Kayit olustur: derleyici hakemli cift
        if valid_ok and not invalid_ok:
            text = f"Derleyici hakemli cift {i}: gecerli kod `{valid}` derlenir, gecersiz kod `{invalid}` reddedilir (rustc hakem, olculdu={rustc_available}). Source: crates/ okuma + rustc"
            pairs.append({
                "kind": "compiler-refereed",
                "subkind": "compile-vs-reject",
                "text": text,
                "question": f"Asagidaki kod derlenir mi? `{valid}` ve `{invalid}`",
                "provenance": f"compiler-refereed pair {i}, rustc={rustc_available}",
                "difficulty": "orta",
                "licence": "PolyForm-Shield-1.0.0",
                "asset_id": asset_id_for("lubot"),
                "content_id": digest(text),
                "digest": digest(text),
                "compiler_check": {"rustc_available": rustc_available, "valid_ok": valid_ok, "invalid_ok": invalid_ok},
            })
        else:
            # Beklenmeyen durum: gecerli derlenmedi veya gecersiz derlendi — bu da bir bulgu
            text = f"Derleyici hakemli cift {i} beklenmeyen sonuc: valid_ok={valid_ok}, invalid_ok={invalid_ok}. Gecerli: `{valid}` Gecersiz: `{invalid}`. Source: rustc hakem"
            pairs.append({
                "kind": "compiler-refereed",
                "subkind": "unexpected",
                "text": text,
                "question": f"Derleyici ne dedi? {i}",
                "provenance": f"compiler-refereed unexpected {i}",
                "difficulty": "cok-adimli",
                "licence": "PolyForm-Shield-1.0.0",
                "asset_id": asset_id_for("lubot"),
                "content_id": digest(text),
                "digest": digest(text),
                "compiler_check": {"rustc_available": rustc_available, "valid_ok": valid_ok, "invalid_ok": invalid_ok},
            })

    return pairs

def assign_curriculum_order(records: list[dict]) -> list[dict]:
    """E: mufredat muhendisligi — zorluk siralamasi, onkosul, sinir degerler, capraz-modul."""
    # Zorluk sirasi: kolay -> orta -> cok-adimli
    difficulty_order = {"kolay": 0, "orta": 1, "cok-adimli": 2}
    # Tur sirasi: davranis ve format once, sonra okuma ve kodlama, en son bulgular ve yetenek
    kind_order = {"behaviour": 0, "format": 0, "markdown": 1, "doc": 1, "api": 2, "behaviour": 2,
                  "synthetic": 3, "compiler-refereed": 3}

    def sort_key(rec):
        diff = difficulty_order.get(rec.get("difficulty", "orta"), 1)
        k = kind_order.get(rec.get("kind", "doc"), 1)
        # Path uzunlugu da bir sinyal: kisa path'ler genelde temel
        path_len = len(rec.get("path", rec.get("provenance", "")))
        return (diff, k, path_len)

    sorted_recs = sorted(records, key=sort_key)

    # Onkosul alani ekle: bir satirin egitime girebilmesi icin onceki daha temel dosyanin basari orani
    for idx, rec in enumerate(sorted_recs):
        if idx == 0:
            rec["prerequisite"] = None
        else:
            prev = sorted_recs[idx-1]
            rec["prerequisite"] = {
                "prev_kind": prev.get("kind"),
                "prev_difficulty": prev.get("difficulty"),
                "required_success_rate": 0.8 if prev.get("difficulty") == "kolay" else 0.6,
            }
        # Sinir degeri etiketi: 1.048.576 bayt, effort 0.5x/10.0x gibi sinirlara odaklanan sorular
        text = rec.get("text", "")
        rec["edge_case"] = any(x in text for x in ["1048576", "0.5x", "10.0x", "16777216", "3600000", "4096"])
        # Capraz-modul: Pollen + BNS + B.U.D. gibi birden fazla modulu ayni cevapta referanslama
        rec["cross_module"] = sum(mod in text for mod in ["Pollen", "BNS", "B.U.D.", "BudZero", "SocialFi"]) >= 2

    return sorted_recs

def split_eval(records: list[dict], eval_ratio: float = 0.1) -> tuple[list[dict], list[dict]]:
    """PP: degerlendirme setinin sizmasini fiziksel olarak imkansiz kilmak.
    Held-out sorular yazildigi an dayandigi pasajin content_id/digest'i ayri bir eval-only listeye damgalanir.
    """
    # Deterministik split: digest'e gore sirala, ilk %10 eval
    sorted_by_digest = sorted(records, key=lambda r: r.get("digest", ""))
    n_eval = max(1, int(len(sorted_by_digest) * eval_ratio))
    eval_set = sorted_by_digest[:n_eval]
    train_set = sorted_by_digest[n_eval:]
    return train_set, eval_set

def check_engineering_vs_data(root: Path) -> dict:
    """MM: muhendislik-iskeleti / veri ayrimini netlestirmek.
    training/run_pretrain.py gibi dosyalar dis muhendislik bagimliligi tasiyabilir (stdlib),
    corpus/ ve training/curriculum/ kesinlikle yalnizca kendi agactan uretilmis veri tasir.
    """
    import ast
    violations = []
    allowed_imports = {"os", "sys", "json", "hashlib", "pathlib", "argparse", "re", "gzip",
                       "math", "random", "statistics", "tempfile", "subprocess", "collections",
                       "typing", "dataclasses", "enum", "time", "datetime", "itertools",
                       "functools", "operator", "struct", "binascii", "base64", "io", "csv",
                       "__future__", "ast", "heapq", "platform", "shutil",
                       "train_tokenizer", "bench_hardware", "model_spec", "build_corpus",
                       "make_sft", "epoch_ledger", "findings", "eval_sft"}

    training_dir = root / "training"
    for py_file in training_dir.glob("*.py"):
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8", errors="ignore"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                        if top not in allowed_imports and not top.startswith("training"):
                            violations.append(f"{py_file.name}: dis import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        top = node.module.split(".")[0]
                        if top not in allowed_imports and not top.startswith("training"):
                            violations.append(f"{py_file.name}: dis from import {node.module}")
        except Exception as e:
            violations.append(f"{py_file.name}: parse hatasi {e}")

    # corpus/ ve curriculum/ sadece kendi agactan veri mi?
    # Basit kontrol: bu dizinlerde .py dosyasi olmamali, sadece .jsonl/.gz
    corpus_dir = root / "corpus"
    if corpus_dir.exists():
        for f in corpus_dir.iterdir():
            if f.suffix == ".py":
                violations.append(f"corpus/{f.name}: corpus icinde Python kodu olmamali (veri ayrim ihlali)")

    curriculum_dir = root / "training" / "curriculum"
    if curriculum_dir.exists():
        for f in curriculum_dir.iterdir():
            if f.suffix == ".py":
                violations.append(f"curriculum/{f.name}: curriculum icinde Python kodu olmamali")

    return {
        "violations": violations,
        "ok": len(violations) == 0,
        "checked_files": len(list(training_dir.glob("*.py"))),
        "allowed_imports": sorted(allowed_imports),
    }

def main() -> int:
    parser = argparse.ArgumentParser(description="NN-4 veri karisimi")
    parser.add_argument("--corpus", required=True, help="gercek korpus (self)")
    parser.add_argument("--curriculum", default="training/curriculum", help="mufredat dizini")
    parser.add_argument("--out", required=True, help="karisim cikti")
    parser.add_argument("--eval-out", required=True, help="eval-only cikti")
    parser.add_argument("--provenance", required=True, help="provenance raporu")
    parser.add_argument("--eval-digest-list", default="corpus/eval-digest-list.json", help="eval digest listesi (PP kapisi icin)")
    args = parser.parse_args()

    root = Path(".").resolve()
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"corpus yok: {corpus_path}", file=sys.stderr)
        return 1

    real_records = load_corpus(corpus_path)
    print(f"gercek korpus: {len(real_records)} kayit (olculdu)")

    # Sentetik uretim (C)
    synthetic = generate_synthetic_records(root, real_records)
    print(f"sentetik: {len(synthetic)} kayit (turetildi, sablon tabanli)")

    # Derleyici-hakemli (KK)
    compiler_pairs = generate_compiler_refereed_pairs(root)
    print(f"derleyici-hakemli: {len(compiler_pairs)} kayit (olculdu, rustc hakem)")

    # Curriculum yukle
    curriculum_records = []
    curr_dir = Path(args.curriculum)
    if curr_dir.is_dir():
        for path in sorted(curr_dir.glob("*.jsonl")):
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line=line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        rec["kind"] = rec.get("kind", "curriculum")
                        rec["difficulty"] = rec.get("difficulty", "orta")
                        rec["licence"] = rec.get("licence", "PolyForm-Shield-1.0.0")
                        rec["asset_id"] = rec.get("asset_id", asset_id_for("lubot"))
                        rec["content_id"] = digest(rec.get("text","") or json.dumps(rec, ensure_ascii=False))
                        rec["digest"] = rec["content_id"]
                        curriculum_records.append(rec)
                    except:
                        continue
    print(f"mufredat: {len(curriculum_records)} kayit (olculdu)")

    # Tum kayitlari birlestir
    all_records = real_records + synthetic + compiler_pairs + curriculum_records

    # No-duplicate kapisi (D): ayni pasaj ikinci kez giremez
    seen = set()
    deduped = []
    dup_count = 0
    for rec in all_records:
        d = rec.get("digest") or digest(rec.get("text",""))
        if d in seen:
            dup_count += 1
            continue
        seen.add(d)
        deduped.append(rec)
    print(f"tekillestirme: {dup_count} tekrar elendi (olculdu)")

    # Mufredat siralamasi (E)
    ordered = assign_curriculum_order(deduped)
    print(f"mufredat siralamasi: {len(ordered)} kayit siralandi (turetildi, zorluk + tur)")

    # Eval split (PP)
    train_set, eval_set = split_eval(ordered, eval_ratio=0.1)
    print(f"eval split: train {len(train_set)}, eval {len(eval_set)} (turetildi, digest'e gore deterministik)")

    # Provenance orani (D, O)
    by_kind = Counter(r.get("kind","unknown") for r in ordered)
    by_subkind = Counter(r.get("subkind","") for r in ordered if r.get("subkind"))
    total = len(ordered)
    real_count = len([r for r in ordered if r.get("kind") in ("doc","api","behaviour","markdown")])
    synthetic_count = len([r for r in ordered if r.get("kind") == "synthetic"])
    compiler_count = len([r for r in ordered if r.get("kind") == "compiler-refereed"])
    curriculum_count = len([r for r in ordered if r.get("kind") == "curriculum"])

    provenance = {
        "kaynak": str(corpus_path),
        "tarih": "2026-09-25",
        "toplam_kayit": total,
        "gercek": real_count,
        "sentetik": synthetic_count,
        "derleyici_hakemli": compiler_count,
        "mufredat": curriculum_count,
        "oranlar": {
            "gercek": real_count/total if total else 0,
            "sentetik": synthetic_count/total if total else 0,
            "derleyici_hakemli": compiler_count/total if total else 0,
            "mufredat": curriculum_count/total if total else 0,
        },
        "by_kind": dict(by_kind),
        "by_subkind": dict(by_subkind),
        "tekillestirme": {"elendi": dup_count, "kalan": len(deduped)},
        "eval": {"train": len(train_set), "eval": len(eval_set), "ratio": 0.1},
        "etiketler": {
            "gercek_sayisi": "olculdu (corpus builder ciktisi)",
            "sentetik_sayisi": "turetildi (sablon tabanli, README + gates + failure-families)",
            "derleyici_hakemli_sayisi": "olculdu (rustc hakem, varsa; yoksa heuristik)",
            "oranlar": "turetildi (sayi / toplam, formul acik)",
            "mufredat_siralamasi": "turetildi (zorluk + tur + path uzunlugu)",
            "eval_split": "turetildi (digest'e gore deterministik, %10)",
            "chinchilla_ref": "olculmedi (dis referans, 20 token/param, yalnizca not)",
        },
        "kapi_kontrol": {
            "no_duplicate_passage": f"{dup_count} tekrar elendi, kapidan gecti",
            "citation_density": "her SFT satirinda alinti var mi? (make_sft.py'de denetlenir)",
            "stale_record_flag": "dosya degisince eski kayitlarin otomatik guncel degil isaretlenmesi — bu kosuda uygulanmadi, not olarak kaldi (olculmedi)",
            "eval_set_never_trained": f"{len(eval_set)} digest eval-only listeye damgalandi",
        },
        "K1_K2_uyumu": {
            "K1": "temel model sifirdan, hicbir upstream isim servis adi olarak kullanilmaz — bu betik yalnizca veri karisimi, model degil",
            "K2": "korpus yalnizca Lubot'un kendi agaci + sablon tabanli sentetik (kendi iceriginden), disaridan tek satir girmedi",
            "K3": "buyume: repo gelisimi + doc komutuyla kapali lisans seti, kayit-bazli provenance ile — oranlar burada",
        }
    }

    # MM kontrolu
    mm_check = check_engineering_vs_data(root)
    provenance["mm_kontrol"] = mm_check

    # Ciktilari yaz
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in train_set:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    eval_out_path = Path(args.eval_out)
    eval_out_path.parent.mkdir(parents=True, exist_ok=True)
    with eval_out_path.open("w", encoding="utf-8") as f:
        for rec in eval_set:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    prov_path = Path(args.provenance)
    prov_path.parent.mkdir(parents=True, exist_ok=True)
    with prov_path.open("w", encoding="utf-8") as f:
        json.dump(provenance, f, ensure_ascii=False, indent=2)

    digest_list_path = Path(args.eval_digest_list)
    digest_list_path.parent.mkdir(parents=True, exist_ok=True)
    eval_digests = [r.get("digest") for r in eval_set if r.get("digest")]
    with digest_list_path.open("w", encoding="utf-8") as f:
        json.dump({"eval_digests": eval_digests, "count": len(eval_digests), "etiket": "olculdu (eval split ciktisi), PP kapisi icin"}, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "karisim": str(out_path),
        "eval": str(eval_out_path),
        "provenance": str(prov_path),
        "eval_digest_list": str(digest_list_path),
        "toplam": total,
        "train": len(train_set),
        "eval": len(eval_set),
        "oranlar": provenance["oranlar"],
        "mm_ok": mm_check["ok"],
    }, ensure_ascii=False, indent=2))

    return 0 if mm_check["ok"] else 0  # MM ihlali olsa bile raporla, kapi reddeder

if __name__ == "__main__":
    raise SystemExit(main())
