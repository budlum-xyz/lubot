#!/usr/bin/env python3
"""Build the knowledge corpus from one or more budlum-xyz checkouts.

What goes in, and why:

* **documentation** - intent. Why a rule exists, which is the part source code
  never states.
* **public signatures** - surface. What can be called, and with what.
* **test names** - proven behaviour. A test name is the one sentence in a
  repository that someone had to make true.
* **gate names** - the rules that block a merge.

What stays out: raw function bodies. A model trained on raw source learns to
autocomplete source; the job here is to explain a protocol.

Every record carries where it came from - source name, path and line range -
so an answer built from it can be walked back to the file. Each source repo
stamps its records with its own licence and its own pre-issuance provenance
pair (asset_id + content_id, one asset_id per source repo), so the share of
each source in the corpus stays measurable (A-section rule). Records built
from several sources in one run are deduplicated across sources: the same
passage (the shared licence text, for example) enters once.

Nothing outside the given source trees enters, and a source whose licence is
not in the closed set is refused at the door, not filtered later.

    # self corpus (CI): one repository
    python3 training/build_corpus.py --repo . --out corpus/knowledge-self.jsonl.gz

    # surface corpus (operator): a sources manifest
    python3 training/build_corpus.py --sources manifest.json --out corpus/budlum-yuzeyi.jsonl.gz

The manifest lists sources explicitly, so the build is a pure function of
the trees and the manifest (paths, names, per-source curation):

    {"sources": [
      {"path": "../lubot", "name": "lubot"},
      {"path": "../budlum", "name": "budlum"},
      {"path": "../workspace", "name": "workspace", "root_only": true}
    ]}
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path

SKIP_DIRS = {".git", "target", "node_modules", "corpus", ".github"}

ALLOWED_LICENCES = {"MIT", "Apache-2.0", "PolyForm-Shield-1.0.0"}


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def asset_id_for(source_name: str) -> str:
    """Kaynak repo basina kanonik (on-issuance) asset kimligi."""
    return digest(f"BDLM_LUBOT_CORPUS_ASSET_V1|{source_name}")


def detect_licence(root: Path) -> str:
    """Kaynak repoya kendi lisans dosyasindan okur; kapali setin disina
    dusen ya da lisanssiz kaynak kapidan reddedilir."""
    for name in ("LICENSE.md", "LICENSE"):
        path = root / name
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="ignore")[:4000]
            if "PolyForm Shield License 1.0.0" in text:
                return "PolyForm-Shield-1.0.0"
            if "MIT License" in text:
                return "MIT"
            if "Apache License" in text and "Version 2.0" in text:
                return "Apache-2.0"
    raise SystemExit(
        f"lisans okunamadi ya da kapali setin disinda: {root} "
        f"(izinli: {sorted(ALLOWED_LICENCES)})"
    )


def walk(root: Path, root_only: bool = False):
    if root_only:
        for path in sorted(root.iterdir()):
            if path.is_file():
                yield path
        return
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        yield path


def rust_records(path: Path, rel: str):
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    doc_block: list[str] = []
    doc_start = 0
    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        if stripped.startswith("//!") or stripped.startswith("///"):
            if not doc_block:
                doc_start = i
            doc_block.append(stripped.lstrip("/!").strip())
            continue

        signature = re.match(r"pub (fn|struct|enum|trait|const|type) ([A-Za-z0-9_]+)", stripped)
        test_name = re.match(r"fn ([a-z0-9_]+)\(\)", stripped) if "#[test]" in "".join(lines[max(0, i - 2) : i]) else None

        if doc_block and (signature or test_name or stripped == ""):
            text = " ".join(w for w in doc_block if w)
            if len(text) > 40:
                yield {
                    "kind": "doc",
                    "text": text,
                    "path": rel,
                    "lines": [doc_start, i - 1],
                }
                # JJ: belge ile imza ayri kayitlar degil, bir cift olarak da girer:
                # "ne cagrilir" ile "neden var" arasindaki bag burada kurulur.
                if signature:
                    yield {
                        "kind": "api-doc-pair",
                        "text": f"{rel}: `{stripped.rstrip(' {')}` - {text}",
                        "path": rel,
                        "lines": [doc_start, i],
                    }
            doc_block = []

        if signature:
            yield {
                "kind": "api",
                "text": f"{rel} exposes `{stripped.rstrip(' {')}`.",
                "path": rel,
                "lines": [i, i],
            }
        if test_name:
            sentence = test_name.group(1).replace("_", " ")
            yield {
                "kind": "behaviour",
                "text": f"Proven in {rel}: {sentence}.",
                "path": rel,
                "lines": [i, i],
            }

        impl_match = re.match(
            r"impl(?:<[^>]*>)?\s+([A-Za-z0-9_:]+)(?:<[^>]*>)?\s+for\s+([A-Za-z0-9_:]+)",
            stripped,
        )
        if impl_match:
            # JJ: "kim neyi uyguluyor" iliskisi; trait tanimi ile impl blogu
            # arasindaki bagi ayri bir kayit turu yapar.
            yield {
                "kind": "trait-impl",
                "text": f"{rel}: `{impl_match.group(2)}` implements `{impl_match.group(1)}`.",
                "path": rel,
                "lines": [i, i],
            }


def markdown_records(path: Path, rel: str):
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    buffer: list[str] = []
    start = 1
    for i, line in enumerate(lines, 1):
        if line.strip():
            if not buffer:
                start = i
            buffer.append(line.strip())
            continue
        if buffer:
            text = " ".join(buffer)
            if len(text) > 60 and not text.startswith("|"):
                yield {"kind": "markdown", "text": text, "path": rel, "lines": [start, i - 1]}
            buffer = []
    if buffer:
        text = " ".join(buffer)
        if len(text) > 60:
            yield {"kind": "markdown", "text": text, "path": rel, "lines": [start, len(lines)]}


def _fonksiyon_govdesi(text: str, ad: str) -> str:
    """Bir fonksiyonun govdesi; sonraki ust duzey def'e kadar."""
    bas = text.find(f"def {ad}(")
    if bas < 0:
        return ""
    kalan = text[bas:]
    son = kalan.find("\ndef ", 10)
    return kalan if son < 0 else kalan[:son]


def gate_records(root: Path):
    gate_file = root / "gates" / "check.py"
    if not gate_file.is_file():
        return
    text = gate_file.read_text(encoding="utf-8")
    for match in re.finditer(r'def (gate_[a-z0-9_]+)\(\) -> str:\n    """(.+?)"""', text, re.S):
        name = match.group(1).removeprefix("gate_").replace("_", "-")
        summary = " ".join(match.group(2).split())
        yield {
            "kind": "doc",
            "text": f"The `{name}` gate blocks a merge unless: {summary}",
            "path": "gates/check.py",
            "lines": [text[: match.start()].count("\n") + 1, text[: match.end()].count("\n") + 1],
        }
    # KK: iddia ile onu curuten kanarya eslesir; hakem mekanik olmali.
    for kayit in re.finditer(
        r'"(?P<ad>[a-z0-9-]+)":\s*\(\s*gate_[a-z0-9_]+,\s*(?P<kanarya>selftest_[a-z0-9_]+)\s*,?\s*\)',
        text,
        re.S,
    ):
        ad = kayit.group("ad")
        kanarya = kayit.group("kanarya")
        govde = _fonksiyon_govdesi(text, kanarya)
        reddeden = [
            satir.strip()
            for satir in govde.splitlines()
            if "assert " in satir or "SystemExit" in satir or "AssertionError" in satir
        ]
        ozet = reddeden[0] if reddeden else "kanaryada red yok"
        satir_no = text[: kayit.start()].count("\n") + 1
        yield {
            "kind": "gate-pair",
            "text": f"`{ad}` gate: checked by `{kanarya}` - {ozet}",
            "path": "gates/check.py",
            "lines": [satir_no, satir_no],
        }


MERMAID_KENAR = re.compile(
    r"^\s*([A-Za-z0-9_]+)(?:\[[^\]]*\]|\{[^}]*\}|\([^)]*\))?\s*-{1,2}>+\s*([A-Za-z0-9_]+)"
)
MERMAID_ETIKET = re.compile(r"([A-Za-z0-9_]+)\[([^\]]+)\]")
SVG_ETIKET = re.compile(r"<text[^>]*>([^<]{2,})</text>")


def mermaid_kayitlari(metin: str, rel: str, satir_temel: int):
    """QQ: metin diyagrami okunur - uretilmez. Kenarlar ve etiketler ayri kayit."""
    for no, satir in enumerate(metin.splitlines(), satir_temel):
        kenar = MERMAID_KENAR.match(satir)
        if kenar:
            yield {
                "kind": "diagram",
                "text": f"{rel}: diagram edge `{kenar.group(1)}` -> `{kenar.group(2)}`.",
                "path": rel,
                "lines": [no, no],
            }
        for ad, etiket in MERMAID_ETIKET.findall(satir):
            yield {
                "kind": "diagram",
                "text": f'{rel}: diagram node `{ad}` is labelled "{etiket}".',
                "path": rel,
                "lines": [no, no],
            }


def mermaid_bloklari(path: Path, rel: str):
    """```mermaid citleri icindeki diyagramlar; satir numaralari dosyadan."""
    icinde = False
    baslangic = 0
    govde: list[str] = []
    for no, satir in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        s = satir.strip()
        if not icinde and s.startswith("```mermaid"):
            icinde, baslangic, govde = True, no, []
        elif icinde and s.startswith("```"):
            icinde = False
            yield from mermaid_kayitlari("\n".join(govde), rel, baslangic + 1)
        elif icinde:
            govde.append(satir)


def svg_kayitlari(path: Path, rel: str):
    """SVG'den yalniz metin etiketleri okunur; kenarlar okunmaz ve bu soylenir."""
    for no, satir in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        for etiket in SVG_ETIKET.findall(satir):
            yield {
                "kind": "diagram",
                "text": f'{rel}: diagram label "{etiket.strip()}" (SVG text; edges are not read).',
                "path": rel,
                "lines": [no, no],
            }


def dependency_records(path: Path, rel: str):
    """JJ: Cargo.toml bagimlilik grafigi ayri bir "modul iliski" korpus turu.

    Derleyicinin kendi okudugu dosyadan turedigi icin %100 mekanik: "X crate'i
    Y'ye mi bagimli" sorusunun cevabi bir yargi degil, manifestin kendisi.
    """
    import tomllib

    metin = path.read_text(encoding="utf-8", errors="ignore")
    satirlar = metin.splitlines()

    def satir_no(ad: str) -> int:
        for no, satir in enumerate(satirlar, 1):
            s = satir.strip()
            if s.startswith(f"{ad} =") or s.startswith(f'"{ad}"') or s.startswith(f"{ad}."):
                return no
        return 1

    veri = tomllib.loads(metin)
    paket = veri.get("package", {}).get("name") or rel
    for bolum in ("dependencies", "dev-dependencies", "build-dependencies"):
        for ad in sorted(veri.get(bolum, {}) or {}):
            yield {
                "kind": "dependency-edge",
                "text": f"`{paket}` depends on `{ad}` ({bolum}).",
                "path": rel,
                "lines": [satir_no(ad), satir_no(ad)],
            }
    for uye in sorted(veri.get("workspace", {}).get("members", []) or []):
        yield {
            "kind": "dependency-edge",
            "text": f"Workspace member declared in `{rel}`: `{uye}`.",
            "path": rel,
            "lines": [satir_no(uye), satir_no(uye)],
        }


def collect(root: Path, source: str, root_only: bool):
    """Tek kaynagin kayitlari; kaynak adi kayda islenir."""
    for path in walk(root, root_only=root_only):
        rel = str(path.relative_to(root))
        if path.suffix == ".rs":
            yield from rust_records(path, rel)
        elif path.suffix == ".md":
            yield from mermaid_bloklari(path, rel)
            yield from markdown_records(path, rel)
        elif path.suffix in (".mmd", ".mermaid"):
            yield from mermaid_kayitlari(
                path.read_text(encoding="utf-8", errors="ignore"), rel, 1
            )
        elif path.suffix == ".svg":
            yield from svg_kayitlari(path, rel)
        elif path.name == "Cargo.toml":
            yield from dependency_records(path, rel)
    yield from gate_records(root)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=None,
                        help="tek kaynak checkout'u (CI self kurulusu)")
    parser.add_argument("--source-name", default=None,
                        help="--repo ile kaynak adi (varsayilan: dizin adi)")
    parser.add_argument("--sources", default=None,
                        help="coklu kaynak manifesti (JSON); --repo ile birlikte kullanilmaz")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.repo and args.sources:
        raise SystemExit("--repo ve --sources ayni anda verilemez; ya tek kaynak ya manifest")

    if args.repo:
        root = Path(args.repo).resolve()
        sources = [{"path": str(root), "name": args.source_name or root.name}]
    elif args.sources:
        manifest = json.loads(Path(args.sources).read_text(encoding="utf-8"))
        sources = manifest["sources"]
        if not sources:
            raise SystemExit("manifest bos: en az bir kaynak gerekir")
    else:
        raise SystemExit("--repo ya da --sources gerekli")

    records = []
    per_source = []
    for spec in sources:
        root = Path(spec["path"]).resolve()
        if not root.is_dir():
            raise SystemExit(f"kaynak dizin yok: {root}")
        name = spec["name"]
        root_only = bool(spec.get("root_only", False))
        licence = spec.get("licence") or detect_licence(root)
        if licence not in ALLOWED_LICENCES:
            raise SystemExit(f"kaynak lisansi kapali setin disinda: {name} -> {licence}")
        attribution = spec.get("attribution") or f"{name} (kendi eser; {licence})"
        asset_id = asset_id_for(name)
        batch = list(collect(root, name, root_only))
        for record in batch:
            record["source"] = name
        records.extend(batch)
        per_source.append({
            "source": name,
            "path": str(root),
            "root_only": root_only,
            "licence": licence,
            "attribution": attribution,
            "asset_id": asset_id,
            "raw_records": len(batch),
        })
    spec_by_name = {s["source"]: s for s in per_source}

    seen: set[str] = set()
    unique = []
    for record in records:
        key = digest(record["text"])
        if key in seen:
            continue
        seen.add(key)
        spec = spec_by_name[record["source"]]
        record["digest"] = key
        record["licence"] = spec["licence"]
        record["attribution"] = spec["attribution"]
        record["content_id"] = key
        record["asset_id"] = spec["asset_id"]
        record["asset_id_pending"] = True
        unique.append(record)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".gz":
        handle = gzip.open(out, "wt", encoding="utf-8")
    else:
        handle = out.open("w", encoding="utf-8")
    with handle:
        for record in unique:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    by_kind: dict[str, int] = {}
    by_source_final: dict[str, int] = {}
    characters = 0
    for record in unique:
        by_kind[record["kind"]] = by_kind.get(record["kind"], 0) + 1
        by_source_final[record["source"]] = by_source_final.get(record["source"], 0) + 1
        characters += len(record["text"])
    print(json.dumps({
        "records": len(unique),
        "by_kind": by_kind,
        "characters": characters,
        "approx_tokens": characters // 4,
        "by_source_raw": {s["source"]: s["raw_records"] for s in per_source},
        "by_source": by_source_final,
        "sources": per_source,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
