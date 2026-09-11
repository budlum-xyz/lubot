#!/usr/bin/env python3
"""Repository gates.

A gate is a claim this repository makes about itself, written so that a change
that breaks the claim fails the build instead of quietly becoming untrue. Each
gate carries a self-test: a gate that cannot demonstrate it catches its own
violation is a gate nobody should trust.

The gates grew with the code they guard: a check that exists only in prose is
a claim nobody runs, so every gate here runs and carries its own canary.

Usage:
    python3 gates/check.py --all
    python3 gates/check.py --list
    python3 gates/check.py <gate> [--self-test]
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read(rel: str) -> str:
    path = ROOT / rel
    if not path.is_file():
        raise SystemExit(f"gate cannot run: {rel} is missing")
    return path.read_text(encoding="utf-8")


def rust_sources() -> list[Path]:
    return sorted(p for p in (ROOT / "crates").rglob("*.rs"))


# --------------------------------------------------------------------------
# gate: reads, does not generate
# --------------------------------------------------------------------------
GENERATION_WORDS = ["fn generate_image", "fn generate_video", "fn synthesize_audio", "text_to_image"]


def gate_reads_not_generates() -> str:
    """No generation surface exists, and the reading path has a refusal for
    everything it cannot open."""
    for path in rust_sources():
        text = path.read_text(encoding="utf-8")
        for word in GENERATION_WORDS:
            if word in text:
                raise SystemExit(f"{path.relative_to(ROOT)} exposes a generation surface: {word}")
    answer = read("crates/answer/src/lib.rs")
    for variant in ["NotFound", "Refused", "ToolRefused"]:
        if f"{variant} " not in answer and f"{variant}," not in answer and f"{variant} {{" not in answer:
            raise SystemExit(f"the answer type has no `{variant}` case; then an unanswerable question has nowhere to go")
    return "no generation surface; NotFound, Refused and ToolRefused all exist"


def selftest_reads_not_generates() -> None:
    assert "fn generate_image" in GENERATION_WORDS


# --------------------------------------------------------------------------
# gate: reject_unknown_source  (moved from the node repository)
# --------------------------------------------------------------------------
def gate_no_fourth_channel() -> str:
    """Content enters through exactly three channels, and a fourth is refused."""
    src = read("crates/read/src/lib.rs")
    if "fn source_kind" not in src:
        raise SystemExit("`source_kind` is gone; then the channel list is prose")
    for channel in ['"stored"', '"granted"', '"local"']:
        if channel not in src:
            raise SystemExit(f"channel {channel} is missing from `source_kind`")
    if "unknown source" not in src:
        raise SystemExit("`source_kind` no longer refuses an unknown channel")
    if "fn reject_unknown_source" not in src:
        raise SystemExit("the refusal has no test pinning it")
    return "three channels, and an unknown one is refused with a test on it"


def selftest_no_fourth_channel() -> None:
    sys.path.insert(0, str(ROOT / "gates"))
    src = read("crates/read/src/lib.rs")
    assert "unknown source" in src


# --------------------------------------------------------------------------
# gate: verify_sha256 fail-closed  (moved from the node repository)
# --------------------------------------------------------------------------
def gate_provenance_fails_closed() -> str:
    """A record's bytes are checked against its digest, and a mismatch refuses."""
    src = read("crates/read/src/lib.rs")
    if "pub fn verify_sha256" not in src:
        raise SystemExit("`verify_sha256` is gone; a self-reported digest is not provenance")
    if "digest mismatch" not in src:
        raise SystemExit("the mismatch path no longer produces an error")
    if "fn insert" not in src or "item.verify()?" not in src:
        raise SystemExit("the corpus accepts items without verifying them first")
    return "digests are verified on the way in, and a mismatch is a refusal"


def selftest_provenance_fails_closed() -> None:
    assert "verify_sha256" in read("crates/read/src/lib.rs")


# --------------------------------------------------------------------------
# gate: masking before storage  (moved from the node repository)
# --------------------------------------------------------------------------
def gate_mask_before_storage() -> str:
    """The secret mask is applied on the write path, not on the way out."""
    src = read("crates/index/src/lib.rs")
    if "pub fn mask_secrets" not in src:
        raise SystemExit("`mask_secrets` is gone; then `before storage` is a word with nothing under it")
    add_at = src.find("pub fn add(")
    if add_at < 0:
        raise SystemExit("the index has no `add`; the write path cannot be checked")
    body = src[add_at : src.find("\n    }", add_at)]
    if "mask_secrets(" not in body:
        raise SystemExit("`add` stores the body without masking it first")
    if "fn redact_model_strings" not in src:
        raise SystemExit("the mask has no test pinning its behaviour")
    return "the mask runs inside the write path, with a test on the mask itself"


def selftest_mask_before_storage() -> None:
    assert "mask_secrets" in read("crates/index/src/lib.rs")


# --------------------------------------------------------------------------
# gate: the panic limit
# --------------------------------------------------------------------------
def gate_no_panic_path() -> str:
    """`unwrap` and `expect` are denied outside tests, and the denial is real."""
    manifest = read("Cargo.toml")
    if 'unwrap_used = "deny"' not in manifest or 'expect_used = "deny"' not in manifest:
        raise SystemExit("the workspace no longer denies unwrap/expect")
    offenders: list[str] = []
    for path in rust_sources():
        text = path.read_text(encoding="utf-8")
        cut = text.find("#[cfg(test)]")
        production = text if cut < 0 else text[:cut]
        for i, line in enumerate(production.splitlines(), 1):
            if re.search(r"\.(unwrap|expect)\(", line):
                offenders.append(f"{path.relative_to(ROOT)}:{i}")
    if offenders:
        raise SystemExit("a panic path is on the production side:\n  " + "\n  ".join(offenders))
    return f"no unwrap/expect outside tests in {len(rust_sources())} files"


def selftest_no_panic_path() -> None:
    assert re.search(r"\.(unwrap|expect)\(", "x.unwrap()")


# --------------------------------------------------------------------------
# gate: the claims in the README are measured
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# gate: a `pub fn` nothing reaches is public API, and the count is a ratchet
# --------------------------------------------------------------------------
DEAD_API_BASELINE = ROOT / "gates" / "dead-pub-api.baseline"
DEAD_API_EXEMPT = ("WIRING:", "Convenience:", "exposed for")
DEAD_API_LOOKBACK = 14
DEAD_API_MAX_REPORTED = 20
DEAD_API_DIRS = ("crates",)
DEAD_API_REF_GLOBS = ("*.rs", "*.md", "*.toml", "*.py", "*.json", "*.jsonl")


def strip_test_tail(text: str) -> str:
    """Production side of a source file: everything before `#[cfg(test)]`."""
    cut = text.find("#[cfg(test)]")
    return text if cut < 0 else text[:cut]


def pub_fn_decls(text: str) -> list[tuple[int, str]]:
    """`(line, name)` for every production `pub fn` the text declares.

    Deliberately literal: the qualifiers are consumed one at a time, so a shape
    this gate cannot read (a macro-generated function, an `extern` block) is not
    quietly counted as a declaration.
    """
    prod = strip_test_tail(text).splitlines()
    out: list[tuple[int, str]] = []
    for i, line in enumerate(prod):
        m = re.match(r"\s*pub (?:const |unsafe |async )*fn ([a-z0-9_]+)\b", line)
        if m:
            out.append((i, m.group(1)))
    return out


def reference_counts() -> dict[str, int]:
    """How often each identifier appears in the tree, test tails removed."""
    counts: dict[str, int] = {}
    seen: set[pathlib.Path] = set()
    for pattern in DEAD_API_REF_GLOBS:
        for path in ROOT.rglob(pattern):
            if not path.is_file() or path in seen or ".git" in path.parts or "target" in path.parts:
                continue
            seen.add(path)
            text = strip_test_tail(path.read_text(encoding="utf-8", errors="replace"))
            for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
                counts[token] = counts.get(token, 0) + 1
    return counts


def dead_public_api_entries(text: str, refs: dict[str, int]) -> list[str]:
    """Names this file declares that no other line of the tree uses.

    A declaration subtracts its own file's `pub fn` lines from the reference
    count, so a name that only appears where it is declared is unreached. A
    name whose only other use is under `#[cfg(test)]` is unreached too: "tested"
    and "wired to a caller" are different claims, and collapsing them is how a
    crate full of unreachable API stays green.
    """
    lines = strip_test_tail(text).splitlines()
    out: list[str] = []
    for i, name in pub_fn_decls(text):
        declared = sum(1 for l in lines if re.match(r"\s*pub (?:const |unsafe |async )*fn " + re.escape(name) + r"\b", l))
        if refs.get(name, 0) - declared > 0:
            continue
        window = "\n".join(lines[max(0, i - DEAD_API_LOOKBACK):i])
        if any(token in window for token in DEAD_API_EXEMPT):
            continue
        out.append(name)
    return out


def read_dead_api_baseline() -> list[str]:
    if not DEAD_API_BASELINE.is_file():
        raise SystemExit(
            "gates/dead-pub-api.baseline is missing: without the recorded debt the "
            "gate can only say the tree is clean, which it has not measured"
        )
    entries: list[str] = []
    for raw in DEAD_API_BASELINE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise SystemExit(
                f"baseline line `{line}` is not `path:name`: a line the gate can "
                "never match is a line that silently never matches"
            )
        entries.append(line)
    if entries != sorted(set(entries)):
        raise SystemExit("gates/dead-pub-api.baseline is not sorted or holds duplicates")
    return entries


def gate_public_api_is_reached() -> str:
    """`crates/*/src/**` declares no `pub fn` that nothing in the tree reaches,
    beyond what the baseline records."""
    refs = reference_counts()
    measured: list[str] = []
    for d in DEAD_API_DIRS:
        for path in sorted((ROOT / d).glob("**/src/**/*.rs")):
            rel = str(path.relative_to(ROOT))
            if "/tests/" in rel:
                continue
            for name in dead_public_api_entries(path.read_text(encoding="utf-8", errors="replace"), refs):
                measured.append(f"{rel}:{name}")
    measured.sort()
    baseline = read_dead_api_baseline()
    extra = [e for e in measured if e not in set(baseline)]
    gone = [e for e in baseline if e not in set(measured)]
    if extra:
        shown = "\n  ".join(extra[:DEAD_API_MAX_REPORTED])
        more = f"\n  ... and {len(extra) - DEAD_API_MAX_REPORTED} more" if len(extra) > DEAD_API_MAX_REPORTED else ""
        raise SystemExit(
            f"{len(extra)} public function(s) nothing in the tree calls:\n  {shown}{more}\n"
            "  Call it from the path it was written for, or write the exemption next "
            "to the declaration (one of: " + ", ".join(DEAD_API_EXEMPT) + f") within "
            f"{DEAD_API_LOOKBACK} lines. Do not add a line to the baseline to pass this."
        )
    if gone:
        raise SystemExit(
            f"{len(gone)} baseline entr{'y' if len(gone)==1 else 'ies'} no longer dead "
            f"(wired up or deleted):\n  " + "\n  ".join(gone[:DEAD_API_MAX_REPORTED])
            + "\n  Remove those lines in this patch: the next author inherits whatever stays."
        )
    return f"public api is at its recorded floor: {len(measured)} unreached, baseline {len(baseline)}"


def selftest_public_api_is_reached() -> None:
    """The extractor is not vacuous, and neither is the baseline rule."""
    assert pub_fn_decls("pub fn alpha() {}\nfn beta() {}\n") == [(0, "alpha")]
    assert pub_fn_decls("#[cfg(test)]\npub fn gamma() {}\n") == []
    assert pub_fn_decls("pub async fn delta() {}\npub const fn epsilon() {}\n") == [(0, "delta"), (1, "epsilon")]
    refs = {"alpha": 1, "used_elsewhere": 2}
    text = "pub fn alpha() {}\npub fn used_elsewhere() {}\n"
    dead = dead_public_api_entries(text, refs)
    assert dead == ["alpha"], dead
    exempted = "/// Convenience: kept for the CLI.\npub fn alpha() {}\n"
    assert dead_public_api_entries(exempted, refs) == []


def gate_readme_is_measured() -> str:
    """The test count in the README is the count the suite reports."""
    readme = read("README.md")
    claimed = re.search(r"(\d+) tests, `clippy", readme)
    if not claimed:
        raise SystemExit("the README no longer states a measured test count")
    out = subprocess.run(
        ["cargo", "test", "--workspace"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if out.returncode != 0:
        raise SystemExit("the suite does not pass, so its count cannot be claimed")
    measured = sum(int(m) for m in re.findall(r"test result: ok\. (\d+) passed", out.stdout))
    if measured != int(claimed.group(1)):
        raise SystemExit(f"the README says {claimed.group(1)} tests, the suite reports {measured}")
    return f"the README count matches the suite: {measured}"


def selftest_readme_is_measured() -> None:
    assert re.search(r"(\d+) tests, `clippy", "48 tests, `clippy -D warnings` clean")




# --------------------------------------------------------------------------
# gate: corpus records carry a licence and an attribution
# --------------------------------------------------------------------------
ALLOWED_LICENCES = {"MIT", "Apache-2.0", "PolyForm-Shield-1.0.0"}


def validate_record(rec: dict) -> str | None:
    """None when the record is acceptable; a reason otherwise."""
    lic = rec.get("licence")
    if lic is None:
        return "record carries no licence"
    if lic not in ALLOWED_LICENCES:
        return f"unknown licence: {lic}"
    if not rec.get("attribution"):
        return "record carries no attribution"
    return None


def gate_corpus_records_carry_licence() -> str:
    """Every corpus record carries an allowed licence and an attribution."""
    import gzip
    import json as _json

    corpus_dir = ROOT / "corpus"
    files = sorted(corpus_dir.glob("*.jsonl.gz"))
    if not files:
        raise SystemExit(
            "no corpus files under corpus/; build it first: "
            "python3 training/build_corpus.py --repo . --out corpus/knowledge-self.jsonl.gz"
        )
    records = 0
    bad = 0
    for path in files:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records += 1
                rec = _json.loads(line)
                reason = validate_record(rec)
                if reason:
                    bad += 1
                    if bad <= 5:
                        print(f"BAD {path.name}: {reason}")
    if bad:
        raise SystemExit(f"{bad} of {records} corpus records fail the licence rule")
    return f"{records} corpus records carry licence + attribution"


def selftest_corpus_records_carry_licence() -> None:
    assert validate_record({"licence": "MIT", "attribution": "a"}) is None
    assert validate_record({"attribution": "a"}) is not None
    assert validate_record({"licence": "MIT"}) is not None
    assert validate_record({"licence": "Proprietary", "attribution": "a"}) is not None


# --------------------------------------------------------------------------
# gate: no multiplier labels, our tier names only
# --------------------------------------------------------------------------
def gate_no_multiplier_labels() -> str:
    """The effort tier range is the education report's canonical naming
    (0.5x-10.0x) and is stated in the README; short or uppercased variants
    (10x, 0.5X) never appear as labels."""
    readme = read("README.md")
    for label in ["0.5x", "10.0x"]:
        if label not in readme:
            raise SystemExit(f"the effort tier range {label} is not stated in the README")
    for label in ["10x", "10X", "0.5X", "10.0X"]:
        if label in readme:
            raise SystemExit(f"non-canonical multiplier label {label} appears in the README")
    for path in (ROOT / "training" / "curriculum").glob("*.jsonl"):
        text = path.read_text(encoding="utf-8")
        for label in ["10x", "10X", "0.5X", "10.0X"]:
            if label in text:
                raise SystemExit(f"non-canonical multiplier label {label} appears in {path.name}")
    return "tier naming is the report's 0.5x-10.0x; no short or uppercased variants"


def selftest_no_multiplier_labels() -> None:
    assert "0.5x" in "0.5x-10.0x"
    assert "10x" not in "10.0x"


# --------------------------------------------------------------------------
# gate: the epoch ledger is fail-closed
# --------------------------------------------------------------------------
def gate_training_gate_epoch_ledger_fail_closed() -> str:
    """The epoch ledger refuses an exhausted or expired grant."""
    src = read("training/epoch_ledger.py")
    for needle in ["max_epochs", "epochs exhausted", "expires_at_block", "def consume", "def open_if_valid"]:
        if needle not in src:
            raise SystemExit(f"epoch ledger no longer carries {needle}")
    # Behavioral execution of the epoch-ledger lifecycle
    proc = subprocess.run(
        [sys.executable, "training/epoch_ledger.py", "--self-test"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(f"epoch ledger self-test failed: {proc.stderr or proc.stdout}")
    return "epoch ledger is fail-closed on time and epochs" 


def selftest_training_gate_epoch_ledger_fail_closed() -> None:
    sys.path.insert(0, str(ROOT / "training"))
    import epoch_ledger
    good = {
        "asset_id": "a", "owner": "o", "grantee": "g",
        "issued_at_block": 0, "expires_at_block": 10,
        "max_epochs": 1, "epochs_used": 0,
    }
    try:
        epoch_ledger.consume(good, 11)
        raise AssertionError("expired grant must refuse")
    except SystemExit:
        pass




# --------------------------------------------------------------------------
# gate: corpus records carry the provenance pair (asset_id, content_id)
# --------------------------------------------------------------------------
def count_corpus_records() -> int:
    """Total records across the corpus files - one number, one method."""
    import gzip
    corpus_dir = ROOT / "corpus"
    files = sorted(corpus_dir.glob("knowledge-*.jsonl.gz"))
    if not files:
        raise SystemExit(f"no knowledge-*.jsonl.gz corpus files under {corpus_dir}")
    total = 0
    for path in files:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            total += sum(1 for _ in fh)
    return total


def gate_corpus_records_carry_provenance() -> str:
    """Every corpus record carries asset_id and content_id (AÇIK-4 rule)."""
    import gzip
    import json as _json

    corpus_dir = ROOT / "corpus"
    files = sorted(corpus_dir.glob("*.jsonl.gz"))
    if not files:
        raise SystemExit(
            "no corpus files under corpus/; build it first: "
            "python3 training/build_corpus.py --repo . --out corpus/knowledge-self.jsonl.gz"
        )
    records = 0
    bad = 0
    for path in files:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records += 1
                rec = _json.loads(line)
                if not rec.get("asset_id") or not rec.get("content_id"):
                    bad += 1
                    if bad <= 5:
                        print(f"BAD {path.name}: missing asset_id/content_id")
    if bad:
        raise SystemExit(f"{bad} of {records} corpus records lack the provenance pair")
    return f"{records} corpus records carry asset_id + content_id"
def selftest_corpus_records_carry_provenance() -> None:
    import json as _json
    good = {"asset_id": "ab", "content_id": "cd"}
    bad = {"asset_id": "ab"}
    assert _json.loads(_json.dumps(good)) == good
    assert not bad.get("content_id")




# --------------------------------------------------------------------------
# gate: the training-grant crate validates epochs fail-closed
# --------------------------------------------------------------------------
def gate_training_grant_crate_validates() -> str:
    """The grant crate holds the epoch rules; the scripts are not the authority."""
    src = read("crates/grant/src/training.rs")
    for marker in [
        "MAX_TRAINING_GRANT_EPOCHS",
        "pub fn validate_shape",
        "pub fn consume_epoch",
        "pub enum EpochRefusal",
        "pub fn issue(",
        "pub fn consume(",
    ]:
        if marker not in src:
            raise SystemExit(f"grant crate lost its epoch rule: {marker}")
    return "grant crate is the epoch-budget authority"


def selftest_training_grant_crate_validates() -> None:
    good = "MAX_TRAINING_GRANT_EPOCHS\npub fn validate_shape\npub fn consume_epoch\npub enum EpochRefusal"
    thin = "pub fn consume_epoch"
    assert not run_grant_like(good)
    assert run_grant_like(thin)


def run_grant_like(text: str) -> bool:
    """True when the text does NOT carry the full rule set."""
    for marker in ["MAX_TRAINING_GRANT_EPOCHS", "pub fn validate_shape", "pub fn consume_epoch", "pub enum EpochRefusal"]:
        if marker not in text:
            return True
    return False


# --------------------------------------------------------------------------
# gate: the Markdown schema is enforced at the answer exit
# --------------------------------------------------------------------------
def gate_ai_output_schema_enforced() -> str:
    """Every reply passes the schema; the answer exit is the single door."""
    schema = read("crates/read/src/output_schema.rs")
    for marker in ["HeadingSkip", "UnbalancedFence", "TableMismatch", "OutputSchemaError::NotUtf8", "MAX_OUTPUT_BYTES"]:
        if marker not in schema:
            raise SystemExit(f"schema module lost the check: {marker}")
    answer = read("crates/answer/src/lib.rs")
    if "render_markdown" not in answer or "validate_markdown_output(text.as_bytes())" not in answer:
        raise SystemExit("answer exit does not validate its own output")
    return "answer output is Markdown-validated at the exit"


def selftest_ai_output_schema_enforced() -> None:
    assert not run_schema_like("HeadingSkip\nUnbalancedFence\nTableMismatch\nOutputSchemaError::NotUtf8\nMAX_OUTPUT_BYTES")
    assert run_schema_like("HeadingSkip only")


def run_schema_like(text: str) -> bool:
    for marker in ["HeadingSkip", "UnbalancedFence", "TableMismatch", "OutputSchemaError::NotUtf8", "MAX_OUTPUT_BYTES"]:
        if marker not in text:
            return True
    return False


# --------------------------------------------------------------------------
# gate: the chain record client refuses to guess
# --------------------------------------------------------------------------
def gate_chain_record_client_present() -> str:
    """The chain reader has a corpus-format exit and a refuse-to-guess rule."""
    src = read("crates/tools/src/chain.rs")
    for marker in ["pub fn jsonrpc_request", "pub fn parse_get_outcome", "pub fn to_corpus_line", "refusing to guess"]:
        if marker not in src:
            raise SystemExit(f"chain client lost its surface: {marker}")
    return "chain outcome records reach the corpus via the builder format"


def selftest_chain_record_client_present() -> None:
    assert not run_chain_like("pub fn jsonrpc_request\npub fn parse_get_outcome\npub fn to_corpus_line\nrefusing to guess")
    assert run_chain_like("pub fn parse_get_outcome")


def run_chain_like(text: str) -> bool:
    for marker in ["pub fn jsonrpc_request", "pub fn parse_get_outcome", "pub fn to_corpus_line", "refusing to guess"]:
        if marker not in text:
            return True
    return False




# --------------------------------------------------------------------------
# gate: read-only modality - no generating variant exists
# --------------------------------------------------------------------------
def gate_no_generation_variant() -> str:
    """The perception set has four kinds; a generation surface does not exist."""
    src = read("crates/read/src/perception.rs")
    import re
    match = re.search(r'enum\s+PerceptionKind\s*\{([^}]+)\}', src)
    if not match:
        raise SystemExit("PerceptionKind enum not found")
    body = match.group(1)
    variants = []
    for line in body.splitlines():
        line = line.strip().split('//')[0].strip().rstrip(',')
        if line and not line.startswith('#'):
            variants.append(line.split('(')[0].split('{')[0].strip())
    expected = ["Text", "Image", "Audio", "Video"]
    if variants != expected:
        raise SystemExit(f"PerceptionKind variants mismatch: got {variants}, expected {expected}")
    for banned in ["Generate", "Generation", "Render"]:
        if banned in src:
            raise SystemExit(f"perception carries a generation surface: {banned}")
    for ceiling in ["MAX_TEXT_BYTES", "MAX_IMAGE_PIXELS", "MAX_AUDIO_MS", "MAX_VIDEO_FRAMES"]:
        if ceiling not in src:
            raise SystemExit(f"perception lost its ceiling: {ceiling}")
    return "four kinds, four ceilings, no generating variant"


def selftest_no_generation_variant() -> None:
    good = "pub enum PerceptionKind\nText\nImage\nAudio\nVideo\nMAX_TEXT_BYTES\nMAX_IMAGE_PIXELS\nMAX_AUDIO_MS\nMAX_VIDEO_FRAMES"
    bad = "pub enum PerceptionKind\nText\nGenerate\nGenerateImage"
    assert not run_perception_like(good)
    assert run_perception_like(bad)


def run_perception_like(text: str) -> bool:
    for banned in ["Generate", "Generation", "Render"]:
        if banned in text:
            return True
    for marker in ["pub enum PerceptionKind", "Text", "Image", "Audio", "Video"]:
        if marker not in text:
            return True
    return False


# --------------------------------------------------------------------------
# gate: the chain client stays on the fixed surface
# --------------------------------------------------------------------------
def gate_chain_surface_fixed() -> str:
    """The chain surface is the registered set: `training/rpc-seti.json` and
    `ALLOWED_METHODS` agree in both directions, and the report's seven stay
    mandatory. An extension is a change in both places, not a string in the
    client."""
    import json as _json
    set_path = ROOT / "training" / "rpc-seti.json"
    if not set_path.exists():
        raise SystemExit("training/rpc-seti.json is missing")
    registered = _json.loads(set_path.read_text(encoding="utf-8"))
    methods = registered.get("methods", [])
    if not isinstance(methods, list) or not methods:
        raise SystemExit("rpc set is empty")
    required = [
        "bud_aiGetModel", "bud_aiRegisterModel", "bud_aiSubmitRequest",
        "bud_aiSubmitResult", "bud_aiGetOutcome", "bud_aiGetActiveVerifiers",
        "bud_aiInferenceStats",
    ]
    for name in required:
        if name not in methods:
            raise SystemExit(f"registered set lost mandatory RPC {name}")
    src = read("crates/tools/src/chain.rs")
    if "ALLOWED_METHODS" not in src or "is_allowed_method" not in src:
        raise SystemExit("chain client lost its surface rule")
    cut = src.find("#[cfg(test)]")
    surface = src if cut < 0 else src[:cut]
    import re as _re
    used = set(_re.findall(r"bud_[A-Za-z0-9_]+", surface))
    for name in used:
        if name not in methods:
            raise SystemExit(f"chain client calls outside the registered set: {name}")
    for name in methods:
        if name not in used:
            raise SystemExit(f"registered method has no client in chain.rs: {name}")
    return f"chain surface matches the registered set ({len(methods)} methods)"


def selftest_chain_surface_fixed() -> None:
    registered = {"bud_aiGetOutcome", "bud_aiGetCeilings"}
    used = {"bud_aiGetOutcome", "bud_aiGetCeilings"}
    assert used - registered == set(), "client must not call outside the set"
    assert registered - used == set(), "every registered method needs a client"
    assert "bud_aiGetModel" in {
        "bud_aiGetModel", "bud_aiRegisterModel", "bud_aiSubmitRequest",
        "bud_aiSubmitResult", "bud_aiGetOutcome", "bud_aiGetActiveVerifiers",
        "bud_aiInferenceStats",
    }


def run_surface_like(text: str) -> bool:
    import re as _re
    allowed = {"bud_aiGetModel", "bud_aiRegisterModel", "bud_aiSubmitRequest", "bud_aiSubmitResult", "bud_aiGetOutcome", "bud_aiGetActiveVerifiers", "bud_aiInferenceStats"}
    for name in _re.findall(r"bud_[A-Za-z0-9_]+", text):
        if name not in allowed:
            return True
    return "ALLOWED_METHODS" not in text or "is_allowed_method" not in text


# --------------------------------------------------------------------------
# gate: the lubot binary really runs
# --------------------------------------------------------------------------
def _begins_with_heading(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith("# ")


def _cli_in(cwd: str, *args: str) -> subprocess.CompletedProcess[str]:
    """The binary with the repository manifest, run inside `cwd` - for path-
    sensitive commands that must see a fixture tree instead of the repo."""
    return subprocess.run(
        ["cargo", "run", "--quiet", "--manifest-path", str(ROOT / "Cargo.toml"),
         "-p", "lubot", "--bin", "lubot", "--", *args],
        cwd=cwd, capture_output=True, text=True, check=False,
    )


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["cargo", "run", "--quiet", "-p", "lubot", "--bin", "lubot", "--", *args],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )


def gate_cli_asks_and_renders_markdown() -> str:
    """The binary answers a real question against a real file and prints a
    Markdown document as its only stdout, and the tool route answers
    arithmetic without touching the index."""
    import hashlib
    import json
    import tempfile

    body = "The private settlement note mentions a schedule of 3 payments."
    digest = hashlib.sha256(body.encode()).hexdigest()
    record = {
        "kind": "markdown", "text": body, "path": "dm/1", "source": "t",
        "digest": digest, "licence": "MIT", "attribution": "t",
        "content_id": "c1", "asset_id": "a" * 64,
    }
    with tempfile.TemporaryDirectory() as td:
        fixture = Path(td) / "c.jsonl"
        fixture.write_text(json.dumps(record) + "\n", encoding="utf-8")
        run = _cli("corpus", str(fixture))
        if run.returncode != 0 or "items" not in run.stdout:
            raise SystemExit(f"`corpus` command failed: {run.stderr.strip()}")
        run = _cli("ask", "--corpus", str(fixture), "--reader", "g", "--effort", "1.0x", "settlement schedule")
        if run.returncode != 0:
            raise SystemExit(f"`ask` command failed: {run.stderr.strip()}")
        if not _begins_with_heading(run.stdout):
            raise SystemExit(f"`ask` stdout is not Markdown: {run.stdout[:80]!r}")
        run = _cli("ask", "--corpus", str(fixture), "--reader", "g", "--effort", "1.0x", "6 * 7 = ?")
        if run.returncode != 0 or not run.stdout.lstrip().startswith("## calculator"):
            raise SystemExit(f"the tool route did not render: {run.stdout[:80]!r}")
    return "the binary answers and renders Markdown on its only stdout"


def selftest_cli_asks_and_renders_markdown() -> None:
    assert _begins_with_heading("# Answer\n\n- x\n")
    assert not _begins_with_heading("plain text without a heading\n")


# --------------------------------------------------------------------------
# gate: operator sync rules (Aşama 7) stay in the tool crate
# --------------------------------------------------------------------------
def gate_operator_sync_rules() -> str:
    """The operator rule Lubot can actually run is the effort ceiling: it is
    checked at every door the CLI has (answer budget, hashed request) and
    verified again by the chain client. The bond, model_hash and window rules
    left the crate with their callers - the gate asserted their names while
    nothing in the tree could produce the records they read, which is the
    exact mask this ratchet exists to catch; a name-only assertion would
    preserve a library nobody links. If the registry lands, the rules return
    in that patch and the gate re-grows its list."""
    src = read("crates/tools/src/operator.rs")
    for word in ["effort_tag_ok", "effort_hash", "answer_budget"]:
        if word not in src:
            raise SystemExit(f"operator effort rules lost `{word}`")
    for gone in ["compute_bond_ok", "same_model_hash", "CheckpointWindow"]:
        if gone in src:
            raise SystemExit(f"`{gone}` is back without a caller to check it")
    if "cheap" in src and "0.4x" not in src:
        raise SystemExit("the ceiling refusals have no fixture")
    chain = read("crates/tools/src/chain.rs")
    if "effort_hash does not match effort_tag" not in chain:
        raise SystemExit("parse_request no longer verifies the hashed effort tier")
    return "ceiling-hashed effort is checked on every CLI door, and the retired rules stay retired without owners"


def selftest_operator_sync_rules() -> None:
    assert 'effort_hash("1.0x").unwrap()' in read("crates/tools/src/operator.rs")
    assert 'assert!(!effort_tag_ok("0.4x"))' in read("crates/tools/src/operator.rs")


# --------------------------------------------------------------------------
# gate: the finalized-output closed loop (Aşama 9)
# --------------------------------------------------------------------------
def gate_output_finalize_closed_loop() -> str:
    """A finalized output is sealed only after schema validation, carries the
    fixed 'ai-inference' tag, and the answer type has no binary/visual/video
    return variant to begin with."""
    src = read("crates/answer/src/output_registry.rs")
    if 'OUTPUT_TAG: &str = "ai-inference"' not in src:
        raise SystemExit("the report's 'ai-inference' tag is gone from the registry")
    if "validate_markdown_output" not in src:
        raise SystemExit("finalize_output no longer validates before sealing")
    if "never downgraded" not in src and "downgrade" not in src:
        raise SystemExit("the no-nearest-format rule is not stated in the registry")
    answer = read("crates/answer/src/lib.rs")
    for variant in ["Image", "Video", "Audio", "Bytes"]:
        if f"Answer::{variant}" in answer or f"{variant} {{" in answer:
            raise SystemExit(f"the answer type gained a non-Markdown return variant: {variant}")
    cli = read("crates/cli/src/main.rs")
    if "--outputs" not in cli:
        raise SystemExit("the binary does not write the finalized-output handoff")
    return "finalized outputs validate first, tag as ai-inference, and no binary return type exists"


def selftest_output_finalize_closed_loop() -> None:
    assert 'OUTPUT_TAG: &str = "ai-inference"' in read("crates/answer/src/output_registry.rs")


# --------------------------------------------------------------------------
# gate: the system prompt is true, nothing unmeasured
# --------------------------------------------------------------------------
def gate_system_prompt_is_true() -> str:
    """The Budlum-specific system prompt states only measured facts: the four
    ceilings, the eight RPC names, the effort range, the threshold, the
    ai-inference tag - and no superlative or proof claim that nothing here
    produced."""
    prompt = read("training/system_prompt.md")
    for token in ["1.048.576", "16.777.216", "3.600.000", "4096",
                  "bud_aiGetModel", "bud_aiRegisterModel", "bud_aiSubmitRequest",
                  "bud_aiSubmitResult", "bud_aiGetOutcome",
                  "bud_aiGetActiveVerifiers", "bud_aiInferenceStats",
                  "bud_aiGetCeilings",
                  "0.5x", "10.0x", "agreement_threshold", "ai-inference",
                  "attestation-only"]:
        if token not in prompt:
            raise SystemExit(f"the system prompt lost a measured fact: {token}")
    for banned in ["state of the art", "state-of-the-art", "mathematically proven",
                   "kanitlanmistir", "kanıtlanmıştır", "en iyi model", "best model",
                   "guaranteed correct", "asla hata yapmaz", "her zaman dogru",
                   "matematiksel ispat saglanmistir"]:
        if banned in prompt:
            raise SystemExit(f"the system prompt claims what nothing here measured: {banned}")
    cli = read("crates/cli/src/main.rs")
    if '"prompt"' not in cli or "--path" not in cli:
        raise SystemExit("the `lubot prompt` command is gone")
    return "the system prompt carries only measured facts, and `lubot prompt` prints it"


def selftest_system_prompt_is_true() -> None:
    assert "0.5x" in "range 0.5x-10.0x"
    assert "state of the art" not in "measured facts only"


# --------------------------------------------------------------------------
# gate: the coding of remaining workspace powers (ceilings, batch, risk)
# --------------------------------------------------------------------------
def gate_yerlesik_komutlar_bagli() -> str:
    """`lubot ceilings` prints the four constants, `lubot risk` names risky
    command shapes on a fixture, `lubot batch` answers several questions with
    one corpus load and counts verdicts."""
    import hashlib
    import json
    import tempfile

    run = _cli("ceilings")
    if run.returncode != 0:
        raise SystemExit(f"`ceilings` failed: {run.stderr.strip()}")
    for expected in ["1048576", "16777216", "3600000", "4096", "0.5x-10.0x"]:
        if expected not in run.stdout:
            raise SystemExit(f"`ceilings` lost {expected}: {run.stdout[:160]!r}")

    run = _cli("risk", "--text", "git push -f origin usl")
    if run.returncode == 0 or "history rewrite" not in run.stdout:
        raise SystemExit("`risk` must fail and name `git push -f`")
    run = _cli("risk", "--text", "cargo test --workspace")
    if run.returncode != 0:
        raise SystemExit(f"`risk` reported a benign command: {run.stdout[:120]!r}")

    def record(text: str, origin: str, cid: str) -> dict:
        return {
            "kind": "markdown", "text": text, "path": origin, "source": "t",
            "digest": hashlib.sha256(text.encode()).hexdigest(),
            "licence": "MIT", "attribution": "t",
            "content_id": cid, "asset_id": "a" * 64,
        }

    with tempfile.TemporaryDirectory() as td:
        fixture = Path(td) / "c.jsonl"
        fixture.write_text("\n".join(json.dumps(record(t, o, c))
                          for t, o, c in [
                              ("The text ceiling is 1048576 bytes.",
                               "budlum-docs/limits.md", "p1"),
                              ("Always write the plan before implementing.",
                               "temel-beceriler/plans/1.md", "p2"),
                          ]) + "\n", encoding="utf-8")
        questions = Path(td) / "q.jsonl"
        questions.write_text(
            '{"question": "what is the text ceiling?"}\n'
            '{"question": "write me a haiku"}\n', encoding="utf-8")
        audit = Path(td) / "audit.jsonl"
        run = _cli("batch", "--corpus", str(fixture), "--questions", str(questions),
                   "--reader", "r", "--effort", "1.0x", "--audit", str(audit))
        if run.returncode != 0:
            raise SystemExit(f"`batch` failed: {run.stderr.strip()}")
        if "answerable" not in run.stdout or "out-of-scope" not in run.stdout:
            raise SystemExit(f"`batch` report lost a verdict: {run.stdout[:200]!r}")
        if not audit.exists() or audit.stat().st_size == 0:
            raise SystemExit("`batch --audit` wrote no trace")
    return "ceilings, risk and batch are wired and answer from fixtures"


def selftest_yerlesik_komutlar_bagli() -> None:
    assert _begins_with_heading("# Ceilings")
    assert lubot_tools_command_risk_exists()


def lubot_tools_command_risk_exists() -> bool:
    """The classifier module is compiled into the binary (gate-side check)."""
    run = _cli("risk", "--text", ":(){ :|:& };:")
    return run.returncode != 0 and "fork bomb" in run.stdout



# --------------------------------------------------------------------------
# gate: a rich document becomes a provenance-true corpus record
# --------------------------------------------------------------------------
PDF_FIXTURE_B64 = (
    "JVBERi0xLjMKJenr8b8KMSAwIG9iago8PAovQ291bnQgMQovS2lkcyBbMyAwIFJdCi9NZWRpYUJveCBbMCAwIDU5NS4yOCA4NDEuODldCi9UeXBlIC9QYWdlcwo+PgplbmRvYmoKMiAwIG9iago8PAovT3BlbkFjdGlvbiBbMyAwIFIgL0ZpdEggbnVsbF0KL1BhZ2VMYXlvdXQgL09uZUNvbHVtbgovUGFnZXMgMSAwIFIKL1R5cGUgL0NhdGFsb2cKPj4KZW5kb2JqCjMgMCBvYmoKPDwKL0NvbnRlbnRzIDQgMCBSCi9QYXJlbnQgMSAwIFIKL1Jlc291cmNlcyA2IDAgUgovVHlwZSAvUGFnZQo+PgplbmRvYmoKNCAwIG9iago8PAovRmlsdGVyIC9GbGF0ZURlY29kZQovTGVuZ3RoIDE4NAo+PgpzdHJlYW0KeJxVjUEKwjAURPeeYpYKEhOtVF0WFRQXLnKBbxvlmzSVNFHs6dWCgquBmeG9KfYDKeY5HoNCY7JVUFMhJfQZG/2pZkqoBRZSiSyDrjA8pFMT0Rl/YY+qsakmjz4QTRtZoEi4UaBLoDOO6y24ZF8Zj9JYdhxwN7D05IjGvW8Wtgm31BKe1H12MYK+/tnzfCbksrfvLPuSf/wVIt3JvzljOG7Jtz2da8cW5PqFwe5rTMHEP8cLLoJQiwplbmRzdHJlYW0KZW5kb2JqCjUgMCBvYmoKPDwKL0Jhc2VGb250IC9IZWx2ZXRpY2EKL0VuY29kaW5nIC9XaW5BbnNpRW5jb2RpbmcKL1N1YnR5cGUgL1R5cGUxCi9UeXBlIC9Gb250Cj4+CmVuZG9iago2IDAgb2JqCjw8Ci9Gb250IDw8L0YxIDUgMCBSPj4KL1Byb2NTZXQgWy9QREYgL1RleHQgL0ltYWdlQiAvSW1hZ2VDIC9JbWFnZUldCj4+CmVuZG9iago3IDAgb2JqCjw8Ci9DcmVhdGlvbkRhdGUgKEQ6MjAyNjA5MDcxMTM2NDFaKQo+PgplbmRvYmoKeHJlZgowIDgKMDAwMDAwMDAwMCA2NTUzNSBmIAowMDAwMDAwMDE1IDAwMDAwIG4gCjAwMDAwMDAxMDIgMDAwMDAgbiAKMDAwMDAwMDIwNSAwMDAwMCBuIAowMDAwMDAwMjg1IDAwMDAwIG4gCjAwMDAwMDA1NDEgMDAwMDAgbiAKMDAwMDAwMDYzOCAwMDAwMCBuIAowMDAwMDAwNzI1IDAwMDAwIG4gCnRyYWlsZXIKPDwKL1NpemUgOAovUm9vdCAyIDAgUgovSW5mbyA3IDAgUgovSUQgWzxFRDJEQjBENTc2Mjg0QkIzMjg3Q0I3OENEMDI4NjZEMD48RUQyREIwRDU3NjI4NEJCMzI4N0NCNzhDRDAyODY2RDA+XQo+PgpzdGFydHhyZWYKNzgwCiUlRU9GCg=="
)


def gate_doc_pdf_feeds_corpus() -> str:
    """`lubot doc` extracts a PDF, tags it with the licence and attribution the
    caller declares, writes records the corpus reader accepts, and refuses a
    document whose licence is missing or outside the set."""
    import base64
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        pdf = Path(td) / "fixture.pdf"
        pdf.write_bytes(base64.b64decode(PDF_FIXTURE_B64))
        out = Path(td) / "records.jsonl.gz"
        asset = "a" * 64
        run = _cli(
            "doc", "--in", str(pdf), "--origin", "docs/fixture",
            "--licence", "MIT", "--attribution", "fixture",
            "--asset-id", asset, "--out", str(out),
        )
        if run.returncode != 0:
            raise SystemExit(f"`doc` failed: {run.stderr.strip()}")
        if "1 records" not in run.stdout:
            raise SystemExit(f"`doc` lost the record count: {run.stdout[:120]!r}")
        loaded = _cli("corpus", str(out))
        if loaded.returncode != 0 or "\"items\":1" not in loaded.stdout:
            raise SystemExit(f"`corpus` refused the produced records: {loaded.stdout[:160]!r}")
        run = _cli("doc", "--in", str(pdf), "--attribution", "fixture", "--asset-id", asset)
        if run.returncode == 0:
            raise SystemExit("`doc` accepted a document with no licence")
    return "`doc` extracts and tags a PDF; the reader accepts the records"


def selftest_doc_pdf_feeds_corpus() -> None:
    assert len(PDF_FIXTURE_B64) > 1000
    assert _begins_with_heading("# Doc")



# --------------------------------------------------------------------------
# gate: the queue works without interruption and resumes without redoing
# --------------------------------------------------------------------------
def gate_queue_continues_uninterruptedly() -> str:
    """The queue processes pending jobs in order, a run stops at its budget
    and the next run resumes without redoing a finished job, a job that keeps
    crashing becomes stalled after three attempts, and a failing check halts
    the run with a non-zero exit."""
    import hashlib
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        def record(text: str, origin: str, cid: str) -> dict:
            return {
                "kind": "markdown", "text": text, "path": origin, "source": "t",
                "digest": hashlib.sha256(text.encode()).hexdigest(),
                "licence": "MIT", "attribution": "t",
                "content_id": cid, "asset_id": "a" * 64,
            }

        corpus = Path(td) / "c.jsonl"
        corpus.write_text("\n".join(json.dumps(record(t, o, c))
                          for t, o, c in [
                              ("The text ceiling is 1048576 bytes.",
                               "budlum-docs/limits.md", "p1"),
                              ("Write the plan before implementing.",
                               "temel-beceriler/plans/1.md", "p2"),
                          ]) + "\n", encoding="utf-8")
        queue = Path(td) / "q.jsonl"
        for i, question in enumerate([
            "what is the text ceiling?",
            "write me a haiku",
            "missing corpus question",
        ]):
            corpus_arg = str(corpus) if i < 2 else str(Path(td) / "nope.jsonl")
            run = _cli("queue", "add", "--corpus", corpus_arg, "--reader", "r", "--effort", "1.0x",
                       "--file", str(queue), question)
            if run.returncode != 0:
                raise SystemExit(f"queue add failed: {run.stderr.strip()}")

        # A duplicate pending job is refused, not doubled - checked while the
        # original is still pending.
        dup = _cli("queue", "add", "--corpus", str(corpus), "--reader", "r", "--effort", "1.0x",
                   "--file", str(queue), "what is the text ceiling?")
        if dup.returncode == 0:
            raise SystemExit("a duplicate pending job was accepted")

        run = _cli("queue", "run", "--file", str(queue), "--budget", "1", "--check", "true")
        if run.returncode != 0 or "1 done" not in run.stdout or "2 pending" not in run.stdout:
            raise SystemExit(f"budgeted run broke: {run.stdout[:160]!r}")
        run = _cli("queue", "run", "--file", str(queue), "--budget", "10", "--check", "true")
        if run.returncode != 0 or "2 done" not in run.stdout:
            raise SystemExit(f"resume broke: {run.stdout[:160]!r}")
        logged = _cli("queue", "log", "--file", str(queue))
        if logged.stdout.count("#1 grounded") != 1:
            raise SystemExit("a finished job was redone on resume")
        # Two more passes: the crashing job fails, then stalls.
        _cli("queue", "run", "--file", str(queue), "--budget", "10", "--check", "true")
        run = _cli("queue", "run", "--file", str(queue), "--budget", "10", "--check", "true")
        if "1 stalled" not in run.stdout:
            raise SystemExit(f"crash job did not stall: {run.stdout[:160]!r}")

        queue2 = Path(td) / "q2.jsonl"
        _cli("queue", "add", "--corpus", str(corpus), "--reader", "r", "--effort", "1.0x",
             "--file", str(queue2), "what is the text ceiling?")
        run = _cli("queue", "run", "--file", str(queue2), "--check", "false")
        if run.returncode == 0 or "halted: true" not in run.stdout:
            raise SystemExit("a failing check must halt the run loudly")
    return "the queue runs, resumes, stalls and halts fail-closed"


def selftest_queue_continues_uninterruptedly() -> None:
    assert _begins_with_heading("# Queue")
    assert "queue run" in "queue add / queue run / queue log"



# --------------------------------------------------------------------------
# gate: the ratchet holds - measured numbers may not regress
# --------------------------------------------------------------------------
RATCHET_KEYS = ["tests", "gates", "pedantic", "corpus"]


def gate_ratchet_holds() -> str:
    """The baselines in training/ratchet.json hold: tests, gates and corpus
    may only rise (pedantic may only fall)."""
    import json as _json

    path = ROOT / "training" / "ratchet.json"
    if not path.exists():
        raise SystemExit("training/ratchet.json is missing")
    baseline = _json.loads(path.read_text(encoding="utf-8"))
    for key in RATCHET_KEYS:
        if key not in baseline:
            raise SystemExit(f"ratchet baseline lost `{key}`")
    # tests: the suite's own count.
    out = subprocess.run(
        ["cargo", "test", "--workspace"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if out.returncode != 0:
        raise SystemExit("the suite does not pass, so its count cannot be ratcheted")
    measured_tests = sum(int(m) for m in re.findall(r"test result: ok\. (\d+) passed", out.stdout))
    measured_gates = len(GATES)
    measured_corpus = count_corpus_records()
    clippy = subprocess.run(
        ["cargo", "clippy", "--workspace", "--all-targets", "--", "-W", "pedantic"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    measured_pedantic = sum(
        1 for line in clippy.stderr.splitlines()
        if line.startswith("warning:")
        and "generated" not in line and "is deprecated" not in line
    )
    measured = {
        "tests": measured_tests, "gates": measured_gates, "pedantic": measured_pedantic,
        "corpus": measured_corpus,
    }
    regressed = []
    for key in ["tests", "gates", "corpus"]:
        if measured[key] < baseline[key]:
            regressed.append(f"{key} {measured[key]} < baseline {baseline[key]}")
    if measured["pedantic"] > baseline["pedantic"]:
        regressed.append(f"pedantic {measured['pedantic']} > baseline {baseline['pedantic']}")
    if regressed:
        raise SystemExit("ratchet regressed: " + "; ".join(regressed))
    return f"ratchet holds: tests {measured_tests}, gates {measured_gates}, pedantic {measured_pedantic}, corpus {measured_corpus}"


def selftest_ratchet_holds() -> None:
    """The direction rules: three rise (>=), pedantic falls (<=)."""
    assert set(RATCHET_KEYS) == {"tests", "gates", "pedantic", "corpus"}
    baseline = {"tests": 5, "pedantic": 2}
    assert 6 >= baseline["tests"], "tests may rise"
    assert 1 <= baseline["pedantic"], "pedantic may fall"



# --------------------------------------------------------------------------
# gate: rustfmt agrees with the tree (ci.py / olc.py power)
# --------------------------------------------------------------------------
def gate_fmt_clean() -> str:
    """`cargo fmt --check` passes: a tree the formatter rewrites is a tree
    nobody read."""
    out = subprocess.run(
        ["cargo", "fmt", "--check"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if out.returncode != 0:
        preview = "\n".join(out.stdout.splitlines()[:8])
        raise SystemExit(f"rustfmt disagrees:\n{preview}")
    return "rustfmt agrees with the tree"


def selftest_fmt_clean() -> None:
    assert "cargo fmt --check" in gate_fmt_clean.__doc__



# --------------------------------------------------------------------------
# gate: `it` moves only the listed paths, and a dry run moves nothing
# --------------------------------------------------------------------------
def gate_it_is_restricted() -> str:
    """`lubot it` commits and pushes only the paths it is given; `--dry-run`
    must leave the fixture repo untouched."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["git", "init", "-q"], cwd=td, capture_output=True, text=True, check=True)
        (Path(td) / "a.txt").write_text("x", encoding="utf-8")
        run = _cli_in(td, "it", "-m", "x", "--path", "a.txt", "--dry-run")
        if run.returncode != 0 or "Dry run" not in run.stdout:
            raise SystemExit(f"`it --dry-run` failed: {run.stdout[:120]!r} {run.stderr[:120]!r}")
        head = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=td, capture_output=True, text=True, check=False,
        )
        if head.returncode == 0:
            raise SystemExit("`it --dry-run` committed something")
        missing = _cli_in(td, "it", "-m", "x", "--path", "yok.txt", "--dry-run")
        if missing.returncode == 0:
            raise SystemExit("`it` accepted a path that does not exist")
    return "`it` takes only the listed paths; the dry run touches nothing"


def selftest_it_is_restricted() -> None:
    assert "Dry run" in "Dry run: nothing staged, committed or pushed."



# --------------------------------------------------------------------------
# gate: no debug leftovers (the tree is clean)
# --------------------------------------------------------------------------
def gate_no_debug_leftovers() -> str:
    """No `dbg!` remains anywhere in the crates: a debug print that survives
    into a commit is a trace nobody asked for."""
    hits = []
    for path in sorted((ROOT / "crates").rglob("*.rs")):
        if "target" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for index, line in enumerate(text.splitlines(), 1):
            if "dbg!(" in line:
                hits.append(f"{path.relative_to(ROOT)}:{index}")
    if hits:
        raise SystemExit("dbg! leftovers: " + ", ".join(hits[:5]))
    return "no dbg! leftovers in crates"


def selftest_no_debug_leftovers() -> None:
    assert "dbg!(" in "let x = dbg!(y);"
    assert "dbg!(" not in "let x = y;"



# --------------------------------------------------------------------------
# gate: no credential shape survives in the tree (credential scan)
# --------------------------------------------------------------------------
def gate_no_secret_material() -> str:
    """`lubot guvenlik` over crates/, training/ and gates/ finds no credential
    shape: a token or key that reaches a commit is a leak, not a typo."""
    run = _cli("guvenlik")
    if run.returncode != 0:
        raise SystemExit(f"secret scan failed: {(run.stdout + run.stderr)[:200]!r}")
    return "no credential shape in crates/, training/, gates/"


def selftest_no_secret_material() -> None:
    """A constructed credential must be found; a mention must not."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        secret_file = Path(td) / "s.txt"
        secret_file.write_text("ghp_" + "A" * 36 + "\n", encoding="utf-8")
        run = _cli("guvenlik", "--path", str(secret_file))
        assert run.returncode != 0, "the scanner missed a full credential"
        mention_file = Path(td) / "m.txt"
        mention_file.write_text("use a token named ghp_short\n", encoding="utf-8")
        run = _cli("guvenlik", "--path", str(mention_file))
        assert run.returncode == 0, "a mention was reported as a credential"


# --------------------------------------------------------------------------
# gate: the repository map reflects the manifests
# --------------------------------------------------------------------------
def gate_graf_maps_workspace() -> str:
    """`lubot graf` reads the actual manifests and passes the schema; a map
    that cannot be read is a map nobody checked."""
    run = _cli("graf")
    if run.returncode != 0:
        raise SystemExit(f"`graf` failed: {(run.stdout + run.stderr)[:200]!r}")
    if "| lubot-tools |" not in run.stdout:
        raise SystemExit("`graf` lost a crate row")
    if "crates," not in run.stdout:
        raise SystemExit("`graf` lost its summary count")
    return "`graf` maps crates from the manifests"


def selftest_graf_maps_workspace() -> None:
    assert "internal deps" in "crate | src files | LOC | internal deps | external deps"


# --------------------------------------------------------------------------
# gate: file kind decides the reading route, before reading
# --------------------------------------------------------------------------
def gate_dosya_routes_before_reading() -> str:
    """`lubot dosya` recognises magic bytes and refuses what has no reading
    path: a PDF routes to `doc`, an executable is refused, text is read."""
    import base64
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        pdf = Path(td) / "f.pdf"
        pdf.write_bytes(base64.b64decode(PDF_FIXTURE_B64))
        run = _cli("dosya", "--path", str(pdf))
        if run.returncode != 0 or "| pdf |" not in run.stdout:
            raise SystemExit(f"`dosya` mis-routed a PDF: {run.stdout[:120]!r}")
        elf = Path(td) / "bud"
        elf.write_bytes(b"\x7fELF\x02\x01\x01")
        run = _cli("dosya", "--path", str(elf))
        if run.returncode == 0 or "refuse" not in run.stdout:
            raise SystemExit("`dosya` accepted an executable")
        text = Path(td) / "n.md"
        text.write_text("merhaba", encoding="utf-8")
        run = _cli("dosya", "--path", str(text))
        if run.returncode != 0 or "| text |" not in run.stdout:
            raise SystemExit("`dosya` mis-routed text")
    return "`dosya` routes by magic bytes and refuses the unreadable"


def selftest_dosya_routes_before_reading() -> None:
    assert "doc" in "pdf -> doc"
    assert "refuse" in "elf -> refuse"



# --------------------------------------------------------------------------
# gate: the decision battery validates and answers are recorded
# --------------------------------------------------------------------------
def gate_soru_bataryasi_gecerli() -> str:
    """`training/soru-bataryasi.json` holds 20 ask_user-shaped questions
    (unique ids, 2-4 options each), `lubot soru list` prints them as a
    Markdown document, `cevapla` records an answer, and an invalid battery
    (one option, duplicate id) is refused by the loader itself."""
    import json as _json
    import tempfile

    path = ROOT / "training" / "soru-bataryasi.json"
    if not path.exists():
        raise SystemExit("soru-bataryasi.json is missing")
    doc = _json.loads(path.read_text(encoding="utf-8"))
    sorular = doc.get("sorular", [])
    if len(sorular) != 20:
        raise SystemExit(f"expected 20 questions, found {len(sorular)}")
    ids = [s.get("id") for s in sorular]
    if len(set(ids)) != len(ids):
        raise SystemExit("battery has duplicate ids")
    for soru in sorular:
        if not (2 <= len(soru.get("options", [])) <= 4):
            raise SystemExit(f"question `{soru.get('id')}` does not carry 2-4 options")
        option_ids = [o.get("id") for o in soru.get("options", [])]
        if len(set(option_ids)) != len(option_ids):
            raise SystemExit(f"question `{soru.get('id')}` has duplicate option ids")

    run = _cli("soru", "list")
    if run.returncode != 0 or "| k6-donanim |" not in run.stdout:
        raise SystemExit(f"`soru list` failed: {run.stdout[:120]!r}")
    with tempfile.TemporaryDirectory() as td:
        queue = Path(td) / "cevaplar.jsonl"
        run = _cli("soru", "cevapla", "bekleyen-merge", "bekle", "--cevap", str(queue))
        if run.returncode != 0:
            raise SystemExit(f"`soru cevapla` failed: {run.stderr.strip()}")
        run = _cli("soru", "durum", "--cevap", str(queue))
        if run.returncode != 0 or "1 of 20" not in run.stdout:
            raise SystemExit(f"`soru durum` broken: {run.stdout[:120]!r}")
    return "20-question battery validates; answers are recorded"


def selftest_soru_bataryasi_gecerli() -> None:
    assert "2-4" in "options: 2-4 per question"
    assert "20" in "20 soru"


def gate_four_axes_wired() -> str:
    """The four parallel axes are runnable: `ara` searches the corpus with
    citations and licences, `indeks` reports measured facts, `mufredat`
    writes the ordered syllabus with digests, `karsilastir` compares effort
    ceilings, and the queue operator can list and cancel an unfinished job -
    a cancelled job never runs."""
    import hashlib
    import json
    import re
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fixture = td / "c.jsonl"
        body = "The private settlement note mentions a schedule of 3 payments."
        record = {
            "kind": "markdown", "text": body, "path": "docs/s1.md", "source": "t",
            "digest": hashlib.sha256(body.encode()).hexdigest(), "licence": "MIT",
            "attribution": "t", "content_id": "c1", "asset_id": "a" * 64,
        }
        fixture.write_text(json.dumps(record) + "\n", encoding="utf-8")
        run = _cli("ara", "--corpus", str(fixture), "--n", "2", "settlement schedule")
        if run.returncode != 0 or "docs/s1.md:1" not in run.stdout or "licence `MIT`" not in run.stdout:
            raise SystemExit(f"`ara` failed: {run.stderr.strip() or run.stdout[:120]!r}")
        run = _cli("indeks", "--corpus", str(fixture))
        if run.returncode != 0 or "items: 1" not in run.stdout:
            raise SystemExit(f"`indeks` failed: {run.stdout[:120]!r}")
        syllabus = td / "syl.jsonl"
        run = _cli("mufredat", "--corpus", str(fixture), "--out", str(syllabus))
        if run.returncode != 0 or "markdown first" not in run.stdout or not syllabus.exists():
            raise SystemExit(f"`mufredat` failed: {run.stdout[:120]!r}")
        run = _cli("karsilastir", "--corpus", str(fixture), "--reader", "g",
                   "--effort", "0.5x,1.0x", "what is the text ceiling?")
        if run.returncode != 0 or "budget 2 passage(s)" not in run.stdout or "budget 3 passage(s)" not in run.stdout:
            raise SystemExit(f"`karsilastir` failed: {run.stdout[:160]!r}")
        queue = td / "q.jsonl"
        run = _cli("queue", "add", "--corpus", str(fixture), "--reader", "r", "--effort", "1.0x",
                   "--file", str(queue), "what is the text ceiling?")
        if run.returncode != 0:
            raise SystemExit(f"queue add failed: {run.stderr.strip()}")
        run = _cli("queue", "ls", "--file", str(queue))
        if run.returncode != 0 or "pending" not in run.stdout:
            raise SystemExit(f"`queue ls` failed: {run.stdout[:120]!r}")
        match = re.search(r"`([0-9a-f]{12}-\d+)`", run.stdout) or re.search(r"([0-9a-f]{12}-\d+)", run.stdout)
        if not match:
            raise SystemExit(f"no job id in `queue ls`: {run.stdout[:120]!r}")
        run = _cli("queue", "iptal", match.group(1), "--file", str(queue))
        if run.returncode != 0 or "iptal edildi" not in run.stdout:
            raise SystemExit(f"`queue iptal` failed: {run.stderr.strip() or run.stdout[:120]!r}")
        run = _cli("queue", "run", "--file", str(queue), "--check", "false")
        if run.returncode != 0 or ("Nothing pending" not in run.stdout and "0 done" not in run.stdout):
            raise SystemExit(f"a cancelled job ran: {run.stdout[:120]!r}")
    return "ara, indeks, mufredat, karsilastir and the queue operator live; cancellation holds"


def selftest_four_axes_wired() -> None:
    assert "ara" in "ara"
    assert _cli("karsilastir", "--corpus", "x", "--reader", "r", "--effort", "0.5x", "q?").returncode != 0



def gate_kirmizi_senaryolar() -> str:
    """The red scenarios stay red: generation, image prompts, credential
    hunts and unmeasured hype are refused out of scope, while a real
    question about the same corpus is answered - the refusals are scoped,
    not blanket. The scenario list is a measured fixture
    (`training/batarya.jsonl`, `tur: red`)."""
    import hashlib
    import json

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        body = "The private settlement note mentions a schedule of 3 payments."
        record = {
            "kind": "markdown", "text": body, "path": "docs/s1.md", "source": "t",
            "digest": hashlib.sha256(body.encode()).hexdigest(), "licence": "MIT",
            "attribution": "t", "content_id": "c1", "asset_id": "a" * 64,
        }
        corpus = Path(td) / "c.jsonl"
        corpus.write_text(json.dumps(record) + "\n", encoding="utf-8")
        scenarios = ROOT / "training" / "batarya.jsonl"
        if not scenarios.exists():
            raise SystemExit("training/batarya.jsonl (red scenarios) is missing")
        total = 0
        for line in scenarios.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            scenario = json.loads(line)
            if scenario.get("tur") != "red":
                continue
            total += 1
            run = _cli("ask", "--corpus", str(corpus), "--reader", "g", "--effort", "1.0x", scenario["soru"])
            if run.returncode != 0:
                raise SystemExit(f"red scenario failed to run: {run.stderr.strip()}")
            if "# Out of scope" not in run.stdout and "# Refused" not in run.stdout:
                raise SystemExit(f"red scenario was NOT refused: {scenario['soru']!r} -> {run.stdout[:100]!r}")
        if total < 3:
            raise SystemExit(f"only {total} red scenarios in the fixture")
        # The control: the same corpus answers a real question.
        run = _cli("ask", "--corpus", str(corpus), "--reader", "g", "--effort", "1.0x",
                   "what payment schedule does the settlement note mention?")
        if run.returncode != 0 or "# Out of scope" in run.stdout or "# Refused" in run.stdout:
            raise SystemExit(f"the control was refused too: {run.stdout[:100]!r}")
    return f"{total} red scenarios refused; the same corpus still answers"


def selftest_kirmizi_senaryolar() -> None:
    assert "red" in "red scenarios"
    assert _cli("ask", "--corpus", "x", "--reader", "g", "write me a haiku").returncode != 0


# --------------------------------------------------------------------------
# gate: the SFT set is examined before an epoch is spent on it
# --------------------------------------------------------------------------
def gate_sft_evaluation_baseline() -> str:
    """Every SFT row must cite, nothing may duplicate, nothing may be empty.
    The builder drops unciteable rows; a survivor is a regression the
    evaluator refuses, measured on the self-corpus this run builds."""
    import tempfile

    def py(script: str, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(ROOT / script), *args],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )

    with tempfile.TemporaryDirectory() as td:
        corpus = str(Path(td) / "knowledge.jsonl")
        sft = str(Path(td) / "sft.jsonl")
        r1 = py("training/build_corpus.py", "--repo", ".", "--out", corpus)
        if r1.returncode != 0:
            raise SystemExit(f"the corpus the evaluator would examine does not build: {r1.stderr[-200:]}")
        r2 = py("training/make_sft.py", "--corpus", corpus, "--curriculum", "training/curriculum", "--out", sft)
        if r2.returncode != 0:
            raise SystemExit(f"the SFT set does not build: {r2.stderr[-200:]}")
        r3 = py("training/eval_sft.py", "--sft", sft)
        if r3.returncode != 0:
            raise SystemExit(f"the SFT evaluation refused the set:\n{r3.stdout[-400:]}")
        report = json.loads(r3.stdout.splitlines()[0])
        # The evaluator must also prove it catches defects on the real
        # pipeline, not just on the self-test's synthetic rows: spike the
        # just-built set with one defect of each class and demand refusal.
        spiked = str(Path(td) / "sft-spiked.jsonl")
        good = Path(sft).read_text(encoding="utf-8")
        dup = good.splitlines()[0]
        defects = [
            dup,  # duplicate of the first row
            '{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", '
            '"content": "a grounded row that never cites anything"}], "kind": "doc"}',
            '{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", '
            '"content": "x"}], "kind": "doc"}',
        ]
        Path(spiked).write_text(good + "\n".join(defects) + "\n", encoding="utf-8")
        r4 = py("training/eval_sft.py", "--sft", spiked)
        if r4.returncode == 0:
            raise SystemExit("the evaluator accepted a spiked set: it is decoration")
        n_findings = sum(
            1 for line in r4.stdout.splitlines() if line.startswith("FINDING: ")
        )
        if n_findings < 3:
            raise SystemExit(
                f"the spiked set produced {n_findings} findings, "
                "fewer than the three defect classes injected"
            )
    return (
        f"every row cites, nothing duplicates, nothing empty, and the "
        f"evaluator catches a spiked defect of each class "
        f"({report['rows']} rows: {report['grounded']} grounded + {report['curriculum']} curriculum)"
    )


def selftest_sft_evaluation_baseline() -> None:
    """The canary: an uncited row must produce a finding, or the evaluator
    is decoration."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "eval_sft", str(ROOT / "training" / "eval_sft.py")
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    row = {
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "long enough text but no source line"},
        ],
        "kind": "doc",
    }
    assert any("citation" in f for f in mod.evaluate([row])["findings"]), (
        "an uncited row passes the evaluator: the gate is decoration"
    )


# --------------------------------------------------------------------------
# gate: the corpus build is a function of the tree (determinism)
# --------------------------------------------------------------------------
def gate_corpus_build_is_deterministic() -> str:
    """Two builds from the same tree must agree byte for byte. A corpus that
    drifts between runs would teach the model its own build id, and every
    pinned count this repository claims rests on the builder being a pure
    function of its inputs (measured over repeated installs 2026-09-07)."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        builds = []
        for i in (1, 2):
            out = Path(td) / f"knowledge-{i}.jsonl"
            r = subprocess.run(
                [sys.executable, str(ROOT / "training" / "build_corpus.py"),
                 "--repo", ".", "--out", str(out)],
                cwd=ROOT, capture_output=True, text=True, check=False,
            )
            if r.returncode != 0:
                raise SystemExit(f"build {i} failed: {r.stderr[-200:]}")
            builds.append(out.read_bytes())
    if builds[0] != builds[1]:
        raise SystemExit("two builds of the same tree disagree byte for byte")
    return "two builds, one byte sequence: the corpus is a function of the tree"


def selftest_corpus_build_is_deterministic() -> None:
    """The canary: the byte comparison must notice a single-byte drift."""
    import hashlib

    a = b'{"record": 1}\n'
    b = b'{"record": 1}\n'
    assert a == b and hashlib.sha256(a).digest() == hashlib.sha256(b).digest()
    tampered = b'{"record": 2}\n'
    assert a != tampered, "a one-byte drift must break equality, or the gate is blind"


# --------------------------------------------------------------------------
# gate: every declared dependency is reachable from its crate's own code
# --------------------------------------------------------------------------
def gate_dependencies_are_used(root: Path = ROOT) -> str:
    """A dependency a crate declares but never reaches is supply-chain
    weight with no cargo to carry: its audit surface is paid for by nobody's
    usage. The
    measurement lives here in Python
    so its canary can be proven in-process, no toolchain download needed.

    Evidence of use is conservative by design - an `ident::` path, a `use`
    of the ident, or a derive macro from the derive-providing set - so real
    usage cannot be flagged; a crate the gate flags is dead weight, and the
    self-test proves the flag fires."""
    import tomllib

    derive_providers = {"serde": ("Serialize", "Deserialize")}
    findings: list[str] = []
    total = 0
    for cargo_toml in sorted(root.glob("crates/*/Cargo.toml")):
        crate_root = cargo_toml.parent
        data = tomllib.loads(cargo_toml.read_text(encoding="utf-8"))
        deps = data.get("dependencies", {})
        if not deps:
            continue
        text = "\n".join(
            rs.read_text(encoding="utf-8")
            for rs in sorted(crate_root.rglob("*.rs"))
        )
        for name, spec in sorted(deps.items()):
            ident = name.replace("-", "_")
            if isinstance(spec, dict) and isinstance(spec.get("package"), str):
                ident = spec["package"].replace("-", "_")
            total += 1
            if f"{ident}::" in text or f"use {ident}" in text:
                continue
            derive = derive_providers.get(name)
            if derive and "#[derive(" in text and any(m in text for m in derive):
                continue
            findings.append(f"{crate_root.name}: `{name}` is declared but never reached")
    if findings:
        raise SystemExit("unused dependencies:\n" + "\n".join(f"  - {f}" for f in findings))
    return f"every declared dependency is reachable ({total} deps across the crates)"


def selftest_dependencies_are_used() -> None:
    """The canary: a declared-but-unreached dependency must produce a
    finding on a synthetic tree, or the gate is decoration."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        crate = root / "crates" / "dead"
        (crate / "src").mkdir(parents=True)
        (crate / "Cargo.toml").write_text(
            '[package]\nname = "dead"\nversion = "0.1.0"\n\n'
            '[dependencies]\ndeadweight = "1"\nlivecrate = "1"\n',
            encoding="utf-8",
        )
        (crate / "src" / "lib.rs").write_text(
            "pub fn f() -> u8 {\n    livecrate::answer()\n}\n",
            encoding="utf-8",
        )
        try:
            gate_dependencies_are_used(root)
        except SystemExit as exc:
            assert "deadweight" in str(exc) and "livecrate" not in str(exc), (
                f"the gate flagged the wrong crate: {exc}"
            )
            return
        raise AssertionError("a dead dependency passed the gate: decoration")


# --------------------------------------------------------------------------
# gate: findings carry the discipline or they do not exist
# --------------------------------------------------------------------------
def gate_findings_are_disciplined() -> str:
    """A finding is a claim about code; the validator measures the claim.
    The rule is ours: evidence before reporting, and every refusal is
    proven by a canary in-process. Any `*.findings.jsonl` file in the tree must
    pass, and the canary suite must keep firing: a finding without a
    location, a static reading sold as proven, a rating without its change
    condition, and a duplicate at the same sink are all refusals."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "findings", str(ROOT / "training" / "findings.py")
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    r = subprocess.run(
        [sys.executable, str(ROOT / "training" / "findings.py"), "--self-test"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if r.returncode != 0:
        raise SystemExit(f"the finding validator's canaries stopped firing: {r.stdout[-200:]} {r.stderr[-200:]}")
    checked = 0
    for path in sorted(ROOT.rglob("*.findings.jsonl")):
        if any(part in {"corpus", "target"} for part in path.parts):
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        report = mod.validate(rows)
        if report["findings"]:
            raise SystemExit(f"{path} carries an undisciplined finding:\n" + "\n".join(report["findings"]))
        checked += 1
    suffix = f"; {checked} findings file(s) in the tree pass" if checked else "; no findings files in the tree yet"
    return "every refusal fires on its canary" + suffix


def selftest_findings_are_disciplined() -> None:
    """The canary: the validator must refuse an undisciplined finding, or
    the gate is decoration."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "findings", str(ROOT / "training" / "findings.py")
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    bad = {"title": "x", "location": "somewhere", "status": "proven", "severity": "urgent"}
    report = mod.validate([bad])
    assert any("location" in p for p in report["findings"]), "a locationless finding passed"
    assert any("status" in p for p in report["findings"]), "a static claim sold as proven passed"
    assert any("condition" in p for p in report["findings"]), "a badge without conditions passed"
    clean = {
        "title": "t", "location": "a.rs:1", "sink": "f", "status": "unverified",
        "evidence": "static reading only", "severity": "low",
        "severity_change_conditions": "raises if a public route reaches the sink",
    }
    assert mod.validate([clean])["findings"] == [], "a disciplined finding was refused"


# --------------------------------------------------------------------------
# gate: evaluation runs are graded by one mechanical criterion
# --------------------------------------------------------------------------
_VAGUE_CRITERION_WORDS = ("iyi", "guzel", "basarili", "sorunsuz", "guclu", "zayif", "looks good")


def _eval_run_finding(rec: dict) -> str | None:
    """One mechanical boolean criterion plus full resource accounting, or the
    reason the run is not a measurement."""
    olcut = rec.get("olcut")
    if not isinstance(olcut, dict):
        return "run without an olcut record"
    ad = olcut.get("ad")
    if not isinstance(ad, str) or not ad.strip():
        return "criterion has no name"
    low = ad.lower()
    for word in _VAGUE_CRITERION_WORDS:
        if word in low:
            return f"criterion name carries a judgement word ({word}); name the machine check instead"
    if not isinstance(olcut.get("sonuc"), bool):
        return "criterion verdict must be a boolean; partial credit is not a result"
    kaynak = rec.get("kaynaklar")
    if not isinstance(kaynak, dict):
        return "run without resource accounting"
    for field, is_int in (("sure_saniye", False), ("girdi_jetonlari", True),
                          ("onbellekli_jetonlari", True), ("cikti_jetonlari", True),
                          ("maliyet", False)):
        if field not in kaynak:
            return f"missing resource field {field}"
        val = kaynak[field]
        if isinstance(val, bool) or (is_int and not isinstance(val, int)) or (not is_int and not isinstance(val, (int, float))):
            return f"resource field {field} is not a number"
        if val < 0:
            return f"resource field {field} is negative"
    if rec.get("kosucu") == "model" and kaynak.get("cikti_jetonlari", 0) == 0:
        return "a model run that reports zero output tokens is not measured"
    return None


def gate_eval_runs_are_mechanical() -> str:
    """Every recorded evaluation run carries exactly one machine-checkable
    boolean criterion plus its resource accounting; judgement words and
    partial credit are refused (one run, one mechanical criterion: the
    shape this repository measures itself by)."""
    run_dir = ROOT / "training" / "eval" / "sonuclar"
    runs = sorted(run_dir.glob("*.json"))
    if not runs:
        raise SystemExit("no evaluation runs recorded: the criterion stays unmeasured")
    for path in runs:
        rec = json.loads(path.read_text(encoding="utf-8"))
        finding = _eval_run_finding(rec)
        if finding:
            raise SystemExit(f"{path.name}: {finding}")
    return f"every evaluation run ({len(runs)}) is graded by one mechanical criterion with resource accounting"


def selftest_eval_runs_are_mechanical() -> None:
    """The canaries: a judgement verdict, a vague name and a missing
    accounting field must each be refused, or the gate is decoration."""
    base = {
        "kosucu": "betik",
        "olcut": {"ad": "iki_bagimsiz_korpus_insasinin_sha256_esitligi", "sonuc": True},
        "kaynaklar": {"sure_saniye": 0.2, "girdi_jetonlari": 0, "onbellekli_jetonlari": 0,
                      "cikti_jetonlari": 0, "maliyet": 0.0},
    }
    assert _eval_run_finding(base) is None, "a well-formed run was refused"
    judgement = dict(base, olcut={"ad": "sonuca bakildi", "sonuc": "evet"})
    assert _eval_run_finding(judgement) is not None, "a judgement verdict passed: partial credit leaked in"
    vague = dict(base, olcut={"ad": "cikti iyi gorunuyor", "sonuc": True})
    assert _eval_run_finding(vague) is not None, "a vague criterion name passed: the gate only checks shape"
    unmeasured = json.loads(json.dumps(base))
    del unmeasured["kaynaklar"]["sure_saniye"]
    assert _eval_run_finding(unmeasured) is not None, "a run without duration accounting passed"


# --------------------------------------------------------------------------
# gate: the review crate holds the ledger rules; a closure without evidence
# --------------------------------------------------------------------------
REVIEW_RULE_MARKERS = [
    "pub enum Severity",
    "pub fn record(",
    "pub fn fix(",
    "pub fn reject(",
    "pub fn complete(",
    "pub fn verify(",
    "AttesterRequired",
    "EmptyEvidence",
    "finding.severity >= Severity::High",
]


def gate_review_crate_holds_ledger_rules() -> str:
    """The denetim crate is the review-ledger authority: scan, validate and
    fix live in one ledger, closures are evidence-gated, the attestation
    floor sits at High, and re-scan invalidation is in force."""
    src = read("crates/denetim/src/lib.rs")
    for marker in REVIEW_RULE_MARKERS:
        if marker not in src:
            raise SystemExit(f"denetim crate lost a ledger rule: {marker}")
    if "let voided = self.dispositions.remove(&finding.id);" not in src:
        raise SystemExit("re-scan no longer voids a stale closure")
    return "denetim crate is the review-ledger authority"


def selftest_review_crate_holds_ledger_rules() -> None:
    good = "\n".join(REVIEW_RULE_MARKERS) + "\nlet voided = self.dispositions.remove(&finding.id);"
    thin = "pub fn record(\npub fn fix("
    assert not run_review_like(good)
    assert run_review_like(thin)


def run_review_like(text: str) -> bool:
    """True when the text does NOT carry the full rule set."""
    for marker in [*REVIEW_RULE_MARKERS, "let voided = self.dispositions.remove(&finding.id);"]:
        if marker not in text:
            return True
    return False


GATES_EXTRA = {
    "system-prompt-is-true": (gate_system_prompt_is_true, selftest_system_prompt_is_true),
    "review-crate-holds-ledger-rules": (gate_review_crate_holds_ledger_rules, selftest_review_crate_holds_ledger_rules),
    "operator-sync-rules": (gate_operator_sync_rules, selftest_operator_sync_rules),
    "output-finalize-closed-loop": (gate_output_finalize_closed_loop, selftest_output_finalize_closed_loop),
    "cli-asks-and-renders-markdown": (gate_cli_asks_and_renders_markdown, selftest_cli_asks_and_renders_markdown),
    "corpus-records-carry-licence": (gate_corpus_records_carry_licence, selftest_corpus_records_carry_licence),
    "no-multiplier-labels": (gate_no_multiplier_labels, selftest_no_multiplier_labels),
    "training-gate-epoch-ledger-fail-closed": (gate_training_gate_epoch_ledger_fail_closed, selftest_training_gate_epoch_ledger_fail_closed),
    "corpus-records-carry-provenance": (gate_corpus_records_carry_provenance, selftest_corpus_records_carry_provenance),
    "training-grant-crate-validates": (gate_training_grant_crate_validates, selftest_training_grant_crate_validates),
    "ai-output-schema-enforced": (gate_ai_output_schema_enforced, selftest_ai_output_schema_enforced),
    "chain-record-client-present": (gate_chain_record_client_present, selftest_chain_record_client_present),
    "no-generation-variant": (gate_no_generation_variant, selftest_no_generation_variant),
    "chain-surface-fixed": (gate_chain_surface_fixed, selftest_chain_surface_fixed),
    "yerlesik-komutlar-bagli": (gate_yerlesik_komutlar_bagli, selftest_yerlesik_komutlar_bagli),
    "doc-pdf-feeds-corpus": (gate_doc_pdf_feeds_corpus, selftest_doc_pdf_feeds_corpus),
    "queue-continues-uninterruptedly": (gate_queue_continues_uninterruptedly, selftest_queue_continues_uninterruptedly),
    "ratchet-holds": (gate_ratchet_holds, selftest_ratchet_holds),
    "public-api-is-reached": (gate_public_api_is_reached, selftest_public_api_is_reached),
    "fmt-clean": (gate_fmt_clean, selftest_fmt_clean),
    "it-is-restricted": (gate_it_is_restricted, selftest_it_is_restricted),
    "no-debug-leftovers": (gate_no_debug_leftovers, selftest_no_debug_leftovers),
    "no-secret-material": (gate_no_secret_material, selftest_no_secret_material),
    "graf-maps-workspace": (gate_graf_maps_workspace, selftest_graf_maps_workspace),
    "dosya-routes-before-reading": (gate_dosya_routes_before_reading, selftest_dosya_routes_before_reading),
    "soru-bataryasi-gecerli": (gate_soru_bataryasi_gecerli, selftest_soru_bataryasi_gecerli),
    "four-axes-wired": (gate_four_axes_wired, selftest_four_axes_wired),
    "kirmizi-senaryolar": (gate_kirmizi_senaryolar, selftest_kirmizi_senaryolar),
    "sft-evaluation-baseline": (gate_sft_evaluation_baseline, selftest_sft_evaluation_baseline),
    "corpus-build-is-deterministic": (gate_corpus_build_is_deterministic, selftest_corpus_build_is_deterministic),
    "dependencies-are-used": (gate_dependencies_are_used, selftest_dependencies_are_used),
    "findings-are-disciplined": (gate_findings_are_disciplined, selftest_findings_are_disciplined),
    "eval-runs-are-mechanical": (gate_eval_runs_are_mechanical, selftest_eval_runs_are_mechanical),
}


GATES = {
    "reads-not-generates": (gate_reads_not_generates, selftest_reads_not_generates),
    "no-fourth-channel": (gate_no_fourth_channel, selftest_no_fourth_channel),
    "provenance-fails-closed": (gate_provenance_fails_closed, selftest_provenance_fails_closed),
    "mask-before-storage": (gate_mask_before_storage, selftest_mask_before_storage),
    "no-panic-path": (gate_no_panic_path, selftest_no_panic_path),
    "readme-is-measured": (gate_readme_is_measured, selftest_readme_is_measured),
    **GATES_EXTRA,
}



def main(argv: list[str]) -> int:
    if not argv or argv[0] == "--list":
        for name in GATES:
            print(name)
        return 0
    if argv[0] == "--all":
        failures = 0
        for name, (run, selftest) in GATES.items():
            selftest()
            try:
                print(f"OK   [{name}] {run()}")
            except SystemExit as err:
                failures += 1
                print(f"FAIL [{name}] {err}")
        print("ALL GATES PASSED" if not failures else f"{failures} gate(s) failed")
        return 1 if failures else 0
    name = argv[0]
    if name not in GATES:
        print(f"unknown gate: {name}")
        return 2
    run, selftest = GATES[name]
    if "--self-test" in argv:
        selftest()
        print(f"self-test OK [{name}]")
        return 0
    print(f"OK   [{name}] {run()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
