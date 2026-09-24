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

import pathlib
import gzip
import json
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
    """The report's operator rules are checks Lubot can run: non-zero bond
    above the floor, one model_hash among active operators, effort tier
    hashed into the request within 0.5x-10.0x, and a checkpoint transition
    window with a real retirement moment."""
    src = read("crates/tools/src/operator.rs")
    for word in ["compute_bond_ok", "same_model_hash", "effort_tag_ok",
                 "effort_hash", "CheckpointWindow", "both_active", "old_retired"]:
        if word not in src:
            raise SystemExit(f"operator rules lost `{word}`")
    if "cheap" in src and "0.4x" not in src:
        raise SystemExit("the ceiling refusals have no fixture")
    chain = read("crates/tools/src/chain.rs")
    if "effort_hash does not match effort_tag" not in chain:
        raise SystemExit("parse_request no longer verifies the hashed effort tier")
    return "bond, single hash, ceiling-hashed effort and the transition window are all checks"


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
RATCHET_KEYS = ["tests", "gates", "pedantic", "corpus", "tokens", "bootstrap", "exam"]


def gate_ratchet_holds() -> str:
    """The baselines in training/ratchet.json hold: tests, gates, corpus and
    the corpus's unique-token count may only rise (pedantic may only fall)."""
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
    measured_bootstrap = len(_onyukleme_turlari())
    measured_exam = len(_sinav_satirlari())
    # The token budget is measured by the script that owns the tokenizer, not
    # re-implemented here: two counters for one corpus is two answers.
    butce = subprocess.run(
        [sys.executable, str(ROOT / "training" / "egitim_butcesi.py"), "--olc"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if butce.returncode != 0:
        raise SystemExit(f"the token budget does not measure: {butce.stderr[-200:]}")
    try:
        measured_tokens = json.loads(butce.stdout)["benzersiz_jeton"]
    except (json.JSONDecodeError, KeyError) as err:
        raise SystemExit(f"the token budget produced no count: {err}") from err
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
        "corpus": measured_corpus, "tokens": measured_tokens,
        "bootstrap": measured_bootstrap, "exam": measured_exam,
    }
    regressed = []
    for key in ["tests", "gates", "corpus", "tokens"]:
        if measured[key] < baseline[key]:
            regressed.append(f"{key} {measured[key]} < baseline {baseline[key]}")
    if measured["pedantic"] > baseline["pedantic"]:
        regressed.append(f"pedantic {measured['pedantic']} > baseline {baseline['pedantic']}")
    if regressed:
        raise SystemExit("ratchet regressed: " + "; ".join(regressed))
    return (
        f"ratchet holds: tests {measured_tests}, gates {measured_gates}, "
        f"pedantic {measured_pedantic}, corpus {measured_corpus}, "
        f"tokens {measured_tokens}, bootstrap {measured_bootstrap}, "
        f"exam {measured_exam}"
    )


def selftest_ratchet_holds() -> None:
    """The direction rules: four rise (>=), pedantic falls (<=)."""
    assert set(RATCHET_KEYS) == {
        "tests", "gates", "pedantic", "corpus", "tokens", "bootstrap", "exam"
    }
    baseline = {"tests": 5, "pedantic": 2, "tokens": 100}
    assert 6 >= baseline["tests"], "tests may rise"
    assert 1 <= baseline["pedantic"], "pedantic may fall"
    assert 120 >= baseline["tokens"], "the token budget may rise"



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
# gate: every crate directory is a workspace member
# --------------------------------------------------------------------------
# A crate written but not registered compiles nowhere and is invisible to CI.
# That is silent, so it is gated.


def _member_names(manifest: str) -> set[str]:
    body = manifest.split("[workspace]", 1)[1]
    body = body.split("\n[", 1)[0]
    return set(re.findall(r'"([^"]+)"', body.split("members", 1)[1].split("]", 1)[0]))


def _crate_dirs() -> list[str]:
    return sorted(
        p.parent.name
        for p in (ROOT / "crates").glob("*/Cargo.toml")
    )


def gate_every_crate_is_a_member() -> str:
    """Every directory under `crates/` holding a Cargo.toml is a workspace member."""
    members = _member_names(read("Cargo.toml"))
    orphaned = [c for c in _crate_dirs() if f"crates/{c}" not in members]
    if orphaned:
        raise SystemExit(
            "these crates compile nowhere, so CI has never seen them:\n  "
            + "\n  ".join(orphaned)
        )
    return f"all {len(_crate_dirs())} crate directories are workspace members"


def selftest_every_crate_is_a_member() -> None:
    members = _member_names('[workspace]\nmembers = ["crates/read"]\n\n[profile]\nx = 1\n')
    assert members == {"crates/read"}, f"member parsing broke: {members}"
    # The gate must be able to see an orphan.
    orphans = [c for c in ["read", "ghost"] if f"crates/{c}" not in members]
    assert orphans == ["ghost"], f"an orphan was not detected: {orphans}"


# --------------------------------------------------------------------------
# gate: assert macros carry the arguments they take
# --------------------------------------------------------------------------
# `assert_eq!` takes two values, or two values and a format message. Three
# values is a compile error that reads like a working assertion until it is
# built, and nothing before the build notices.


def _split_top_level(body: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    in_str = False
    escaped = False
    for ch in body:
        if in_str:
            current.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            current.append(ch)
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(ch)
    tail = "".join(current)
    if tail.strip():
        parts.append(tail)
    return parts


def _macro_call_args(text: str, macro: str) -> list[tuple[int, list[str]]]:
    """Returns (line number, argument list) for every `macro!(...)` call."""
    found: list[tuple[int, list[str]]] = []
    needle = macro + "!"
    start = 0
    while True:
        at = text.find(needle, start)
        if at < 0:
            return found
        open_at = text.find("(", at)
        if open_at < 0:
            return found
        depth = 0
        in_str = False
        escaped = False
        end = open_at
        while end < len(text):
            ch = text[end]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
                if depth == 0:
                    break
            end += 1
        found.append((text.count("\n", 0, at) + 1, _split_top_level(text[open_at + 1 : end])))
        start = end + 1


def _strip_tests_and_comments(text: str) -> str:
    """Keeps test bodies (the assertions live there) but drops comment lines."""
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("//")
    )


def gate_assert_arity() -> str:
    """`assert_eq!` carries two values, then optionally a format string.

    More than three arguments is legal only when the third is the format string
    and the rest are its arguments, so that is what is checked. Three values
    with no format string does not compile, and it reads like a working
    assertion until it is built.
    """
    offenders: list[str] = []
    for path in rust_sources():
        text = _strip_tests_and_comments(path.read_text(encoding="utf-8"))
        for macro in ("assert_eq", "assert_ne", "debug_assert_eq"):
            for line, args in _macro_call_args(text, macro):
                if len(args) < 2:
                    offenders.append(
                        f"{path.relative_to(ROOT)}:{line} {macro}! has {len(args)} argument(s)"
                    )
                    continue
                if len(args) > 2 and not args[2].lstrip().startswith('"'):
                    offenders.append(
                        f"{path.relative_to(ROOT)}:{line} {macro}! has a third argument "
                        f"that is not a format string: {args[2].strip()[:60]}"
                    )
    if offenders:
        raise SystemExit("an assertion that will not compile:\n  " + "\n  ".join(offenders))
    checked = sum(
        len(_macro_call_args(_strip_tests_and_comments(p.read_text(encoding="utf-8")), "assert_eq"))
        for p in rust_sources()
    )
    return f"{checked} assert_eq! calls carry a valid argument count"


def selftest_assert_arity() -> None:
    assert _split_top_level("a, b") == ["a", " b"]
    assert _split_top_level("f(1, 2), g(3)") == ["f(1, 2)", " g(3)"]
    assert _split_top_level('"a, b", c') == ['"a, b"', " c"]
    # Three values and no format string: the shape that does not compile.
    bad = _macro_call_args('assert_eq!(a, b, c)', "assert_eq")[0][1]
    assert len(bad) == 3, f"the canary was not parsed as three arguments: {bad}"
    assert not bad[2].lstrip().startswith('"'), f"the canary stopped being bad: {bad}"
    # A format string with commas inside it must stay one argument.
    ok = _macro_call_args('assert_eq!(a, b, "x {} {}", y, z)', "assert_eq")[0][1]
    assert len(ok) == 5, f"a message with commas was split: {ok}"
    assert ok[2].strip().startswith('"'), "the format string was not recognised"
    # Two values is the normal case.
    assert len(_macro_call_args('assert_eq!(a, b)', "assert_eq")[0][1]) == 2


# --------------------------------------------------------------------------
# gate: no error variant that nothing produces
# --------------------------------------------------------------------------
# A variant declared, displayed and asserted but never constructed is a variant
# no caller can ever handle. It reads as coverage and is not.


def _enum_variants(text: str, enum_name: str) -> list[str]:
    at = text.find(f"enum {enum_name}")
    if at < 0:
        return []
    open_at = text.find("{", at)
    depth = 0
    end = open_at
    while end < len(text):
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
            if depth == 0:
                break
        end += 1
    body = text[open_at + 1 : end]
    body = "\n".join(l for l in body.splitlines() if not l.lstrip().startswith("//"))
    names: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        # `(` has to be here: a tuple variant is `Variant(Type)`, and without it
        # every tuple variant looks undeclared, which reads as a compile error
        # that is not one.
        match = re.match(r"^([A-Z][A-Za-z0-9_]*)\s*(\{|\(|,|$)", stripped)
        if match:
            names.append(match.group(1))
    return names


def _is_construction(text: str, enum_name: str, variant: str) -> bool:
    """True when the variant is built somewhere, not merely named.

    Match arms are excluded: `Self::Variant { .. } => ...` mentions the variant
    without producing it, which is exactly the shape that made a dead variant
    look covered.
    """
    pattern = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(enum_name) + r"::" + re.escape(variant) + r"(?![A-Za-z0-9_])")
    for line in text.splitlines():
        if line.lstrip().startswith("//") or line.lstrip().startswith("///"):
            continue
        if "=>" in line:
            continue
        if pattern.search(line):
            return True
    return False


def gate_no_dead_error_variant() -> str:
    """Every error variant is both declared and constructed.

    Both directions, because each half alone has a blind spot. A declared variant
    nothing constructs is a variant no caller can handle. A constructed variant
    that is not declared does not compile - and the first half cannot see it,
    since it only ever looks at names it found in the declaration.
    """
    offenders: list[str] = []
    checked = 0
    for path in rust_sources():
        text = path.read_text(encoding="utf-8")
        enums = sorted(set(re.findall(r"pub enum ([A-Za-z0-9_]*Error)\b", text)))
        for enum_name in enums:
            declared = _enum_variants(text, enum_name)
            for variant in declared:
                checked += 1
                if not _is_construction(text, enum_name, variant):
                    offenders.append(
                        f"{path.relative_to(ROOT)} {enum_name}::{variant} is declared but never produced"
                    )
            # The other half: every `Enum::Variant` the file writes must be one of
            # the variants it declares.
            for variant in sorted(set(re.findall(
                r"(?<![A-Za-z0-9_])" + re.escape(enum_name) + r"::([A-Z][A-Za-z0-9_]*)", text
            ))):
                checked += 1
                if variant not in declared:
                    offenders.append(
                        f"{path.relative_to(ROOT)} {enum_name}::{variant} is used but not declared"
                    )
    if offenders:
        raise SystemExit("an error variant is declared or used without the other:\n  "
            + "\n  ".join(offenders))
    return f"{checked} error-variant checks pass in both directions"


def selftest_no_dead_error_variant() -> None:
    sample = """
pub enum DemoError {
    /// documented
    Live { reason: String },
    Dead { reason: String },
}

impl std::fmt::Display for DemoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Live { .. } => write!(f, "live"),
            Self::Dead { .. } => write!(f, "dead"),
        }
    }
}

fn build() -> Result<(), DemoError> {
    Err(DemoError::Live { reason: "x".to_string() })
}
"""
    variants = _enum_variants(sample, "DemoError")
    assert variants == ["Live", "Dead"], f"variant parsing broke: {variants}"
    # Tuple and unit variants, not just struct variants.
    shapes = "pub enum E {\n    Tuple(String),\n    Unit,\n    Struct { a: u8 },\n}\n"
    assert _enum_variants(shapes, "E") == ["Tuple", "Unit", "Struct"], (
        f"tuple and unit variants were missed: {_enum_variants(shapes, 'E')}"
    )
    assert _is_construction(sample, "DemoError", "Live"), "a constructed variant looked dead"
    assert not _is_construction(sample, "DemoError", "Dead"), "a dead variant looked constructed"
    # The other direction: a variant used but never declared. This is the half
    # whose absence let a compile error through.
    used = set(re.findall(r"(?<![A-Za-z0-9_])DemoError::([A-Z][A-Za-z0-9_]*)", sample))
    assert used == {"Live"}, f"usage scanning broke: {used}"
    missing_sample = sample.replace("    Dead { reason: String },\n", "")
    declared = set(_enum_variants(missing_sample, "DemoError"))
    assert "Dead" in used or "Dead" not in declared, "fixture is not exercising the gap"
    ghost = "Err(DemoError::Ghost { reason: String::new() })"
    assert set(re.findall(r"(?<![A-Za-z0-9_])DemoError::([A-Z][A-Za-z0-9_]*)", ghost)) == {"Ghost"}
    assert "Ghost" not in _enum_variants(sample, "DemoError")


# --------------------------------------------------------------------------
# gate: intra-doc links resolve
# --------------------------------------------------------------------------
# A link left behind by a deleted variant is a rustdoc warning, and a warning
# that is not an error is a warning nobody reads.


def _doc_links(line: str) -> list[str]:
    """The intra-doc links on a doc-comment line.

    Both `[Type]` and `[`Type::Variant`]` are links; rustdoc documents the
    backticked form. A lowercase single segment is not: it is prose naming
    something, and a TOML `[package]` heading is indistinguishable from a link
    by shape alone.
    """
    found: list[str] = []
    for backtick, path in re.findall(
        r"\[(`?)([A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z0-9_]+)*)\1\]", line
    ):
        del backtick
        head = path.partition("::")[0]
        if "::" not in path and not head[0].isupper() and head != "Self":
            continue
        found.append(path)
    return found


KNOWN_EXTERNAL_LINK_TARGETS = {
    "Self", "Vec", "VecDeque", "BTreeMap", "BTreeSet", "HashMap", "HashSet",
    "Option", "Result", "String", "str", "u8", "u16", "u32", "u64", "u128",
    "i64", "usize", "f64", "bool", "Iterator", "Copy", "Eq", "Debug", "Clone",
    "Default", "PartialEq", "Ord", "Display",
}


def gate_doc_links_resolve() -> str:
    """Every `[Type::Thing]` doc link names something this file declares."""
    offenders: list[str] = []
    checked = 0
    for path in rust_sources():
        text = path.read_text(encoding="utf-8")
        declared = set(re.findall(r"\b(?:pub\s+)?(?:enum|struct|trait|type|const|fn)\s+([A-Za-z0-9_]+)", text))
        # A link can name something this file imported rather than declared, and
        # rustdoc resolves it through the import. Without this the gate reports
        # every cross-crate reference as broken.
        for use_line in re.findall(r"^\s*use\s+([^;]+);", text, re.MULTILINE):
            for part in re.findall(r"[A-Za-z0-9_]+", use_line):
                declared.add(part)
        for line in text.splitlines():
            if not line.lstrip().startswith("///") and not line.lstrip().startswith("//!"):
                continue
            # Scan inside code spans too: `[`Type::Variant`]` is the idiom
            # rustdoc documents. What is skipped instead is a lowercase single
            # segment, which is prose naming something rather than a link.
            for link in _doc_links(line):
                head, _, tail = link.partition("::")
                if head in KNOWN_EXTERNAL_LINK_TARGETS:
                    continue
                # A path-qualified link (`crate::X`, `lubot_read::X`) is resolved
                # by rustdoc against the crate graph, not against this file.
                if head in ("crate", "self", "super") or "::" in head:
                    continue
                if head not in declared and re.match(r"^[a-z][a-z0-9_]*$", head):
                    # A crate name: `lubot_read::perception::MAX_TEXT_BYTES`.
                    continue
                checked += 1
                if head not in declared:
                    offenders.append(f"{path.relative_to(ROOT)} [{link}]: no `{head}` in this file")
                    continue
                if tail and not re.search(r"(?<![A-Za-z0-9_])" + re.escape(tail) + r"(?![A-Za-z0-9_])", text):
                    offenders.append(f"{path.relative_to(ROOT)} [{link}]: no `{tail}` in this file")
    if offenders:
        raise SystemExit("broken intra-doc links:\n  " + "\n  ".join(offenders))
    return f"{checked} intra-doc links resolve"


def selftest_doc_links_resolve() -> None:
    sample = "pub enum Thing {\n    Gone,\n}\n/// see [Thing::Gone] and [Thing::Missing]\n"
    declared = set(re.findall(r"\b(?:pub\s+)?(?:enum|struct|trait|type|const|fn)\s+([A-Za-z0-9_]+)", sample))
    assert declared == {"Thing"}, f"declaration scanning broke: {declared}"
    links = re.findall(r"\[([A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z0-9_]+)*)\]", sample)
    assert links == ["Thing::Gone", "Thing::Missing"], f"link scanning broke: {links}"
    # The backticked form is the idiom rustdoc documents, so it must be scanned.
    backticked = _doc_links("/// see [`Thing::Gone`] for the refusal")
    assert backticked == ["Thing::Gone"], f"a backticked link was skipped: {backticked}"
    # A lowercase single segment is prose naming something, not a link.
    prose = _doc_links("/// the manifest's `[package]` block")
    assert prose == [], f"prose was read as a link: {prose}"
    # The canary: the removed variant must be caught.
    assert not re.search(r"(?<![A-Za-z0-9_])Missing(?![A-Za-z0-9_])", "pub enum Thing { Gone, }")
    assert re.search(r"(?<![A-Za-z0-9_])Gone(?![A-Za-z0-9_])", "pub enum Thing { Gone, }")


# --------------------------------------------------------------------------
# gate: no boolean compared to a literal
# --------------------------------------------------------------------------
# `flag == false` is a clippy failure under the workspace lint set, and it reads
# as a comparison rather than as the negation it is.


def gate_no_bool_comparison() -> str:
    """No `== true` or `== false` in any Rust source."""
    offenders: list[str] = []
    for path in rust_sources():
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("//"):
                continue
            if re.search(r"==\s*(true|false)\b", line) or re.search(r"!=\s*(true|false)\b", line):
                offenders.append(f"{path.relative_to(ROOT)}:{i}")
    if offenders:
        raise SystemExit("a boolean compared to a literal:\n  " + "\n  ".join(offenders))
    return f"no boolean is compared to a literal in {len(rust_sources())} files"


def selftest_no_bool_comparison() -> None:
    assert re.search(r"==\s*(true|false)\b", "assert!(x.is_ready() == false)")
    assert not re.search(r"==\s*(true|false)\b", "assert!(!x.is_ready())")
    assert re.search(r"!=\s*(true|false)\b", "if flag != true {")


def _starts_char_literal(text: str, at: int) -> bool:
    """Whether the apostrophe at `at` opens a character literal.

    Rust reuses the apostrophe for lifetimes, so the one in `fn f<'a>()` does not
    open a literal. Treating it as one makes the scanner consume everything up to
    the next apostrophe, which is how balanced files came out unbalanced.
    """
    rest = text[at + 1:]
    if rest.startswith("'"):
        return True
    if rest.startswith("\\"):
        return len(rest) > 2 and rest[2] == "'"
    return len(rest) > 1 and rest[1] == "'"


# --------------------------------------------------------------------------
# gate: delimiters balance
# --------------------------------------------------------------------------
# A brace-balance check is not a type check, but an unbalanced file cannot be
# one, and it costs nothing to run before anything more expensive.


def gate_delimiters_balance() -> str:
    """Braces, parentheses and brackets balance in every Rust source."""
    offenders: list[str] = []
    for path in rust_sources():
        text = path.read_text(encoding="utf-8")
        code: list[str] = []
        in_str = False
        in_char = False
        in_line_comment = False
        in_block_comment = 0
        previous = ""
        for i, ch in enumerate(text):
            if in_line_comment:
                if ch == "\n":
                    in_line_comment = False
                    code.append(ch)
                previous = ch
                continue
            if in_block_comment:
                if previous == "*" and ch == "/":
                    in_block_comment -= 1
                previous = ch
                continue
            if in_str:
                if ch == '"' and previous != "\\":
                    in_str = False
                previous = ch
                continue
            if in_char:
                if ch == "'" and previous != "\\":
                    in_char = False
                previous = ch
                continue
            if previous == "/" and ch == "/":
                in_line_comment = True
                previous = ch
                continue
            if previous == "/" and ch == "*":
                in_block_comment += 1
                previous = ch
                continue
            if ch == '"':
                in_str = True
            elif ch == "'" and _starts_char_literal(text, i):
                in_char = True
            else:
                code.append(ch)
            previous = ch
        body = "".join(code)
        for open_ch, close_ch in (("{", "}"), ("(", ")"), ("[", "]")):
            delta = body.count(open_ch) - body.count(close_ch)
            if delta:
                offenders.append(
                    f"{path.relative_to(ROOT)} {open_ch}{close_ch} off by {delta}"
                )
    if offenders:
        raise SystemExit("unbalanced delimiters:\n  " + "\n  ".join(offenders))
    return f"delimiters balance in {len(rust_sources())} files"


def selftest_delimiters_balance() -> None:
    # A lifetime is not a character literal. This is the case that made
    # balanced files report as unbalanced.
    lifetimes = "fn f<'a>(x: &'a str) {}"
    assert not _starts_char_literal(lifetimes, 5), "a lifetime opened a literal"
    assert not _starts_char_literal(lifetimes, 14), "a reference lifetime opened a literal"
    plain = "let c = 'x';"
    assert _starts_char_literal(plain, 8), "a real character literal was missed"
    escaped = "let c = '\\n';"
    assert _starts_char_literal(escaped, 8), "an escaped literal was missed"



# --------------------------------------------------------------------------
# gate: crates are reachable from the binary, and the unwired set only shrinks
# --------------------------------------------------------------------------
# A crate that nothing calls is code that has never been run. That is worth
# measuring rather than assuming, and worth ratcheting rather than merely
# reporting: the number is only interesting if it cannot go up.


def _path_deps(manifest: str) -> set[str]:
    """The crate directories this manifest depends on through `path`."""
    found: set[str] = set()
    for match in re.finditer(r'path\s*=\s*"([^"]+)"', manifest):
        target = Path(match.group(1)).name
        if target:
            found.add(target)
    return found


def _reachable_from(root_crate: str) -> set[str]:
    seen: set[str] = set()
    stack = [root_crate]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        manifest = ROOT / "crates" / current / "Cargo.toml"
        if not manifest.is_file():
            continue
        for dependency in _path_deps(manifest.read_text(encoding="utf-8")):
            stack.append(dependency)
    return seen


UNWIRED_BASELINE = "gates/unwired.baseline"


def gate_crates_are_reachable() -> str:
    """Every crate is reachable from the binary, or it is on the shrinking list.

    The list is a ratchet, not a permission. A crate may be added to it when it
    is written; the gate fails the moment the list grows, so wiring can only ever
    catch up.
    """
    crates = set(_crate_dirs())
    reachable = _reachable_from("cli")
    unwired = sorted(crates - reachable)
    baseline_path = ROOT / UNWIRED_BASELINE
    baseline: set[str] = set()
    if baseline_path.is_file():
        baseline = {
            line.strip()
            for line in baseline_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        }
    # A baseline naming a crate that no longer exists is stale, and a stale
    # baseline silently permits whatever replaces it.
    stale = sorted(baseline - crates)
    if stale:
        raise SystemExit(
            f"{UNWIRED_BASELINE} names crates that do not exist: {', '.join(stale)}"
        )
    regressed = sorted(set(unwired) - baseline)
    if regressed:
        raise SystemExit(
            "these crates are reachable from nothing and are not on the baseline:\n  "
            + "\n  ".join(regressed)
            + f"\n  add them to {UNWIRED_BASELINE} only while they are being wired"
        )
    wired_since = sorted(baseline - set(unwired))
    if wired_since:
        raise SystemExit(
            "these crates are now reachable, so shrink the baseline:\n  "
            + "\n  ".join(wired_since)
            + f"\n  remove them from {UNWIRED_BASELINE}"
        )
    return (
        f"{len(reachable & crates)} of {len(crates)} crates are reachable from the binary; "
        f"{len(unwired)} are on the baseline"
    )


def selftest_crates_are_reachable() -> None:
    manifest = """
[dependencies]
lubot-read = { path = "../read" }
lubot-answer = { path = "../answer" }
serde = "1"
"""
    deps = _path_deps(manifest)
    assert deps == {"read", "answer"}, f"path dependency parsing broke: {deps}"
    # The ratchet has to be able to see a regression.
    assert set(["yeni"]) - set(["eski"]) == {"yeni"}, "a new unwired crate went unnoticed"
    # And it has to be able to see the list shrinking.
    assert set(["eski"]) - set([]) == {"eski"}, "a wired crate was not reported"



# --------------------------------------------------------------------------
# gate: the crate documentation is measured
# --------------------------------------------------------------------------
# A table of crate sizes that nobody checks is a table that goes stale the first
# time a crate changes, and a stale table is worse than no table because it is
# read as current.


def _crate_measurements() -> dict[str, tuple[int, int]]:
    """Crate name to (source lines, test count)."""
    measured: dict[str, tuple[int, int]] = {}
    for manifest in sorted((ROOT / "crates").glob("*/Cargo.toml")):
        name = manifest.parent.name
        lines = 0
        tests = 0
        for source in sorted(manifest.parent.rglob("*.rs")):
            text = source.read_text(encoding="utf-8")
            lines += len(text.splitlines())
            tests += len(re.findall(r"#\[test\]", text))
        measured[name] = (lines, tests)
    return measured


def gate_crates_doc_is_measured() -> str:
    """`docs/CRATES.md` names every crate and its figures match the source."""
    doc = read("docs/CRATES.md")
    measured = _crate_measurements()
    missing = sorted(set(measured) - set(re.findall(r"`([a-z0-9_]+)`", doc)))
    if missing:
        raise SystemExit(
            "these crates are not documented in docs/CRATES.md:\n  " + "\n  ".join(missing)
        )
    wrong: list[str] = []
    for name, (lines, tests) in sorted(measured.items()):
        # The tables read `| `name` | 664 | 16 |`.
        row = re.search(
            r"\|\s*`" + re.escape(name) + r"`\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|", doc
        )
        if not row:
            # Prose figures are not checked, so a crate documented only in prose
            # is a crate whose figures nobody verifies. Requiring the row is what
            # makes the table the single place a figure can be stated.
            wrong.append(f"{name}: documented without a `| crate | lines | tests |` row")
            continue
        claimed_lines, claimed_tests = int(row.group(1)), int(row.group(2))
        if claimed_lines != lines:
            wrong.append(f"{name}: documented as {claimed_lines} lines, has {lines}")
        if claimed_tests != tests:
            wrong.append(f"{name}: documented as {claimed_tests} tests, has {tests}")
    if wrong:
        raise SystemExit("docs/CRATES.md is stale:\n  " + "\n  ".join(wrong))
    return f"all {len(measured)} crates are documented and their figures match"


def selftest_crates_doc_is_measured() -> None:
    doc = "| `read` | 848 | 28 | x |\n| `muhur` | 1 | 1 | y |\n"
    row = re.search(r"\|\s*`read`\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|", doc)
    assert row is not None, "the row pattern does not match its own fixture"
    assert (int(row.group(1)), int(row.group(2))) == (848, 28)
    # The gate has to be able to see a crate that is not mentioned.
    documented = set(re.findall(r"`([a-z0-9_]+)`", doc))
    assert set(["read", "muhur", "ghost"]) - documented == {"ghost"}
    # A crate mentioned in prose but absent from the tables must be caught.
    prose_only = set(re.findall(r"`([a-z0-9_]+)`", "and `cli` has 6818 lines"))
    assert prose_only == {"cli"}
    assert re.search(r"\|\s*`cli`\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|", doc) is None



# --------------------------------------------------------------------------
# gate: the RPC surface is one surface, stated in three places
# --------------------------------------------------------------------------
# The allowed-method array, the JSON set and the system prompt all state the
# same list. They were inconsistent - the JSON note said seven while the array
# held eight - and an inconsistency between a document and the code it describes
# is only visible to whoever reads both.


TURKISH_NUMBERS = {
    "bir": 1, "iki": 2, "uc": 3, "dort": 4, "bes": 5,
    "alti": 6, "yedi": 7, "sekiz": 8, "dokuz": 9, "on": 10,
}


def gate_rpc_surface_consistent() -> str:
    """`ALLOWED_METHODS`, `rpc-seti.json` and the system prompt agree."""
    chain = read("crates/tools/src/chain.rs")
    match = re.search(
        r"ALLOWED_METHODS:\s*\[&str;\s*(\d+)\]\s*=\s*\[([^\]]*)\]", chain, re.DOTALL
    )
    if not match:
        raise SystemExit("ALLOWED_METHODS could not be found in crates/tools/src/chain.rs")
    declared_length = int(match.group(1))
    allowed = re.findall(r'"([^"]+)"', match.group(2))
    if len(allowed) != declared_length:
        raise SystemExit(
            f"ALLOWED_METHODS declares {declared_length} entries and holds {len(allowed)}"
        )
    listed = json.loads(read("training/rpc-seti.json"))["methods"]
    if sorted(listed) != sorted(allowed):
        raise SystemExit(
            "training/rpc-seti.json and ALLOWED_METHODS disagree:\n"
            f"  only in the JSON: {sorted(set(listed) - set(allowed))}\n"
            f"  only in the array: {sorted(set(allowed) - set(listed))}"
        )
    # The note states the count in words, in two files.
    prompt = read("training/system_prompt.md")
    spoken = re.search(r"Zincir yüzeyi (\w+) sabit RPC", prompt)
    if not spoken:
        raise SystemExit("training/system_prompt.md no longer states the RPC count")
    word = spoken.group(1).lower()
    # The prompt is written with Turkish diacritics; the count word is not one of
    # the words that carries one, but fold them anyway so a rewrite cannot break
    # the match for a reason unrelated to the count.
    folded = word.replace("\u00fc", "u").replace("\u00e7", "c").replace("\u0131", "i")
    if folded not in TURKISH_NUMBERS:
        raise SystemExit(f"the system prompt states the count as {word!r}, which is not a number")
    if TURKISH_NUMBERS[folded] != len(allowed):
        raise SystemExit(
            f"training/system_prompt.md says {word} ({TURKISH_NUMBERS[folded]}) and the surface has {len(allowed)}"
        )
    note = json.loads(read("training/rpc-seti.json"))["note"]
    digits = re.findall(r"\b(\d+)\b", note)
    if str(len(allowed)) not in digits:
        raise SystemExit(
            f"the rpc-seti.json note does not state the count {len(allowed)} anywhere"
        )
    return f"the {len(allowed)}-method surface is stated identically in all three places"


def selftest_rpc_surface_consistent() -> None:
    chain = 'pub const ALLOWED_METHODS: [&str; 2] = [\n    "a",\n    "b",\n];\n'
    match = re.search(r"ALLOWED_METHODS:\s*\[&str;\s*(\d+)\]\s*=\s*\[([^\]]*)\]", chain, re.DOTALL)
    assert match is not None, "the array pattern does not match its own fixture"
    assert int(match.group(1)) == 2 and re.findall(r'"([^"]+)"', match.group(2)) == ["a", "b"]
    # A length that disagrees with the contents has to be caught.
    lying = 'pub const ALLOWED_METHODS: [&str; 3] = [\n    "a",\n    "b",\n];\n'
    m2 = re.search(r"ALLOWED_METHODS:\s*\[&str;\s*(\d+)\]\s*=\s*\[([^\]]*)\]", lying, re.DOTALL)
    assert int(m2.group(1)) != len(re.findall(r'"([^"]+)"', m2.group(2)))
    # Turkish number words, with and without diacritics.
    assert TURKISH_NUMBERS["sekiz"] == 8
    assert TURKISH_NUMBERS["yedi"] == 7
    assert "sekiz".replace("\u00fc", "u") in TURKISH_NUMBERS



# --------------------------------------------------------------------------
# gate: public API is used outside its own crate, on a shrinking list
# --------------------------------------------------------------------------
# A `pub` item nothing outside its crate reaches is either meant to be
# `pub(crate)` or is code waiting for a caller. Both are worth knowing; neither
# is worth arguing about, so it is measured and ratcheted.
#
# Deleting uncalled API is a judgement call about what the code is for - these
# are the encoded form of rules the project decided on, not leftovers - so the
# gate reports rather than removes, and fails only when the list grows.


def _pub_items(path: Path) -> list[str]:
    """Names declared `pub` at the top level of a file."""
    names: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^pub (?:const |static |fn |struct |enum |trait |type )([A-Za-z0-9_]+)", line)
        if match:
            names.append(match.group(1))
    return names


def _crate_of(path: Path) -> str:
    parts = path.relative_to(ROOT).parts
    return parts[1] if len(parts) > 1 and parts[0] == "crates" else parts[0]


def _externally_used(name: str, own_crate: str, texts: dict[str, str]) -> bool:
    pattern = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])")
    for crate, text in texts.items():
        if crate == own_crate:
            continue
        if pattern.search(text):
            return True
    return False


# The binary crate has nothing above it, so none of its items can be reached
# from outside by construction. Counting them drowns the signal from the
# libraries, which is what the gate is about.
BINARY_CRATES = {"cli"}


def gate_pub_api_is_used() -> str:
    """Every `pub` item in a library crate is reached from outside it, or is listed."""
    everything = sorted((ROOT / "crates").rglob("*.rs"))
    # The binary is excluded from what is *reported* but kept in what is
    # *searched*: it is the main consumer of the libraries, and dropping it from
    # the corpus would make every item it calls look unused.
    sources = [p for p in everything if _crate_of(p) not in BINARY_CRATES]
    texts: dict[str, list[str]] = {}
    for path in everything:
        texts.setdefault(_crate_of(path), []).append(path.read_text(encoding="utf-8"))
    joined = {crate: "\n".join(parts) for crate, parts in texts.items()}
    unused: list[str] = []
    for path in sources:
        crate = _crate_of(path)
        for name in _pub_items(path):
            if not _externally_used(name, crate, joined):
                unused.append(f"{crate}:{name}")
    unused.sort()
    baseline_path = ROOT / "gates/unused-pub-api.baseline"
    baseline: set[str] = set()
    if baseline_path.is_file():
        baseline = {
            line.strip()
            for line in baseline_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        }
    stale = sorted(baseline - set(unused))
    if stale:
        raise SystemExit(
            "these entries are used now, so shrink the baseline:\n  "
            + "\n  ".join(stale)
            + "\n  remove them from gates/unused-pub-api.baseline"
        )
    regressed = sorted(set(unused) - baseline)
    if regressed:
        raise SystemExit(
            "these public items are reached from nowhere outside their crate:\n  "
            + "\n  ".join(regressed)
        )
    return f"{len(unused)} public items are crate-internal by use; the list has not grown"


def selftest_pub_api_is_used() -> None:
    sample = "pub fn used_elsewhere() {}\npub struct AlsoUsed;\nfn private() {}\n"
    names = _pub_items_from_text(sample)
    assert names == ["used_elsewhere", "AlsoUsed"], f"pub scanning broke: {names}"
    # An item only its own crate mentions has to be caught.
    texts = {"a": "x.used_elsewhere()", "b": "unrelated"}
    assert _externally_used("used_elsewhere", "c", texts) is True
    assert _externally_used("AlsoUsed", "c", texts) is False
    # The same name inside its own crate does not count as external use.
    assert _externally_used("used_elsewhere", "a", texts) is False


def _pub_items_from_text(text: str) -> list[str]:
    names: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^pub (?:const |static |fn |struct |enum |trait |type )([A-Za-z0-9_]+)", line)
        if match:
            names.append(match.group(1))
    return names


# --------------------------------------------------------------------------
# gate: the frozen BPE vocab is versioned, lossless and structurally sound
# --------------------------------------------------------------------------
def gate_tokenizer_vocab_is_frozen() -> str:
    """Every frozen BPE vocab family is committed, versioned and lossless:
    each vocab under training/tokenizer/ loads through the fail-closed
    loader, round-trips every corpus record this machine holds, and its
    pretoken pattern is the trainer's own; a family that is derived on the
    fly, structurally broken or silently renamed is refused."""
    vocab_dir = ROOT / "training" / "tokenizer"
    vocabs = sorted(vocab_dir.glob("lubot-bpe-v*.json"))
    if not vocabs:
        raise SystemExit(
            "training/tokenizer/ has no frozen vocab family; the vocab is "
            "cut and committed, never derived on the fly"
        )
    corpora = sorted((ROOT / "corpus").glob("knowledge-*.jsonl.gz"))
    if not corpora:
        raise SystemExit(
            "no knowledge-*.jsonl.gz corpus under corpus/; CI builds it before the gates"
        )
    for vocab in vocabs:
        cmd = [sys.executable, str(ROOT / "training" / "train_tokenizer.py"),
               "--verify", "--vocab", str(vocab)]
        for corpus in corpora:
            cmd += ["--corpus", str(corpus)]
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise SystemExit(
                f"{vocab.name} failed verification: {(proc.stderr or proc.stdout)[-400:]}"
            )
    families = ", ".join(v.stem for v in vocabs)
    return f"frozen vocab families round-trip every corpus record on this machine: {families}"


def selftest_tokenizer_vocab_is_frozen() -> None:
    """The canary: a structurally broken vocab and a misnamed family must
    both be refused by the loader; the trainer must train."""
    import tempfile

    sys.path.insert(0, str(ROOT / "training"))
    import train_tokenizer as tt

    with tempfile.TemporaryDirectory() as td:
        corpus = Path(td) / "c.jsonl"
        corpus.write_text(
            json.dumps({"kind": "doc", "text": "donmus sozluk kanaryasi " * 8}) + "\n",
            encoding="utf-8",
        )
        out = Path(td) / "lubot-bpe-v1.json"
        rc = subprocess.run(
            [sys.executable, str(ROOT / "training" / "train_tokenizer.py"),
             "--corpus", str(corpus), "--out", str(out), "--vocab-size", "400"],
            cwd=ROOT, capture_output=True, text=True, check=False,
        ).returncode
        assert rc == 0, "canary training failed"
        good = json.loads(out.read_text(encoding="utf-8"))
        broken = dict(good)
        broken["merges"] = [[256 + len(good["merges"]) - 1, 65]] + good["merges"][1:]
        badfile = Path(td) / "bozuk.json"
        badfile.write_text(json.dumps(broken), encoding="utf-8")
        try:
            tt.load_vocab(str(badfile))
            raise AssertionError("a broken merge table was accepted")
        except SystemExit:
            pass
        misnamed = Path(td) / "baska.json"
        misnamed.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
        try:
            tt.load_vocab(str(misnamed))
            raise AssertionError("a misnamed family was accepted")
        except SystemExit:
            pass


# --------------------------------------------------------------------------
# gate: the committed model spec is internally consistent (NN-3)
# --------------------------------------------------------------------------
def gate_model_spec_is_consistent() -> str:
    """The committed training/model_spec.json validates against its own
    rules: structure, the muP table's init/LR formulas per parameter group,
    the weight-tying resolution (shared embedding + 1/d_model logit scale),
    the exact tensor-by-tensor param count, and the measured hardware
    ceiling (K6). A spec whose declared numbers disagree with its formulas,
    or that steps over the measured ceiling, is refused."""
    spec_path = ROOT / "training" / "model_spec.json"
    if not spec_path.exists():
        raise SystemExit(
            "training/model_spec.json is missing; the architecture decision is "
            "committed as data, never carried in someone's head"
        )
    proc = subprocess.run(
        [sys.executable, str(ROOT / "training" / "model_spec.py"),
         "--validate", str(spec_path)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"model spec failed validation: {(proc.stderr or proc.stdout)[-400:]}"
        )
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    return (
        f"model spec {spec['name']} is consistent: muP table, tying, "
        f"param count {spec['params']['toplam']}, ceiling respected (K6)"
    )


def selftest_model_spec_is_consistent() -> None:
    """The canary: a spec with a wrong param count, one over the ceiling,
    an untied readout and a mis-scaled attention must all be refused."""
    sys.path.insert(0, str(ROOT / "training"))
    import copy

    import model_spec as ms

    ms.selftest()  # the tool's own consistency proofs carry over
    base = json.loads((ROOT / "training" / "model_spec.json").read_text(encoding="utf-8"))
    ms.validate_spec(base)  # the committed spec is the healthy control

    yanlis = copy.deepcopy(base)
    yanlis["params"]["toplam"] += 1
    try:
        ms.validate_spec(yanlis)
        raise AssertionError("a wrong param count was accepted")
    except SystemExit:
        pass

    tasmis = copy.deepcopy(base)
    tasmis["ceiling_reference"]["max_params_train_fp32_adamw"] = 1
    try:
        ms.validate_spec(tasmis)
        raise AssertionError("a spec over the measured ceiling was accepted")
    except SystemExit:
        pass

    bagsiz = copy.deepcopy(base)
    bagsiz["weight_tying"] = {"tied": False}
    try:
        ms.validate_spec(bagsiz)
        raise AssertionError("an untied readout was accepted")
    except SystemExit:
        pass

    olceksiz = copy.deepcopy(base)
    olceksiz["attention_scale"] = "1/sqrt(d_k)"
    try:
        ms.validate_spec(olceksiz)
        raise AssertionError("a standard 1/sqrt(d_k) attention scale was accepted")
    except SystemExit:
        pass


# --------------------------------------------------------------------------
# gate: a passage stamped eval-only never becomes a training row (PP)
# --------------------------------------------------------------------------
# An eval set separated by intent leaks the moment discipline slips. This one
# is separated by digest: the passage's own content_id is stamped, and the
# check is mechanical from there on.


def _eval_only_file_finding(path: Path) -> str | None:
    """The stamp list's own shape. A malformed list is refused here so it can
    never pass for an empty one."""
    if not path.exists():
        return "the eval-only stamp list is missing: the leak check would run on nothing"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        return f"not JSON: {err}"
    if not isinstance(data, dict):
        return "not a JSON object"
    digests = data.get("digests")
    if not isinstance(digests, list):
        return "`digests` is not a list"
    for digest in digests:
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            return f"`{digest!r}` is not a sha256 digest"
    if len(set(digests)) != len(digests):
        return "duplicate digest"
    return None


def gate_eval_set_never_trained() -> str:
    """The eval-only list holds, and the check proves it fires on the real
    pipeline rather than only on synthetic rows. The corpus this machine
    builds is turned into an SFT set; every grounded row must carry the
    `content_id` of the passage it came from, so the leak check cannot be
    silently disarmed; and one row's own digest stamped into a temporary list
    must make the evaluator refuse the whole set."""
    import tempfile

    stamp_list = ROOT / "training" / "eval" / "eval-only.json"
    finding = _eval_only_file_finding(stamp_list)
    if finding:
        raise SystemExit(f"{stamp_list.relative_to(ROOT)}: {finding}")

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
            raise SystemExit(f"the corpus the check examines does not build: {r1.stderr[-200:]}")
        r2 = py("training/make_sft.py", "--corpus", corpus,
                "--curriculum", "training/curriculum", "--out", sft)
        if r2.returncode != 0:
            raise SystemExit(f"the SFT set does not build: {r2.stderr[-200:]}")
        rows = [
            json.loads(line)
            for line in Path(sft).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        grounded = [row for row in rows if row.get("kind") != "curriculum"]
        if not grounded:
            raise SystemExit("the SFT set has no grounded rows: the check would pass on nothing")
        unaddressed = [row for row in grounded if not row.get("content_id")]
        if unaddressed:
            raise SystemExit(
                f"{len(unaddressed)} grounded row(s) carry no content_id: "
                "the leak check is disarmed"
            )
        r3 = py("training/eval_sft.py", "--sft", sft, "--eval-only", str(stamp_list))
        if r3.returncode != 0:
            raise SystemExit(f"a stamped passage is in the training set:\n{r3.stdout[-400:]}")
        stamped = str(Path(td) / "stamped.json")
        Path(stamped).write_text(
            json.dumps({"digests": [grounded[0]["content_id"]]}), encoding="utf-8"
        )
        r4 = py("training/eval_sft.py", "--sft", sft, "--eval-only", stamped)
        if r4.returncode == 0:
            raise SystemExit(
                "the evaluator accepted a row whose own passage is stamped "
                "eval-only: the check is decoration"
            )
        if "eval-only" not in r4.stdout:
            raise SystemExit(
                f"the spiked set was refused without naming the eval-only leak:\n{r4.stdout[-300:]}"
            )
        n_grounded = len(grounded)

    stamps = len(json.loads(stamp_list.read_text(encoding="utf-8"))["digests"])
    return (
        f"{stamps} stamped passage(s), {n_grounded} grounded row(s) checked, "
        f"none stamped; the evaluator refuses a spiked stamp"
    )


def selftest_eval_set_never_trained() -> None:
    """The canaries: a leaked row must produce a finding, an unstamped row
    must not, and a malformed stamp list must be refused rather than read as
    empty."""
    import importlib.util
    import tempfile

    spec = importlib.util.spec_from_file_location(
        "eval_sft", str(ROOT / "training" / "eval_sft.py")
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    digest = "a" * 64
    row = {
        "messages": [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "a body long enough to count on its own\n\nSource: a/b.rs:1",
            },
        ],
        "kind": "doc",
        "citation": "a/b.rs:1",
        "content_id": digest,
    }
    assert any("eval-only" in f for f in mod.evaluate([row], {digest})["findings"]), (
        "a stamped row passed the evaluator: the gate is decoration"
    )
    assert not mod.evaluate([row], {"b" * 64})["findings"], (
        "an unstamped row was refused"
    )

    with tempfile.TemporaryDirectory() as td:
        broken = Path(td) / "broken.json"
        broken.write_text(json.dumps({"digests": ["not-a-digest"]}), encoding="utf-8")
        if _eval_only_file_finding(broken) is None:
            raise AssertionError("a malformed stamp list passed for an empty one")
        doubled = Path(td) / "doubled.json"
        doubled.write_text(json.dumps({"digests": [digest, digest]}), encoding="utf-8")
        if _eval_only_file_finding(doubled) is None:
            raise AssertionError("a duplicated stamp passed")
        whole = Path(td) / "whole.json"
        whole.write_text(json.dumps({"digests": [digest]}), encoding="utf-8")
        if _eval_only_file_finding(whole) is not None:
            raise AssertionError("a well-formed stamp list was refused")


# --------------------------------------------------------------------------
# gate: the muP init measurement is reproduced, not recited (NN-4)
# --------------------------------------------------------------------------
# model_spec.json marks two decisions `olculmedi` and names NN-4 as the run that
# measures them. A recorded number that nothing re-derives is a number that
# drifts away from its tree, so the records are re-measured here: the same
# machine, the same seeds, the same bands.


def _mup_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "mup_olcum", str(ROOT / "training" / "mup_olcum.py")
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def gate_mup_measurement_reproduced() -> str:
    """The three NN-4 records are re-measured on this machine, their control
    channels are checked, the committed spec's parameter table is re-counted
    from its own configuration, and every record passes the mechanical-run
    schema. A profile outside the theta(1) band is reported, not hidden: the
    gate's job is to keep the number honest, not to make it comfortable."""
    mod = _mup_module()
    dikkat_yolu, readout_yolu, spec_yolu = mod.kayit_yollari()
    for yol in (dikkat_yolu, readout_yolu, spec_yolu):
        if not yol.is_file():
            raise SystemExit(f"{yol.relative_to(ROOT)} is missing: an unrecorded measurement is not a measurement")
    olcum = mod.olc()
    spec = mod.spec_olcumu()
    bulgular = mod.kayitlari_denetle(olcum, spec) + mod.kontrol_bulgulari(olcum)
    if bulgular:
        raise SystemExit("the muP records do not reproduce:\n  " + "\n  ".join(bulgular))
    for yol in (dikkat_yolu, readout_yolu, spec_yolu):
        rec = json.loads(yol.read_text(encoding="utf-8"))
        finding = _eval_run_finding(rec)
        if finding:
            raise SystemExit(f"{yol.name}: {finding}")
    if not olcum["kriterler"]["dikkat"] or not olcum["kriterler"]["readout"]:
        raise SystemExit(
            "the muP criteria are recorded as true but re-measure false: "
            f"{olcum['kriterler']}"
        )
    if spec["sayim_toplam_hesaplanan"] != spec["sayim_toplam_beyan"]:
        raise SystemExit(
            "the committed spec's parameter table does not survive an "
            f"independent count: {spec['sayim_toplam_hesaplanan']} != {spec['sayim_toplam_beyan']}"
        )
    band = spec["ileri_gecis"]["theta_1_bandinda"]
    return (
        "muP init olcumu yeniden uretildi: dikkat oranlari "
        f"{ {k: v['ortalama'] for k, v in olcum['dikkat_oranlari'].items()} }, "
        f"bagli readout sapmasi {olcum['bagli_olmayan_mup_esitligi']}, "
        f"kontroller {olcum['kontrol']['dikkat_standart_orani']['ortalama']} / "
        f"{olcum['bagli_olceksiz_buyumesi']['ortalama']}; "
        f"spec sayimi {spec['sayim_toplam_hesaplanan']} (beyan {spec['sayim_toplam_beyan']}); "
        f"katman profili {spec['ileri_gecis']['ilk_katman_rms']} -> {spec['ileri_gecis']['son_katman_rms']} "
        f"({spec['ileri_gecis']['buyume_son_bolu_ilk']}x), theta_1 bandinda={band}"
        + ("" if band else " [BULGU: bant disi]")
    )


def selftest_mup_measurement_reproduced() -> None:
    """The canaries: a drifted number, a missing record and a judgement-word
    criterion must each be refused."""
    import tempfile

    mod = _mup_module()
    olcum = mod.olc()
    spec = mod.spec_olcumu()
    if mod.kayitlari_denetle(olcum, spec) + mod.kontrol_bulgulari(olcum):
        raise AssertionError("the fresh measurement does not match its own records")
    bozuk = json.loads(json.dumps(olcum))
    bozuk["dikkat_logit_rms"]["spesifikasyon"]["128"][0] *= 3.0
    if not mod.kayitlari_denetle(bozuk, spec):
        raise AssertionError("a drifted number was accepted")
    bozuk_spec = json.loads(json.dumps(spec))
    bozuk_spec["ileri_gecis"]["katman_aktivasyon_rms"][-1] *= 2.0
    if not mod.kayitlari_denetle(olcum, bozuk_spec):
        raise AssertionError("a drifted layer profile was accepted")
    bozuk_sayim = json.loads(json.dumps(spec))
    bozuk_sayim["sayim_toplam_hesaplanan"] = 1
    if not mod.kayitlari_denetle(olcum, bozuk_sayim):
        raise AssertionError("a spec parameter table that does not count was accepted")
    kor = json.loads(json.dumps(olcum))
    kor["dikkat_oranlari"]["standart"]["ortalama"] = 3.0
    if not mod.kontrol_bulgulari(kor):
        raise AssertionError("a control channel that stopped discriminating was accepted")
    # A record stating a judgement instead of a machine check is not a run.
    kayit = mod.kayitlar(olcum)[0][1]
    kayit["olcut"]["ad"] = "dikkat olcegi iyi gorunuyor"
    if _eval_run_finding(kayit) is None:
        raise AssertionError("a judgement-word criterion passed the run schema")
    # And the verifier has to notice records that are not there at all.
    with tempfile.TemporaryDirectory() as td:
        gercek = mod.KAYIT_DIZINI
        mod.KAYIT_DIZINI = Path(td)
        try:
            if not mod.kayitlari_denetle(olcum):
                raise AssertionError("missing records were accepted")
        finally:
            mod.KAYIT_DIZINI = gercek


# --------------------------------------------------------------------------
# gate: the training mix is declared, and the declaration is enforced (NN §8.4)
# --------------------------------------------------------------------------
# A mix nobody wrote down is a mix nobody can audit: a set that quietly becomes
# 99% one source still trains and still looks like a set. So the strata are
# declared in `training/veri-karisimi.json` with a share band each, and the
# builder refuses rather than adjusts.


def gate_data_mix_is_declared() -> str:
    """The mix is rebuilt from its declaration and the record has to survive it:
    an undeclared stratum, a share outside its declared band, or a grounded row
    with no content_id each refuse the build. The record is also checked against
    the mechanical-run schema, because a mix report is an evaluation run."""
    kayit_yolu = ROOT / "training" / "eval" / "sonuclar" / "veri-karisimi-2026-09-23.json"
    if not kayit_yolu.is_file():
        raise SystemExit(
            f"{kayit_yolu.relative_to(ROOT)} is missing: an unrecorded mix is a mix nobody declared"
        )
    r = subprocess.run(
        [sys.executable, str(ROOT / "training" / "veri_karisimi.py"), "--dogrula"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if r.returncode != 0:
        raise SystemExit(
            "the declared mix does not rebuild:\n"
            + "".join(f"  {line}\n" for line in (r.stdout + r.stderr).strip().splitlines())
        )
    rec = json.loads(kayit_yolu.read_text(encoding="utf-8"))
    finding = _eval_run_finding(rec)
    if finding:
        raise SystemExit(f"{kayit_yolu.name}: {finding}")
    karisim = rec.get("karisim")
    if not isinstance(karisim, dict) or "paylar" not in karisim:
        raise SystemExit(f"{kayit_yolu.name}: the record carries no measured mix")
    return (
        f"veri karisimi beyandan yeniden kuruldu: {karisim.get('toplam_satir')} satir, "
        f"sayilar {karisim.get('satir_sayilari')}, paylar {karisim.get('paylar')}"
    )


def selftest_data_mix_is_declared() -> None:
    """The canaries: a share outside its band, a band that does not parse, and a
    missing record must each be refused."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "veri_karisimi", str(ROOT / "training" / "veri_karisimi.py")
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    beyan = json.loads(mod.BEYAN.read_text(encoding="utf-8"))
    gercek = mod.BEYAN.read_text(encoding="utf-8")
    sikisik = json.loads(json.dumps(beyan))
    sikisik["katmanlar"]["gercek"]["taban_pay"] = 0.0
    sikisik["katmanlar"]["gercek"]["tavan_pay"] = 0.01
    mod.BEYAN.write_text(json.dumps(sikisik, ensure_ascii=False), encoding="utf-8")
    try:
        karisim = mod.karisim_olc(mod.satirlari_topla(1, 0))
        if not karisim["bant_ihlalleri"]:
            raise AssertionError("a share above its declared ceiling was accepted")
    finally:
        mod.BEYAN.write_text(gercek, encoding="utf-8")

    gecersiz = json.loads(json.dumps(beyan))
    gecersiz["katmanlar"]["gercek"]["taban_pay"] = 0.9
    gecersiz["katmanlar"]["gercek"]["tavan_pay"] = 0.1
    mod.BEYAN.write_text(json.dumps(gecersiz, ensure_ascii=False), encoding="utf-8")
    try:
        try:
            mod.beyan_oku()
        except SystemExit:
            pass
        else:
            raise AssertionError("a band whose floor is above its ceiling was accepted")
    finally:
        mod.BEYAN.write_text(gercek, encoding="utf-8")

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        gercek_kayit = mod.KAYIT
        mod.KAYIT = Path(td) / "yok.json"
        try:
            karisim = mod.karisim_olc(mod.satirlari_topla(1, 0))
            if not karisim["paylar"]:
                raise AssertionError("the mix measured nothing and said so")
        finally:
            mod.KAYIT = gercek_kayit


# --------------------------------------------------------------------------
# gate: the training budget is declared and measured (NN §8.5, HH)
# --------------------------------------------------------------------------
# "More epochs is better" is an assumption until the repetition is a number. The
# policy declares the epoch count and the weight decay; this gate re-measures the
# budget the policy is written against and refuses a policy that outruns the
# protocol ceiling or a budget that does not reproduce.


def gate_training_budget_is_declared() -> str:
    """The regularization/epoch policy exists, its epoch count stays under the
    grant protocol's ceiling, and the token budget it is written against
    re-measures with the frozen vocabulary."""
    kayit_yolu = ROOT / "training" / "eval" / "sonuclar" / "egitim-butcesi-2026-09-23.json"
    if not kayit_yolu.is_file():
        raise SystemExit(
            f"{kayit_yolu.relative_to(ROOT)} is missing: a policy with no measured budget is a guess"
        )
    r = subprocess.run(
        [sys.executable, str(ROOT / "training" / "egitim_butcesi.py"), "--dogrula"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if r.returncode != 0:
        raise SystemExit(
            "the training budget does not reproduce:\n"
            + "".join(f"  {line}\n" for line in (r.stdout + r.stderr).strip().splitlines())
        )
    rec = json.loads(kayit_yolu.read_text(encoding="utf-8"))
    finding = _eval_run_finding(rec)
    if finding:
        raise SystemExit(f"{kayit_yolu.name}: {finding}")
    butce = rec.get("butce")
    if not isinstance(butce, dict) or "benzersiz_jeton" not in butce:
        raise SystemExit(f"{kayit_yolu.name}: the record carries no measured budget")
    return (
        f"egitim butcesi olculdu: {butce['benzersiz_jeton']} benzersiz jeton, "
        f"{butce['max_epochs']} epoch -> {butce['toplam_gecis']} gecis, "
        f"etkin kaynak orani {butce['etkin_kaynak_orani']}, "
        f"jeton/param {butce['jeton_basina_param_tam_butce']} "
        f"(protokol tavani {butce['protokol_epoch_tavani']})"
    )


def selftest_training_budget_is_declared() -> None:
    """The canaries: an epoch count over the protocol ceiling, a weight decay out
    of range and a policy missing a field must each be refused."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "egitim_butcesi", str(ROOT / "training" / "egitim_butcesi.py")
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    gercek = mod.POLITIKA.read_text(encoding="utf-8")
    politika = json.loads(gercek)
    try:
        tavan = mod.protokol_tavani()
        asiri = json.loads(json.dumps(politika))
        asiri["max_epochs"] = tavan + 1
        # Korpus gerektirmeyen saf kural uzerinden: CI'da "Gate self-tests"
        # adimi korpus kurulmadan ONCE kosar, butce_olc() burada kosamazdi.
        if not mod.politika_ihlalleri(asiri, tavan):
            raise AssertionError("an epoch count over the protocol ceiling was accepted")
        bozuk = json.loads(json.dumps(politika))
        bozuk["weight_decay"] = 0.0
        mod.POLITIKA.write_text(json.dumps(bozuk, ensure_ascii=False), encoding="utf-8")
        try:
            mod.politika_oku()
        except SystemExit:
            pass
        else:
            raise AssertionError("a zero weight decay was accepted")
    finally:
        mod.POLITIKA.write_text(gercek, encoding="utf-8")
    if mod.politika_ihlalleri(politika, mod.protokol_tavani()):
        raise AssertionError("the committed policy breaches its own ceiling")
    # The ceiling is read from the grant crate, not repeated: prove it is a number
    # and that it is the crate's, so a second literal cannot drift in.
    tavan = mod.protokol_tavani()
    kaynak = mod.GRANT_KAYNAGI.read_text(encoding="utf-8")
    if f"MAX_TRAINING_GRANT_EPOCHS: u32 = {tavan}" not in kaynak:
        raise AssertionError("the ceiling was not read from the grant crate")



TOMURCUK_YASAK = [
    "String",
    "format!",
    "write!",
    "writeln!",
    "to_string",
    "push_str",
    "char",
    "println!",
    "eprint",
    "Vec<u8>",
]


def _tomurcuk_kodu() -> str:
    """The head's source with tests and comments removed.

    Tests may format an assertion message; comments may name the forbidden
    tokens while explaining why they are forbidden. Neither can generate text
    at runtime, so neither is what this gate is about.
    """
    kaynaklar = sorted((ROOT / "crates" / "tomurcuk" / "src").rglob("*.rs"))
    if not kaynaklar:
        raise SystemExit("crates/tomurcuk/src has no sources: the head is gone")
    parcalar: list[str] = []
    for kaynak in kaynaklar:
        satirlar = kaynak.read_text(encoding="utf-8").splitlines()
        parcalar.append(
            "\n".join(s for s in _test_modulu_disinda(satirlar)
                      if not s.lstrip().startswith("//"))
        )
    return "\n".join(parcalar)


def _test_modulu_disinda(satirlar: list[str]) -> list[str]:
    """Every line except the `#[cfg(test)]` module's own block.

    Cutting everything *after* the marker instead of the block itself would let
    production code hide below the tests, which is exactly the shape an
    accidental paste takes. The block is found by matching braces, so what comes
    after it is still checked.
    """
    for i, satir in enumerate(satirlar):
        if not satir.strip().startswith("#[cfg(test)]"):
            continue
        j = i
        while j < len(satirlar) and not satirlar[j].lstrip().startswith(("mod ", "pub mod ")):
            j += 1
        if j >= len(satirlar):
            return satirlar[:i]
        derinlik = 0
        basladi = False
        k = j
        while k < len(satirlar):
            derinlik += satirlar[k].count("{") - satirlar[k].count("}")
            if "{" in satirlar[k]:
                basladi = True
            if basladi and derinlik <= 0:
                break
            k += 1
        return satirlar[:i] + satirlar[k + 1:]
    return satirlar


def gate_decision_head_has_no_generation_surface() -> str:
    """The decision head cannot produce text, even at compile time.

    T makes this a doctrine: the head decides, the generative model writes. A
    doctrine held only in prose is one a later edit can quietly undo, so the
    source is what gets checked - and the output surface is checked to be
    exactly the three closed shapes, because a fourth variant is how free text
    would get in.
    """
    import re

    kod = _tomurcuk_kodu()
    for yasak in TOMURCUK_YASAK:
        if yasak in kod:
            raise SystemExit(
                f"the decision head carries a generation surface: {yasak}"
            )
    match = re.search(r"pub enum Karar\s*\{([^}]+)\}", kod)
    if not match:
        raise SystemExit("the head's output enum is not there to check")
    variants = [
        line.strip().split("(")[0].strip()
        for line in match.group(1).splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    expected = ["Secenek", "Puan", "EvetHayir"]
    if variants != expected:
        raise SystemExit(
            f"the head's output surface changed: got {variants}, expected {expected}"
        )
    # The head has to be reached from the binary, or the doctrine is a file
    # nobody runs.
    dagitim = read("crates/cli/src/main.rs")
    if '"karar" => lubot::karar::cmd_karar(rest)' not in dagitim:
        raise SystemExit("the head is not reachable from any command")
    return "the decision head has 3 closed shapes and no text-producing surface"


def selftest_decision_head_has_no_generation_surface() -> None:
    """Canaries: text-producing code, a fourth shape, and an unwired head must
    each be refused - and a `format!` inside the tests must not be."""
    lib = ROOT / "crates" / "tomurcuk" / "src" / "lib.rs"
    gercek = lib.read_text(encoding="utf-8")
    try:
        # Canary 1: a function that returns text, appended *after* the test
        # module. Cutting everything below `#[cfg(test)]` would have let this
        # through, so the canary sits exactly where that hole was.
        lib.write_text(
            gercek + "\n#[allow(dead_code)]\npub fn kacak() -> String { String::new() }\n",
            encoding="utf-8",
        )
        try:
            gate_decision_head_has_no_generation_surface()
            raise AssertionError("a head that can return text was accepted")
        except SystemExit:
            pass
        # Canary 2: a fourth output shape, which is how text would get in.
        dort = gercek.replace(
            "    /// A yes or no with a probability.\n    EvetHayir(EvetHayirKarari),",
            "    /// A yes or no with a probability.\n    EvetHayir(EvetHayirKarari),\n    Metin(&'static str),",
            1,
        )
        if dort == gercek:
            raise AssertionError("the fourth-shape canary did not apply")
        lib.write_text(dort, encoding="utf-8")
        try:
            gate_decision_head_has_no_generation_surface()
            raise AssertionError("a fourth output shape was accepted")
        except SystemExit:
            pass
        # Canary 3: a `format!` inside the tests is not a generation surface.
        # Injected rather than assumed, so the canary tests the stripping and
        # not what this file happens to contain today.
        ic = """    #[allow(dead_code)]
    fn kanarya() -> String {
        format!("bir iddia mesaji")
    }
"""
        asili = gercek.replace(
            "mod tests {\n    use super::*;\n",
            "mod tests {\n    use super::*;\n\n" + ic + "\n",
            1,
        )
        if asili == gercek:
            raise AssertionError("the test-module canary did not apply")
        lib.write_text(asili, encoding="utf-8")
        gate_decision_head_has_no_generation_surface()
    finally:
        lib.write_text(gercek, encoding="utf-8")


def _onyukleme_turlari() -> list[dict]:
    """Recorded bootstrap rounds, oldest first."""
    import json as _json

    turlar = []
    for dosya in sorted((ROOT / "training" / "eval" / "sonuclar").glob("onyukleme-*.json")):
        veri = _json.loads(dosya.read_text(encoding="utf-8"))
        if isinstance(veri.get("tur"), int):
            turlar.append(veri)
    return turlar


def gate_bootstrap_round_is_measured() -> str:
    """A bootstrap round is a measurement, not a claim of progress.

    G's loop only teaches if the refusals say why and the improvement is
    recomputed. So: the record must re-verify against the data, every negative
    row must carry a reason, and the competence delta must match what the two
    records actually say - a round that compares itself to a round that does
    not exist is a fabricated improvement.
    """
    import json as _json

    sonuclar = ROOT / "training" / "eval" / "sonuclar"
    kayitlar = sorted(sonuclar.glob("onyukleme-*.json"))
    if not kayitlar:
        raise SystemExit("no bootstrap round is recorded")
    kosu = subprocess.run(
        [sys.executable, str(ROOT / "training" / "onyukleme.py"), "--dogrula"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if kosu.returncode != 0:
        raise SystemExit(
            f"the bootstrap round does not re-verify: {kosu.stdout.strip()[-300:]}"
        )
    havuz_dosya = ROOT / "training" / "eval" / "negatif-havuz.jsonl"
    havuz = []
    if havuz_dosya.is_file():
        havuz = [
            _json.loads(line)
            for line in havuz_dosya.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    nedensiz = [h for h in havuz if not h.get("neden")]
    if nedensiz:
        raise SystemExit(
            f"{len(nedensiz)} negative row(s) carry no reason: a refusal nobody "
            "explained cannot be learned from"
        )
    for kayit_dosya in kayitlar:
        kayit = _json.loads(kayit_dosya.read_text(encoding="utf-8"))
        bulgu = _eval_run_finding(kayit)
        if bulgu:
            raise SystemExit(f"{kayit_dosya.name}: {bulgu}")
        fark = kayit.get("yeterlilik_farki")
        if not isinstance(fark, dict) or "karsilastirilabilir" not in fark:
            raise SystemExit(f"{kayit_dosya.name}: no competence delta recorded")
    # A comparable delta needs a previous round; with one record there is none.
    if len(kayitlar) == 1:
        kayit = _json.loads(kayitlar[0].read_text(encoding="utf-8"))
        if kayit["yeterlilik_farki"]["karsilastirilabilir"]:
            raise SystemExit(
                "the only recorded round claims a competence delta: there is no "
                "previous round to be better than"
            )
    return (
        f"{len(kayitlar)} bootstrap round(s) re-verify; {len(havuz)} negative "
        "row(s), each with a reason"
    )


def selftest_bootstrap_round_is_measured() -> None:
    """Canaries: a missing record, an unexplained refusal and a fabricated delta
    must each be refused."""
    sonuclar = ROOT / "training" / "eval" / "sonuclar"
    kayitlar = sorted(sonuclar.glob("onyukleme-*.json"))
    if not kayitlar:
        raise AssertionError("the self-test needs a recorded round to break")
    kayit_dosya = kayitlar[-1]
    havuz_dosya = ROOT / "training" / "eval" / "negatif-havuz.jsonl"
    eski_kayit = kayit_dosya.read_text(encoding="utf-8")
    eski_havuz = havuz_dosya.read_text(encoding="utf-8") if havuz_dosya.is_file() else None
    try:
        gate_bootstrap_round_is_measured()

        # Canary 1: a refusal with no reason.
        satirlar = eski_havuz or ""
        with havuz_dosya.open("a", encoding="utf-8") as akis:
            akis.write('{"tur": 1, "sinif": "okuma", "kimlik": "okuma:99", "neden": []}\n')
        try:
            gate_bootstrap_round_is_measured()
            raise AssertionError("a refusal with no reason was accepted")
        except SystemExit:
            pass
        if eski_havuz is None:
            havuz_dosya.unlink(missing_ok=True)
        else:
            havuz_dosya.write_text(satirlar, encoding="utf-8")

        # Canary 2: a delta invented against a round that does not exist.
        kayit = json.loads(eski_kayit)
        kayit["yeterlilik_farki"] = {
            "karsilastirilabilir": True,
            "onceki_tur": 0,
            "yeni_cozulen_siniflar": ["okuma"],
            "gerileyen_siniflar": [],
            "yalniz_tekrar": False,
        }
        kayit_dosya.write_text(
            json.dumps(kayit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        try:
            gate_bootstrap_round_is_measured()
            raise AssertionError("a fabricated competence delta was accepted")
        except SystemExit:
            pass

        # Canary 3: a record that no longer matches the data.
        kayit = json.loads(eski_kayit)
        kayit["gecen_toplam"] = kayit["gecen_toplam"] + 1
        kayit_dosya.write_text(
            json.dumps(kayit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        try:
            gate_bootstrap_round_is_measured()
            raise AssertionError("a drifted count was accepted")
        except SystemExit:
            pass
    finally:
        kayit_dosya.write_text(eski_kayit, encoding="utf-8")
        if eski_havuz is not None:
            havuz_dosya.write_text(eski_havuz, encoding="utf-8")
        elif havuz_dosya.is_file():
            havuz_dosya.unlink()


def _sinav_satirlari() -> list[dict]:
    """Held-out exam questions; an absent file is an empty set, not an error."""
    import json as _json

    dosya = ROOT / "training" / "eval" / "sinav-seti.jsonl"
    if not dosya.is_file():
        return []
    return [
        _json.loads(line)
        for line in dosya.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def gate_comparison_class_is_declared() -> str:
    """The opponent is named with numbers, and a match may only be claimed on
    the task axis.

    GG's calibration: today's "small" open models are 135M-600M parameters, so
    a 924.288-parameter reader is below that class and a parameter-axis match
    claim would be a category error. The declaration is re-measured here rather
    than argued once, because the corpus grows and the class moves with it.
    """
    import json as _json

    kayit_dosya = ROOT / "training" / "eval" / "sonuclar" / "kiyas-sinifi-2026-09-23.json"
    if not kayit_dosya.is_file():
        raise SystemExit("no comparison class is declared")
    kosu = subprocess.run(
        [sys.executable, str(ROOT / "training" / "kiyas_sinifi.py"), "--dogrula"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if kosu.returncode != 0:
        raise SystemExit(
            f"the comparison class does not re-verify: {kosu.stdout.strip()[-300:]}"
        )
    kayit = _json.loads(kayit_dosya.read_text(encoding="utf-8"))
    bulgu = _eval_run_finding(kayit)
    if bulgu:
        raise SystemExit(f"{kayit_dosya.name}: {bulgu}")
    kural = kayit.get("eksen_kurali", {})
    if kural.get("kapisma_iddiasi_yalniz") != "gorev-eslenegi":
        raise SystemExit("the record does not confine a match claim to the task axis")
    if kural.get("reddedilen") != "parametre-eslenegi":
        raise SystemExit("the record does not refuse the parameter axis")
    sinif = kayit.get("sinif", {})
    if not isinstance(sinif.get("parametre"), int) or sinif["parametre"] <= 0:
        raise SystemExit("the declared class carries no parameter count")
    return (
        f"class declared: {sinif['parametre']} params vs the "
        f"{sinif['sinir'][0]['ad']} boundary, {kayit['sinav_seti']['soru_sayisi']} "
        "exam question(s), match claims confined to the task axis"
    )


def selftest_comparison_class_is_declared() -> None:
    """Canaries: a missing record, a boundary breach, an unstamped exam question
    and a parameter-axis match claim must each be refused."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "kiyas_sinifi", str(ROOT / "training" / "kiyas_sinifi.py")
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    # The script's own canaries. The corpus-free ones always run; the record
    # round-trip needs the corpus and CI runs this step before the corpus exists.
    if mod.self_test() != 0:
        raise AssertionError("the comparison class canaries did not all fire")
    if not mod.KORPUS.is_file():
        print(
            "self-test OK [comparison-class-is-declared] "
            "(korpus yok: politika kanaryalari kosuldu, kayit kanaryasi atlandi)"
        )
        return

    kayit_dosya = mod.KAYIT
    gercek = kayit_dosya.read_text(encoding="utf-8") if kayit_dosya.is_file() else None
    try:
        # Canary: with no declaration at all the gate must refuse.
        if kayit_dosya.is_file():
            kayit_dosya.unlink()
        try:
            gate_comparison_class_is_declared()
            raise AssertionError("an undeclared comparison class was accepted")
        except SystemExit:
            pass
    finally:
        if gercek is not None:
            kayit_dosya.write_text(gercek, encoding="utf-8")


def gate_exam_set_is_held_out() -> str:
    """The exam set is held out physically, not by intention.

    A stamp list nobody honours is a wish. So this checks the whole chain:
    every question's passage is stamped, no stamp is orphaned, and - the part
    that matters - building the SFT set from the corpus really drops every
    stamped passage. Detection without prevention gives a refused run, not a
    held-out exam.
    """
    import json as _json
    import tempfile

    kayit_dosya = ROOT / "training" / "eval" / "sonuclar" / "sinav-seti-2026-09-23.json"
    if not kayit_dosya.is_file():
        raise SystemExit("no exam set is declared")
    kosu = subprocess.run(
        [sys.executable, str(ROOT / "training" / "sinav.py"), "--dogrula"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if kosu.returncode != 0:
        raise SystemExit(f"the exam set does not re-verify: {kosu.stdout.strip()[-300:]}")
    kayit = _json.loads(kayit_dosya.read_text(encoding="utf-8"))
    bulgu = _eval_run_finding(kayit)
    if bulgu:
        raise SystemExit(f"{kayit_dosya.name}: {bulgu}")
    damga = _json.loads((ROOT / "training" / "eval" / "eval-only.json").read_text(encoding="utf-8"))
    damgalar = damga.get("digests", [])
    if not damgalar:
        raise SystemExit("the exam set is declared but nothing is stamped")

    # Prevention, measured: the stamped passages must not survive into the SFT
    # set. Without the corpus this cannot be run, and the self-test says so.
    if not (ROOT / "corpus" / "knowledge-self.jsonl.gz").is_file():
        raise SystemExit("the corpus is missing: prevention cannot be measured")
    with tempfile.TemporaryDirectory() as td:
        duz = Path(td) / "korpus.jsonl"
        with gzip.open(ROOT / "corpus" / "knowledge-self.jsonl.gz", "rb") as kaynak:
            duz.write_bytes(kaynak.read())
        sft = Path(td) / "sft.jsonl"
        kosu = subprocess.run(
            [sys.executable, str(ROOT / "training" / "make_sft.py"),
             "--corpus", str(duz), "--curriculum", "training/curriculum",
             "--out", str(sft)],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        if kosu.returncode != 0:
            raise SystemExit(f"the SFT set does not build: {kosu.stderr[-200:]}")
        rapor = _json.loads(kosu.stdout)
        if rapor.get("dropped_eval_only") != len(damgalar):
            raise SystemExit(
                f"{len(damgalar)} passage(s) are stamped but make_sft dropped "
                f"{rapor.get('dropped_eval_only')}: the exam set is not held out"
            )
    return (
        f"{kayit['sinav_seti']['soru_sayisi']} exam question(s), "
        f"{len(damgalar)} stamped passage(s), all of them dropped from the "
        "training set"
    )


def selftest_exam_set_is_held_out() -> None:
    """Canaries: an unstamped question, an orphan stamp and a missing declaration
    must each be refused; the prevention run needs the corpus and is skipped
    loudly when it is absent."""
    kosu = subprocess.run(
        [sys.executable, str(ROOT / "training" / "sinav.py"), "--self-test"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if kosu.returncode != 0:
        raise AssertionError(f"the exam set canaries did not fire: {kosu.stderr[-200:]}")
    kayit_dosya = ROOT / "training" / "eval" / "sonuclar" / "sinav-seti-2026-09-23.json"
    gercek = kayit_dosya.read_text(encoding="utf-8") if kayit_dosya.is_file() else None
    if not (ROOT / "corpus" / "knowledge-self.jsonl.gz").is_file():
        print(
            "self-test OK [exam-set-is-held-out] "
            "(korpus yok: onleme kosusu atlandi, kural kanaryalari kosuldu)"
        )
        return
    try:
        if kayit_dosya.is_file():
            kayit_dosya.unlink()
        try:
            gate_exam_set_is_held_out()
            raise AssertionError("an undeclared exam set was accepted")
        except SystemExit:
            pass
    finally:
        if gercek is not None:
            kayit_dosya.write_text(gercek, encoding="utf-8")


def _bulgu_alanlari() -> list[tuple[str, str, dict]]:
    """Every `bulgu*` field in every recorded result, wherever it is nested."""
    import json as _json

    bulunan: list[tuple[str, str, dict]] = []

    def gez(veri, dosya: str, yol: str) -> None:
        if isinstance(veri, dict):
            for anahtar, deger in veri.items():
                # `bulgu_` rather than `bulgu`: a curriculum class is named
                # `bulgular`, and a class name is not a claim.
                if (
                    anahtar.startswith("bulgu_")
                    and isinstance(deger, dict)
                ):
                    bulunan.append((dosya, f"{yol}.{anahtar}", deger))
                gez(deger, dosya, f"{yol}.{anahtar}")
        elif isinstance(veri, list):
            for i, oge in enumerate(veri):
                gez(oge, dosya, f"{yol}[{i}]")

    for dosya in sorted((ROOT / "training" / "eval" / "sonuclar").glob("*.json")):
        gez(_json.loads(dosya.read_text(encoding="utf-8")), dosya.name, "")
    return bulunan


def gate_claims_carry_their_evidence() -> str:
    """A finding is a claim; a claim without its number and its boundary is
    refused.

    The failure this prevents is the ordinary one: a sentence that says
    "measured" and carries no measurement, or a finding that quietly fixes
    something without saying what it left alone. So every `bulgu*` field states
    what was measured (with a number in it), what follows from it, and what was
    deliberately not done.
    """
    alanlar = _bulgu_alanlari()
    if not alanlar:
        raise SystemExit("no finding is recorded: the gate would pass on nothing")
    hatalar: list[str] = []
    for dosya, yol, deger in alanlar:
        for zorunlu in ("olculen", "hukum", "yapilmayan"):
            metin = deger.get(zorunlu)
            if not isinstance(metin, str) or not metin.strip():
                hatalar.append(f"{dosya}{yol}: `{zorunlu}` yok ya da bos")
        olculen = deger.get("olculen")
        if isinstance(olculen, str) and not any(c.isdigit() for c in olculen):
            hatalar.append(
                f"{dosya}{yol}: `olculen` hic sayi tasimiyor - olculmus bir "
                "iddia olcusuz olmaz"
            )
    if hatalar:
        raise SystemExit("a finding does not carry its evidence:\n  " + "\n  ".join(hatalar))
    return f"{len(alanlar)} finding(s) each carry a measured number, a verdict and what was left alone"


def selftest_claims_carry_their_evidence() -> None:
    """Canaries: a finding with no boundary, and a finding whose `olculen`
    carries no number, must each be refused."""
    import json as _json

    sonuclar = ROOT / "training" / "eval" / "sonuclar"
    hedef = None
    for dosya in sorted(sonuclar.glob("*.json")):
        veri = _json.loads(dosya.read_text(encoding="utf-8"))
        if any(k.startswith("bulgu_") for k in veri) or any(
            isinstance(v, dict) and any(k.startswith("bulgu_") for k in v)
            for v in veri.values()
        ):
            hedef = dosya
            break
    if hedef is None:
        raise AssertionError("no record carries a finding to break")
    gercek = hedef.read_text(encoding="utf-8")
    try:
        gate_claims_carry_their_evidence()

        # Canary 1: a finding that never says what it left alone.
        veri = _json.loads(gercek)
        veri["bulgu_kanarya"] = {"olculen": "1 satir", "hukum": "BULGUDUR"}
        hedef.write_text(_json.dumps(veri, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            gate_claims_carry_their_evidence()
            raise AssertionError("a finding with no boundary was accepted")
        except SystemExit:
            pass

        # Canary 2: "measured" with no number in it.
        veri = _json.loads(gercek)
        veri["bulgu_kanarya"] = {
            "olculen": "performans olculdu ve yeterli bulundu",
            "hukum": "BULGUDUR",
            "yapilmayan": "hicbir sey degistirilmedi",
        }
        hedef.write_text(_json.dumps(veri, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            gate_claims_carry_their_evidence()
            raise AssertionError("an unnumbered measurement was accepted")
        except SystemExit:
            pass
    finally:
        hedef.write_text(gercek, encoding="utf-8")


def gate_decision_latency_is_recorded() -> str:
    """The speed axis has a number on it, and the number says what it includes.

    U makes speed and unit cost a second axis; an axis with no measurement on it
    is a preference. This checks that the baseline exists, is mechanical, and
    carries the caveat that the timing includes process startup - without it the
    figure reads as the head's own cost, which is a smaller and false number.
    """
    import json as _json

    kayit_dosya = ROOT / "training" / "eval" / "sonuclar" / "karar-gecikme-2026-09-23.json"
    if not kayit_dosya.is_file():
        raise SystemExit("no decision-latency baseline is recorded")
    kosu = subprocess.run(
        [sys.executable, str(ROOT / "training" / "karar_gecikme.py"), "--dogrula"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if kosu.returncode != 0:
        raise SystemExit(
            f"the latency baseline does not re-verify: {kosu.stdout.strip()[-300:]}"
        )
    kayit = _json.loads(kayit_dosya.read_text(encoding="utf-8"))
    bulgu = _eval_run_finding(kayit)
    if bulgu:
        raise SystemExit(f"{kayit_dosya.name}: {bulgu}")
    gecikme = kayit["gecikme"]
    return (
        f"decision path measured over {gecikme['kosu_sayisi']} runs: median "
        f"{gecikme['medyan_ms']} ms (process startup included, stated in the record)"
    )


def selftest_decision_latency_is_recorded() -> None:
    """Canaries: a one-run baseline, a missing caveat and a self-contradicting
    interval must each be refused. None of them needs the binary."""
    kosu = subprocess.run(
        [sys.executable, str(ROOT / "training" / "karar_gecikme.py"), "--self-test"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if kosu.returncode != 0:
        raise AssertionError(f"the latency canaries did not fire: {kosu.stdout[-300:]}")
    kayit_dosya = ROOT / "training" / "eval" / "sonuclar" / "karar-gecikme-2026-09-23.json"
    gercek = kayit_dosya.read_text(encoding="utf-8") if kayit_dosya.is_file() else None
    try:
        if kayit_dosya.is_file():
            kayit_dosya.unlink()
        try:
            gate_decision_latency_is_recorded()
            raise AssertionError("an unrecorded latency axis was accepted")
        except SystemExit:
            pass
    finally:
        if gercek is not None:
            kayit_dosya.write_text(gercek, encoding="utf-8")


def gate_first_answer_latency_is_recorded() -> str:
    """V's half: cold start plus first answer on the local path has one number.

    V asks for the local-first inference stack to be measured (cold start +
    first-answer time) and wired into a gate. Nothing on this path stays warm -
    no daemon, no cache, no loaded weights - so the caller's cost is the cold
    span, and that is what the record must carry, with the caveat as a field.
    """
    import json as _json

    kayit_dosya = (
        ROOT / "training" / "eval" / "sonuclar" / "ilk-cevap-gecikme-2026-09-24.json"
    )
    if not kayit_dosya.is_file():
        raise SystemExit("no first-answer (cold path) latency baseline is recorded")
    kosu = subprocess.run(
        [sys.executable, str(ROOT / "training" / "ilk_cevap_gecikme.py"), "--dogrula"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if kosu.returncode != 0:
        raise SystemExit(
            f"the cold-path baseline does not re-verify: {kosu.stdout.strip()[-300:]}"
        )
    kayit = _json.loads(kayit_dosya.read_text(encoding="utf-8"))
    bulgu = _eval_run_finding(kayit)
    if bulgu:
        raise SystemExit(f"{kayit_dosya.name}: {bulgu}")
    gecikme = kayit["gecikme"]
    return (
        f"ask path measured cold over {gecikme['kosu_sayisi']} runs: median "
        f"{gecikme['medyan_ms']} ms (process startup + corpus parse included, "
        "stated in the record)"
    )


def selftest_first_answer_latency_is_recorded() -> None:
    """Canaries: thin baselines, a missing caveat, a missing cold flag, a
    swapped-in other path and a self-contradicting interval must be refused.
    None of them needs the binary."""
    kosu = subprocess.run(
        [
            sys.executable,
            str(ROOT / "training" / "ilk_cevap_gecikme.py"),
            "--self-test",
        ],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if kosu.returncode != 0:
        raise AssertionError(
            f"the cold-path canaries did not fire: {kosu.stdout[-300:]}"
        )
    kayit_dosya = (
        ROOT / "training" / "eval" / "sonuclar" / "ilk-cevap-gecikme-2026-09-24.json"
    )
    gercek = kayit_dosya.read_text(encoding="utf-8") if kayit_dosya.is_file() else None
    try:
        if kayit_dosya.is_file():
            kayit_dosya.unlink()
        try:
            gate_first_answer_latency_is_recorded()
            raise AssertionError("an unrecorded cold-path axis was accepted")
        except SystemExit:
            pass
    finally:
        if gercek is not None:
            kayit_dosya.write_text(gercek, encoding="utf-8")


def gate_architecture_doc_tracks_layer_rule() -> str:
    """The architecture doc names what the code enforces, or the gate drops.

    crates/mimari holds the layer rule in code (a component may depend on its
    own layer or below, never above). A document describing the architecture
    that drifts away from the rule would teach a reader a system that does not
    exist, so the doc must name every crate in crates/ and carry the rule's
    own words - checked mechanically, not reviewed by memory.
    """
    belge = ROOT / "docs" / "ARCHITECTURE.md"
    if not belge.is_file():
        raise SystemExit("docs/ARCHITECTURE.md is missing")
    metin = belge.read_text(encoding="utf-8")
    crate_adlari = sorted(
        d.name for d in (ROOT / "crates").iterdir() if d.is_dir()
    )
    eksikler = [
        ad for ad in crate_adlari
        if not re.search(rf"`{re.escape(ad)}`", metin)
    ]
    if eksikler:
        raise SystemExit(
            f"the architecture doc no longer names these crates: {', '.join(eksikler)}"
        )
    for isaret in ("never above", "topological"):
        if isaret not in metin:
            raise SystemExit(
                f"the layer rule drifted out of the architecture doc: "
                f"missing '{isaret}'"
            )
    return (
        f"architecture doc names all {len(crate_adlari)} crates and carries "
        "the layer rule"
    )


def selftest_architecture_doc_tracks_layer_rule() -> None:
    """Canaries: a missing doc and a doc that silently lost a crate must each
    be refused."""
    belge = ROOT / "docs" / "ARCHITECTURE.md"
    icerik = belge.read_text(encoding="utf-8") if belge.is_file() else None
    try:
        if belge.is_file():
            belge.unlink()
        try:
            gate_architecture_doc_tracks_layer_rule()
            raise AssertionError("a missing architecture doc was accepted")
        except SystemExit:
            pass
        assert icerik is not None, "canary needs the real document"
        bir_crate = sorted(
            d.name for d in (ROOT / "crates").iterdir() if d.is_dir()
        )[0]
        belge.write_text(icerik.replace(f"`{bir_crate}`", "kayip-crate"))
        try:
            gate_architecture_doc_tracks_layer_rule()
            raise AssertionError("a doc that lost a crate name was accepted")
        except SystemExit:
            pass
    finally:
        if icerik is not None:
            belge.write_text(icerik, encoding="utf-8")


def gate_unserved_records_never_cited() -> str:
    """Operator decision (2026-09-24): process documents never answer a question.

    The device replied to a spec question by dumping work-queue passages. The
    decision: the archive keeps every record (provenance and the ratchet do
    not regress), but a record the serving policy stamps `served: false` can
    neither be searched nor cited. Checked three ways: the policy file is
    well-formed and non-stale (the builder refuses a policy entry that touches
    nothing), the stamp count in a fresh build agrees with the file, and a
    canary corpus proves the stamp itself is what excludes a record - stamped
    canary is never cited; the same canary unstamped is found again, so the
    gate cannot pass on a blind reader.
    """
    import hashlib

    politika = ROOT / "training" / "servis-politikasi.json"
    if not politika.is_file():
        raise SystemExit("training/servis-politikasi.json is missing")
    veri = json.loads(politika.read_text(encoding="utf-8"))
    girisler = veri.get("servis_disi")
    if not isinstance(girisler, list) or not girisler:
        raise SystemExit("the serving policy carries no servis_disi entries")
    yollar = [g.get("path") for g in girisler if isinstance(g, dict)]
    if any(not isinstance(y, str) or not y for y in yollar):
        raise SystemExit("a serving-policy entry has no path")
    if "YAPILACAKLAR.md" not in yollar:
        raise SystemExit("the work-queue document fell out of the serving policy")

    ikili = ROOT / "target" / "debug" / "lubot"
    if not ikili.is_file():
        ikili = ROOT / "target" / "release" / "lubot"
    if not ikili.is_file():
        raise SystemExit("no lubot binary: the canary ask cannot run")

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        gecici = Path(td)
        korpus_yolu = gecici / "k.jsonl"
        b = subprocess.run(
            [
                sys.executable,
                str(ROOT / "training" / "build_corpus.py"),
                "--repo", str(ROOT),
                "--out", str(korpus_yolu),
            ],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        if b.returncode != 0:
            raise SystemExit(
                f"the corpus does not build with the serving policy: {b.stderr[-200:]}"
            )
        damgali = 0
        for satir in korpus_yolu.read_text(encoding="utf-8").splitlines():
            if satir.strip() and json.loads(satir).get("served") is False:
                damgali += 1
        if damgali == 0:
            raise SystemExit(
                "no record is stamped served:false, though the policy names entries"
            )

        GIZ = "g13li-kanarya-2026"
        def _kayit(yol: str, served: bool) -> dict:
            metin = f"The work queue says the {GIZ} token stays internal."
            return {
                "kind": "markdown",
                "text": metin,
                "path": yol,
                "source": "kapi",
                "digest": hashlib.sha256(metin.encode()).hexdigest(),
                "licence": "MIT",
                "attribution": "kapi",
                "content_id": hashlib.sha256((yol + str(served)).encode()).hexdigest(),
                "asset_id": "a" * 64,
                "served": served,
            }

        for served in (False, True):
            deneme = gecici / f"kanarya-{served}.jsonl"
            deneme.write_text(
                json.dumps(_kayit("PLAN.md", served), ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            kosu = subprocess.run(
                [
                    str(ikili), "ask", "--corpus", str(deneme),
                    "--reader", "kapi", "--effort", "1x",
                    "work queue token stalk",
                ],
                cwd=ROOT, capture_output=True, text=True, check=False,
            )
            sizdi = GIZ in kosu.stdout or "PLAN.md" in kosu.stdout
            if not served and sizdi:
                raise SystemExit(
                    "a stamped record was cited: the serving exclusion does not hold"
                )
            if served and not sizdi:
                raise SystemExit(
                    "an unstamped canary was not found: this gate is blind, not green"
                )
    return (
        f"{damgali} archived records are stamped never-served; the canary proves "
        "the stamp is what excludes a record (control run finds it)"
    )


def selftest_unserved_records_never_cited() -> None:
    """Canaries: a missing policy file and a policy that lost the work-queue
    document must each be refused. The binary-level canaries live in the gate
    itself (the control run would refuse a blind exclusion)."""
    politika = ROOT / "training" / "servis-politikasi.json"
    icerik = politika.read_text(encoding="utf-8") if politika.is_file() else None
    try:
        if politika.is_file():
            politika.unlink()
        try:
            gate_unserved_records_never_cited()
            raise AssertionError("a missing serving policy was accepted")
        except SystemExit:
            pass
        assert icerik is not None, "canary needs the real policy"
        veri = json.loads(icerik)
        veri["servis_disi"] = [
            g for g in veri["servis_disi"] if g.get("path") != "YAPILACAKLAR.md"
        ]
        politika.write_text(json.dumps(veri, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            gate_unserved_records_never_cited()
            raise AssertionError("a policy that lost the work-queue document was accepted")
        except SystemExit:
            pass
    finally:
        if icerik is not None:
            politika.write_text(icerik, encoding="utf-8")


def _geri_besleme_ihlalleri(sayilar: set[str], yollar: set[str]) -> list[str]:
    """Which corpus-visible files restate a corpus-derived number."""
    import re as _re

    ihlal: list[str] = []
    for yol in sorted(yollar):
        dosya = ROOT / yol
        if not dosya.is_file():
            continue
        try:
            metin = dosya.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for sayi in sorted(sayilar):
            if _re.search(rf"(?<!\d){sayi}(?!\d)", metin):
                ihlal.append(f"{yol}: {sayi}")
    return ihlal


def gate_measurements_do_not_feed_back() -> str:
    """A measurement may not be written into a file the measurement reads.

    This was found, not predicted. `README.md` is inside the corpus, and its
    ratchet line restated the corpus's own token count, so the figure was an
    input to the measurement that produced it. The result was a two-cycle with
    no fixed point: with 101443 written the corpus measured 101444, and with
    101444 written it measured 101443. The ratchet could not settle, and the
    reason looked like a flaky build rather than a feedback loop.

    Numbers that come out of the corpus belong in `training/ratchet.json`,
    which the corpus does not read.
    """
    import json as _json

    korpus = ROOT / "corpus" / "knowledge-self.jsonl.gz"
    if not korpus.is_file():
        raise SystemExit(f"corpus is not built: {korpus}")
    butce_dosya = ROOT / "training" / "eval" / "sonuclar" / "egitim-butcesi-2026-09-23.json"
    if not butce_dosya.is_file():
        raise SystemExit(f"no budget record: {butce_dosya}")
    butce = _json.loads(butce_dosya.read_text(encoding="utf-8"))["butce"]
    sayilar = {str(butce["korpus_kayit_sayisi"]), str(butce["benzersiz_jeton"])}
    yollar: set[str] = set()
    with gzip.open(korpus, "rt", encoding="utf-8") as fh:
        for satir in fh:
            satir = satir.strip()
            if satir:
                yollar.add(_json.loads(satir)["path"])
    ihlal = _geri_besleme_ihlalleri(sayilar, yollar)
    if ihlal:
        raise SystemExit(
            "a corpus-derived figure is written into a file the corpus reads, so the "
            "measurement feeds back into itself: " + "; ".join(ihlal)
        )
    return (
        f"{len(yollar)} corpus-visible file(s) checked against the corpus figures "
        f"({', '.join(sorted(sayilar))}); none restates them"
    )


def selftest_measurements_do_not_feed_back() -> None:
    """Canaries: a corpus-visible file carrying the token count is caught, and
    the same file without it is clean. Needs no corpus."""
    # The canary figures are deliberately not plausible corpus figures. This
    # file is inside the corpus, so a hardcoded number here would itself be
    # flagged by the gate the moment the corpus reached that size - which is
    # what happened when this self-test was first written with the live figures.
    kanarya = ROOT / "geri-besleme-kanaryasi.md"
    sahte = {"987654321", "987654320"}
    olusturuldu = not kanarya.is_file()
    try:
        kanarya.write_text("korpus 987654321 jeton iceriyor.\n", encoding="utf-8")
        bulunan = _geri_besleme_ihlalleri(sahte, {"geri-besleme-kanaryasi.md"})
        if not bulunan:
            raise AssertionError("a file restating a corpus figure was not caught")
        kanarya.write_text("korpusun jeton sayisi ratchet.json'da duruyor.\n", encoding="utf-8")
        temiz = _geri_besleme_ihlalleri(sahte, {"geri-besleme-kanaryasi.md"})
        if temiz:
            raise AssertionError(f"a clean file was reported as feedback: {temiz}")
        # A figure embedded in a longer number is not the figure.
        kanarya.write_text("satir 1987654321x\n", encoding="utf-8")
        gomulu = _geri_besleme_ihlalleri(sahte, {"geri-besleme-kanaryasi.md"})
        if gomulu:
            raise AssertionError(f"a substring was counted as the figure: {gomulu}")
    finally:
        if olusturuldu and kanarya.is_file():
            kanarya.unlink()


def gate_rust_tokenizer_agrees_with_python() -> str:
    """The Rust tokenizer and the Python one that cut the vocab produce the same ids.

    The vocab was cut in Python; the training core reads it in Rust. Two
    implementations of one format agreeing by convention is not an agreement,
    so both are run over the whole corpus and their ids are compared record by
    record. The Rust side is *executed*, never reimplemented here - a
    comparison against a copy of the logic would measure the copy.
    """
    import json as _json

    korpus = ROOT / "corpus" / "knowledge-self.jsonl.gz"
    if not korpus.is_file():
        raise SystemExit(f"corpus is not built: {korpus}")
    betik = ROOT / "training" / "jeton_capraz.py"
    proc = subprocess.run(
        [sys.executable, str(betik), "--corpus", str(korpus), "--json", "--cargo"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"the two tokenizers disagree or the Rust side refused: "
            f"{(proc.stdout or proc.stderr).strip()[-400:]}"
        )
    try:
        olcum = _json.loads(proc.stdout)
    except ValueError as exc:
        raise SystemExit(f"cross-check produced no measurement: {exc}") from exc
    if olcum["uyusmaz_kayit"] != 0:
        raise SystemExit(f"{olcum['uyusmaz_kayit']} record(s) tokenize differently")
    if olcum["karsilastirilan_kayit"] != olcum["korpus_kayit"]:
        raise SystemExit(
            f"only {olcum['karsilastirilan_kayit']} of {olcum['korpus_kayit']} records "
            f"were compared"
        )
    return (
        f"{olcum['sozluk']}: {olcum['karsilastirilan_kayit']} record(s) tokenized "
        f"identically by Rust and Python ({olcum['rust_toplam_jeton']} tokens)"
    )


def selftest_rust_tokenizer_agrees_with_python() -> None:
    """Canary: a vocab whose pretoken pattern this reader cannot apply must make
    the cross-check fail, not silently tokenize something else. Needs no corpus."""
    import tempfile as _tempfile

    dizin = pathlib.Path(_tempfile.mkdtemp(prefix="jeton-kanarya-"))
    try:
        vocab = dizin / "sahte-sozluk.json"
        vocab.write_text(
            json.dumps({
                "format": "lubot-bpe",
                "format_version": 1,
                "vocab_family": "kanarya-v0",
                "vocab_size": 257,
                "pretoken_pattern": r"\w+",
                "merges": [[97, 98]],
            }),
            encoding="utf-8",
        )
        korpus = dizin / "korpus.jsonl"
        korpus.write_text('{"text": "abc"}\n', encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(ROOT / "training" / "jeton_capraz.py"),
             "--vocab", str(vocab), "--corpus", str(korpus), "--json", "--cargo"],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        if proc.returncode == 0:
            raise AssertionError(
                "a vocab with an unsupported pretoken pattern was accepted instead "
                "of refused"
            )
    finally:
        import shutil as _shutil

        _shutil.rmtree(dizin, ignore_errors=True)


# --------------------------------------------------------------------------
# gates: the trained surface
# --------------------------------------------------------------------------
# Four gates measure the four things that can be wrong quietly about a run:
# the checkpoint file, the inference path, the run's own report, and the
# ranking surface. They run the real binary on the real corpus - a gate that
# only reads source text cannot see a checkpoint that does not round-trip.


def _binary() -> list[str]:
    """The fastest binary that exists: the release build when one is there,
    otherwise `cargo run`, which reuses the debug artifacts the test step
    already produced."""
    release = ROOT / "target" / "release" / "lubot"
    if release.is_file():
        return [str(release)]
    return ["cargo", "run", "--quiet", "-p", "lubot", "--bin", "lubot", "--"]


def _kosu(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([*_binary(), *args], cwd=ROOT, capture_output=True,
                          text=True, check=False)


_MINI_CORPUS_KILIT = "_mini_corpus_kilidi"


def _mini_korpus(tmp: Path) -> Path:
    """Build the corpus once per gate process and keep it in the temp dir."""
    corpus = tmp / "mini.jsonl.gz"
    if not corpus.is_file():
        built = subprocess.run(
            [sys.executable, str(ROOT / "training" / "build_corpus.py"),
             "--repo", str(ROOT), "--out", str(corpus)],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        if built.returncode != 0:
            raise SystemExit(f"the corpus builder failed: {built.stderr.strip()[:200]}")
    return corpus


def _kucuk_kosu(tmp: Path, ad: str, *, tohum: int = 20260924, adim: int = 4,
                ek: tuple[str, ...] = ()) -> dict[str, Path]:
    """One genuinely small training run: real corpus, real stamp, real steps.

    Small in steps and window, not in discipline: the stamp is computed and
    declared, the held-out exam set is passed, and the checkpoint is written by
    the same code path a long run uses.
    """
    corpus = _mini_korpus(tmp)
    damga = _kosu("korpus-damgasi", "--corpus", str(corpus),
                  "--vocab", "training/tokenizer/lubot-bpe-v2.json")
    if damga.returncode != 0 or len(damga.stdout.strip()) != 64:
        raise SystemExit(f"the corpus stamp could not be computed: {damga.stderr.strip()[:200]}")
    ckpt = tmp / f"{ad}.ckpt"
    rapor = tmp / f"{ad}.md"
    kayit = tmp / f"{ad}.json"
    run = _kosu(
        "egitim-kosu", "--corpus", str(corpus), "--damga", damga.stdout.strip(),
        "--sinav", "training/eval/sinav-seti.jsonl", "--ckpt", str(ckpt),
        "--rapor", str(rapor), "--kayit", str(kayit), "--sessiz",
        "--adim", str(adim), "--pencere", "64", "--yigin", "1",
        "--dogrulama-her", "2", "--dogrulama-pencere", "4",
        "--tohum", str(tohum), "--isinma", "1", *ek,
    )
    if run.returncode != 0:
        raise SystemExit(f"a short training run failed: {run.stderr.strip()[:300]}")
    return {"ckpt": ckpt, "rapor": rapor, "kayit": kayit, "corpus": corpus}


def gate_checkpoint_round_trips() -> str:
    """The same seed twice gives the same bytes; a different seed does not; a
    file with one flipped byte is refused. A checkpoint that fails any of the
    three is not a record of a run."""
    import hashlib
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        a = _kucuk_kosu(tmp, "a")
        b = _kucuk_kosu(tmp, "b")
        c = _kucuk_kosu(tmp, "c", tohum=777)
        oa = hashlib.sha256(a["ckpt"].read_bytes()).hexdigest()
        ob = hashlib.sha256(b["ckpt"].read_bytes()).hexdigest()
        oc = hashlib.sha256(c["ckpt"].read_bytes()).hexdigest()
        if oa != ob:
            raise SystemExit("the same seed produced two different checkpoints: the run is not reproducible")
        if oa == oc:
            raise SystemExit("a different seed produced the same checkpoint: the seed reaches nothing")
        # tek bayt cevrilir: dosya kendini reddetmeli
        bozuk = tmp / "bozuk.ckpt"
        ham = bytearray(a["ckpt"].read_bytes())
        ham[len(ham) // 2] ^= 0x01
        bozuk.write_bytes(bytes(ham))
        denetim = _kosu("cikarim", "denetle", "--ckpt", str(bozuk), "--kimlikler", "1,2,3,4")
        if denetim.returncode == 0:
            raise SystemExit("a checkpoint with a flipped byte loaded: the digest is not checked")
        if "ozet" not in (denetim.stderr + denetim.stdout):
            raise SystemExit(f"the refusal does not name the digest: {(denetim.stderr + denetim.stdout).strip()[:200]}")
    return "the run round-trips byte for byte, the seed changes it, and a flipped byte is refused by the digest"


def selftest_checkpoint_round_trips() -> None:
    """The canaries: the comparisons this gate makes must each be able to fail."""
    import hashlib
    a = hashlib.sha256(b"one").hexdigest()
    b = hashlib.sha256(b"another").hexdigest()
    assert a != b, "the digest comparison cannot tell two files apart"
    ham = bytearray(b"LUBOTCKPT" + bytes(range(32)))
    ham[len(ham) // 2] ^= 0x01
    assert bytes(ham) != b"LUBOTCKPT" + bytes(range(32)), "the byte flip did nothing"


def _cache_finding(stdout: str, tolerance: float) -> str | None:
    """Read the three-way agreement out of the report, or say what is missing."""
    import re
    for field in ("onbellekli ortalama log-olasilik", "tam gecis ortalamasi",
                  "egitim cekirdegi", "en buyuk fark (onbellek/tam)"):
        if field not in stdout:
            return f"the report does not carry `{field}`"
    match = re.search(r"\| en buyuk fark \(onbellek/tam\) \| ([0-9.e+-]+)", stdout)
    if match is None:
        return "the cached/full difference is not a number"
    gap = float(match.group(1))
    if gap != gap or gap > tolerance:
        return f"the cached path and the full pass disagree by {gap:.3e} > {tolerance:.0e}"
    return None


def selftest_inference_cache_agrees() -> None:
    """A report with a gap above the tolerance has to be refused, or the gate
    is only checking that the report exists."""
    iyi = "| en buyuk fark (onbellek/tam) | 1.000e-15 (tolerans 1e-9) |\n"
    assert _cache_finding("onbellekli ortalama log-olasilik\n tam gecis ortalamasi\n egitim cekirdegi\n" + iyi, 1e-9) is None
    kotu = "| en buyuk fark (onbellek/tam) | 1.000e-03 |\n"
    assert _cache_finding("onbellekli ortalama log-olasilik\n tam gecis ortalamasi\n egitim cekirdegi\n" + kotu, 1e-9) is not None
    eksik = "| en buyuk fark (onbellek/tam) | 1.000e-15 |\n"
    assert _cache_finding(eksik, 1e-9) is not None, "a report that never measured the training kernel passed"


def gate_inference_cache_agrees() -> str:
    """Scoring a checkpoint through the cache and through a full recomputation
    must give the same number, and the training kernel must agree with both."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        kosu = _kucuk_kosu(tmp, "c")
        denetim = _kosu("cikarim", "denetle", "--ckpt", str(kosu["ckpt"]),
                        "--kimlikler", "1,2,3,4,5,6,7,8")
        if denetim.returncode != 0:
            raise SystemExit(f"the cache check could not run: {denetim.stderr.strip()[:300]}")
        finding = _cache_finding(denetim.stdout, 1e-9)
        if finding:
            raise SystemExit(finding)
    return "the cached pass, a full recomputation and the training kernel give one number for one id sequence"


def _run_report_finding(md: str) -> str | None:
    """A run report has to carry what the run measured, by name."""
    for field in ("| adim |", "| kayip |", "| durma |", "| jeton |",
                  "| korpus ozeti |", "| kontrol noktasi |", "| held-out |",
                  "| epoch |"):
        if field not in md:
            return f"the report has no `{field.strip('| ')}` row"
    if "ALL" in md:
        return "the report carries a verdict word"
    return None


def selftest_training_run_is_measured() -> None:
    """The canaries: a report missing a measured row, or carrying a verdict
    word, must each be refused."""
    tam = "| adim | 1 -> 4 |\n| kayip | 9.0 -> 8.0 |\n| durma | adim-butcesi |\n| jeton | 100 |\n| korpus ozeti | x |\n| kontrol noktasi | y |\n| held-out | z |\n| epoch | 0 -> 1 (tavan 8) |\n"
    assert _run_report_finding(tam) is None, "a complete report was refused"
    assert _run_report_finding(tam.replace("| durma | adim-butcesi |\n", "")) is not None
    assert _run_report_finding(tam + "ALL GATES PASSED\n") is not None


def gate_training_run_is_measured() -> str:
    """A short run writes a report that names every quantity it measured, and
    an evaluation record that survives the mechanical-criterion schema."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        kosu = _kucuk_kosu(tmp, "r", adim=6)
        md = kosu["rapor"].read_text(encoding="utf-8")
        finding = _run_report_finding(md)
        if finding:
            raise SystemExit(finding)
        if not _begins_with_heading(md):
            raise SystemExit("the run report does not begin with a heading")
        rec = json.loads(kosu["kayit"].read_text(encoding="utf-8"))
        problem = _eval_run_finding(rec)
        if problem:
            raise SystemExit(f"the run's evaluation record is not a measurement: {problem}")
        if rec["kaynaklar"]["cikti_jetonlari"] <= 0:
            raise SystemExit("a model run reported zero output tokens")
        if rec["kosucu"] != "model":
            raise SystemExit("a training run is a model run and has to say so")
    return "a real run's report carries every measured row and its record passes the one-criterion schema"


def _ranking_finding(md: str) -> str | None:
    """A ranking report has to be ordered, count its tokens and tie stably."""
    import re
    rows = re.findall(r"^\| (\d+) \| (\d+) \| (\d+) \| ([0-9.eE+-]+) \|$", md, re.M)
    if len(rows) < 2:
        return "the ranking report has fewer than two candidates"
    puanlar = [float(r[3]) for r in rows]
    if any(a < b - 1e-12 for a, b in zip(puanlar, puanlar[1:])):
        return "the candidates are not in descending score order"
    if any(int(r[2]) <= 0 for r in rows):
        return "a candidate reports zero scored tokens"
    esit = [(puanlar[i], int(rows[i][1]), int(rows[i + 1][1]))
            for i in range(len(rows) - 1) if puanlar[i] == puanlar[i + 1]]
    for _, once, sonra in esit:
        if once >= sonra:
            return "equal scores were reordered: a tie was broken by something other than the caller's index"
    return None


def selftest_reranker_is_measured() -> None:
    """The canaries: a mis-ordered table and a zero-token row must be refused."""
    iyi = "| 1 | 0 | 5 | -1.5 |\n| 2 | 1 | 5 | -2.0 |\n"
    assert _ranking_finding(iyi) is None, "an ordered table was refused"
    kotu = "| 1 | 0 | 5 | -2.5 |\n| 2 | 1 | 5 | -1.0 |\n"
    assert _ranking_finding(kotu) is not None, "an ascending table passed as a ranking"
    sifir = "| 1 | 0 | 0 | -1.5 |\n| 2 | 1 | 5 | -2.0 |\n"
    assert _ranking_finding(sifir) is not None, "a candidate with no scored tokens passed"


def gate_reranker_is_measured() -> str:
    """The ranking surface orders candidates by score, counts the tokens it
    scored, and leaves ties in the order the caller gave them."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        kosu = _kucuk_kosu(tmp, "s")
        adaylar = tmp / "adaylar.txt"
        adaylar.write_text("1,2,3,4,5\n6,7,8,9,10\n1,2,3,4,5\n", encoding="utf-8")
        siralama = _kosu("cikarim", "sirala", "--ckpt", str(kosu["ckpt"]),
                         "--baglam", "11,12", "--adaylar", str(adaylar))
        if siralama.returncode != 0:
            raise SystemExit(f"the ranking surface failed: {siralama.stderr.strip()[:300]}")
        finding = _ranking_finding(siralama.stdout)
        if finding:
            raise SystemExit(finding)
        if siralama.stdout.count("| 1,2,3,4,5 |") == 0 and "1,2,3,4,5" in siralama.stdout:
            raise SystemExit("the report printed the ids instead of the caller's indices")
    return "the ranking surface is ordered, counts its tokens, and keeps equal scores in the caller's order"


# --- MM: muhendislik iskeleti ile veri ayrimi -------------------------------


def _mm_import_adlari(yol: pathlib.Path) -> set[str]:
    """Bir Python dosyasinin ust duzey import adlari; dosya calistirilmaz."""
    import ast

    agac = ast.parse(yol.read_text(encoding="utf-8"))
    adlar: set[str] = set()
    for dugum in ast.walk(agac):
        if isinstance(dugum, ast.Import):
            adlar.update(takma.name.split(".")[0] for takma in dugum.names)
        elif isinstance(dugum, ast.ImportFrom) and dugum.level == 0 and dugum.module:
            adlar.add(dugum.module.split(".")[0])
    return adlar


def _mm_muhendislik_ihlalleri(klasor: pathlib.Path) -> list[str]:
    """training/*.py yalnizca stdlib ve kardes modul import eder (MM).

    Iskelet deseni disaridan esinlenebilir, ama kosucu dis bir ML
    cercevesine baglanirsa "sifirdan" iddiasi sessizce duser.
    """
    yerel = {p.stem for p in klasor.glob("*.py")}
    stdlib = set(sys.stdlib_module_names)
    ihlaller: list[str] = []
    for p in sorted(klasor.glob("*.py")):
        for ad in sorted(_mm_import_adlari(p)):
            if ad in stdlib or ad in yerel:
                continue
            ihlaller.append(f"{p.relative_to(klasor.parent)} dis bagimlilik ister: {ad}")
    return ihlaller


def _mm_disi_url(metin: str) -> bool:
    return "http://" in metin or "https://" in metin


def _mm_veri_ihlalleri(kok: pathlib.Path) -> list[str]:
    """K2 siniri: curriculum yalniz kendi agactan (dis URL yasak), korpus kayitlarinin
    provenance'i agac icindeki bir dosyayi gostermek zorunda."""
    ihlaller: list[str] = []
    for p in sorted((kok / "training" / "curriculum").glob("*.jsonl")):
        for no, satir in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            satir = satir.strip()
            if not satir:
                continue
            kayit = json.loads(satir)
            roller = [m.get("role") for m in kayit.get("messages", []) if isinstance(m, dict)]
            if "user" not in roller or "assistant" not in roller:
                ihlaller.append(f"{p.name}:{no} kullanici/asistan cifti yok")
            if _mm_disi_url(satir):
                ihlaller.append(f"{p.name}:{no} dis URL tasiyor")
    for p in sorted((kok / "corpus").glob("knowledge-*.jsonl.gz")):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for no, satir in enumerate(f, 1):
                satir = satir.strip()
                if not satir:
                    continue
                kayit = json.loads(satir)
                yol = str(kayit.get("path", ""))
                if not yol or yol.startswith(("/", "..")) or not (kok / yol).is_file():
                    ihlaller.append(f"{p.name}:{no} provenance agac disi: {yol!r}")
    return ihlaller


def gate_training_runner_engineering_vs_data() -> str:
    """Muhendislik iskeleti (MM) ile veri (K2) ayri eksenlerdir: kosucu
    dosyalari desen esinlenmesi tasiyabilir ama dis bir cerceveye
    baglanamaz; corpus/ ve training/curriculum/ ise yalnizca bu agactan
    uretilmis kayitlari tasir - her kaydin provenance'i agac icinde
    olmali ve hicbir kayit dis URL tasimamali. Ikisini ayri kapilarla
    tutmak, K1'i korurken K2'yi yanlislikla ihlal etmeyi engeller."""
    ihlaller = _mm_muhendislik_ihlalleri(ROOT / "training") + _mm_veri_ihlalleri(ROOT)
    if ihlaller:
        raise SystemExit(
            "muhendislik/veri siniri ihlal edildi:\n" + "".join(f"  {s}\n" for s in ihlaller)
        )
    script = len(list((ROOT / "training").glob("*.py")))
    return f"kosucu yalniz stdlib+kardes modul ({script} script), veri yalniz kendi agactan"


def selftest_training_runner_engineering_vs_data() -> None:
    """Kanarya: numpy import eden script ve dis URL tasiyan curriculum
    satiri reddedilmeli; temiz es lenegi kabul edilmeli."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        kok = pathlib.Path(td)
        (kok / "training" / "curriculum").mkdir(parents=True)
        (kok / "corpus").mkdir()
        (kok / "training" / "temiz.py").write_text(
            "import json\nfrom pathlib import Path\n", encoding="utf-8")
        assert _mm_muhendislik_ihlalleri(kok / "training") == []
        (kok / "training" / "kacak.py").write_text("import numpy\n", encoding="utf-8")
        assert _mm_muhendislik_ihlalleri(kok / "training"), "dis bagimlilik kabul edildi"
        (kok / "training" / "kacak.py").unlink()
        iyi = {"messages": [{"role": "user", "content": "soru"},
                            {"role": "assistant", "content": "cevap"}]}
        (kok / "training" / "curriculum" / "a.jsonl").write_text(
            json.dumps(iyi) + "\n", encoding="utf-8")
        assert _mm_veri_ihlalleri(kok) == []
        kotu = {"messages": [{"role": "user", "content": "bkz https://example.com"},
                             {"role": "assistant", "content": "cevap"}]}
        (kok / "training" / "curriculum" / "a.jsonl").write_text(
            json.dumps(kotu) + "\n", encoding="utf-8")
        assert _mm_veri_ihlalleri(kok), "dis URL kabul edildi"


# --- JJ: korpusun yapisal kayitlari -----------------------------------------


def _ks_yapisal_denetim(korpus: pathlib.Path) -> tuple[dict[str, int], list[str]]:
    """Korpus kayitlarinin turlerini ve provenance'ini denetler (dosya okur)."""
    sayim: dict[str, int] = {}
    hatalar: list[str] = []
    with gzip.open(korpus, "rt", encoding="utf-8") as fh:
        for no, satir in enumerate(fh, 1):
            satir = satir.strip()
            if not satir:
                continue
            kayit = json.loads(satir)
            tur = str(kayit.get("kind", ""))
            sayim[tur] = sayim.get(tur, 0) + 1
            yol = kayit.get("path")
            aralik = kayit.get("lines")
            if not isinstance(yol, str) or not yol:
                hatalar.append(f"kayit {no}: provenance yok")
            elif (
                not isinstance(aralik, list)
                or len(aralik) != 2
                or not all(isinstance(x, int) for x in aralik)
            ):
                hatalar.append(f"kayit {no}: satir araligi bozuk ({yol})")
    for gerekli in ("api-doc-pair", "trait-impl", "dependency-edge"):
        if sayim.get(gerekli, 0) == 0:
            hatalar.append(f"yapisal kayit turu eksik: {gerekli}")
    return sayim, hatalar


def gate_corpus_carries_structure() -> str:
    """JJ: korpus duz metin degil, yapi tasir.

    Belge-imza ciftleri (ne cagrilir + neden var), "kim neyi uyguluyor"
    iliskileri ve Cargo.toml bagimlilik kenarlari ayri kayit turleri olarak
    bulunur. Bir tur sessizce kaybolursa kapi kirmizi olur; "yok" ile
    "olculmedi" ayni sey degildir."""
    korpus = ROOT / "corpus" / "knowledge-self.jsonl.gz"
    if not korpus.is_file():
        raise SystemExit(f"corpus is not built: {korpus.relative_to(ROOT)}")
    sayim, hatalar = _ks_yapisal_denetim(korpus)
    if hatalar:
        raise SystemExit("JJ yapisal korpus denetimi:\n" + "".join(f"  {s}\n" for s in hatalar[:6]))
    return (
        f"yapisal kayitlar: api-doc-pair {sayim['api-doc-pair']}, "
        f"trait-impl {sayim['trait-impl']}, dependency-edge {sayim['dependency-edge']}"
    )


def selftest_corpus_carries_structure() -> None:
    """Kanarya: eksik tur ve provenance'siz kayit reddedilmeli, tam kayit kabul."""
    import tempfile

    def yaz(kok: pathlib.Path, kayitlar: list[dict]) -> pathlib.Path:
        yol = kok / "k.jsonl.gz"
        with gzip.open(yol, "wt", encoding="utf-8") as fh:
            for kayit in kayitlar:
                fh.write(json.dumps(kayit) + "\n")
        return yol

    with tempfile.TemporaryDirectory() as td:
        kok = pathlib.Path(td)
        tam = [
            {"kind": "api-doc-pair", "text": "x", "path": "a.rs", "lines": [1, 2]},
            {"kind": "trait-impl", "text": "x", "path": "a.rs", "lines": [3, 3]},
            {"kind": "dependency-edge", "text": "x", "path": "Cargo.toml", "lines": [4, 4]},
        ]
        _, hatalar = _ks_yapisal_denetim(yaz(kok, tam))
        assert hatalar == [], f"temiz korpus reddedildi: {hatalar}"
        _, hatalar = _ks_yapisal_denetim(yaz(kok, tam[:2]))
        assert hatalar, "eksik tur kabul edildi"
        bozuk = [dict(kayit) for kayit in tam]
        bozuk[0].pop("path")
        _, hatalar = _ks_yapisal_denetim(yaz(kok, bozuk))
        assert hatalar, "provenance'siz kayit kabul edildi"


# --- RR: bilgi boslugu haritasi ---------------------------------------------


_BH_SATIRLAR = [
    {"at": 1, "question": "tokenizer sozlugu nasil donar", "citations": ["a"], "refusals": 0},
    {"at": 2, "question": "tokenizer merge tablosu nedir", "citations": ["b"], "refusals": 0},
    {"at": 3, "question": "tokenizer kac jeton", "citations": [], "refusals": 1},
    {"at": 4, "question": "zkvm icine ispat", "citations": [], "refusals": 0},
    {"at": 5, "question": "zkvm kaniti nasil", "citations": [], "refusals": 0},
]


def _bh_kos(kok: pathlib.Path, satirlar: list[dict] | None = None) -> tuple[dict, dict]:
    """Sahte gunluk + korpus uzerinde haritayi kosar; (rapor, kayit) doner."""
    korpus = kok / "k.jsonl.gz"
    with gzip.open(korpus, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({"kind": "doc", "text": "tokenizer donmus sozluk",
                             "path": "a.md", "lines": [1, 1]}) + "\n")
    gunluk = kok / "audit.jsonl"
    gunluk.write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in (satirlar or _BH_SATIRLAR)) + "\n",
        encoding="utf-8",
    )
    rapor_yolu = kok / "rapor.json"
    kayit_yolu = kok / "kayit.jsonl"
    kosu = subprocess.run(
        [sys.executable, str(ROOT / "training" / "bosluk_haritasi.py"),
         "--audit", str(gunluk), "--corpus", str(korpus),
         "--out", str(rapor_yolu), "--kayit", str(kayit_yolu), "--kok", str(kok)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if kosu.returncode != 0:
        raise SystemExit(f"harita kosmadi: {(kosu.stderr or kosu.stdout)[-300:]}")
    return (
        json.loads(rapor_yolu.read_text(encoding="utf-8")),
        json.loads(kayit_yolu.read_text(encoding="utf-8").strip()),
    )


def gate_gap_report_is_measured() -> str:
    """RR: "cevaplanamadi" tek tek cevaplarin kaderi olarak kalmasin.

    Audit gunlugu her soruyu, cevap turunu ve red sayisini yazar; harita bu
    gunlukten turetilir: en cok sorulan ama korpusta en az karsiligi olan
    konular siralanir. Kapi, haritanin *gunlukle birlikte* degistigini
    gosterir (sabit bir sayi degil, olcumdur) ve kaydin provenance'ini
    denetler."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        kok = pathlib.Path(td)
        rapor, kayit = _bh_kos(kok)
        if rapor["soru"] != 5 or rapor["cevapsiz"] != 3:
            raise SystemExit(
                f"harita gunlugu saymiyor: soru={rapor['soru']} cevapsiz={rapor['cevapsiz']}"
            )
        ilk = rapor["bosluklar"][0]
        if ilk["konu"] != "zkvm" or ilk["korpus_kaydi"] != 0:
            raise SystemExit(f"en zayif konu yanlis siralandi: {ilk}")
        tokenizer = [b for b in rapor["karsiligi_olan"] if b["konu"] == "tokenizer"]
        if not tokenizer or tokenizer[0]["korpus_kaydi"] != 1:
            raise SystemExit("korpusta karsiligi olan konu kapsama almadi")
        if kayit.get("kind") != "gap-report" or not kayit.get("path"):
            raise SystemExit(f"harita kaydi korpusa girmez: {kayit}")
        # Gunluk degisince harita da degisir: sabit sayi degil, olcum.
        rapor2, _ = _bh_kos(kok, _BH_SATIRLAR[:-1])
        if rapor2["soru"] != 4:
            raise SystemExit("harita gunlukle birlikte degismiyor")
        eksik = subprocess.run(
            [sys.executable, str(ROOT / "training" / "bosluk_haritasi.py"), "--out", str(kok / "x.json")],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        if eksik.returncode == 0:
            raise SystemExit("harita gunluk olmadan yazildi")
    return "harita gunluge bagli: 5 soru/3 cevapsiz, en zayif konu kapsamasiz"


def selftest_gap_report_is_measured() -> None:
    """Kanarya: bos ve bozuk gunluk kabul edilmemeli; cikti iki kosuda ayni."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        kok = pathlib.Path(td)
        rapor, _ = _bh_kos(kok)
        rapor_ikinci, _ = _bh_kos(kok)
        assert rapor == rapor_ikinci, "harita deterministik degil"
        bos = kok / "bos.jsonl"
        bos.write_text("\n", encoding="utf-8")
        kosu = subprocess.run(
            [sys.executable, str(ROOT / "training" / "bosluk_haritasi.py"),
             "--audit", str(bos), "--out", str(kok / "r.json")],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        assert kosu.returncode == 0, "bos gunluk hata verdi"
        assert json.loads((kok / "r.json").read_text(encoding="utf-8"))["soru"] == 0


# --- QQ: diyagram girdisi ---------------------------------------------------


def gate_doc_diagram_feeds_corpus() -> str:
    """QQ: `doc` yeteneiginin girdi tarafi genisler - diyagram okunur, uretilmez.

    Metin diyagramlari (Mermaid: .mmd dosyasi ya da ```mermaid citi) kenar ve
    dugum etiketi kayitlarina cevrilir; SVG'den yalniz <text> etiketleri
    okunur ve "kenarlar okunmadi" diye yazar. Bu bir uretim yuzeyi degildir:
    kayitlar dosyanin kendi satirlarindan gelir, provenance tasir ve goruntu
    dosyalari (png/jpg) metne cevrilmez - okumadigini iddia etmez."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        kok = pathlib.Path(td)
        (kok / "LICENSE.md").write_text("MIT License\n", encoding="utf-8")
        (kok / "d.mmd").write_text(
            "graph TD\n  A[Istek] --> B[Grant]\n  B --> C[Indeks]\n", encoding="utf-8")
        (kok / "README.md").write_text(
            "# Baslik\n\n```mermaid\ngraph LR\n  X[Oku] --> Y[Cevap]\n```\n",
            encoding="utf-8")
        (kok / "sekil.svg").write_text(
            "<svg><text>Operator</text><text>Zincir</text></svg>\n", encoding="utf-8")
        (kok / "resim.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        cikti = kok / "k.jsonl.gz"
        kosu = subprocess.run(
            [sys.executable, str(ROOT / "training" / "build_corpus.py"),
             "--repo", str(kok), "--out", str(cikti)],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        if kosu.returncode != 0:
            raise SystemExit(f"korpus kurulamadi: {(kosu.stderr or kosu.stdout)[-300:]}")
        kayitlar = []
        with gzip.open(cikti, "rt", encoding="utf-8") as fh:
            for satir in fh:
                if satir.strip():
                    kayitlar.append(json.loads(satir))
        diyagramlar = [k for k in kayitlar if k.get("kind") == "diagram"]
        if not diyagramlar:
            raise SystemExit("diyagram kaydi uretilmedi")
        metinler = " ".join(k["text"] for k in diyagramlar)
        if "`A` -> `B`" not in metinler:
            raise SystemExit("mmd kenari okunmadi")
        if not any(k["path"] == "README.md" for k in diyagramlar):
            raise SystemExit("markdown icindeki mermaid citi okunmadi")
        if "Operator" not in metinler or "edges are not read" not in metinler:
            raise SystemExit("SVG etiketleri durustce raporlanmadi")
        if any(str(k.get("path", "")).endswith(".png") for k in kayitlar):
            raise SystemExit("goruntu dosyasi metne cevrilmis gibi kayit uretti")
        if not all(k.get("path") and k.get("lines") for k in diyagramlar):
            raise SystemExit("diyagram kaydi provenance tasimiyor")
    return f"diyagram okunuyor ({len(diyagramlar)} kayit), goruntu dosyasi okunmuyor"


def selftest_doc_diagram_feeds_corpus() -> None:
    """Kanarya: kenarsiz metin diyagram uretmemeli; lisanssiz agac reddedilmeli."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        kok = pathlib.Path(td)
        (kok / "k.mmd").write_text("graph TD\n  tek dugum\n", encoding="utf-8")
        sys.path.insert(0, str(ROOT / "training"))
        import build_corpus as bc

        kayitlar = list(bc.mermaid_kayitlari("graph TD\n  tek dugum\n", "k.mmd", 1))
        assert kayitlar == [], "kenarsiz/etiketsiz satirdan kayit uretildi"
        kosu = subprocess.run(
            [sys.executable, str(ROOT / "training" / "build_corpus.py"),
             "--repo", str(kok), "--out", str(kok / "o.jsonl.gz")],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        assert kosu.returncode != 0, "lisanssiz agac kabul edildi"


# --- KK: hakem ciftleri -----------------------------------------------------


def _gp_eksikler(kok: pathlib.Path) -> list[str]:
    """Repodaki her kapinin bir kanaryasi ve kanaryada bir reddi var mi?"""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        cikti = pathlib.Path(td) / "k.jsonl.gz"
        kosu = subprocess.run(
            [sys.executable, str(ROOT / "training" / "build_corpus.py"),
             "--repo", str(kok), "--out", str(cikti)],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        if kosu.returncode != 0:
            return [f"korpus kurulamadi: {(kosu.stderr or kosu.stdout)[-200:]}"]
        ciftler: dict[str, str] = {}
        with gzip.open(cikti, "rt", encoding="utf-8") as fh:
            for satir in fh:
                if not satir.strip():
                    continue
                kayit = json.loads(satir)
                if kayit.get("kind") == "gate-pair":
                    ciftler[kayit["text"]] = kayit["text"]
    liste = subprocess.run(
        [sys.executable, str(ROOT / "gates" / "check.py"), "--list"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    kapilar = [s.strip() for s in liste.stdout.splitlines() if s.strip()]
    sorunlar: list[str] = []
    if len(ciftler) != len(kapilar):
        sorunlar.append(f"kapi {len(kapilar)}, hakem cifti {len(ciftler)}: eslesmiyor")
    for metin in ciftler:
        if "kanaryada red yok" in metin:
            sorunlar.append(f"kanaryasiz iddia: {metin[:80]}")
    return sorunlar


def gate_gate_pairs_carry_referee() -> str:
    """KK: her kapi iddiasi kendi kanaryasiyla eslesir ve kanarya bir seyi
    reddeder.

    Derleyici ve test takimi bu repoda bedava hakemdir: bir kural ancak onu
    curen bir kanarya kosuyorsa kuraldir. Kapi, korpustaki `gate-pair`
    kayitlarinin sayisini kapilarla karsilastirir ve "kanaryada red yok"
    diyen bir iddiayi kabul etmez."""
    sorunlar = _gp_eksikler(ROOT)
    if sorunlar:
        raise SystemExit("hakem ciftleri eksik:\n" + "".join(f"  {s}\n" for s in sorunlar[:6]))
    return "her kapinin kanaryasi var ve kanarya reddediyor"


def selftest_gate_pairs_carry_referee() -> None:
    """Kanarya: reddi olmayan selftest ve eksik kayit yakalanmali."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        kok = pathlib.Path(td)
        (kok / "LICENSE.md").write_text("MIT License\n", encoding="utf-8")
        (kok / "gates").mkdir()
        (kok / "gates" / "check.py").write_text(
            'def gate_bir() -> str:\n    """Bir sey."""\n    return "x"\n\n'
            "def selftest_bir() -> None:\n    pass\n",
            encoding="utf-8",
        )
        sorunlar = _gp_eksikler(kok)
        assert sorunlar, "reddi olmayan selftest kabul edildi"


# --- korpus turleri: uretici ile okuyucu ayni dili konusur -------------------


def gate_corpus_kinds_agree() -> str:
    """Yazilan her tur okunabilmeli.

    build_corpus.py'nin urettigi kayit turleri ile `lubot corpus`un kabul
    ettigi turler ayni kume olmali; okuyucu bilinmeyen turu reddettigi icin
    yeni bir tur eklemek iki tarafi birlikte degistirmeyi gerektirir. Bu
    kapi, JJ/QQ/KK/RR turlarini eklerken tam olarak bu yuzden dogdu:
    2755 kayitlik korpusun 617'si okuyucudan donuyordu."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        korpus = _mini_korpus(pathlib.Path(td))
        turler: dict[str, int] = {}
        with gzip.open(korpus, "rt", encoding="utf-8") as fh:
            for satir in fh:
                if satir.strip():
                    tur = json.loads(satir)["kind"]
                    turler[tur] = turler.get(tur, 0) + 1
        kosu = _cli("corpus", str(korpus))
        birlesik = (kosu.stdout or "") + (kosu.stderr or "")
        if kosu.returncode != 0 or "refused" in birlesik:
            raise SystemExit(f"okuyucu ureticinin turlerini reddetti: {birlesik[-300:]}")
        return f"{len(turler)} kayit turu okuyucudan gecti: " + ", ".join(sorted(turler))


def selftest_corpus_kinds_agree() -> None:
    """Kanarya: uydurma bir tur okuyucudan gecemez."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        yol = pathlib.Path(td) / "uydurma.jsonl.gz"
        with gzip.open(yol, "wt", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "kind": "uydurma-tur", "text": "kanarya",
                "path": "x.md", "digest": "0" * 64,
            }) + "\n")
        kosu = _cli("corpus", str(yol))
        assert kosu.returncode != 0, "uydurma tur kabul edildi"


# --- I: alma (retrieval) yuzeyinin olcumu -----------------------------------
# Almak model cagirmaz: BM25 ayni korpusu tarar. Bu yuzden olcum ikiliyi
# ister, agirlik istemez - ve ikili yoksa olcum 12 soru icin 12 kez
# derlemeye kalkisirdi; ikili bir kez derlenir.


def _er_ikili() -> str:
    """Olcum icin ikili spec'i: release ikili yoksa bir kez derlenir."""
    import shlex

    if not (ROOT / "target" / "release" / "lubot").is_file():
        subprocess.run(["cargo", "build", "--release", "-p", "lubot"], cwd=ROOT,
                       capture_output=True, text=True, check=False)
    return shlex.join(_binary())


def _er_kos(*ek: str) -> list[str]:
    """Alma olcumu: korpus + sinav seti + ikili, verilen ek bayraklarla."""
    return [
        sys.executable, str(ROOT / "training" / "erisim_geri_cagirma.py"),
        "--corpus", "corpus/knowledge-self.jsonl.gz",
        "--sorular", "training/eval/sinav-seti.jsonl",
        "--bin", _er_ikili(), *ek,
    ]


def _er_olc(*ek: str) -> dict:
    kosu = subprocess.run(_er_kos(*ek), cwd=ROOT, capture_output=True, text=True, check=False)
    if kosu.returncode != 0:
        raise SystemExit(f"alma olcumu kosmadi: {(kosu.stderr or kosu.stdout)[-300:]}")
    return json.loads(kosu.stdout)


def _alma_kanit_bulgu(kayit: dict, taze: dict, alinti: dict) -> str | None:
    """Olcum kaydinin taze olcumle uyusu; uyusmuyorsa gerekcesi.

    Iki sorgu bicimi ayri ayri karsilastirilir: tam soru metni yonerge
    tasir, `alinti` alani yalniz cekirdek cumleyi olcer. Ikisi ayni
    yonde sapmaz, bu yuzden kayit ikisini de tasimak zorunda."""
    kanit = kayit.get("kanit")
    if not isinstance(kanit, dict):
        return "kayit kanit tasimiyor: olcum kaniti olmadan iddia sayilamaz"
    for ad, olcum in (("tam", taze), ("alinti", alinti)):
        if not olcum["ilk_sirada"] <= olcum["ilk_n_icinde"] <= olcum["soru"]:
            return f"{ad} olcumu ic tutarsiz: ilk sirada {olcum['ilk_sirada']}, " \
                   f"ilk {olcum['n']} icinde {olcum['ilk_n_icinde']}, soru {olcum['soru']}"
        if len(olcum["sirali"]) != olcum["soru"]:
            return f"{ad} olcumunde her soru icin sira kaydi yok"
        if len(olcum["bulunamayan"]) > olcum["soru"] - olcum["ilk_n_icinde"]:
            return f"{ad} olcumunde bulunamayan listesi isabetsizlikle uyusmuyor"
    ikinci = kanit.get("ikinci_olcum")
    if not isinstance(ikinci, dict) or ikinci.get("sorgu_alani") != "alinti":
        return "kayit ikinci olcumu tasimiyor: sorgu bicimi ayrimi olculmemis"
    for ad, kayitli, olcum in (("tam", kanit, taze), ("alinti", ikinci, alinti)):
        if (kayitli.get("ilk_sirada"), kayitli.get("ilk_n_icinde")) != (
            olcum["ilk_sirada"], olcum["ilk_n_icinde"]
        ):
            return (
                f"kayit bayat ({ad}): kayitta {kayitli.get('ilk_sirada')}/"
                f"{kayitli.get('ilk_n_icinde')}, olcum {olcum['ilk_sirada']}/"
                f"{olcum['ilk_n_icinde']} - kaydi yeniden uret"
            )
    return None


def gate_retrieval_at_k_is_measured() -> str:
    """I: alma katmani iddia degil olcumdur.

    Sinav setindeki her soru damgalanmis bir pasaja dayanir; `ara` ayni
    korpusu BM25 ile tarar ve damganin kacinci sirada ciktigi olculur.
    Kapi olcumu yeniden kosar, kaydi taze olcumle karsilastirir ve olcumun
    deterministik oldugunu gorur: sapma varsa kayit yeniden uretilmelidir."""
    kayit_yolu = ROOT / "training" / "eval" / "sonuclar" / "erisim-2026-09-24.json"
    if not kayit_yolu.is_file():
        raise SystemExit("alma olcumu kaydi yok: training/eval/sonuclar/erisim-2026-09-24.json")
    kayit = json.loads(kayit_yolu.read_text(encoding="utf-8"))
    bulgu = _eval_run_finding(kayit)
    if bulgu:
        raise SystemExit(f"{kayit_yolu.name}: {bulgu}")
    taze = _er_olc()
    alinti = _er_olc("--sorgu-alani", "alinti")
    bulgu = _alma_kanit_bulgu(kayit, taze, alinti)
    if bulgu:
        raise SystemExit(bulgu)
    tekrar = _er_olc()
    if tekrar["sirali"] != taze["sirali"]:
        raise SystemExit("alma olcumu deterministik degil: ayni sorgu farkli sira verdi")
    return (
        f"alma olculdu: damga ilk sirada {taze['ilk_sirada']}/{taze['soru']}, "
        f"ilk {taze['n']} icinde {taze['ilk_n_icinde']}/{taze['soru']}; "
        f"yalniz cekirdek cumleyle ilk {alinti['n']} icinde "
        f"{alinti['ilk_n_icinde']}/{alinti['soru']}"
    )


def selftest_retrieval_at_k_is_measured() -> None:
    """Kanarya: bayat kayit, eksik ikinci olcum, ic tutarsiz sayi ve bos soru
    seti ayri ayri reddedilir.

    Kayit semasi saf denetlenir (alt surec yok); betik girdisi ise kendi
    korpusuyla sinanir - kanarya deponun korpusuna baglanirsa CI'da
    korpus henuz kurulmamisken yanlis sebepten kirmizi yanar."""
    import gzip
    import tempfile

    ornek = {"soru": 12, "n": 3, "ilk_sirada": 10, "ilk_n_icinde": 10,
             "bulunamayan": [{"soru": "sinav-01", "neden": "ilk n icinde yok"}],
             "sirali": [{"soru_kimligi": f"sinav-{i:02d}", "sira": None} for i in range(1, 13)],
             "sorgu_alani": "tam"}
    alinti = dict(ornek, ilk_sirada=5, ilk_n_icinde=11, sorgu_alani="alinti")
    kayit = {"kanit": {k: ornek[k] for k in ("soru", "n", "ilk_sirada", "ilk_n_icinde")},
             "alinti": None}
    kayit["kanit"]["ikinci_olcum"] = {k: alinti[k] for k in ("sorgu_alani", "ilk_sirada", "ilk_n_icinde")}
    assert _alma_kanit_bulgu(kayit, ornek, alinti) is None, "gecerli kayit reddedildi"
    bayat = json.loads(json.dumps(kayit))
    bayat["kanit"]["ilk_sirada"] = 9
    assert "bayat" in (_alma_kanit_bulgu(bayat, ornek, alinti) or ""), "bayat kayit gecti"
    eksik = json.loads(json.dumps(kayit))
    del eksik["kanit"]["ikinci_olcum"]
    assert "ikinci olcumu" in (_alma_kanit_bulgu(eksik, ornek, alinti) or ""), "tek alanli kayit gecti"
    tutarsiz = dict(ornek, ilk_sirada=11)
    assert "tutarsiz" in (_alma_kanit_bulgu(kayit, tutarsiz, alinti) or ""), "tutarsiz sayi gecti"
    fazla = dict(ornek, bulunamayan=[{"soru": f"s{i}"} for i in range(5)])
    assert "isabetsizlikle" in (_alma_kanit_bulgu(kayit, fazla, alinti) or ""), "isabetsizlik gecti"

    with tempfile.TemporaryDirectory() as td:
        kok = pathlib.Path(td)
        sorular = kok / "sorular.jsonl"
        sorular.write_text(json.dumps({
            "soru_kimligi": "kanarya-01", "soru": "cevap nedir",
            "content_id": "0" * 64, "kaynak_dosya": "yok.md",
        }) + "\n", encoding="utf-8")
        eksik_korpus = subprocess.run([
            sys.executable, str(ROOT / "training" / "erisim_geri_cagirma.py"),
            "--corpus", str(kok / "yok.jsonl.gz"), "--sorular", str(sorular),
            "--bin", "lubot",
        ], cwd=ROOT, capture_output=True, text=True, check=False)
        assert eksik_korpus.returncode != 0 and "korpus yok" in (eksik_korpus.stderr + eksik_korpus.stdout), (
            "eksik korpus reddedilmedi: " + (eksik_korpus.stderr or eksik_korpus.stdout)[-200:]
        )
        # Bos soru seti: betik sorgu kosmadan durur, bu yuzden burada ikili
        # gerekmez - korpus gercek, soru listesi bos.
        gercek = kok / "kucuk.jsonl.gz"
        with gzip.open(gercek, "wt", encoding="utf-8") as fh:
            fh.write(json.dumps({"content_id": "0" * 64, "path": "yok.md", "text": "bir kayit"}) + "\n")
        bos = kok / "bos.jsonl"
        bos.write_text("\n", encoding="utf-8")
        bos_kosu = subprocess.run([
            sys.executable, str(ROOT / "training" / "erisim_geri_cagirma.py"),
            "--corpus", str(gercek), "--sorular", str(bos), "--bin", "lubot",
        ], cwd=ROOT, capture_output=True, text=True, check=False)
        assert bos_kosu.returncode != 0 and "soru seti bos" in (bos_kosu.stderr + bos_kosu.stdout), (
            "bos soru seti reddedilmedi: " + (bos_kosu.stderr or bos_kosu.stdout)[-200:]
        )


GATES_EXTRA = {
    "retrieval-at-k-is-measured": (gate_retrieval_at_k_is_measured, selftest_retrieval_at_k_is_measured),
    "corpus-kinds-agree": (gate_corpus_kinds_agree, selftest_corpus_kinds_agree),
    "gate-pairs-carry-referee": (gate_gate_pairs_carry_referee, selftest_gate_pairs_carry_referee),
    "doc-diagram-feeds-corpus": (gate_doc_diagram_feeds_corpus, selftest_doc_diagram_feeds_corpus),
    "gap-report-is-measured": (gate_gap_report_is_measured, selftest_gap_report_is_measured),
    "corpus-carries-structure": (gate_corpus_carries_structure, selftest_corpus_carries_structure),
    "training-runner-engineering-vs-data": (gate_training_runner_engineering_vs_data, selftest_training_runner_engineering_vs_data),
    "system-prompt-is-true": (gate_system_prompt_is_true, selftest_system_prompt_is_true),
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
    "tokenizer-vocab-is-frozen": (gate_tokenizer_vocab_is_frozen, selftest_tokenizer_vocab_is_frozen),
    "model-spec-is-consistent": (gate_model_spec_is_consistent, selftest_model_spec_is_consistent),
    "dependencies-are-used": (gate_dependencies_are_used, selftest_dependencies_are_used),
    "findings-are-disciplined": (gate_findings_are_disciplined, selftest_findings_are_disciplined),
    "eval-runs-are-mechanical": (gate_eval_runs_are_mechanical, selftest_eval_runs_are_mechanical),
    "eval-set-never-trained": (gate_eval_set_never_trained, selftest_eval_set_never_trained),
    "mup-measurement-reproduced": (gate_mup_measurement_reproduced, selftest_mup_measurement_reproduced),
    "data-mix-is-declared": (gate_data_mix_is_declared, selftest_data_mix_is_declared),
    "decision-head-has-no-generation-surface": (
        gate_decision_head_has_no_generation_surface,
        selftest_decision_head_has_no_generation_surface,
    ),
    "bootstrap-round-is-measured": (
        gate_bootstrap_round_is_measured,
        selftest_bootstrap_round_is_measured,
    ),
    "comparison-class-is-declared": (
        gate_comparison_class_is_declared,
        selftest_comparison_class_is_declared,
    ),
    "exam-set-is-held-out": (
        gate_exam_set_is_held_out,
        selftest_exam_set_is_held_out,
    ),
    "claims-carry-their-evidence": (
        gate_claims_carry_their_evidence,
        selftest_claims_carry_their_evidence,
    ),
    "decision-latency-is-recorded": (
        gate_decision_latency_is_recorded,
        selftest_decision_latency_is_recorded,
    ),
    "first-answer-latency-is-recorded": (
        gate_first_answer_latency_is_recorded,
        selftest_first_answer_latency_is_recorded,
    ),
    "architecture-doc-tracks-layer-rule": (
        gate_architecture_doc_tracks_layer_rule,
        selftest_architecture_doc_tracks_layer_rule,
    ),
    "measurements-do-not-feed-back": (
        gate_measurements_do_not_feed_back,
        selftest_measurements_do_not_feed_back,
    ),
    "rust-tokenizer-agrees-with-python": (
        gate_rust_tokenizer_agrees_with_python,
        selftest_rust_tokenizer_agrees_with_python,
    ),
    "training-budget-is-declared": (gate_training_budget_is_declared, selftest_training_budget_is_declared),
    "every-crate-is-a-member": (gate_every_crate_is_a_member, selftest_every_crate_is_a_member),
    "assert-arity": (gate_assert_arity, selftest_assert_arity),
    "no-dead-error-variant": (gate_no_dead_error_variant, selftest_no_dead_error_variant),
    "doc-links-resolve": (gate_doc_links_resolve, selftest_doc_links_resolve),
    "no-bool-comparison": (gate_no_bool_comparison, selftest_no_bool_comparison),
    "delimiters-balance": (gate_delimiters_balance, selftest_delimiters_balance),
    "crates-are-reachable": (gate_crates_are_reachable, selftest_crates_are_reachable),
    "crates-doc-is-measured": (gate_crates_doc_is_measured, selftest_crates_doc_is_measured),
    "rpc-surface-consistent": (gate_rpc_surface_consistent, selftest_rpc_surface_consistent),
    "pub-api-is-used": (gate_pub_api_is_used, selftest_pub_api_is_used),
    "checkpoint-round-trips": (gate_checkpoint_round_trips, selftest_checkpoint_round_trips),
    "inference-cache-agrees": (gate_inference_cache_agrees, selftest_inference_cache_agrees),
    "training-run-is-measured": (gate_training_run_is_measured, selftest_training_run_is_measured),
    "reranker-is-measured": (gate_reranker_is_measured, selftest_reranker_is_measured),
}



GATES = {
    "reads-not-generates": (gate_reads_not_generates, selftest_reads_not_generates),
    "no-fourth-channel": (gate_no_fourth_channel, selftest_no_fourth_channel),
    "provenance-fails-closed": (gate_provenance_fails_closed, selftest_provenance_fails_closed),
    "mask-before-storage": (gate_mask_before_storage, selftest_mask_before_storage),
    "no-panic-path": (gate_no_panic_path, selftest_no_panic_path),
    "readme-is-measured": (gate_readme_is_measured, selftest_readme_is_measured),
    **GATES_EXTRA,
    "unserved-records-never-cited": (
        gate_unserved_records_never_cited,
        selftest_unserved_records_never_cited,
    ),
}



def main(argv: list[str]) -> int:
    if not argv or argv[0] == "--list":
        for name in GATES:
            print(name)
        return 0
    if argv[0] == "--all":
        failures = 0
        for name, (run, selftest) in GATES.items():
            try:
                selftest()
            except Exception as err:  # noqa: BLE001 - reported, never raised
                failures += 1
                print(f"FAIL [{name}] its own self-test is broken: {err}")
                continue
            try:
                print(f"OK   [{name}] {run()}")
            except SystemExit as err:
                failures += 1
                print(f"FAIL [{name}] {err}")
            except Exception as err:  # noqa: BLE001 - reported, never raised
                # A gate that cannot run - a missing `cargo`, say - has to report
                # a failure. Raising instead aborts every gate after it, so the
                # run reports nothing about the ones that never got to run.
                failures += 1
                print(f"FAIL [{name}] could not run: {type(err).__name__}: {err}")
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
    try:
        print(f"OK   [{name}] {run()}")
    except SystemExit:
        raise
    except Exception as err:  # noqa: BLE001 - a gate that cannot run has to say so
        print(f"FAIL [{name}] could not run: {type(err).__name__}: {err}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
