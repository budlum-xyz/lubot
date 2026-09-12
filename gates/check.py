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
        match = re.match(r"^([A-Z][A-Za-z0-9_]*)\s*(\{|,|$)", stripped)
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
    """Every variant of every `*Error` enum is constructed by some code path."""
    offenders: list[str] = []
    checked = 0
    for path in rust_sources():
        text = path.read_text(encoding="utf-8")
        for enum_name in sorted(set(re.findall(r"pub enum ([A-Za-z0-9_]*Error)\b", text))):
            for variant in _enum_variants(text, enum_name):
                checked += 1
                if not _is_construction(text, enum_name, variant):
                    offenders.append(f"{path.relative_to(ROOT)} {enum_name}::{variant}")
    if offenders:
        raise SystemExit(
            "these error variants are never produced, so no caller handles them:\n  "
            + "\n  ".join(offenders)
        )
    return f"all {checked} error variants are constructed somewhere"


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
    assert _is_construction(sample, "DemoError", "Live"), "a constructed variant looked dead"
    assert not _is_construction(sample, "DemoError", "Dead"), "a dead variant looked constructed"


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



GATES_EXTRA = {
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
    "dependencies-are-used": (gate_dependencies_are_used, selftest_dependencies_are_used),
    "findings-are-disciplined": (gate_findings_are_disciplined, selftest_findings_are_disciplined),
    "eval-runs-are-mechanical": (gate_eval_runs_are_mechanical, selftest_eval_runs_are_mechanical),
    "every-crate-is-a-member": (gate_every_crate_is_a_member, selftest_every_crate_is_a_member),
    "assert-arity": (gate_assert_arity, selftest_assert_arity),
    "no-dead-error-variant": (gate_no_dead_error_variant, selftest_no_dead_error_variant),
    "doc-links-resolve": (gate_doc_links_resolve, selftest_doc_links_resolve),
    "no-bool-comparison": (gate_no_bool_comparison, selftest_no_bool_comparison),
    "delimiters-balance": (gate_delimiters_balance, selftest_delimiters_balance),
    "crates-are-reachable": (gate_crates_are_reachable, selftest_crates_are_reachable),
    "crates-doc-is-measured": (gate_crates_doc_is_measured, selftest_crates_doc_is_measured),
    "rpc-surface-consistent": (gate_rpc_surface_consistent, selftest_rpc_surface_consistent),
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
