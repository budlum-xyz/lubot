#!/usr/bin/env python3
"""NN-2: Budlum'a ozel, donmus ve surumlu bayt-duzeyi BPE sozlugu.

Yontem: bayt-duzeyi BPE. Yontem bilinir; implementasyon bu depoda sifirdan
ve yalnizca standart kutuphane ile yazildi. Disaridan kod, veri, agirlik ya
da hazir sozluk girmez (K1/K2). Onislem deseni metni harf / sayi / bosluk /
diger olarak dort sinifa ayirir; Turkce harfler harf sinifindadir, bayt
duzeyi calistigi icin hicbir karakter bozulmaz.

Donmus sozluk kurali (EE): sozluk kesildigi an donar ve surumlenir. Dosya
(korpus icerigi, hedef sozluk boyutu, onislem deseni)nun saf bir
fonksiyonudur; zaman damgasi tasimaz, ayni girdiyle iki egitim birebir ayni
dosyayi verir. Korpus bilincli olarak buyutuldugunde yeni surum (v2, v3,
...) kesilir; her yeni surum yeni bir model ailesidir. Eski surum silinmez.

Kapi: tokenizer-vocab-is-frozen (gates/check.py). Yukleyici yapisindan
suphe duyulan sozlugu kapidan reddeder; --verify gecerli korpusun her
kaydini kayipsiz geri dondurur ve kaynaktan sapmayi raporlar.

Kullanim:
    python3 train_tokenizer.py --corpus corpus/knowledge-self.jsonl.gz
        --out training/tokenizer/lubot-bpe-v1.json
    python3 train_tokenizer.py --verify --corpus ... --vocab ...
    python3 train_tokenizer.py --self-test
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import heapq
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

FORMAT = "lubot-bpe"
FORMAT_VERSION = 1
DEFAULT_VOCAB_SIZE = 4096
PRETOKEN_PATTERN = r"[^\W\d_]+|\d+|\s+|[\W_]+"
PRETOKEN_RE = re.compile(PRETOKEN_PATTERN)


# --------------------------------------------------------------------------
# korpus okuma
# --------------------------------------------------------------------------

def open_corpus(path: str):
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"korpus yok: {path}")
    return (gzip.open if p.suffix == ".gz" else open)(p, "rt", encoding="utf-8")


def corpus_texts(paths: list[str]) -> tuple[list[str], dict]:
    """Kayit metinleri ve olculen korpus istatistikleri. Birden fazla korpus
    dosyasi verilebilir (yuzey korpusu); ozet sayilar toplanir, sha256 ise
    dosya sirasiyla birlestirilmis satir akisinin ozetidir."""
    texts: list[str] = []
    records = 0
    characters = 0
    sha = hashlib.sha256()
    for path in paths:
        with open_corpus(path) as handle:
            for line in handle:
                sha.update(line.encode("utf-8"))
                text = json.loads(line).get("text", "")
                texts.append(text)
                records += 1
                characters += len(text)
    return texts, {
        "records": records,
        "characters": characters,
        "corpus_files": list(paths),
        "corpus_sha256": sha.hexdigest(),
    }


# --------------------------------------------------------------------------
# egitim
# --------------------------------------------------------------------------

def pretoken_counts(texts: list[str]) -> Counter:
    """Her benzersiz ontokeni agirlikla sayar: ayni ontoken bir kez durur."""
    counts: Counter = Counter()
    for text in texts:
        for pre in PRETOKEN_RE.findall(text):
            counts[tuple(pre.encode("utf-8"))] += 1
    return counts


def apply_merge(seq: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Cifti soldan saga, ust uste binmeden uygular (deterministik)."""
    out: list[int] = []
    i = 0
    while i < len(seq):
        if i + 1 < len(seq) and seq[i] == pair[0] and seq[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def train_merges(pre_counts: Counter, target_vocab: int) -> list[tuple[int, int]]:
    """Artan sayimli BPE. Secim: en yuksek cift sayimi; esitlikte en kucuk
    cift (deterministik). Sayim 2'nin altina duserse egitim ac kalar durur;
    hedefe varmak icin veriyi zorlamaz."""
    pair_counts: Counter = Counter()
    pair_pos: dict[tuple[int, int], set] = defaultdict(set)
    for key, cnt in pre_counts.items():
        for p in zip(key, key[1:]):
            pair_counts[p] += cnt
            pair_pos[p].add(key)

    heap: list[tuple[int, tuple[int, int]]] = []
    for p, c in pair_counts.items():
        if c >= 2:
            heapq.heappush(heap, (-c, p))

    merges: list[tuple[int, int]] = []
    while 256 + len(merges) < target_vocab:
        best = None
        while heap:
            negc, p = heapq.heappop(heap)
            if pair_counts.get(p, 0) == -negc and -negc >= 2:
                best = p
                break
        if best is None:
            break
        new_id = 256 + len(merges)
        merges.append(best)
        for key in list(pair_pos.pop(best, set())):
            cnt = pre_counts.pop(key, 0)
            if cnt == 0:
                continue
            seq = list(key)
            for p in zip(seq, seq[1:]):
                pair_counts[p] -= cnt
            merged = apply_merge(seq, best, new_id)
            mkey = tuple(merged)
            for p in zip(merged, merged[1:]):
                pair_counts[p] += cnt
                if pair_counts[p] >= 2:
                    heapq.heappush(heap, (-pair_counts[p], p))
                pair_pos[p].add(mkey)
            pre_counts[mkey] = pre_counts.get(mkey, 0) + cnt
    return merges


# --------------------------------------------------------------------------
# kodlama ve sozluk dosyasi
# --------------------------------------------------------------------------

def token_bytes(merges: list[tuple[int, int]]) -> dict[int, bytes]:
    """Token kimliginden bayt dizisine donusum tablosu (DAG: her birlestirme
    yalnizca kendinden once var olan kimliklere deginebilir)."""
    table = {i: bytes([i]) for i in range(256)}
    for idx, (a, b) in enumerate(merges):
        if a >= 256 + idx or b >= 256 + idx:
            raise SystemExit(
                f"gecersiz birlestirme #{idx}: ({a}, {b}) kendinden once tanimli degil"
            )
        table[256 + idx] = table[a] + table[b]
    return table


def load_vocab(path: str) -> dict:
    """Fail-closed yukleyici: yapisindan suphe duyulan sozluk reddedilir,
    asla en yakin gecerli bicime dusurulmez."""
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"sozluk yok: {path} (donmus sozluk turetilemez, kesilmis olmali)")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise SystemExit(f"sozluk JSON degil: {err}")
    for field in ["format", "format_version", "vocab_family", "vocab_size",
                  "target_vocab_size", "pretoken_pattern", "merges", "trained_from"]:
        if field not in raw:
            raise SystemExit(f"sozlukta zorunlu alan yok: {field}")
    if raw["format"] != FORMAT:
        raise SystemExit(f"bilinmeyen bicim: {raw['format']} (beklenen {FORMAT})")
    if raw["format_version"] != FORMAT_VERSION:
        raise SystemExit(f"bilinmeyen bicim surumu: {raw['format_version']}")
    if raw["vocab_family"] != p.stem:
        raise SystemExit(
            f"aile adi ile dosya adi uyusmuyor: {raw['vocab_family']} != {p.stem}"
        )
    merges = [(int(a), int(b)) for a, b in raw["merges"]]
    if raw["vocab_size"] != 256 + len(merges):
        raise SystemExit(
            f"vocab_size {raw['vocab_size']} birlestirme sayisiyla uyusmuyor "
            f"(256 + {len(merges)})"
        )
    try:
        pre_re = re.compile(raw["pretoken_pattern"])
    except re.error as err:
        raise SystemExit(f"onislem deseni derlenmiyor: {err}")
    return {
        "vocab_family": raw["vocab_family"],
        "vocab_size": raw["vocab_size"],
        "target_vocab_size": raw["target_vocab_size"],
        "pretoken_pattern": raw["pretoken_pattern"],
        "pretoken_re": pre_re,
        "merges": merges,
        "ranks": {pair: i for i, pair in enumerate(merges)},
        "bytes": token_bytes(merges),
        "trained_from": raw["trained_from"],
    }


def encode(text: str, vocab: dict) -> list[int]:
    """Gready en kucuk rank'li birlestirme; her ontoken kendi icinde kalir."""
    ranks = vocab["ranks"]
    ids: list[int] = []
    for pre in vocab["pretoken_re"].findall(text):
        seq = list(pre.encode("utf-8"))
        while len(seq) >= 2:
            best_rank = None
            best_i = -1
            for i, p in enumerate(zip(seq, seq[1:])):
                r = ranks.get(p)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_i = i
            if best_rank is None:
                break
            seq[best_i:best_i + 2] = [256 + best_rank]
        ids.extend(seq)
    return ids


def decode(ids: list[int], vocab: dict) -> str:
    table = vocab["bytes"]
    try:
        raw = b"".join(table[i] for i in ids)
    except KeyError as err:
        raise SystemExit(f"sozlukte olmayan token kimligi: {err}")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as err:
        raise SystemExit(f"token baytlari gecerli UTF-8 degil: {err}")


def write_vocab(out_path: str, merges: list[tuple[int, int]], target_vocab: int,
                trained_from: dict) -> None:
    payload = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "vocab_family": Path(out_path).stem,
        "vocab_size": 256 + len(merges),
        "target_vocab_size": target_vocab,
        "pretoken_pattern": PRETOKEN_PATTERN,
        "merges": [[a, b] for a, b in merges],
        "trained_from": trained_from,
        "notes": (
            "Donmus ve surumlu sozluk (EE): korpus bilincli buyutuldugunde yeni "
            "surum kesilir, eskisi silinmez. Dosya (korpus icerigi, hedef boyut, "
            "onislem deseni)nun saf bir fonksiyonudur; zaman damgasi tasimaz."
        ),
    }
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8")


# --------------------------------------------------------------------------
# dogrulama
# --------------------------------------------------------------------------

def verify(vocab_path: str, corpus_paths: list[str]) -> dict:
    """Donmus sozlugu verilen korpus dosyalarinin her kaydina karsi olcer:
    kayipsiz geri donus, gercek token sayimi ve kesildigi kaynaktan sapma
    raporu."""
    vocab = load_vocab(vocab_path)
    if vocab["pretoken_pattern"] != PRETOKEN_PATTERN:
        raise SystemExit(
            "sozlugun onislem deseni egiticinin gecerli deseninden farkli; "
            "farkli desen farkli ailedir, bilincli surum kesimi gerekir"
        )
    texts, stats = corpus_texts(corpus_paths)

    failures = 0
    total_tokens = 0
    used: set[int] = set()
    total_bytes = 0
    for text in texts:
        ids = encode(text, vocab)
        if decode(ids, vocab) != text:
            failures += 1
            if failures <= 3:
                print(f"kayipsiz geri donus HATASI: {text[:60]!r}", file=sys.stderr)
        total_tokens += len(ids)
        used.update(ids)
        total_bytes += len(text.encode("utf-8"))

    if failures:
        raise SystemExit(f"kayipsiz geri donus bozuk: {failures} kayit")

    trained = vocab["trained_from"]
    return {
        "vocab_family": vocab["vocab_family"],
        "vocab_size": vocab["vocab_size"],
        "roundtrip_records": stats["records"],
        "roundtrip_failures": 0,
        "bpe_tokens": total_tokens,
        "unique_tokens_used": len(used),
        "bytes_per_token": round(total_bytes / total_tokens, 2) if total_tokens else None,
        "approx_tokens_chars_over_4": stats["characters"] // 4,
        "trained_from_records": trained.get("records"),
        "current_records": stats["records"],
        "record_drift": stats["records"] - int(trained.get("records", 0)),
    }


# --------------------------------------------------------------------------
# komut satiri
# --------------------------------------------------------------------------

def _script(argv: list[str]) -> int:
    return subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), *argv],
        capture_output=True, text=True, check=False,
    ).returncode


def cmd_train(args: argparse.Namespace) -> int:
    texts, stats = corpus_texts(args.corpus)  # --corpus tekrarlanabilir
    counts = pretoken_counts(texts)
    merges = train_merges(counts, args.vocab_size)
    write_vocab(args.out, merges, args.vocab_size, {
        "corpus_files": stats["corpus_files"],
        "corpus_sha256": stats["corpus_sha256"],
        "records": stats["records"],
        "characters": stats["characters"],
    })
    vocab = load_vocab(args.out)
    total = 0
    total_bytes = 0
    for text in texts:
        ids = encode(text, vocab)
        total += len(ids)
        total_bytes += len(text.encode("utf-8"))
    print(json.dumps({
        "written": args.out,
        "vocab_family": vocab["vocab_family"],
        "vocab_size": vocab["vocab_size"],
        "target_vocab_size": args.vocab_size,
        "starved_early": vocab["vocab_size"] < args.vocab_size,
        "records": stats["records"],
        "characters": stats["characters"],
        "bpe_tokens": total,
        "bytes_per_token": round(total_bytes / total, 2) if total else None,
        "approx_tokens_chars_over_4": stats["characters"] // 4,
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    print(json.dumps(verify(args.vocab, args.corpus), ensure_ascii=False, indent=2))
    return 0  # --corpus tekrarlanabilir


def cmd_self_test() -> int:
    """Kanaryalar: egitim deterministik, bozuk yapi ve yabanci aile reddedilir,
    ac kalan egitim hedefi asmaz, sapma raporu olculur."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        corpus = Path(td) / "c.jsonl"
        lines = [
            {"kind": "doc", "text": "lubot okur, uretmez ve her cumleye alinti gosterir. " * 6},
            {"kind": "doc", "text": "view grant bir grantee ve bir content key id adlandirir. " * 6},
            {"kind": "doc", "text": "Türkçe çğıöşü ve kod: fn denetle() -> Result<(), Hata> {} " * 6},
        ]
        corpus.write_text(
            "".join(json.dumps(l, ensure_ascii=False) + "\n" for l in lines),
            encoding="utf-8",
        )

        out = Path(td) / "lubot-bpe-v1.json"
        assert _script(["--corpus", str(corpus), "--out", str(out),
                        "--vocab-size", "512"]) == 0, "kucuk korpus egitimi basarisiz"

        # 1) determinizm: ayni girdiyle iki egitim birebir ayni dosyayi verir
        #    (ayni aile adi, farkli dizin; aile adi dosya adindan turer)
        alt = Path(td) / "alt"
        alt.mkdir()
        out2 = alt / "lubot-bpe-v1.json"
        _script(["--corpus", str(corpus), "--out", str(out2), "--vocab-size", "512"])
        assert out.read_bytes() == out2.read_bytes(), \
            "iki egitim ayni korpusu farkli dosyaya cevirdi"

        # 2) saglam sozluk dogrulamayi gecer
        assert _script(["--verify", "--corpus", str(corpus), "--vocab", str(out)]) == 0, \
            "saglam sozluk dogrulamayi gecmedi"

        # 3) DAG bozuklugu reddedilir: kendinden sonraki kimlige deginen birlestirme
        broken = json.loads(out.read_text(encoding="utf-8"))
        broken["merges"][0] = [256 + len(broken["merges"]) - 1, 65]
        bad_dag = Path(td) / "bozuk-dag.json"
        bad_dag.write_text(json.dumps(broken, ensure_ascii=False), encoding="utf-8")
        try:
            load_vocab(str(bad_dag))
            raise AssertionError("DAG bozuklugu kabul edildi")
        except SystemExit:
            pass

        # 4) vocab_size birlestirme sayisiyla celisirse reddedilir
        mismatch = json.loads(out.read_text(encoding="utf-8"))
        mismatch["vocab_size"] += 1
        bad_size = Path(td) / "bozuk-sayi.json"
        bad_size.write_text(json.dumps(mismatch, ensure_ascii=False), encoding="utf-8")
        try:
            load_vocab(str(bad_size))
            raise AssertionError("celisen vocab_size kabul edildi")
        except SystemExit:
            pass

        # 5) aile adi dosya adiyla uyusmazsa reddedilir
        misnamed = Path(td) / "baska-aile.json"
        misnamed.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
        try:
            load_vocab(str(misnamed))
            raise AssertionError("uyusmayan aile adi kabul edildi")
        except SystemExit:
            pass

        # 6) ac kalan egitim hedefi asmaz ve yine de gecerli kalir
        huge = Path(td) / "aclik.json"
        _script(["--corpus", str(corpus), "--out", str(huge), "--vocab-size", "50000"])
        starved = json.loads(huge.read_text(encoding="utf-8"))
        assert starved["vocab_size"] < 50000, \
            "kucuk korpus buyuk hedefi doldurdu; aclik kontrolu yok"
        assert _script(["--verify", "--corpus", str(corpus), "--vocab", str(huge)]) == 0, \
            "aclikta kesilen sozluk dogrulamayi gecmedi"

        # 7) yabanci korpus: bayt-duzeyi guaranti kayipsizligi korur, sapma raporu gorunur
        other = Path(td) / "diger.jsonl"
        other.write_text(
            json.dumps({"kind": "doc", "text": "tamamen baska bir metin " * 10},
                       ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        report = verify(str(huge), [str(other)])
        assert report["roundtrip_failures"] == 0
        assert report["record_drift"] != 0, "kaynak sapmasi olculmedi"

    print("self-test OK: determinizm, DAG, sayi uyumu, aile adi, aclik, sapma raporu")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", action="append", help="korpus jsonl(.gz); tekrarlanabilir")
    ap.add_argument("--out", help="kesilecek sozluk dosyasi (egitim)")
    ap.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    ap.add_argument("--verify", action="store_true", help="donmus sozlugu olc")
    ap.add_argument("--vocab", help="dogrulanacak sozluk dosyasi")
    ap.add_argument("--self-test", action="store_true", help="kanaryalar")
    args = ap.parse_args(argv)

    if args.self_test:
        return cmd_self_test()
    if args.verify:
        if not args.corpus or not args.vocab:
            raise SystemExit("--verify icin en az bir --corpus ve --vocab gerekli")
        return cmd_verify(args)
    if args.corpus and args.out:
        return cmd_train(args)
    ap.error("--corpus + --out (egitim) ya da --verify --corpus --vocab (dogrulama) "
             "ya da --self-test gerekli")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
