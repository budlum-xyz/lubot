#![forbid(unsafe_code)]
//! The `lubot` binary: load a corpus, answer questions, keep the book and
//! the audit. `ask` writes ONLY the rendered Markdown to stdout; everything
//! else goes to stderr or to files, so a pipeline can trust stdout.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use lubot::{
    ask as run_ask, corpus_summary, load_corpus, now_seconds, run_batch, BookFile, StoredGrant,
};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("lubot: {message}");
            ExitCode::FAILURE
        }
    }
}

fn usage() -> String {
    [
        "usage:",
        "  lubot corpus <file.jsonl.gz>...",
        "  lubot egitim   (trainer self-check: spec, epoch ceiling, measured descent)",
        "  lubot jetonla --vocab <v.json> --corpus <c.jsonl.gz> [--limit N] [--tam]",
        "  lubot egitim-veri --corpus <c.jsonl.gz> [--uzunluk N]  (window measurement vs the spec)",
        "  lubot egitim-karsilastir --corpus <c.jsonl.gz> [--ckpt c.ckpt] [--pencere N]  (f64 vs f32 kernel, measured)",
        "  lubot egitim-kosu --corpus c.jsonl.gz --damga sha256 --sinav training/eval/sinav-seti.jsonl --ckpt out.ckpt [--rapor f.md] [--kayit f.json] [--adim N] [--iplik N] ...",
        "  lubot korpus-damgasi --corpus c.jsonl.gz [--vocab v.json]  (the stamp a run declares)",
        "  lubot cikarim denetle --ckpt <f> --kimlikler 1,2,3  |  cikarim puanla --ckpt <f> --baglam 1,2 --metin 3,4  |  cikarim sirala --ckpt <f> --baglam 1,2 --adaylar a.txt",
        "  lubot sinav-kosu --ckpt <f> --sinav training/eval/sinav-seti.jsonl --corpus c.jsonl.gz [--aday 4] [--rapor f.md] [--kayit f.json]",
        "  lubot ask --corpus <f1,f2> --reader <r> --effort 0.5x..10.0x [--audit f] [--outputs f] [--book b] <question>",
        "  lubot grant issue --reader <r> --key <k> --expires-at <sec> [--book b]",
        "  lubot grant revoke --reader <r> --key <k> [--book b]",
        "  lubot grant list [--book b]",
        "  lubot audit --path <f> [--limit n]",
        "  lubot prompt [--path training/system_prompt.md]",
        "  lubot ceilings",
        "  lubot batch --corpus <f1,f2> --questions <jsonl> --reader <r> --effort 0.5x..10.0x [--audit f] [--outputs f]",
        "  lubot risk --text <command> | --path <file>",
        "  lubot doc --in <f.pdf|f.md> [--origin o] --licence L --attribution A --asset-id <hex> [--kind doc] [--out f.jsonl.gz]",
        "  lubot queue add --corpus <f> --reader <r> --effort 0.5x..10.0x [--file q.jsonl] <question>",
        "  lubot queue list [--file q.jsonl]",
        "  lubot queue run [--file q.jsonl] [--budget n] [--check cmd] [--audit f] [--outputs f] [--book b] [--watch --poll s --idle n]",
        "  lubot queue log [--file q.jsonl] [--limit n]",
        "  lubot sohbet --ckpt <f> --sorgu \"...\" [--tohum N] [--sicaklik T] [--top-k K] [--top-p P] [--en-cok N] [--tekrar-cezasi C] [--kac-gram N] [--en-az-jeton N] [--kaydirma yeniden|onbellek] [--kayit f.json]",
        "  lubot ratchet [--set] [--baseline training/ratchet.json]",
        "  lubot envanter [--corpus-dir corpus]",
        "  lubot it -m <msg> --path <p> [--path p2 ...] [--dry-run] [--branch b]",
        "  lubot olc",
        "  lubot durum",
        "  lubot guvenlik [--path <f|dir> ...]",
        "  lubot graf",
        "  lubot dosya --path <f>",
        "  lubot soru list [--batarya f]",
        "  lubot soru get <id> [--batarya f]",
        "  lubot soru cevapla <id> <secim> [--not n] [--batarya f] [--cevap f]",
        "  lubot soru durum [--batarya f] [--cevap f]",
        "  lubot ara --corpus <f1,f2> [--n 3] <soru>",
        "  lubot indeks --corpus <f1,f2>",
        "  lubot mufredat --corpus <f1,f2> [--out syllabus.jsonl]",
        "  lubot karsilastir --corpus <f1,f2> --reader <r> --effort 0.5x,1.0x,5.0x [--book b] <soru>",
        "  lubot kosum denetle --reader <r> --corpus <digest> --epoch <n> [--budget n] [--lifetime s] [--restricted-ceiling n] [--scope k1,k2]  (stdin: oncelik<TAB>anahtar<TAB>R|O<TAB>govde)",
        "  lubot kosum dogrula --record <f.json>",
        "  lubot olcum mimari",
        "  lubot olcum esik --members 1,2,3,4 --threshold 3 --signers 2,3,4 [--requester n]",
        "  lubot olcum takip [--bound n]  (stdin: gorev[,bagimlilik...])",
        "  lubot odeme yaz --seq n --chain n --fee 1.50 [--open h --close h] [--payout a:10.00:ref,...] [--out f]",
        "  lubot odeme dogrula --media <f>",
        "  lubot olcum olcek --up 0.8 --down 0.4 [--cooldown 3 --min 1 --max 10 --step 0.5 --replicas n --window n --load 1.0,0.9,...]",
        "  lubot olcum sinif --weights kategori=sinyal:agirlik,...;kategori=... --signals ad=gucluluk,... [--floor 0.6]",
        "  lubot sikistir --path <f> [--igne desen ...] [--depo dir]",
        "  lubot sikistir --geri-getir <ozet-dosya> [--depo dir]",
        "  lubot ogren --log <f> [--ogren-dir outputs/ogren]",
        "  lubot karar [doktrin|tek <evet|hayir>:<olasilik>|oyla <evet:0.9,hayir:0.7,...>]",
    ]
    .join("\n")
}

fn run(args: &[String]) -> Result<(), String> {
    let Some(command) = args.first() else {
        return Err(usage());
    };
    let rest = &args[1..];
    match command.as_str() {
        "corpus" => {
            if rest.is_empty() {
                return Err(usage());
            }
            let paths: Vec<PathBuf> = rest.iter().map(PathBuf::from).collect();
            let corpus = load_corpus(&paths)?;
            println!(
                "{}",
                serde_json::to_string(&corpus_summary(&corpus)).map_err(|e| e.to_string())?
            );
            Ok(())
        }
        "ask" => cmd_ask(rest),
        "grant" => cmd_grant(rest),
        "audit" => cmd_audit(rest),
        "prompt" => cmd_prompt(rest),
        "ceilings" => cmd_ceilings(rest),
        "batch" => cmd_batch(rest),
        "risk" => cmd_risk(rest),
        "doc" => cmd_doc(rest),
        "queue" => cmd_queue(rest),
        "sohbet" => lubot::sohbet::cmd_sohbet(rest),
        "ratchet" => cmd_ratchet(rest),
        "envanter" => cmd_envanter(rest),
        "it" => cmd_it(rest),
        "olc" => cmd_olc(rest),
        "egitim" => cmd_egitim(rest),
        "jetonla" => cmd_jetonla(rest),
        "egitim-veri" => cmd_egitim_veri(rest),
        "egitim-kosu" => lubot::egitim_kosu::cmd_egitim_kosu(rest),
        "egitim-karsilastir" => lubot::egitim_kosu::cmd_egitim_karsilastir(rest),
        "cikarim" => lubot::egitim_kosu::cmd_cikarim(rest),
        "korpus-damgasi" => lubot::egitim_kosu::cmd_korpus_damgasi(rest),
        "sinav-kosu" => lubot::egitim_kosu::cmd_sinav_kosu(rest),
        "durum" => cmd_durum(rest),
        "guvenlik" => cmd_guvenlik(rest),
        "graf" => cmd_graf(rest),
        "dosya" => cmd_dosya(rest),
        "soru" => cmd_soru(rest),
        "ara" => cmd_ara(rest),
        "indeks" => cmd_indeks(rest),
        "mufredat" => cmd_mufredat(rest),
        "karsilastir" => cmd_karsilastir(rest),
        "kosum" => lubot::kosum::cmd_kosum(rest),
        "olcum" => lubot::olcum::cmd_olcum(rest),
        "odeme" => lubot::odeme::cmd_odeme(rest),
        "sikistir" => lubot::sikistir::cmd_sikistir(rest),
        "karar" => lubot::karar::cmd_karar(rest),
        "ogren" => lubot::sikistir::cmd_ogren(rest),
        other => Err(format!("unknown command `{other}`\n{}", usage())),
    }
}

/// Pick flag values from `args`, leaving the rest in `leftover`.
fn flags(args: &[String], names: &[&str]) -> (Vec<(String, String)>, Vec<String>) {
    let mut found: Vec<(String, String)> = Vec::new();
    let mut leftover: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if let Some(name) = names.iter().find(|n| arg == **n) {
            if let Some(value) = args.get(i + 1) {
                found.push(((*name).to_string(), value.clone()));
                i += 2;
                continue;
            }
            leftover.push(arg.clone());
            i += 1;
            continue;
        }
        leftover.push(arg.clone());
        i += 1;
    }
    (found, leftover)
}

/// Pick boolean flags out of `args`, leaving everything else (including
/// other flags) untouched. A boolean flag never consumes the next argument -
/// the value-taking parser above must never see it.
fn flags_bool(args: &[String], names: &[&str]) -> (bool, Vec<String>) {
    let mut set = false;
    let mut kept = Vec::new();
    for arg in args {
        if names.contains(&arg.as_str()) {
            set = true;
        } else {
            kept.push(arg.clone());
        }
    }
    (set, kept)
}

/// `--effort` is required at the CLI boundary (is-basi kurali: her is kendi
/// etiketini tasir; varsayilan yok). The library keeps its own internal
/// default; the workforce never guesses a tier.
fn effort_required(found: &[(String, String)], context: &str) -> Result<String, String> {
    let tag = one(found, "--effort").ok_or_else(|| {
        format!("{context}: --effort required (0.5x-10.0x; is-basi etiket kurali)")
    })?;
    lubot_tools::operator::answer_budget(&tag).map_err(|e| format!("--effort: {e}"))?;
    Ok(tag)
}

/// The closed-loop output target: `--outputs` when given, otherwise the
/// `outputs/` directory (the chosen default). The directory is created on
/// the write path, so a missing directory is never a silent no-op.
fn outputs_or_default(found: &[(String, String)]) -> Option<PathBuf> {
    if let Some(path) = one(found, "--outputs") {
        return Some(PathBuf::from(path));
    }
    let _ = std::fs::create_dir_all("outputs");
    Some(PathBuf::from("outputs/kayit.jsonl"))
}

fn one(found: &[(String, String)], name: &str) -> Option<String> {
    found
        .iter()
        .find(|(n, _)| n == name)
        .map(|(_, v)| v.clone())
}

fn cmd_ask(args: &[String]) -> Result<(), String> {
    let (found, leftover) = flags(
        args,
        &[
            "--corpus",
            "--reader",
            "--audit",
            "--outputs",
            "--effort",
            "--book",
        ],
    );
    let corpus_arg =
        one(&found, "--corpus").ok_or_else(|| format!("missing --corpus\n{}", usage()))?;
    let reader = one(&found, "--reader").ok_or_else(|| format!("missing --reader\n{}", usage()))?;
    if let Some(stray) = leftover.iter().find(|s| s.starts_with("--")) {
        return Err(format!("ask: unexpected `{stray}`\n{}", usage()));
    }
    let question = leftover.join(" ").trim().to_string();
    if question.is_empty() {
        return Err("ask: question is empty".to_string());
    }
    let paths: Vec<PathBuf> = corpus_arg
        .split(',')
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .collect();
    if paths.is_empty() {
        return Err("ask: --corpus is empty".to_string());
    }
    let corpus = load_corpus(&paths)?;
    let book_path = one(&found, "--book")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("lubot-grants.json"));
    let mut grants = BookFile::load(&book_path)?.to_book();
    let now = now_seconds()?;
    let audit_path = one(&found, "--audit").map(PathBuf::from);
    let outputs_path = outputs_or_default(&found);
    let effort = effort_required(&found, "ask")?;
    let markdown = run_ask(
        &corpus,
        &reader,
        &question,
        &mut grants,
        now,
        audit_path.as_deref(),
        outputs_path.as_deref(),
        Some(&effort),
    )?;
    println!("{markdown}");
    if let Some(path) = &audit_path {
        eprintln!("audit: {}", path.display());
    }
    if let Some(path) = &outputs_path {
        eprintln!("outputs: {}", path.display());
    }
    Ok(())
}

fn cmd_grant(args: &[String]) -> Result<(), String> {
    let Some(action) = args.first() else {
        return Err(format!("missing grant action\n{}", usage()));
    };
    let rest = &args[1..];
    match action.as_str() {
        "issue" => {
            let (found, leftover) = flags(rest, &["--reader", "--key", "--expires-at", "--book"]);
            if !leftover.is_empty() {
                return Err(format!("grant issue: unexpected `{}`", leftover[0]));
            }
            let reader = one(&found, "--reader").ok_or("grant issue: missing --reader")?;
            let key = one(&found, "--key").ok_or("grant issue: missing --key")?;
            let expires = one(&found, "--expires-at")
                .ok_or("grant issue: missing --expires-at")?
                .parse::<u64>()
                .map_err(|e| format!("grant issue: --expires-at: {e}"))?;
            let book_path = one(&found, "--book")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("lubot-grants.json"));
            let mut file = BookFile::load(&book_path)?;
            file.revoked.retain(|(k, r)| !(k == &key && r == &reader));
            file.grants
                .retain(|g| !(g.key_id == key && g.grantee == reader));
            file.grants.push(StoredGrant {
                key_id: key,
                grantee: reader,
                expires_at: expires,
            });
            file.save(&book_path)?;
            println!("issued");
            Ok(())
        }
        "revoke" => {
            let (found, leftover) = flags(rest, &["--reader", "--key", "--book"]);
            if !leftover.is_empty() {
                return Err(format!("grant revoke: unexpected `{}`", leftover[0]));
            }
            let reader = one(&found, "--reader").ok_or("grant revoke: missing --reader")?;
            let key = one(&found, "--key").ok_or("grant revoke: missing --key")?;
            let book_path = one(&found, "--book")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("lubot-grants.json"));
            let mut file = BookFile::load(&book_path)?;
            file.grants
                .retain(|g| !(g.key_id == key && g.grantee == reader));
            if !file.revoked.contains(&(key.clone(), reader.clone())) {
                file.revoked.push((key, reader));
            }
            file.save(&book_path)?;
            println!("revoked");
            Ok(())
        }
        "list" => {
            let (found, leftover) = flags(rest, &["--book"]);
            if !leftover.is_empty() {
                return Err(format!("grant list: unexpected `{}`", leftover[0]));
            }
            let book_path = one(&found, "--book")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("lubot-grants.json"));
            let file = BookFile::load(&book_path)?;
            for grant in &file.grants {
                println!("{} {}\t{}", grant.key_id, grant.grantee, grant.expires_at);
            }
            for (key, reader) in &file.revoked {
                println!("{key} {reader}\trevoked");
            }
            Ok(())
        }
        other => Err(format!("unknown grant action `{other}`\n{}", usage())),
    }
}

fn cmd_ceilings(args: &[String]) -> Result<(), String> {
    if !args.is_empty() {
        return Err(format!("ceilings takes no arguments\n{}", usage()));
    }
    let md = lubot::ceilings_doc();
    lubot::validate_output(md.as_bytes(), "ceilings")?;
    println!("{md}");
    Ok(())
}

fn cmd_batch(args: &[String]) -> Result<(), String> {
    let (found, leftover) = flags(
        args,
        &[
            "--corpus",
            "--questions",
            "--reader",
            "--effort",
            "--book",
            "--audit",
            "--outputs",
        ],
    );
    if !leftover.is_empty() {
        return Err(format!("batch: unexpected `{}`", leftover[0]));
    }
    let corpus_arg =
        one(&found, "--corpus").ok_or_else(|| format!("batch: missing --corpus\n{}", usage()))?;
    let questions_path = PathBuf::from(
        one(&found, "--questions")
            .ok_or_else(|| format!("batch: missing --questions\n{}", usage()))?,
    );
    let reader =
        one(&found, "--reader").ok_or_else(|| format!("batch: missing --reader\n{}", usage()))?;

    let text = std::fs::read_to_string(&questions_path)
        .map_err(|e| format!("questions {}: {e}", questions_path.display()))?;
    let mut questions: Vec<String> = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let question: serde_json::Value = serde_json::from_str(line)
            .map_err(|e| format!("questions {}:{}: {e}", questions_path.display(), index + 1))?;
        let q = question
            .get("question")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                format!(
                    "questions {}:{}: missing `question`",
                    questions_path.display(),
                    index + 1
                )
            })?;
        if q.trim().is_empty() {
            return Err(format!(
                "questions {}:{}: empty question",
                questions_path.display(),
                index + 1
            ));
        }
        questions.push(q.to_string());
    }
    let paths: Vec<PathBuf> = corpus_arg
        .split(',')
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .collect();
    let corpus = load_corpus(&paths)?;
    let book_path = one(&found, "--book")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("lubot-grants.json"));
    let mut grants = BookFile::load(&book_path)?.to_book();
    let audit_path = one(&found, "--audit").map(PathBuf::from);
    let outputs_path = outputs_or_default(&found);
    let effort = effort_required(&found, "batch")?;
    let md = run_batch(
        &corpus,
        &reader,
        &questions,
        &mut grants,
        now_seconds()?,
        Some(&effort),
        audit_path.as_deref(),
        outputs_path.as_deref(),
    )?;
    println!("{md}");
    Ok(())
}

/// Split a `--corpus f1,f2` argument into paths, refusing an empty list.
fn corpus_paths(list: &str) -> Result<Vec<PathBuf>, String> {
    let paths: Vec<PathBuf> = list
        .split(',')
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .collect();
    if paths.is_empty() {
        return Err("--corpus is empty".to_string());
    }
    Ok(paths)
}

fn cmd_ara(args: &[String]) -> Result<(), String> {
    let (found, leftover) = flags(args, &["--corpus", "--n"]);
    let corpus =
        one(&found, "--corpus").ok_or_else(|| format!("ara: missing --corpus\n{}", usage()))?;
    let n: usize = one(&found, "--n")
        .unwrap_or_else(|| "3".to_string())
        .parse()
        .map_err(|_| "ara: --n must be a number".to_string())?;
    let question = leftover.join(" ").trim().to_string();
    if question.is_empty() {
        return Err("ara: question is empty".to_string());
    }
    let paths = corpus_paths(&corpus)?;
    let loaded = lubot::load_corpus(&paths)?;
    let hits = lubot::corpus_search(&loaded, &question, n)?;
    let mut md = format!("# Arama\n\nSoru: `{question}`\n\n");
    if hits.is_empty() {
        md.push_str("_no passage matched_\n");
    }
    for hit in &hits {
        let licence = loaded
            .meta()
            .iter()
            .find(|m| m.id == hit.item_id)
            .map(|m| m.licence.as_str())
            .unwrap_or("-");
        let pasaj = hit.text.trim();
        let cit = lubot_read::output_schema::fence_for(pasaj);
        md.push_str(&format!(
            "- `{}` (licence `{licence}`)\n\n{cit}\n{pasaj}\n{cit}\n\n",
            hit.citation()
        ));
    }
    lubot::validate_output(md.as_bytes(), "ara")?;
    println!("{md}");
    Ok(())
}

fn cmd_indeks(args: &[String]) -> Result<(), String> {
    let (found, leftover) = flags(args, &["--corpus"]);
    if !leftover.is_empty() {
        return Err(format!("indeks: unexpected `{}`", leftover[0]));
    }
    let corpus =
        one(&found, "--corpus").ok_or_else(|| format!("indeks: missing --corpus\n{}", usage()))?;
    let paths = corpus_paths(&corpus)?;
    let loaded = lubot::load_corpus(&paths)?;
    let stats = lubot::index_stats(&loaded);
    let by_kind = stats["by_kind"].as_object().map(|o| o.len()).unwrap_or(0);
    let by_licence = stats["by_licence"]
        .as_object()
        .map(|o| o.len())
        .unwrap_or(0);
    let md = format!(
        "# Indeks\n\n- items: {}\n- bytes: {}\n- kinds: {}\n- licences: {}\n- origins: {}\n",
        stats["items"], stats["bytes"], by_kind, by_licence, stats["origins"]
    );
    lubot::validate_output(md.as_bytes(), "indeks")?;
    println!("{md}");
    Ok(())
}

fn cmd_mufredat(args: &[String]) -> Result<(), String> {
    let (found, leftover) = flags(args, &["--corpus", "--out"]);
    if !leftover.is_empty() {
        return Err(format!("mufredat: unexpected `{}`", leftover[0]));
    }
    let corpus = one(&found, "--corpus")
        .ok_or_else(|| format!("mufredat: missing --corpus\n{}", usage()))?;
    let out = one(&found, "--out").map(PathBuf::from);
    let paths = corpus_paths(&corpus)?;
    let loaded = lubot::load_corpus(&paths)?;
    let md = lubot::curriculum_md(&loaded, out.as_deref())?;
    lubot::validate_output(md.as_bytes(), "mufredat")?;
    println!("{md}");
    Ok(())
}

fn cmd_karsilastir(args: &[String]) -> Result<(), String> {
    let (found, leftover) = flags(args, &["--corpus", "--reader", "--effort", "--book"]);
    let corpus = one(&found, "--corpus")
        .ok_or_else(|| format!("karsilastir: missing --corpus\n{}", usage()))?;
    let reader = one(&found, "--reader")
        .ok_or_else(|| format!("karsilastir: missing --reader\n{}", usage()))?;
    let effort_list = one(&found, "--effort")
        .ok_or_else(|| format!("karsilastir: missing --effort\n{}", usage()))?;
    let question = leftover.join(" ").trim().to_string();
    if question.is_empty() {
        return Err("karsilastir: question is empty".to_string());
    }
    let efforts: Vec<String> = effort_list
        .split(',')
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect();
    for tag in &efforts {
        lubot_tools::operator::answer_budget(tag)?;
    }
    let book_path =
        PathBuf::from(one(&found, "--book").unwrap_or_else(|| "lubot-grants.json".to_string()));
    let grants = lubot::BookFile::load(&book_path)?.to_book();
    let paths = corpus_paths(&corpus)?;
    let loaded = lubot::load_corpus(&paths)?;
    let md = lubot::compare_efforts(
        &loaded,
        &reader,
        &question,
        &grants,
        lubot::now_seconds()?,
        &efforts,
    )?;
    println!("{md}");
    Ok(())
}

fn cmd_soru(args: &[String]) -> Result<(), String> {
    let Some(action) = args.first() else {
        return Err(format!("missing soru action\n{}", usage()));
    };
    let rest = &args[1..];
    let (found, leftover) = flags(rest, &["--not", "--batarya", "--cevap"]);
    let battery_path = PathBuf::from(
        one(&found, "--batarya").unwrap_or_else(|| "training/soru-bataryasi.json".to_string()),
    );
    let answers_path =
        PathBuf::from(one(&found, "--cevap").unwrap_or_else(|| "soru-cevaplari.jsonl".to_string()));
    let battery = || -> Result<lubot::soru::Batarya, String> {
        let text = std::fs::read_to_string(&battery_path)
            .map_err(|e| format!("{}: {e}", battery_path.display()))?;
        lubot::soru::Batarya::load(&text)
    };
    let args_ok = |expected: usize| -> Result<(), String> {
        if leftover.len() != expected {
            Err(format!(
                "soru {action}: {expected} arguman bekleniyor\n{}",
                usage()
            ))
        } else {
            Ok(())
        }
    };
    match action.as_str() {
        "list" => {
            args_ok(0)?;
            let batarya = battery()?;
            let md = batarya.list_md();
            lubot::validate_output(md.as_bytes(), "soru list")?;
            println!("{md}");
            Ok(())
        }
        "get" => {
            args_ok(1)?;
            let batarya = battery()?;
            let soru = batarya
                .by_id(&leftover[0])
                .ok_or_else(|| format!("soru `{}` bataryada yok", leftover[0]))?;
            let mut md = format!(
                "# Soru: {}\n\n{}\n\n| secenek | aciklama |\n|---|---|\n",
                soru.id, soru.question
            );
            for option in &soru.options {
                md.push_str(&format!(
                    "| {} | {} |\n",
                    option.label,
                    option.description.replace('|', "/")
                ));
            }
            md.push_str(&format!(
                "\nOzel cevap: {}\n\nJSON: `{}`\n",
                if soru.allow_custom_response {
                    "izinli"
                } else {
                    "kapali"
                },
                batarya.question_json(&soru.id).unwrap_or_default()
            ));
            lubot::validate_output(md.as_bytes(), "soru get")?;
            println!("{md}");
            Ok(())
        }
        "cevapla" => {
            args_ok(2)?;
            let batarya = battery()?;
            let not = one(&found, "--not").unwrap_or_default();
            lubot::soru::record_answer(
                &answers_path,
                &leftover[0],
                &batarya,
                &leftover[1],
                &not,
                lubot::now_seconds()?,
            )?;
            let md = format!(
                "# Soru\n\n`{}` -> `{}` kaydedildi.\n",
                leftover[0], leftover[1]
            );
            lubot::validate_output(md.as_bytes(), "soru cevapla")?;
            println!("{md}");
            Ok(())
        }
        "durum" => {
            args_ok(0)?;
            let batarya = battery()?;
            let (answered, total) = lubot::soru::durum(&answers_path, batarya.len());
            let md = format!("# Soru durum\n\n{answered} of {total} answered\n");
            lubot::validate_output(md.as_bytes(), "soru durum")?;
            println!("{md}");
            Ok(())
        }
        other => Err(format!("unknown soru action `{other}`\n{}", usage())),
    }
}

fn cmd_risk(args: &[String]) -> Result<(), String> {
    let (found, leftover) = flags(args, &["--text", "--path"]);
    if !leftover.is_empty() {
        return Err(format!("risk: unexpected `{}`", leftover[0]));
    }
    let mut hits: Vec<(String, usize, &'static str)> = Vec::new();
    if let Some(text) = one(&found, "--text") {
        for (line, reason) in lubot_tools::command_risk::classify_lines(&text) {
            hits.push(("(text)".to_string(), line, reason));
        }
    }
    if let Some(path) = one(&found, "--path") {
        let body = std::fs::read_to_string(&path).map_err(|e| format!("risk: {path}: {e}"))?;
        for (line, reason) in lubot_tools::command_risk::classify_lines(&body) {
            hits.push((format!("{path}:{line}"), line, reason));
        }
    }
    if hits.is_empty() {
        let md = "# Command risk\n\nNo risky shape found.\n";
        lubot::validate_output(md.as_bytes(), "risk")?;
        println!("{md}");
        return Ok(());
    }
    let mut doc = String::from("# Command risk\n\n| where | line | reason |\n|---|---|---|\n");
    for (where_, line, reason) in &hits {
        doc.push_str(&format!("| {where_} | {line} | {reason} |\n"));
    }
    lubot::validate_output(doc.as_bytes(), "risk")?;
    println!("{doc}");
    Err(format!("{} risky command shape(s) found", hits.len()))
}

fn cmd_doc(args: &[String]) -> Result<(), String> {
    let (found, leftover) = flags(
        args,
        &[
            "--in",
            "--origin",
            "--licence",
            "--attribution",
            "--asset-id",
            "--kind",
            "--out",
        ],
    );
    if !leftover.is_empty() {
        return Err(format!("doc: unexpected `{}`", leftover[0]));
    }
    let input = PathBuf::from(
        one(&found, "--in").ok_or_else(|| format!("doc: missing --in\n{}", usage()))?,
    );
    let licence =
        one(&found, "--licence").ok_or_else(|| format!("doc: missing --licence\n{}", usage()))?;
    if !lubot::ALLOWED_LICENCES.contains(&licence.as_str()) {
        return Err(format!(
            "doc: licence `{licence}` is outside the allowed set"
        ));
    }
    let attribution = one(&found, "--attribution")
        .ok_or_else(|| format!("doc: missing --attribution\n{}", usage()))?;
    let asset_id =
        one(&found, "--asset-id").ok_or_else(|| format!("doc: missing --asset-id\n{}", usage()))?;
    let origin = one(&found, "--origin").unwrap_or_else(|| input.to_string_lossy().to_string());
    let default_kind = if input
        .extension()
        .map(|e| matches!(e.to_str(), Some("md" | "txt")))
        .unwrap_or(false)
    {
        "markdown"
    } else {
        "doc"
    };
    let kind = one(&found, "--kind").unwrap_or_else(|| default_kind.to_string());
    let bytes = std::fs::read(&input).map_err(|e| format!("doc: {}: {e}", input.display()))?;
    let is_pdf = input
        .extension()
        .map(|e| e.eq_ignore_ascii_case("pdf"))
        .unwrap_or(false);
    let text = if is_pdf {
        lubot_doc::pdf_text(&bytes)?
    } else {
        String::from_utf8(bytes).map_err(|e| format!("doc: not UTF-8 text: {e}"))?
    };
    let records = lubot::doc_records(
        &text,
        &origin,
        "doc",
        &licence,
        &attribution,
        &asset_id,
        &kind,
    )?;
    let mut summary = format!(
        "# Doc\n\n{} records from {} ({}, licence: {})\n",
        records.len(),
        input.display(),
        origin,
        licence
    );
    if let Some(out) = one(&found, "--out") {
        lubot::write_records_gz(PathBuf::from(&out).as_path(), &records)?;
        summary.push_str(&format!("\nWrote {out}\n"));
    } else {
        summary.push_str("\nNo --out: records were validated but not written.\n");
    }
    lubot::validate_output(summary.as_bytes(), "doc")?;
    println!("{summary}");
    Ok(())
}

fn cmd_queue(args: &[String]) -> Result<(), String> {
    let Some(action) = args.first() else {
        return Err(format!("missing queue action\n{}", usage()));
    };
    let rest = &args[1..];
    let queue_path = |found: &[(String, String)]| -> PathBuf {
        one(found, "--file")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("lubot-kuyruk.jsonl"))
    };
    match action.as_str() {
        "add" => {
            let (found, leftover) = flags(rest, &["--corpus", "--reader", "--effort", "--file"]);
            let question = leftover.join(" ").trim().to_string();
            if question.is_empty() {
                return Err("queue add: question is empty".to_string());
            }
            let corpus_arg = one(&found, "--corpus")
                .ok_or_else(|| format!("queue add: missing --corpus\n{}", usage()))?;
            let reader = one(&found, "--reader")
                .ok_or_else(|| format!("queue add: missing --reader\n{}", usage()))?;
            let effort = Some(effort_required(&found, "queue add")?);
            let mut jobs = lubot::queue::load_queue(&queue_path(&found))?;
            let corpus_key = corpus_arg.to_string();
            if let Some(existing) = jobs.iter().find(|j| {
                j.is_pending() && j.question == question && j.corpus.join(",") == corpus_key
            }) {
                return Err(format!(
                    "queue add: duplicate pending job `{}`",
                    existing.job_id
                ));
            }
            let now = lubot::now_seconds()?;
            let job_id = format!(
                "{}-{}",
                &lubot_read::sha256_hex(question.as_bytes())[..12],
                jobs.len() + 1
            );
            let job = lubot::queue::Job::pending(
                &job_id,
                now,
                &question,
                corpus_arg
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .collect(),
                &reader,
                effort,
            );
            jobs.push(job);
            lubot::queue::save_queue(&queue_path(&found), &jobs)?;
            let md = format!(
                "# Queue\n\nAdded `{}` ({} pending)\n",
                job_id,
                jobs.iter().filter(|j| j.is_pending()).count()
            );
            lubot::validate_output(md.as_bytes(), "queue add")?;
            println!("{md}");
            Ok(())
        }
        "list" => {
            let (found, leftover) = flags(rest, &["--file"]);
            if !leftover.is_empty() {
                return Err(format!("queue list: unexpected `{}`", leftover[0]));
            }
            let jobs = lubot::queue::load_queue(&queue_path(&found))?;
            let mut md = String::from("# Queue\n\n| id | state | attempts | verdict | question |\n|---|---|---|---|---|\n");
            for job in &jobs {
                md.push_str(&format!(
                    "| {} | {:?} | {} | {} | {} |\n",
                    job.job_id,
                    job.state,
                    job.attempts,
                    job.verdict.as_deref().unwrap_or("-"),
                    job.question.replace('|', "/").replace('`', "'"),
                ));
            }
            let pending = jobs.iter().filter(|j| j.is_pending()).count();
            md.push_str(&format!("\n{pending} pending of {}\n", jobs.len()));
            lubot::validate_output(md.as_bytes(), "queue list")?;
            println!("{md}");
            Ok(())
        }
        "run" => {
            let (watch, kept) = flags_bool(rest, &["--watch"]);
            let (found, leftover) = flags(
                &kept,
                &[
                    "--file",
                    "--budget",
                    "--check",
                    "--audit",
                    "--outputs",
                    "--book",
                    "--poll",
                    "--idle",
                ],
            );
            if !leftover.is_empty() {
                return Err(format!("queue run: unexpected `{}`", leftover[0]));
            }
            let path = queue_path(&found);
            let budget: usize = match one(&found, "--budget") {
                Some(raw) => raw
                    .parse()
                    .map_err(|_| format!("--budget: `{raw}` is not a number"))?,
                None => 1_000_000,
            };
            let default_check = "python3 gates/check.py --all --self-test";
            let check_cmd = one(&found, "--check").unwrap_or_else(|| default_check.to_string());
            let poll_secs: u64 = match one(&found, "--poll") {
                Some(raw) => raw
                    .parse()
                    .map_err(|_| format!("--poll: `{raw}` is not a number"))?,
                None => 5,
            };
            let max_idle: u64 = match one(&found, "--idle") {
                Some(raw) => raw
                    .parse()
                    .map_err(|_| format!("--idle: `{raw}` is not a number"))?,
                None => 3,
            };
            let book_path = one(&found, "--book")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("lubot-grants.json"));
            let audit_path = one(&found, "--audit").map(PathBuf::from);
            let outputs_path = outputs_or_default(&found);
            let mut idle = 0u64;
            loop {
                let mut jobs = lubot::queue::load_queue(&path)?;
                if jobs.iter().all(|j| !j.is_pending()) {
                    if !watch || idle >= max_idle {
                        let md = format!(
                            "# Queue run\n\nNothing pending: {} job(s) in the queue.\n",
                            jobs.len()
                        );
                        lubot::validate_output(md.as_bytes(), "queue run")?;
                        println!("{md}");
                        return Ok(());
                    }
                    idle += 1;
                    std::thread::sleep(std::time::Duration::from_secs(poll_secs));
                    continue;
                }
                // One shared corpus cache per pass: each distinct corpus list
                // is loaded once, and one grant book carries the whole pass.
                let mut cache: std::collections::HashMap<String, lubot::LoadedCorpus> =
                    std::collections::HashMap::new();
                let mut grants = BookFile::load(&book_path)?.to_book();
                let mut check_pass = || -> Result<(), String> {
                    let out = std::process::Command::new("sh")
                        .arg("-c")
                        .arg(&check_cmd)
                        .output()
                        .map_err(|e| format!("check: {e}"))?;
                    if !out.status.success() {
                        return Err(format!(
                            "check exited {}: {}",
                            out.status,
                            String::from_utf8_lossy(&out.stderr).trim()
                        ));
                    }
                    Ok(())
                };
                let mut process = |job: &lubot::queue::Job| -> Result<(String, usize), String> {
                    let key = job.corpus.join(",");
                    let corpus = if !cache.contains_key(&key) {
                        let paths: Vec<PathBuf> = job.corpus.iter().map(PathBuf::from).collect();
                        let loaded = lubot::load_corpus(&paths)?;
                        cache.insert(key.clone(), loaded);
                        cache
                            .get(&key)
                            .ok_or_else(|| "corpus cache: miss after insert".to_string())?
                    } else {
                        cache
                            .get(&key)
                            .ok_or_else(|| "corpus cache: miss".to_string())?
                    };
                    lubot::ask_verdict(
                        corpus,
                        &job.reader,
                        &job.question,
                        &mut grants,
                        lubot::now_seconds()?,
                        job.effort.as_deref(),
                        audit_path.as_deref(),
                        outputs_path.as_deref(),
                    )
                };
                let report =
                    lubot::queue::run_queue(&mut jobs, budget, &mut process, &mut check_pass);
                lubot::queue::save_queue(&path, &jobs)?;
                let done = jobs
                    .iter()
                    .filter(|j| j.state == lubot::queue::JobState::Done)
                    .count();
                let failed = jobs
                    .iter()
                    .filter(|j| j.state == lubot::queue::JobState::Failed)
                    .count();
                let stalled = jobs
                    .iter()
                    .filter(|j| j.state == lubot::queue::JobState::Stalled)
                    .count();
                let pending = jobs.iter().filter(|j| j.is_pending()).count();
                let md = format!(
                    "# Queue run\n\n{done} done, {failed} failed (retried next run), {stalled} stalled, {pending} pending; halted: {}\n",
                    report.halted
                );
                lubot::validate_output(md.as_bytes(), "queue run")?;
                println!("{md}");
                if report.halted {
                    return Err("queue halted: the check failed".to_string());
                }
                if !watch {
                    return Ok(());
                }
                idle = 0;
            }
        }
        "ls" => {
            let (found, leftover) = flags(rest, &["--file"]);
            if !leftover.is_empty() {
                return Err(format!("queue ls: unexpected `{}`", leftover[0]));
            }
            let jobs = lubot::queue::load_queue(&queue_path(&found))?;
            let md = lubot::queue::list_md(&jobs);
            lubot::validate_output(md.as_bytes(), "queue ls")?;
            println!("{md}");
            Ok(())
        }
        "iptal" => {
            let (found, leftover) = flags(rest, &["--file"]);
            if leftover.len() != 1 {
                return Err(format!("queue iptal: 1 arguman bekleniyor\n{}", usage()));
            }
            let mut jobs = lubot::queue::load_queue(&queue_path(&found))?;
            let id = lubot::queue::cancel(&mut jobs, &leftover[0])?;
            lubot::queue::save_queue(&queue_path(&found), &jobs)?;
            let md = format!("# Kuyruk\n\n`{id}` iptal edildi.\n");
            lubot::validate_output(md.as_bytes(), "queue iptal")?;
            println!("{md}");
            Ok(())
        }
        "log" => {
            let (found, leftover) = flags(rest, &["--file", "--limit"]);
            if !leftover.is_empty() {
                return Err(format!("queue log: unexpected `{}`", leftover[0]));
            }
            let limit: usize = match one(&found, "--limit") {
                Some(raw) => raw
                    .parse()
                    .map_err(|_| format!("--limit: `{raw}` is not a number"))?,
                None => 30,
            };
            let jobs = lubot::queue::load_queue(&queue_path(&found))?;
            let mut entries: Vec<String> = Vec::new();
            for job in &jobs {
                for entry in &job.journal {
                    entries.push(format!("{} | {}", job.job_id, entry));
                }
            }
            let shown = entries.len().min(limit);
            let mut md = String::from("# Queue log\n\n");
            for entry in &entries[entries.len() - shown..] {
                md.push_str(&format!("- `{}`\n", entry.replace('`', "'")));
            }
            md.push_str(&format!("\n{shown} of {} entries\n", entries.len()));
            lubot::validate_output(md.as_bytes(), "queue log")?;
            println!("{md}");
            Ok(())
        }
        other => Err(format!("unknown queue action `{other}`\n{}", usage())),
    }
}

fn cmd_ratchet(args: &[String]) -> Result<(), String> {
    let (set, kept) = flags_bool(args, &["--set"]);
    let (found, leftover) = flags(&kept, &["--baseline"]);
    if !leftover.is_empty() {
        return Err(format!("ratchet: unexpected `{}`", leftover[0]));
    }
    let baseline_path = PathBuf::from(
        one(&found, "--baseline").unwrap_or_else(|| "training/ratchet.json".to_string()),
    );
    let baseline = lubot::ratchet::load(&baseline_path)?;
    let measured = measure_all()?;
    let diffs = lubot::ratchet::compare(&measured.as_baseline(), &baseline);
    let regressed: Vec<&lubot::ratchet::Diff> = diffs.iter().filter(|d| d.regressed).collect();
    let mut md =
        String::from("# Ratchet\n\n| measure | baseline | measured | |\n|---|---|---|---|\n");
    for diff in &diffs {
        md.push_str(&format!(
            "| {} | {} | {} | {} |\n",
            diff.field,
            diff.baseline,
            diff.measured,
            if diff.regressed { "GERILEME" } else { "ok" }
        ));
    }
    if set {
        let kept = lubot::ratchet::save_measured_keys(&baseline_path, &measured.as_baseline())?;
        md.push_str(&format!(
            "\nBaseline rewritten to the measurement ({}){}.\n",
            baseline_path.display(),
            if kept.is_empty() {
                String::new()
            } else {
                format!(
                    "; keys this program does not measure kept: {}",
                    kept.join(", ")
                )
            }
        ));
    } else if regressed.is_empty() {
        md.push_str("\nNo regression: every measured number holds its baseline.\n");
    } else {
        md.push_str(&format!(
            "\n{} regression(s): the baseline holds, the measurement does not.\n",
            regressed.len()
        ));
    }
    lubot::validate_output(md.as_bytes(), "ratchet")?;
    println!("{md}");
    if regressed.is_empty() {
        Ok(())
    } else {
        Err("ratchet: measurement regressed".to_string())
    }
}

fn measure_all() -> Result<lubot::ratchet::Measured, String> {
    Ok(lubot::ratchet::Measured {
        tests: lubot::ratchet::measure_tests("cargo")?,
        gates: lubot::ratchet::measure_gates("python3", "gates/check.py")?,
        pedantic: lubot::ratchet::measure_pedantic("cargo")?,
        corpus: lubot::ratchet::measure_corpus(&PathBuf::from("corpus"))?,
    })
}

fn cmd_envanter(args: &[String]) -> Result<(), String> {
    let (found, leftover) = flags(args, &["--corpus-dir"]);
    if !leftover.is_empty() {
        return Err(format!("envanter: unexpected `{}`", leftover[0]));
    }
    let corpus_dir =
        PathBuf::from(one(&found, "--corpus-dir").unwrap_or_else(|| "corpus".to_string()));
    let mut crate_names: Vec<String> = Vec::new();
    let mut rs_files = 0usize;
    let mut loc = 0usize;
    if let Ok(entries) = std::fs::read_dir("crates") {
        for entry in entries.flatten() {
            let dir = entry.path();
            if dir.join("Cargo.toml").is_file() {
                if let Some(name) = dir.file_name().and_then(|n| n.to_str()) {
                    crate_names.push(name.to_string());
                }
            }
        }
    }
    walk_rs(&PathBuf::from("crates"), &mut rs_files, &mut loc);
    crate_names.sort();
    let measured = measure_all()?;
    let corpus_count = lubot::ratchet::measure_corpus(&corpus_dir)?;
    let mut md = String::from("# Envanter\n\n");
    md.push_str("| olcum | deger |\n|---|---|\n");
    md.push_str(&format!("| crates | {} |\n", crate_names.len()));
    md.push_str(&format!("| Rust dosyasi | {rs_files} |\n"));
    md.push_str(&format!("| Rust LOC | {loc} |\n"));
    md.push_str(&format!("| test (olcum) | {} |\n", measured.tests));
    md.push_str(&format!("| kapi (olcum) | {} |\n", measured.gates));
    md.push_str(&format!("| corpus kaydi | {corpus_count} |\n"));
    md.push_str(&format!("| pedantic uyari | {} |\n", measured.pedantic));
    md.push_str("\nKok: `");
    md.push_str(&crate_names.join("`, `"));
    md.push_str("`\n");
    lubot::validate_output(md.as_bytes(), "envanter")?;
    println!("{md}");
    Ok(())
}

fn walk_rs(dir: &Path, files: &mut usize, loc: &mut usize) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk_rs(&path, files, loc);
        } else if path.extension().map(|e| e == "rs").unwrap_or(false) {
            *files += 1;
            if let Ok(body) = std::fs::read_to_string(&path) {
                *loc += body.lines().count();
            }
        }
    }
}

fn cmd_it(args: &[String]) -> Result<(), String> {
    let (dry_run, kept) = flags_bool(args, &["--dry-run"]);
    let (found, leftover) = flags(&kept, &["-m", "--message", "--path", "--branch"]);
    if !leftover.is_empty() {
        return Err(format!("it: unexpected `{}`", leftover[0]));
    }
    let message = one(&found, "-m")
        .or_else(|| one(&found, "--message"))
        .ok_or_else(|| format!("it: missing -m\n{}", usage()))?;
    if message.trim().is_empty() {
        return Err("it: message is empty".to_string());
    }
    let mut paths: Vec<PathBuf> = Vec::new();
    for (name, value) in &found {
        if name == "--path" {
            paths.push(PathBuf::from(value));
        }
    }
    if paths.is_empty() {
        return Err("it: at least one --path is required".to_string());
    }
    for path in &paths {
        if !path.exists() {
            return Err(format!("it: {} does not exist", path.display()));
        }
    }
    let branch = one(&found, "--branch");
    let mut md = String::from("# It\n\n");
    md.push_str(&format!("message: `{}`\n\n", message.replace('`', "'")));
    md.push_str("paths:\n\n");
    for path in &paths {
        md.push_str(&format!("- {}\n", path.display()));
    }
    if dry_run {
        md.push_str("\nDry run: nothing staged, committed or pushed.\n");
        lubot::validate_output(md.as_bytes(), "it")?;
        println!("{md}");
        return Ok(());
    }
    let branch = match branch {
        Some(b) => b,
        None => git_stdout(&["rev-parse", "--abbrev-ref", "HEAD"])?,
    };
    // Only the listed paths enter the commit; everything else stays exactly
    // where it was - the tree is never swept or shifted.
    let mut add_args = vec!["add", "--"];
    for path in &paths {
        add_args.push(path.to_str().ok_or("it: path is not UTF-8")?);
    }
    git_run(&add_args)?;
    let mut commit_args = vec!["commit", "-m", message.as_str(), "--"];
    for path in &paths {
        commit_args.push(path.to_str().ok_or("it: path is not UTF-8")?);
    }
    git_run(&commit_args)?;
    let pushed = git_stdout(&["push", "origin", branch.as_str()])?;
    md.push_str(&format!("\nPushed to `{branch}`:\n\n```\n{pushed}\n```\n"));
    lubot::validate_output(md.as_bytes(), "it")?;
    println!("{md}");
    Ok(())
}

fn git_run(args: &[&str]) -> Result<(), String> {
    let out = std::process::Command::new("git")
        .args(args)
        .output()
        .map_err(|e| format!("git: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "git {}: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(())
}

fn git_stdout(args: &[&str]) -> Result<String, String> {
    let out = std::process::Command::new("git")
        .args(args)
        .output()
        .map_err(|e| format!("git: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "git {}: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Trainer self-check: does the from-scratch core forward, back-propagate and
/// actually descend?
///
/// Everything printed here is measured at runtime. The descent runs on a short
/// in-memory token sequence, so it is evidence that the training path works -
/// it is not a corpus measurement and is not reported as one. The gradient
/// itself is checked against finite differences in `lubot-egitim`'s own tests,
/// not here.
/// Apply the frozen vocab to the training corpus and print the ids.
///
/// This exists so the Rust tokenizer can be cross-checked against the Python
/// one that cut the vocab: same file, same records, ids compared one by one.
/// Two tokenizers that agree by convention is not an agreement.
/// Measure the corpus against the spec's window length.
///
/// The spec's `max_seq_len` was chosen from the *surface* corpus (p95 ≈ 246).
/// This measures the corpus the model will actually train on and says plainly
/// whether that number still holds. It does not adjust anything: a spec whose
/// assumption is falsified is a finding, not something to patch in passing.
fn cmd_egitim_veri(args: &[String]) -> Result<(), String> {
    let mut korpus_yolu: Option<String> = None;
    let mut vocab_yolu = String::from("training/tokenizer/lubot-bpe-v2.json");
    let mut uzunluk: Option<usize> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--corpus" => {
                i += 1;
                korpus_yolu = args.get(i).cloned();
            }
            "--vocab" => {
                i += 1;
                vocab_yolu = args.get(i).cloned().unwrap_or(vocab_yolu);
            }
            "--uzunluk" => {
                i += 1;
                uzunluk = Some(
                    args.get(i)
                        .and_then(|v| v.parse::<usize>().ok())
                        .ok_or_else(|| "--uzunluk bir sayi istiyor".to_string())?,
                );
            }
            other => {
                return Err(format!(
                    "egitim-veri: bilinmeyen secenek {other}\n{}",
                    usage()
                ))
            }
        }
        i += 1;
    }
    let korpus_yolu = korpus_yolu.ok_or_else(|| format!("--corpus zorunlu\n{}", usage()))?;
    let spec = lubot_egitim::Spec::lubot_a1();
    let uzunluk = uzunluk.unwrap_or(spec.max_seq_len);

    // Spec'in beyani dosyadan okunur ve Rust sabitiyle karsilastirilir.
    let spec_metin = std::fs::read_to_string("training/model_spec.json")
        .map_err(|e| format!("model_spec.json okunamadi: {e}"))?;
    let spec_json: serde_json::Value = serde_json::from_str(&spec_metin)
        .map_err(|e| format!("model_spec.json JSON degil: {e}"))?;
    let beyan = spec_json["max_seq_len"]
        .as_u64()
        .ok_or_else(|| "model_spec.json: max_seq_len yok".to_string())? as usize;
    if beyan != spec.max_seq_len {
        return Err(format!(
            "spec beyani {} ama lubot-egitim::Spec {} diyor: iki yer anlasmıyor",
            beyan, spec.max_seq_len
        ));
    }

    let sozluk = lubot_jeton::Sozluk::yukle(std::path::Path::new(&vocab_yolu))
        .map_err(|e| format!("sozluk reddedildi: {e}"))?;
    let dosya = std::fs::File::open(&korpus_yolu)
        .map_err(|e| format!("korpus acilamadi: {korpus_yolu} ({e})"))?;
    let okuyucu: Box<dyn std::io::BufRead> = if std::path::Path::new(&korpus_yolu)
        .extension()
        .is_some_and(|e| e == "gz")
    {
        Box::new(std::io::BufReader::new(flate2::read::GzDecoder::new(dosya)))
    } else {
        Box::new(std::io::BufReader::new(dosya))
    };
    let mut sayilar: Vec<usize> = Vec::new();
    let mut diziler: Vec<Vec<u32>> = Vec::new();
    for (sira, satir) in std::io::BufRead::lines(okuyucu).enumerate() {
        let satir = satir.map_err(|e| format!("korpus okunamadi ({sira}): {e}"))?;
        let satir = satir.trim();
        if satir.is_empty() {
            continue;
        }
        let deger: serde_json::Value = serde_json::from_str(satir)
            .map_err(|e| format!("korpus kaydi {sira} JSON degil: {e}"))?;
        let metin = deger["text"]
            .as_str()
            .ok_or_else(|| format!("korpus kaydi {sira}: `text` alani yok"))?;
        let kimlikler = sozluk.kodla(metin);
        sayilar.push(kimlikler.len());
        diziler.push(kimlikler);
    }
    let rapor: lubot_egitim::PencereRaporu = lubot_egitim::pencere_olcu(&sayilar, uzunluk)
        .map_err(|e| {
            format!(
                "pencere olcumu reddedildi: {}",
                match e {
                    lubot_egitim::PencereHatasi::SifirUzunluk => "pencere uzunlugu sifir",
                    lubot_egitim::PencereHatasi::BosKorpus => "korpus bos",
                }
            )
        })?;

    let (paketler, paket_raporu): (Vec<lubot_egitim::PaketPencere>, lubot_egitim::PaketRaporu) =
        lubot_egitim::paketle(&diziler, uzunluk).map_err(|e| {
            // Reddin sebebi adıyla söylenir; `{e:?}` reddi bir hata
            // ayıklama dizesine çevirir, sebebi söylemez.
            format!(
                "paketleme reddedildi: {}",
                match e {
                    lubot_egitim::PaketHatasi::SifirUzunluk => "pencere uzunlugu sifir",
                    lubot_egitim::PaketHatasi::BosKorpus => "paketlenecek kayit yok",
                }
            )
        })?;
    let mut md = String::from("# Egitim veri yolu\n\n| olcu | deger |\n|---|---|\n");
    md.push_str(&format!(
        "| korpus | {} kayit, {} jeton |\n",
        rapor.kayit, rapor.toplam_jeton
    ));
    md.push_str(&format!(
        "| kayit uzunlugu (jeton) | p50 {}, p95 {}, p99 {}, en uzun {} |\n",
        rapor.p50, rapor.p95, rapor.p99, rapor.en_uzun
    ));
    md.push_str(&format!(
        "| pencere | uzunluk {}, {} tam pencere, {} jeton kapsandi, {} jeton artik kuyruklarda |\n",
        uzunluk, rapor.pencere, rapor.kapsanan_jeton, rapor.artan_jeton
    ));
    let kayit_kapsama = 100.0 * rapor.kapsanan_jeton as f64 / rapor.toplam_jeton as f64;
    let paket_kapsama = 100.0 * (rapor.paket_pencere * uzunluk) as f64 / rapor.toplam_jeton as f64;
    md.push_str(&format!(
        "| kapsama | kayit basina pencereleme {:.4}% ({} jeton atilir); paketleme {:.4}% ({} jeton atilir) |\n",
        kayit_kapsama,
        rapor.artan_jeton,
        paket_kapsama,
        rapor.paket_artan
    ));
    if paket_kapsama - kayit_kapsama > 1.0 {
        md.push_str(&format!(
            "| bulgu | kayit basina pencereleme jetonlarin {:.2}%'ini atiyor: medyan kayit {} jeton, pencere {} jeton. Kayitlar arasi paketleme olmadan egitim korpusun kucuk bir parcasiyla kosar |\n",
            100.0 - kayit_kapsama,
            rapor.p50,
            uzunluk
        ));
    }
    let hukum = if rapor.p95 <= beyan {
        format!(
            "p95 {} <= spec'in max_seq_len beyani {}: beyan bu korpusta DURUYOR",
            rapor.p95, beyan
        )
    } else {
        format!(
            "p95 {} > spec'in max_seq_len beyani {}: beyan bu korpusta YANLIS, spec yeniden dogrulanmali",
            rapor.p95, beyan
        )
    };
    md.push_str(&format!("| hukum | {hukum} |\n"));
    md.push_str(&format!(
        "| paketleme | {} pencere: {} tek kaynakli, {} birden cok kaydi birlestiriyor (bir pencerede en cok {} kayit) |\n",
        paket_raporu.pencere,
        paket_raporu.tek_kaynakli,
        paket_raporu.cok_kaynakli,
        paket_raporu.en_cok_kaynak
    ));
    md.push_str(&format!(
        "| provenans | her pencere konum basina kaynak izi tasiyor ({} pencere, kimlik ve kaynak dizileri esit uzunlukta); alinti hangi kayda ait oldugunu kaybetmiyor |\n",
        paketler.len()
    ));
    // Paketli adim gerçek korpus verisiyle koşulur: pencere içindeki kayıt
    // sınırlarında dikkat kesiliyor mu, burada ölçülür. Tek adımın kaybı bir
    // korpus ölçümü değildir ve öyle raporlanmaz; raporlanan şey, maskenin
    // kaç konumda devreye girdiği ve adımın sonlu bir kayıp verdiği.
    let paket_spec = lubot_egitim::Spec {
        vocab: sozluk.boyut(),
        d_model: 16,
        n_layers: 2,
        n_heads: 2,
        d_ff: 32,
        max_seq_len: uzunluk,
    };
    paket_spec.dogrula().map_err(|e| {
        format!(
            "paketli adim spec reddedildi: {}",
            match e {
                lubot_egitim::SpecHatasi::BosBoyut => "sifir boyutlu bir eksen",
                lubot_egitim::SpecHatasi::BasSayisiBolmuyor => "bas sayisi genisligi bolmuyor",
            }
        )
    })?;
    let paket_param = lubot_egitim::Parametreler::belirgin_doldur(paket_spec, 20_260_923);
    let pencere = paketler
        .first()
        .ok_or_else(|| "paketleme pencere uretmedi".to_string())?;
    let girdi: Vec<usize> = pencere.kimlikler.iter().map(|k| *k as usize).collect();
    let hedef: Vec<usize> = (1..pencere.kimlikler.len())
        .map(|i| pencere.kimlikler[i] as usize)
        .chain(std::iter::once(pencere.kimlikler[0] as usize))
        .collect();
    let (paket_kayip, _) = lubot_egitim::ileri_ve_geri_paket(
        paket_spec,
        &paket_param,
        &girdi,
        &hedef,
        &pencere.kaynak,
    );
    if !paket_kayip.is_finite() {
        return Err(format!("paketli adim sonlu kayip vermedi: {paket_kayip}"));
    }
    let sinir = pencere
        .kaynak
        .windows(2)
        .filter(|cift| cift[0] != cift[1])
        .count();
    md.push_str(&format!(
        "| paketli adim | ilk pencere: {} jeton, {} kayit siniri maskelendi, kayip {:.6} (tek adim, korpus olcumu degil) |\n",
        pencere.kimlikler.len(),
        sinir,
        paket_kayip
    ));
    md.push_str(
        "| yontem | yuzdelikler en yakin-rank (rank = ceil(p*n)); spec beyani dosyadan okundu ve Rust sabitiyle karsilastirildi |\n",
    );
    lubot::validate_output(md.as_bytes(), "egitim-veri")?;
    print!("{md}");
    Ok(())
}

fn cmd_jetonla(args: &[String]) -> Result<(), String> {
    let mut vocab_yolu: Option<String> = None;
    let mut korpus_yolu: Option<String> = None;
    let mut limit: usize = 250;
    let mut tam = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--vocab" => {
                i += 1;
                vocab_yolu = args.get(i).cloned();
            }
            "--corpus" => {
                i += 1;
                korpus_yolu = args.get(i).cloned();
            }
            "--limit" => {
                i += 1;
                limit = args
                    .get(i)
                    .and_then(|v| v.parse::<usize>().ok())
                    .ok_or_else(|| "--limit bir sayi istiyor".to_string())?;
            }
            "--tam" => tam = true,
            other => return Err(format!("jetonla: bilinmeyen secenek {other}\n{}", usage())),
        }
        i += 1;
    }
    let vocab_yolu = vocab_yolu.ok_or_else(|| format!("--vocab zorunlu\n{}", usage()))?;
    let korpus_yolu = korpus_yolu.ok_or_else(|| format!("--corpus zorunlu\n{}", usage()))?;
    let sozluk = lubot_jeton::Sozluk::yukle(std::path::Path::new(&vocab_yolu)).map_err(|e| {
        // Reddin sebebi adıyla söylenir: "yaklaşık olarak uyguladım" ile
        // "uygulayamam" aynı cümleyle geçiştirilemez.
        let tur = match e {
            lubot_jeton::SozlukHatasi::DesenDesteklenmiyor(_) => "desen-desteklenmiyor",
            lubot_jeton::SozlukHatasi::DagBozuk { .. } => "birlestirme-dag-bozuk",
            lubot_jeton::SozlukHatasi::BoyutUyusmuyor { .. } => "boyut-uyusmuyor",
            lubot_jeton::SozlukHatasi::BilinmeyenBicim(_) => "bilinmeyen-bicim",
            lubot_jeton::SozlukHatasi::BirlestirmeBicimiBozuk(_) => "birlestirme-bicimi-bozuk",
            lubot_jeton::SozlukHatasi::EksikAlan(_) => "eksik-alan",
            lubot_jeton::SozlukHatasi::BozukJson(_) => "bozuk-json",
            lubot_jeton::SozlukHatasi::Yok(_) => "sozluk-yok",
        };
        format!("sozluk reddedildi [{tur}]: {e}")
    })?;

    let dosya = std::fs::File::open(&korpus_yolu)
        .map_err(|e| format!("korpus acilamadi: {korpus_yolu} ({e})"))?;
    let okuyucu: Box<dyn std::io::BufRead> = if std::path::Path::new(&korpus_yolu)
        .extension()
        .is_some_and(|e| e == "gz")
    {
        Box::new(std::io::BufReader::new(flate2::read::GzDecoder::new(dosya)))
    } else {
        Box::new(std::io::BufReader::new(dosya))
    };

    let mut cikti = String::new();
    let mut kayit = 0usize;
    let mut toplam_jeton = 0usize;
    let mut toplam_on_jeton = 0usize;
    for (sira, satir) in std::io::BufRead::lines(okuyucu).enumerate() {
        let satir = satir.map_err(|e| format!("korpus okunamadi ({sira}): {e}"))?;
        let satir = satir.trim();
        if satir.is_empty() {
            continue;
        }
        let deger: serde_json::Value = serde_json::from_str(satir)
            .map_err(|e| format!("korpus kaydi {sira} JSON degil: {e}"))?;
        let metin = deger["text"]
            .as_str()
            .ok_or_else(|| format!("korpus kaydi {sira}: `text` alani yok"))?;
        let kimlikler = sozluk.kodla(metin);
        toplam_on_jeton += lubot_jeton::on_token_sayisi(metin);
        let geri = sozluk
            .coz(&kimlikler)
            .map_err(|e| format!("kayit {sira} geri cozulemedi: {e}"))?;
        if geri != metin {
            return Err(format!(
                "kayit {sira}: kodla/coz kayipsiz degil ({} bayt -> {} bayt)",
                metin.len(),
                geri.len()
            ));
        }
        toplam_jeton += kimlikler.len();
        if kayit < limit {
            if tam {
                let liste: Vec<String> = kimlikler.iter().map(|k| k.to_string()).collect();
                cikti.push_str(&format!(
                    "{{\"i\":{sira},\"n\":{},\"ids\":[{}]}}\n",
                    kimlikler.len(),
                    liste.join(",")
                ));
            } else {
                cikti.push_str(&format!("{{\"i\":{sira},\"n\":{}}}\n", kimlikler.len()));
            }
        }
        kayit += 1;
    }
    eprintln!(
        "jetonla: {} kayit, {} on-jeton, {} jeton, sozluk {} ({} birlestirme, boyut {}, desen {})",
        kayit,
        toplam_on_jeton,
        toplam_jeton,
        sozluk.aile(),
        sozluk.birlestirme_sayisi(),
        sozluk.boyut(),
        lubot_jeton::DESTEKLENEN_DESEN
    );
    print!("{cikti}");
    Ok(())
}

fn cmd_egitim(args: &[String]) -> Result<(), String> {
    if !args.is_empty() {
        return Err(format!("egitim takes no arguments\n{}", usage()));
    }
    let spec = lubot_egitim::Spec::lubot_a1();
    spec.dogrula().map_err(|e| {
        format!(
            "egitim spec reddedildi: {}",
            match e {
                lubot_egitim::SpecHatasi::BosBoyut => "sifir boyutlu bir eksen",
                lubot_egitim::SpecHatasi::BasSayisiBolmuyor => "bas sayisi genisligi bolmuyor",
            }
        )
    })?;
    let tavan = lubot_grant::training::MAX_TRAINING_GRANT_EPOCHS;
    let mut md = String::from("# Egitim oz-denetimi\n\n| adim | sonuc |\n|---|---|\n");
    md.push_str(&format!(
        "| spec | {} parametre, {} katman, d_model {}, {} bas, vocab {} |\n",
        spec.parametre_sayisi(),
        spec.n_layers,
        spec.d_model,
        spec.n_heads,
        spec.vocab
    ));
    md.push_str(&format!(
        "| epoch tavani | {} (lubot-grant), istenen 1 -> {} |\n",
        tavan,
        lubot_egitim::epoch_butcesi(1)?
    ));
    md.push_str(&format!(
        "| tavan asimi | {} |\n",
        lubot_egitim::epoch_butcesi(tavan + 1)
            .map(|n| format!("KABUL EDILDI: {n}"))
            .unwrap_or_else(|e| format!("reddedildi ({e})"))
    ));

    // Kisa bir inis: kucuk bir spec, bellek ici dizi.
    let kucuk = lubot_egitim::Spec {
        vocab: 64,
        d_model: 16,
        n_layers: 2,
        n_heads: 2,
        d_ff: 32,
        max_seq_len: 16,
    };
    kucuk
        .dogrula()
        .map_err(|e| format!("oz-denetim spec reddedildi: {e:?}"))?;
    let mut p = lubot_egitim::Parametreler::belirgin_doldur(kucuk, 20_260_923);
    let girdi = [0usize, 7, 3, 11, 5, 1, 9, 2];
    let hedef = [7usize, 3, 11, 5, 1, 9, 2, 4];
    let (baslangic, _) = lubot_egitim::ileri_ve_geri(kucuk, &p, &girdi, &hedef);
    let adim_sayisi = 30u32;
    let mut embed_durum = lubot_egitim::Adamw::yeni(p.embedding.len(), 0.05, 0.1)?;
    let mut wq_durum = lubot_egitim::Adamw::yeni(p.wq.len(), 0.05, 0.1)?;
    let mut son = baslangic;
    for _ in 0..adim_sayisi {
        let (kayip, grad) = lubot_egitim::ileri_ve_geri(kucuk, &p, &girdi, &hedef);
        son = kayip;
        embed_durum.adim(&mut p.embedding, &grad.embedding, false)?;
        wq_durum.adim(&mut p.wq, &grad.wq, true)?;
    }
    if !son.is_finite() || son >= baslangic {
        return Err(format!(
            "egitim yolu kaybi dusurmedi: {baslangic:.6} -> {son:.6}"
        ));
    }
    md.push_str(&format!(
        "| inis | {adim_sayisi} adim, kayip {baslangic:.6} -> {son:.6} (bellek ici 8 jetonluk dizi, korpus olcumu degil) |\n"
    ));
    md.push_str(&format!(
        "| LayerNorm eps | {:.0e} (lubot-egitim::LN_EPS) |\n",
        lubot_egitim::LN_EPS
    ));
    md.push_str(&format!(
        "| gradyan denetimi | lubot-egitim testlerinde: her parametre sonlu farkla karsilastirilir, goreli tolerans {:.0e}, mutlak taban {:.0e} |\n",
        lubot_egitim::GRADIENT_CHECK_TOLERANCE,
        lubot_egitim::GRADIENT_CHECK_MUTLAK_TABAN
    ));
    md.push_str(
        "| olculmeyen | egitilmis kontrol noktasi yok (K6): sinav skoru, alinti dogrulugu ve kapisma sonucu bu komutun kapsami disinda |\n",
    );
    lubot::validate_output(md.as_bytes(), "egitim")?;
    print!("{md}");
    Ok(())
}

fn cmd_olc(args: &[String]) -> Result<(), String> {
    if !args.is_empty() {
        return Err(format!("olc takes no arguments\n{}", usage()));
    }
    let mut failures: Vec<String> = Vec::new();
    let mut md = String::from("# Olc\n\n| adim | sonuc |\n|---|---|\n");
    // 1. Biçim: rustfmt çalıştırılabilirse kontrol edilir; yoksa "ölçülmedi".
    let fmt = std::process::Command::new("cargo")
        .args(["fmt", "--check"])
        .output();
    match fmt {
        Ok(out) if out.status.success() => md.push_str("| fmt | ok |\n"),
        Ok(out) => {
            md.push_str(&format!(
                "| fmt | KIRMIZI: {} |\n",
                String::from_utf8_lossy(&out.stderr)
                    .lines()
                    .next()
                    .unwrap_or("")
                    .trim()
            ));
            failures.push("fmt".to_string());
        }
        Err(e) => md.push_str(&format!("| fmt | olculmedi: {e} |\n")),
    }
    // 2. clippy katı.
    let clippy = std::process::Command::new("cargo")
        .args([
            "clippy",
            "--workspace",
            "--all-targets",
            "--",
            "-D",
            "warnings",
        ])
        .output();
    match clippy {
        Ok(out) if out.status.success() => md.push_str("| clippy -D warnings | ok |\n"),
        Ok(_out) => {
            md.push_str("| clippy -D warnings | KIRMIZI |\n");
            failures.push("clippy".to_string());
        }
        Err(e) => md.push_str(&format!("| clippy | olculmedi: {e} |\n")),
    }
    // 3. Kapılar.
    let gates = std::process::Command::new("python3")
        .args(["gates/check.py", "--all", "--self-test"])
        .output();
    match gates {
        Ok(out) if out.status.success() => md.push_str("| kapilar | ok |\n"),
        Ok(_out) => {
            md.push_str("| kapilar | KIRMIZI |\n");
            failures.push("kapilar".to_string());
        }
        Err(e) => md.push_str(&format!("| kapilar | olculmedi: {e} |\n")),
    }
    // 4. Ratchet.
    match cmd_ratchet(&[]) {
        Ok(()) => md.push_str("| ratchet | ok |\n"),
        Err(_) => {
            md.push_str("| ratchet | KIRMIZI |\n");
            failures.push("ratchet".to_string());
        }
    }
    md.push_str(&format!(
        "\n{kirmizi} kirmizi adim, {toplam} adim\n",
        kirmizi = failures.len(),
        toplam = 4
    ));
    lubot::validate_output(md.as_bytes(), "olc")?;
    println!("{md}");
    if failures.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "olc: {} kirmizi adim: {}",
            failures.len(),
            failures.join(", ")
        ))
    }
}

fn cmd_durum(args: &[String]) -> Result<(), String> {
    if !args.is_empty() {
        return Err(format!("durum takes no arguments\n{}", usage()));
    }
    let branch = git_stdout(&["rev-parse", "--abbrev-ref", "HEAD"])?;
    let porcelain = git_stdout(&["status", "--porcelain"])?;
    let lines: Vec<&str> = porcelain.lines().filter(|l| !l.trim().is_empty()).collect();
    let mut md = format!("# Durum\n\nbranch: `{branch}`\n\n");
    if lines.is_empty() {
        md.push_str("Agac temiz: calisma agaci branch ile ayni.\n");
    } else {
        md.push_str(&format!("{} degisiklik:\n\n", lines.len()));
        for line in lines.iter().take(30) {
            md.push_str(&format!("- `{}`\n", line.replace('`', "'")));
        }
        if lines.len() > 30 {
            md.push_str(&format!("- ... {} daha\n", lines.len() - 30));
        }
    }
    lubot::validate_output(md.as_bytes(), "durum")?;
    println!("{md}");
    Ok(())
}

fn cmd_guvenlik(args: &[String]) -> Result<(), String> {
    let (found, leftover) = flags(args, &["--path"]);
    if !leftover.is_empty() {
        return Err(format!("guvenlik: unexpected `{}`", leftover[0]));
    }
    let targets: Vec<PathBuf> = found
        .iter()
        .filter(|(name, _)| name == "--path")
        .map(|(_, value)| PathBuf::from(value))
        .collect();
    let targets = if targets.is_empty() {
        vec![
            PathBuf::from("crates"),
            PathBuf::from("training"),
            PathBuf::from("gates"),
        ]
    } else {
        targets
    };
    let mut files: Vec<PathBuf> = Vec::new();
    for target in &targets {
        if target.is_file() {
            files.push(target.clone());
        } else if target.is_dir() {
            collect_text_files(target, &mut files);
        } else {
            return Err(format!("guvenlik: {} does not exist", target.display()));
        }
    }
    let mut hits: Vec<(String, lubot_tools::secrets::Hit)> = Vec::new();
    let mut scanned = 0usize;
    for path in &files {
        // The scanner never scans its own source: the shapes live there as
        // literals, and scanning them would be a tautology.
        if path.ends_with("src/secrets.rs") {
            continue;
        }
        let Ok(bytes) = std::fs::read(path) else {
            continue;
        };
        let Ok(text) = String::from_utf8(bytes) else {
            continue;
        };
        scanned += 1;
        for hit in lubot_tools::secrets::scan(&text) {
            hits.push((path.display().to_string(), hit));
        }
    }
    let mut md = String::from("# Guvenlik\n\n| file | line | kind | vendor |\n|---|---|---|---|\n");
    for (file, hit) in &hits {
        md.push_str(&format!(
            "| {file} | {} | {} | {} |\n",
            hit.line, hit.kind, hit.vendor
        ));
    }
    let found = hits.len();
    md.push_str(&format!(
        "\n{scanned} files scanned; {found} credential shape(s)\n"
    ));
    lubot::validate_output(md.as_bytes(), "guvenlik")?;
    println!("{md}");
    if found > 0 {
        Err(format!("guvenlik: {found} credential shape(s) in the tree"))
    } else {
        Ok(())
    }
}

fn collect_text_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let dir_name = path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            if ["target", ".git", "node_modules"].contains(&dir_name.as_str()) {
                continue;
            }
            collect_text_files(&path, out);
        } else {
            let keep = path
                .extension()
                .map(|e| {
                    matches!(
                        e.to_str(),
                        Some("rs" | "md" | "py" | "json" | "toml" | "txt" | "jsonl")
                    )
                })
                .unwrap_or(false);
            if keep {
                out.push(path);
            }
        }
    }
}

fn cmd_graf(args: &[String]) -> Result<(), String> {
    if !args.is_empty() {
        return Err(format!("graf takes no arguments\n{}", usage()));
    }
    let md = lubot::graph::repo_graph(&PathBuf::from("."))?;
    lubot::validate_output(md.as_bytes(), "graf")?;
    println!("{md}");
    Ok(())
}

fn cmd_dosya(args: &[String]) -> Result<(), String> {
    let (found, leftover) = flags(args, &["--path"]);
    if !leftover.is_empty() {
        return Err(format!("dosya: unexpected `{}`", leftover[0]));
    }
    let Some(path_raw) = one(&found, "--path") else {
        return Err(format!("dosya: missing --path\n{}", usage()));
    };
    let path = PathBuf::from(&path_raw);
    let bytes = std::fs::read(&path).map_err(|e| format!("dosya: {}: {e}", path.display()))?;
    let kind = lubot_read::file_kind::file_kind(&bytes);
    let route = lubot_read::file_kind::route(kind, &path_raw);
    let way = match &route {
        lubot_read::file_kind::Route::Doc => "doc (`lubot doc`)",
        lubot_read::file_kind::Route::Corpus => "corpus (`lubot corpus`)",
        lubot_read::file_kind::Route::Text => "text (read directly)",
        lubot_read::file_kind::Route::Refuse { reason } => {
            return refuse_dosya(&path_raw, kind, reason)
        }
    };
    let kind_label = format!("{kind:?}").to_lowercase();
    let md = format!("# Dosya\n\n| file | kind | route |\n|---|---|---|\n| {path_raw} | {kind_label} | {way} |\n");
    lubot::validate_output(md.as_bytes(), "dosya")?;
    println!("{md}");
    Ok(())
}

fn refuse_dosya(
    path: &str,
    kind: lubot_read::file_kind::FileKind,
    reason: &'static str,
) -> Result<(), String> {
    let kind_label = format!("{kind:?}").to_lowercase();
    let md = format!(
        "# Dosya\n\n| file | kind | route |\n|---|---|---|\n| {path} | {kind_label} | refuse: {reason} |\n"
    );
    lubot::validate_output(md.as_bytes(), "dosya")?;
    println!("{md}");
    Err(format!("dosya: refused: {reason}"))
}

fn cmd_prompt(args: &[String]) -> Result<(), String> {
    let (found, leftover) = flags(args, &["--path"]);
    if !leftover.is_empty() {
        return Err(format!("prompt: unexpected `{}`", leftover[0]));
    }
    let path = one(&found, "--path")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("training/system_prompt.md"));
    let text =
        std::fs::read_to_string(&path).map_err(|e| format!("prompt: {}: {e}", path.display()))?;
    // The system prompt itself is a Lubot output: it must pass the same
    // schema the answers pass, so a prompt that breaks the schema never
    // reaches a model.
    lubot::validate_output(text.as_bytes(), &format!("prompt {}", path.display()))?;
    println!("{text}");
    Ok(())
}

fn cmd_audit(args: &[String]) -> Result<(), String> {
    let (found, leftover) = flags(args, &["--path", "--limit"]);
    if !leftover.is_empty() {
        return Err(format!("audit: unexpected `{}`", leftover[0]));
    }
    let path = one(&found, "--path").ok_or("audit: missing --path")?;
    let limit = one(&found, "--limit")
        .map(|s| {
            s.parse::<usize>()
                .map_err(|e| format!("audit: --limit: {e}"))
        })
        .transpose()?;
    let text = std::fs::read_to_string(&path).map_err(|e| format!("audit: {e}"))?;
    let mut lines: Vec<&str> = text.lines().collect();
    if let Some(limit) = limit {
        if lines.len() > limit {
            lines = lines[lines.len() - limit..].to_vec();
        }
    }
    for line in lines {
        println!("{line}");
    }
    Ok(())
}
