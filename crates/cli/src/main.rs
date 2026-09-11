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
        "  lubot ratchet [--set] [--baseline training/ratchet.json]",
        "  lubot envanter [--corpus-dir f] [--at EPOCH]",
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
        "  lubot sikistir --path <f> [--igne desen ...] [--depo dir]",
        "  lubot sikistir --geri-getir <ozet-dosya> [--depo dir]",
        "  lubot ogren --log <f> [--ogren-dir outputs/ogren]",
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
        "ratchet" => cmd_ratchet(rest),
        "envanter" => cmd_envanter(rest),
        "it" => cmd_it(rest),
        "olc" => cmd_olc(rest),
        "durum" => cmd_durum(rest),
        "guvenlik" => cmd_guvenlik(rest),
        "graf" => cmd_graf(rest),
        "dosya" => cmd_dosya(rest),
        "soru" => cmd_soru(rest),
        "ara" => cmd_ara(rest),
        "indeks" => cmd_indeks(rest),
        "mufredat" => cmd_mufredat(rest),
        "karsilastir" => cmd_karsilastir(rest),
        "sikistir" => lubot::sikistir::cmd_sikistir(rest),
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
        md.push_str(&format!(
            "- `{}` (licence `{licence}`)\n\n```\n{}\n```\n\n",
            hit.citation(),
            hit.text.trim()
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
        lubot::ratchet::save(&baseline_path, &measured.as_baseline())?;
        md.push_str(&format!(
            "\nBaseline rewritten to the measurement ({}).\n",
            baseline_path.display()
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
    let (found, leftover) = flags(args, &["--corpus-dir", "--at"]);
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
    let at: u64 = match one(&found, "--at") {
        Some(s) => s
            .parse()
            .map_err(|_| format!("envanter: --at must be a u64, got `{s}`"))?,
        None => 0,
    };
    let reg = lubot::activation::load(&PathBuf::from(lubot::activation::DEFAULT_LEDGER))
        .map_err(|e| format!("envanter: {e}"))?;
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
    md.push_str(&format!(
        "| aktivasyon defteri ({at}) | {} |\n",
        lubot::activation::report(&reg, at)
    ));
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
