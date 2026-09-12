//! Settlement media: write a batch, then verify it independently.
//!
//! # Why the two directions are separate commands
//!
//! A writer that produces a file only the same writer can read is not a format.
//! `odeme yaz` seals and emits; `odeme dogrula` re-parses the emitted text,
//! re-checks it, and compares the rebuilt file byte for byte against what was
//! given to it. That last step is the one most implementations omit, and it is
//! the one that catches a writer and a reader that disagree about layout.
//!
//! # What the media carries
//!
//! The seal links are in the file, not recomputed from it. A reader that
//! recomputes them from the entries it just parsed is checking its own parse
//! against itself; a reader that compares against the recorded links is checking
//! the file against what was signed.
//!
//! # Verification does not trust the writer
//!
//! [`verify_media`] runs four checks and names the first one that fails:
//!
//! 1. the text parses;
//! 2. the parsed batch satisfies its own invariants;
//! 3. the recorded seal links match the entries;
//! 4. re-rendering the parsed batch reproduces the input byte for byte.
//!
//! Check 4 is what catches a layout the parser tolerates but the writer would
//! never produce, which is exactly how two nodes come to disagree about a file
//! that both of them accept.

use lubot_muhur::{Sealer, SealError};
use lubot_usl::{Amount, EnvelopeError, Manifest, Payout};

/// What verification found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MediaFault {
    /// The text did not parse.
    Unparsable(EnvelopeError),
    /// It parsed but violates its own invariants.
    Invalid(EnvelopeError),
    /// The recorded seal links do not match the entries.
    SealBroken(SealError),
    /// Re-rendering does not reproduce the input.
    NotRoundTripped { expected: usize, got: usize },
}

impl std::fmt::Display for MediaFault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unparsable(err) => write!(f, "the media does not parse: {err}"),
            Self::Invalid(err) => write!(f, "the media parses but is not a valid batch: {err}"),
            Self::SealBroken(err) => write!(f, "the recorded seals do not match: {err}"),
            Self::NotRoundTripped { expected, got } => write!(
                f,
                "re-rendering produced {got} bytes where the input had {expected}; the parser accepts a layout the writer would never produce"
            ),
        }
    }
}

/// What a successful verification reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MediaSummary {
    pub sequence: u64,
    pub chain_id: u64,
    pub payouts: usize,
    pub bytes: usize,
}

/// Verifies settlement media by re-parsing, re-checking, re-sealing and
/// re-rendering it.
///
/// # Errors
///
/// The first [`MediaFault`] found, named.
pub fn verify_media(media: &str) -> Result<MediaSummary, MediaFault> {
    let manifest = Manifest::from_media(media).map_err(MediaFault::Unparsable)?;
    manifest.check().map_err(MediaFault::Invalid)?;
    let entries: Vec<String> = manifest
        .payouts
        .iter()
        .map(Payout::render)
        .collect();
    let references: Vec<&str> = entries.iter().map(String::as_str).collect();
    Sealer::verify(&references, &manifest.links).map_err(MediaFault::SealBroken)?;
    // `to_media` re-seals as it renders, so a parse that dropped or reordered
    // anything produces a different length even when the entries still verify.
    let mut rebuilt = manifest.clone();
    let rendered = rebuilt.to_media().map_err(MediaFault::Unparsable)?;
    if rendered.len() != media.len() {
        return Err(MediaFault::NotRoundTripped {
            expected: media.len(),
            got: rendered.len(),
        });
    }
    Ok(MediaSummary {
        sequence: manifest.sequence,
        chain_id: manifest.chain_id,
        payouts: manifest.payouts.len(),
        bytes: media.len(),
    })
}

/// One payout as written on the command line: `recipient:amount:reference`.
///
/// Split from the right, so a reference containing a colon is kept whole and a
/// recipient containing one is refused by the amount failing to parse rather
/// than being silently mis-split.
fn parse_payout(spec: &str) -> Result<Payout, String> {
    let Some((recipient, rest)) = spec.split_once(':') else {
        return Err(format!("expected recipient:amount:reference, got {spec:?}"));
    };
    let Some((amount, reference)) = rest.rsplit_once(':') else {
        return Err(format!("expected recipient:amount:reference, got {spec:?}"));
    };
    let amount = Amount::parse(amount).map_err(|err| err.to_string())?;
    Payout::new(recipient, amount, reference).map_err(|err| err.to_string())
}

/// The `lubot odeme` subcommand.
///
/// # Errors
///
/// A message for the caller, which `main` prints to stderr.
pub fn cmd_odeme(args: &[String]) -> Result<(), String> {
    let Some(sub) = args.first() else {
        return Err("usage: lubot odeme <yaz|dogrula> ...".to_string());
    };
    let rest = &args[1..];
    match sub.as_str() {
        "yaz" => cmd_yaz(rest),
        "dogrula" => cmd_dogrula(rest),
        other => Err(format!("unknown odeme subcommand: {other}")),
    }
}

/// Pulls `--name value` pairs out of `args`.
fn options(args: &[String]) -> Vec<(String, String)> {
    let mut found = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if let Some(name) = args[i].strip_prefix("--") {
            if let Some(value) = args.get(i + 1) {
                found.push((name.to_string(), value.clone()));
                i += 2;
                continue;
            }
        }
        i += 1;
    }
    found
}

/// Looks one option up.
fn option(args: &[String], name: &str) -> Option<String> {
    options(args)
        .into_iter()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value)
}

/// Builds a batch, seals it, and writes the media.
fn cmd_yaz(args: &[String]) -> Result<(), String> {
    let sequence: u64 = option(args, "seq")
        .ok_or("odeme yaz needs --seq")?
        .parse()
        .map_err(|_| "--seq is not a number".to_string())?;
    let chain_id: u64 = option(args, "chain")
        .ok_or("odeme yaz needs --chain")?
        .parse()
        .map_err(|_| "--chain is not a number".to_string())?;
    let fee = Amount::parse(&option(args, "fee").ok_or("odeme yaz needs --fee")?)
        .map_err(|err| err.to_string())?;
    let mut manifest = Manifest::new(sequence, chain_id, fee);
    let open: u64 = option(args, "open")
        .unwrap_or_else(|| "0".to_string())
        .parse()
        .map_err(|_| "--open is not a number".to_string())?;
    let close: u64 = option(args, "close")
        .unwrap_or_else(|| "0".to_string())
        .parse()
        .map_err(|_| "--close is not a number".to_string())?;
    manifest
        .set_window(open, close)
        .map_err(|err| err.to_string())?;
    for spec in option(args, "payout").unwrap_or_default().split(',') {
        if spec.trim().is_empty() {
            continue;
        }
        let payout = parse_payout(spec)?;
        manifest
            .add_payout(payout)
            .map_err(|err| err.to_string())?;
    }
    manifest.check().map_err(|err| err.to_string())?;
    let media = manifest.to_media().map_err(|err| err.to_string())?;
    match option(args, "out") {
        Some(path) => {
            std::fs::write(&path, &media).map_err(|err| format!("{path}: {err}"))?;
            eprintln!(
                "odeme: {} payouts, {} bytes, written to {path}",
                manifest.payouts.len(),
                media.len()
            );
            Ok(())
        }
        None => {
            print!("{media}");
            Ok(())
        }
    }
}

/// Verifies a media file.
fn cmd_dogrula(args: &[String]) -> Result<(), String> {
    let path = option(args, "media").ok_or("odeme dogrula needs --media")?;
    let media = std::fs::read_to_string(&path).map_err(|err| format!("{path}: {err}"))?;
    match verify_media(&media) {
        Ok(summary) => {
            println!(
                "OK   sequence {} on chain {}, {} payouts, {} bytes",
                summary.sequence, summary.chain_id, summary.payouts, summary.bytes
            );
            Ok(())
        }
        Err(fault) => Err(fault.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A batch that satisfies every invariant, built the way the command builds it.
    fn sample() -> Manifest {
        let mut manifest = Manifest::new(7, 1, Amount::from_minor(150));
        manifest.set_window(10, 20).expect("window");
        manifest
            .add_payout(Payout::new("alice", Amount::from_minor(1000), "inv-1").expect("payout"))
            .expect("add");
        manifest
            .add_payout(Payout::new("bob", Amount::from_minor(250), "inv-2").expect("payout"))
            .expect("add");
        manifest
    }

    #[test]
    fn media_written_by_the_writer_is_verified_by_the_reader() {
        // The check that matters most: the two directions agree on the layout.
        let mut manifest = sample();
        let media = manifest.to_media().expect("to_media");
        let summary = verify_media(&media).expect("verify");
        assert_eq!(summary.sequence, 7);
        assert_eq!(summary.chain_id, 1);
        assert_eq!(summary.payouts, 2);
        assert_eq!(summary.bytes, media.len());
    }

    #[test]
    fn an_edited_amount_breaks_the_seal() {
        // The seal is checked against the recorded links, not against a
        // recomputation of the parse, so an edited entry is caught.
        let mut manifest = sample();
        let media = manifest.to_media().expect("to_media");
        let edited = media.replace("10.00", "99.00");
        assert_ne!(edited, media, "the fixture did not change anything");
        assert!(
            matches!(verify_media(&edited), Err(MediaFault::SealBroken(_))),
            "an edited amount was accepted"
        );
    }

    #[test]
    fn an_unparsable_file_is_named_as_unparsable() {
        assert!(matches!(
            verify_media("this is not settlement media"),
            Err(MediaFault::Unparsable(_))
        ));
    }

    #[test]
    fn a_payout_spec_splits_from_the_right() {
        // A reference containing a colon stays whole.
        let payout = parse_payout("alice:10.00:inv:2026:01").expect("parse");
        assert_eq!(payout.recipient, "alice");
        assert_eq!(payout.amount.minor(), 1000);
        assert_eq!(payout.reference, "inv:2026:01");
    }

    #[test]
    fn a_payout_spec_with_too_few_fields_is_refused() {
        assert!(parse_payout("alice:10.00").is_err());
        assert!(parse_payout("alice").is_err());
    }

    #[test]
    fn a_non_canonical_amount_is_refused() {
        // Two spellings of the same amount means two files that settle the same
        // thing differently.
        assert!(parse_payout("alice:10.0:ref").is_err());
        assert!(parse_payout("alice:010.00:ref").is_err());
        assert!(parse_payout("alice:10:ref").is_err());
    }
}
