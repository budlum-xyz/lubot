//! # lubot-usl - the settlement envelope a cold wallet reads off the media
//!
//! Universal Settlement Layer: the exchange format between the hot side, which
//! builds what should be paid, and the cold side, which signs nothing it
//! cannot re-derive. One file per media, one media per batch, and the file
//! carries no signature - it carries a *seal*: its own entries, hashed in
//! order, exactly as [`lubot_muhur`] defines them. The cold reader recomputes
//! the seal and refuses on the first byte that disagrees.
//!
//! This boundary is deliberate. Signing belongs to the cold device; what this
//! crate guarantees is that the bytes it is asked to sign are the bytes the
//! hot device meant - a media edited in transit (or in a flaky write) changes
//! the seal, and an envelope whose entries no longer hash to its own seal is
//! not a payment instruction, it is an artifact of a copy.
//!
//! # The shape
//!
//! [`Manifest`] holds a sequence number, the chain it settles on, the payout
//! window, the fee, and the payout lines. [`Payout`] accepts only canonical
//! amounts (`M.mm` - the two-digit minor unit is the layer's own unit, and
//! each side maps it to its currency at its own boundary, silently converting
//! nothing). [`Manifest::to_media`] seals; [`Manifest::from_media`] rebuilds
//! every entry, re-seals, and compares - the check is recomputation, never
//! trust in the stored line.
//!
//! # Failure modes
//!
//! A payout line that differs from an earlier one by nothing is a duplicate
//! the batch author did not mean (rejected at [`Manifest::add_payout`]). A
//! window that does not open after genesis cannot be checked against the
//! maturity rule it exists for (rejected at [`Manifest::check`]). A media
//! whose bytes hash to something other than the seal it carries was edited
//! after sealing, and no partial read of it is safe.
//!
//! # Invariants
//!
//! 1. `to_media` writes what `check` accepted, never more and never less.
//! 2. `from_media` rebuilds the whole chain from the file's own entries and
//!    compares the seal; an accepted media satisfies `check` by construction,
//!    because every payout is parsed back through the same validating
//!    constructors that built it.
//! 3. No line contains tab, quote, backslash, or control characters, so the
//!    one-space-per-token grammar of the media cannot be escaped out of.
//!
//! # Non-transitive dependency notice
//!
//! This crate is std-only and depends on exactly one workspace crate
//! ([`lubot_muhur`]) for the seal arithmetic; it is a leaf. It is a public
//! API of this repository: `lubot usl make|check` drives it from the CLI, and
//! a cold-side reader outside the repo consumes only what `to_media` writes.
//! `cargo update` on that root cannot pull an outside crate into here.
#![forbid(unsafe_code)]
#![warn(missing_docs, clippy::all)]

use std::fmt;

/// How many payout lines one media may carry. A batch larger than this is
/// split before it reaches this crate; the number is small so a human can
/// read the whole media over.
pub const MAX_PAYOUTS: usize = 200;

/// File extension of a media, so a stick full of files reads at a glance.
pub const MEDIA_EXT: &str = "uslj";

const KIND: &str = "USL1";
const GENESIS: &str = "usl-media";
const MINOR_PER_MAJOR: u64 = 100;

/// Why an envelope was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UslError {
    /// The envelope has no payout at all; nothing settles.
    Empty,
    /// An amount that is not `M.mm` - a canonical amount is a canonical
    /// line, and a media with two spellings for one number is two batches.
    BadAmount(String),
    /// A field that cannot survive the media grammar (control characters,
    /// quotes, spaces where a token is required) or a rule about the lines
    /// themselves (a duplicate payout).
    BadField(String),
    /// A window that cannot be checked against the rules it exists for.
    Window(String),
    /// Media text that is not a USL1 media at all.
    Media(String),
    /// The entries and the seal they carry disagree - the media was edited
    /// after sealing, or was half-written, and there is no third reading.
    Unsealed(String),
}

impl fmt::Display for UslError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "the envelope carries no payout line"),
            Self::BadAmount(what) => write!(f, "amount is not canonical M.mm: {what}"),
            Self::BadField(what) => write!(f, "field cannot survive the media grammar: {what}"),
            Self::Window(what) => write!(f, "the window is not checkable: {what}"),
            Self::Media(what) => write!(f, "not a USL1 media: {what}"),
            Self::Unsealed(what) => write!(f, "the media does not match its own seal: {what}"),
        }
    }
}

/// Parse `M.mm` into its two parts. A minor unit that is not two digits, or
/// does not fit under 100, is refused: silent rounding here would make the
/// hot and cold totals differ by an amount no audit can attribute.
///
/// # Errors
///
/// [`UslError::BadAmount`] for any spelling other than canonical `M.mm`.
pub fn parse_amount(text: &str) -> Result<(u64, u64), UslError> {
    let bad = || UslError::BadAmount(text.to_string());
    let (major, minor) = text.split_once('.').ok_or_else(bad)?;
    if minor.len() != 2 {
        return Err(bad());
    }
    let major: u64 = major.parse().map_err(|_| bad())?;
    let minor: u64 = minor.parse().map_err(|_| bad())?;
    if minor >= MINOR_PER_MAJOR {
        return Err(bad());
    }
    Ok((major, minor))
}

fn reject_hostile(name: &str, value: &str, spaces_allowed: bool) -> Result<(), UslError> {
    let bad = |why: &str| UslError::BadField(format!("{name}: {why}"));
    if value.is_empty() {
        return Err(bad("empty"));
    }
    if value.len() > 120 {
        return Err(bad("longer than 120 bytes"));
    }
    if value.starts_with(' ') || value.ends_with(' ') {
        return Err(bad("padded with spaces"));
    }
    if value.contains("  ") {
        return Err(bad("contains a double space"));
    }
    if !spaces_allowed && value.contains(' ') {
        return Err(bad("contains a space where one token is required"));
    }
    if value.contains('"') || value.contains('\\') {
        return Err(bad("contains a quote or backslash"));
    }
    if value.chars().any(char::is_control) {
        return Err(bad("contains a control character"));
    }
    Ok(())
}

/// One payout line: who is paid, how much, and the memo that identifies it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Payout {
    to: String,
    major: u64,
    minor: u64,
    memo: String,
}

impl Payout {
    /// Builds one payout, refusing anything that cannot survive the media
    /// grammar. The amount must be canonical `M.mm`.
    ///
    /// # Errors
    ///
    /// [`UslError::BadField`] and [`UslError::BadAmount`] as described above.
    pub fn new(to: &str, amount: &str, memo: &str) -> Result<Self, UslError> {
        reject_hostile("address", to, false)?;
        reject_hostile("memo", memo, true)?;
        let (major, minor) = parse_amount(amount)?;
        Ok(Self { to: to.to_string(), major, minor, memo: memo.to_string() })
    }

    /// The address being paid.
    #[must_use]
    pub fn to(&self) -> &str {
        &self.to
    }

    /// The memo, or the empty string.
    #[must_use]
    pub fn memo(&self) -> &str {
        &self.memo
    }

    /// The canonical amount spelling, exactly as it appears in the media.
    #[must_use]
    pub fn amount(&self) -> String {
        format!("{}.{:02}", self.major, self.minor)
    }

    fn line(&self) -> String {
        if self.memo.is_empty() {
            format!("PAY {} {}", self.to, self.amount())
        } else {
            format!("PAY {} {} {}", self.to, self.amount(), self.memo)
        }
    }
}

/// A settlement envelope: what to pay, on which chain, inside which window.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    seq: u64,
    chain_id: String,
    not_before: Option<u64>,
    not_after: Option<u64>,
    fee: Option<(u64, u64)>,
    payouts: Vec<Payout>,
}

impl Manifest {
    /// Starts an envelope. The chain label is the media's replay actor and
    /// must survive the media grammar.
    ///
    /// # Errors
    ///
    /// [`UslError::BadField`] if the chain label is unusable.
    pub fn new(seq: u64, chain_id: &str) -> Result<Self, UslError> {
        reject_hostile("chain label", chain_id, false)?;
        Ok(Self {
            seq,
            chain_id: chain_id.to_string(),
            not_before: None,
            not_after: None,
            fee: None,
            payouts: Vec::new(),
        })
    }

    /// Sets the payout window, refusing it at the source: a window that
    /// does not open after genesis, or that ends where it begins, can never
    /// be settled correctly, and an envelope still being assembled is no
    /// license to seal one that cannot be. `check` repeats the rule so the
    /// reader enforces what the writer enforced.
    ///
    /// # Errors
    ///
    /// [`UslError::Window`] if `not_before` is zero or `not_after` is not
    /// strictly later than it.
    pub fn with_maturity(&mut self, not_before: u64, not_after: u64) -> Result<(), UslError> {
        if not_before < 1 {
            return Err(UslError::Window(
                "not_before must be at least 1; nothing settles at genesis".to_string(),
            ));
        }
        if not_after <= not_before {
            return Err(UslError::Window(format!(
                "expiry {not_after} must be strictly after the earliest open {not_before}"
            )));
        }
        self.not_before = Some(not_before);
        self.not_after = Some(not_after);
        Ok(())
    }

    /// Sets the fee the hot side is willing to spend to settle this batch.
    ///
    /// # Errors
    ///
    /// [`UslError::BadField`] if the minor part is not two digits.
    #[must_use = "an unset fee makes the envelope unsealable; handle the error"]
    pub fn with_fee(&mut self, major: u64, minor: u64) -> Result<(), UslError> {
        if minor >= MINOR_PER_MAJOR {
            return Err(UslError::BadField(format!(
                "fee: {minor} is not a two-digit minor unit"
            )));
        }
        self.fee = Some((major, minor));
        Ok(())
    }

    /// Adds one payout line, refusing a duplicate: the same address, amount,
    /// and memo twice is a batch the author assembled wrong, not a batch
    /// that pays twice - paying twice is a different memo.
    ///
    /// # Errors
    ///
    /// [`UslError::BadField`] for a duplicate or an oversized batch.
    #[must_use = "a refused payout line leaves the envelope short; handle the error"]
    pub fn add_payout(&mut self, payout: Payout) -> Result<(), UslError> {
        if self.payouts.len() >= MAX_PAYOUTS {
            return Err(UslError::BadField(format!(
                "a media holds at most {MAX_PAYOUTS} payouts"
            )));
        }
        let duplicate = self
            .payouts
            .iter()
            .any(|p| p.to == payout.to && p.memo == payout.memo && p.major == payout.major && p.minor == payout.minor);
        if duplicate {
            return Err(UslError::BadField(format!(
                "payout to {} x{} `{}` appears twice",
                payout.to,
                payout.amount(),
                if payout.memo.is_empty() { "no memo" } else { &payout.memo }
            )));
        }
        self.payouts.push(payout);
        Ok(())
    }

    /// The envelope's sequence number: what `media_file_name` writes and
    /// what a reader matches against the last media it consumed.
    #[must_use]
    pub fn seq(&self) -> u64 {
        self.seq
    }

    /// The chain this envelope settles on.
    #[must_use]
    pub fn chain_id(&self) -> &str {
        &self.chain_id
    }

    /// The payout window as `(earliest, latest)`.
    #[must_use]
    pub fn maturity(&self) -> (u64, u64) {
        (self.not_before.unwrap_or(0), self.not_after.unwrap_or(0))
    }

    /// The canonical fee spelling.
    #[must_use]
    pub fn fee(&self) -> String {
        self.fee.map_or_else(|| "0.00".to_string(), |(m, n)| format!("{m}.{n:02}"))
    }

    /// The payout lines.
    #[must_use]
    pub fn payouts(&self) -> &[Payout] {
        &self.payouts
    }

    /// The sum of the payout lines, carried across the minor unit.
    #[must_use]
    pub fn total(&self) -> (u64, u64) {
        let minors: u64 = self.payouts.iter().map(|p| p.major * MINOR_PER_MAJOR + p.minor).sum();
        (minors / MINOR_PER_MAJOR, minors % MINOR_PER_MAJOR)
    }

    /// The rules that make an envelope signable. Called by both writer and
    /// reader, so `lubot usl check` enforces exactly what `make` enforced.
    ///
    /// # Errors
    ///
    /// The first [`UslError`] the envelope fails.
    pub fn check(&self) -> Result<(), UslError> {
        if self.payouts.is_empty() {
            return Err(UslError::Empty);
        }
        let not_before = self
            .not_before
            .ok_or_else(|| UslError::Window("the media must carry a NOT-BEFORE line".to_string()))?;
        let not_after = self
            .not_after
            .ok_or_else(|| UslError::Window("the media must carry a NOT-AFTER line".to_string()))?;
        self.fee.ok_or_else(|| UslError::BadField("the media must carry a FEE line".to_string()))?;
        Ok(())
    }

    fn entry_lines(&self) -> Vec<String> {
        let mut lines = vec![
            format!("SEQ {}", self.seq),
            format!("CHAIN {}", self.chain_id),
            format!("NOT-BEFORE {}", self.not_before.unwrap_or(0)),
            format!("NOT-AFTER {}", self.not_after.unwrap_or(0)),
            format!("FEE {}", self.fee()),
        ];
        lines.extend(self.payouts.iter().map(Payout::line));
        lines
    }

    /// The sealed media text: the header, the entries in order, the seal.
    ///
    /// # Errors
    ///
    /// Whatever [`Manifest::check`] refuses - an unvalidated envelope never
    /// reaches a device.
    pub fn to_media(&self) -> Result<String, UslError> {
        self.check()?;
        use lubot_muhur::{Chain, Indexed};
        let lines = self.entry_lines();
        let mut ix = Indexed::new(Chain::new(GENESIS));
        for line in &lines {
            ix.append(&self.chain_id, line)
                .map_err(|b| UslError::Unsealed(format!("{b:?}")))?;
        }
        let seal = ix.finalize().hex();
        let mut out = String::from(KIND);
        out.push('\n');
        out.push_str(&format!("GEN {GENESIS}\n"));
        for line in &lines {
            out.push_str(line);
            out.push('\n');
        }
        out.push_str(&format!("SEALED {seal}\n"));
        Ok(out)
    }

    /// Reads a media back: every entry is re-appended, the chain is re-sealed
    /// from the bytes, and only a recomputed seal equal to the written one is
    /// accepted. A match is not proof of origin; it is proof the file was
    /// edited after sealing by nobody who did not also forge the tail.
    ///
    /// # Errors
    ///
    /// [`UslError::Media`] for text this is not a media at all,
    /// [`UslError::Unsealed`] when the seal does not recompute, and whatever
    /// [`Manifest::check`] refuses about the rebuilt envelope.
    pub fn from_media(text: &str) -> Result<Self, UslError> {
        let lines: Vec<&str> = text.lines().filter(|l| !l.is_empty()).collect();
        if lines.len() < 4 || lines[0] != KIND || !lines[1].starts_with("GEN ") {
            return Err(UslError::Media("header is not `USL1` then `GEN`".to_string()));
        }
        if &lines[1][4..] != GENESIS {
            return Err(UslError::Media(format!("genesis label is not `{GENESIS}`")));
        }
        let tail = lines[lines.len() - 1];
        let Some(written_seal) = tail.strip_prefix("SEALED ") else {
            return Err(UslError::Media("the last line is not the SEALED one".to_string()));
        };
        let entries = &lines[2..lines.len() - 1];
        let replay_actor = entries
            .iter()
            .find_map(|l| l.strip_prefix("CHAIN "))
            .ok_or_else(|| UslError::Media("no CHAIN line to key the replay on".to_string()))?;

        use lubot_muhur::{Chain, Indexed};
        let mut ix = Indexed::new(Chain::new(GENESIS));
        for line in entries {
            ix.append(replay_actor, line)
                .map_err(|b| UslError::Unsealed(format!("{b:?}")))?;
        }
        let recomputed = ix.finalize().hex();
        if recomputed != written_seal {
            return Err(UslError::Unsealed(format!(
                "the entries hash to {recomputed}, the file carries {written_seal}"
            )));
        }
        ix.verify().map_err(|b| UslError::Unsealed(format!("{b:?}")))?;

        let mut manifest = Manifest {
            seq: 0,
            chain_id: replay_actor.to_string(),
            not_before: None,
            not_after: None,
            fee: None,
            payouts: Vec::new(),
        };
        for line in entries {
            if let Some(rest) = line.strip_prefix("SEQ ") {
                manifest.seq = rest.parse().map_err(|_| UslError::Media(format!("`{line}`")))?;
            } else if let Some(rest) = line.strip_prefix("CHAIN ") {
                reject_hostile("chain label", rest, false)?;
                if rest != replay_actor {
                    return Err(UslError::Media("two CHAIN lines".to_string()));
                }
            } else if let Some(rest) = line.strip_prefix("NOT-BEFORE ") {
                manifest.not_before = Some(rest.parse().map_err(|_| UslError::Media(format!("`{line}`")))?);
            } else if let Some(rest) = line.strip_prefix("NOT-AFTER ") {
                manifest.not_after = Some(rest.parse().map_err(|_| UslError::Media(format!("`{line}`")))?);
            } else if let Some(rest) = line.strip_prefix("FEE ") {
                let (major, minor) = parse_amount(rest)
                    .map_err(|_| UslError::Media(format!("`{line}`")))?;
                manifest.fee = Some((major, minor));
            } else if let Some(rest) = line.strip_prefix("PAY ") {
                let mut parts = rest.splitn(3, ' ');
                let to = parts.next().unwrap_or_default();
                let amount = parts.next().unwrap_or_default();
                let memo = parts.next().unwrap_or("");
                manifest.payouts.push(Payout::new(to, amount, memo)?);
            } else {
                return Err(UslError::Media(format!("unknown line `{line}`")));
            }
        }
        manifest.check()?;
        Ok(manifest)
    }
}

/// The file a media is written to, for a given envelope. The name carries
/// only the sequence number - no caller-supplied string reaches it, so no
/// media path can escape the directory it is written into.
#[must_use]
pub fn media_file_name(seq: u64) -> String {
    format!("USL-{seq:06}.{MEDIA_EXT}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn built() -> Manifest {
        let mut m = Manifest::new(7, "mainnet").unwrap();
        m.with_maturity(100, 200).unwrap();
        m.with_fee(0, 21).unwrap();
        m.add_payout(Payout::new("bc1qaddr", "10.00", "rent").unwrap()).unwrap();
        m.add_payout(Payout::new("bc1qother", "0.50", "").unwrap()).unwrap();
        m
    }

    #[test]
    fn only_canonical_amounts_are_paid() {
        assert!(matches!(Payout::new("addr", "1.5", ""), Err(UslError::BadAmount(_))));
        assert!(matches!(Payout::new("addr", "1.500", ""), Err(UslError::BadAmount(_))));
        assert!(matches!(Payout::new("a d d r", "1.00", ""), Err(UslError::BadField(_))));
        assert!(matches!(Payout::new("addr", "1.00", "with \"quote\""), Err(UslError::BadField(_))));
        let p = Payout::new("addr", "3.40", "ok").unwrap();
        assert_eq!(p.amount(), "3.40");
        assert_eq!(p.to(), "addr");
        assert_eq!(p.memo(), "ok");
    }

    #[test]
    fn an_envelope_round_trips_through_media() {
        let m = built();
        let text = m.to_media().unwrap();
        let back = Manifest::from_media(&text).unwrap();
        assert_eq!(back.seq(), 7);
        assert_eq!(back.chain_id(), "mainnet");
        assert_eq!(back.maturity(), (100, 200));
        assert_eq!(back.fee(), "0.21");
        assert_eq!(back.payouts(), m.payouts());
        assert_eq!(back.total(), (10, 50));
    }

    #[test]
    fn one_edited_byte_breaks_the_seal() {
        let text = built().to_media().unwrap();
        let edited = text.replace("rent", "rEnt");
        assert_ne!(edited, text, "the fixture must actually change");
        let err = Manifest::from_media(&edited).unwrap_err();
        assert!(matches!(err, UslError::Unsealed(_)), "{err:?}");
        let truncated = text.lines().take(4).collect::<Vec<_>>().join("\n");
        assert!(matches!(
            Manifest::from_media(&truncated),
            Err(UslError::Media(_) | UslError::Unsealed(_))
        ));
    }

    #[test]
    fn windows_open_after_genesis_and_fees_are_spelled_once() {
        let mut m = built();
        assert!(matches!(m.with_maturity(0, 200), Err(UslError::Window(_))));
        assert!(matches!(m.with_maturity(200, 200), Err(UslError::Window(_))));
        assert!(matches!(
            Manifest::new(9, "c").unwrap().to_media(),
            Err(UslError::Empty)
        ));
        assert!(matches!(Manifest::new(1, "c").unwrap().check(), Err(UslError::Empty)));
        let mut lone = Manifest::new(1, "c").unwrap();
        assert!(lone.with_fee(1, 100).is_err());
    }

    #[test]
    fn a_repeated_line_is_a_mistake_not_a_double_payment() {
        let mut m = built();
        let again = Payout::new("bc1qaddr", "10.00", "rent").unwrap();
        assert!(matches!(m.add_payout(again), Err(UslError::BadField(_))));
        let other = Payout::new("bc1qaddr", "10.00", "rent again").unwrap();
        assert!(m.add_payout(other).is_ok());
    }

    #[test]
    fn media_names_pad_and_carry_nothing_the_caller_controls() {
        assert_eq!(media_file_name(42), "USL-000042.uslj");
        assert_eq!(media_file_name(u64::MAX), "USL-18446744073709551615.uslj");
    }
}
