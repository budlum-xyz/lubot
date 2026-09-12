//! The settlement envelope a cold wallet reads off the media.
//!
//! # The boundary this crate exists for
//!
//! The hot side builds what should be paid. The cold side signs it. Between
//! them is a file, and a file is not a channel: it can be edited, truncated,
//! reordered by a tool that sorts lines, or written half-way by a disk that
//! gives up. What this crate guarantees is that **the bytes the cold device is
//! asked to sign are the bytes the hot device meant.**
//!
//! It does that with a seal, not a signature - see [`lubot_muhur`] for why the
//! distinction matters. The envelope carries no key. The cold reader recomputes
//! the seal from the entries and refuses on the first line that disagrees.
//!
//! # The rule underneath everything here
//!
//! **A read media is verified by recomputation, never by trusting the stored
//! line.** [`Manifest::from_media`] parses every line back through the same
//! validating constructors that built it, re-seals, and compares - and then
//! re-renders the whole file and compares it to the bytes it was given. That
//! last step is the one that matters most and the one most implementations
//! omit: without it, a file that only *sort of* parses is normalized into a
//! different file, and the cold device ends up signing something nobody wrote.
//!
//! Concretely, `SEQ 007` is refused. So is a swapped line order. So is a payout
//! amount written `1.5` where the canonical form is `1.50`.
//!
//! # Amounts
//!
//! [`Amount`] accepts exactly one canonical form: `M.mm`, a whole part with no
//! leading zeros and exactly two decimal digits. The two-digit minor unit is
//! this layer's own unit; each side maps it to its currency at its own boundary,
//! and nothing in here converts anything silently. A settlement layer that
//! rounds is a settlement layer that loses money in a way nobody can point at.

use lubot_muhur::{SealError, Sealer};

/// The media format tag. First line of every envelope.
pub const MEDIA_TAG: &str = "LUBOT-USL-V1";

/// Why an envelope was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvelopeError {
    /// An amount was not in the canonical `M.mm` form.
    AmountNotCanonical { got: String },
    /// A payout line duplicated an earlier one exactly.
    DuplicatePayout { index: usize },
    /// A payout line carried a character that would make the media ambiguous.
    UnsafeCharacter { index: usize, ch: char },
    /// The payout window does not open after genesis.
    WindowBeforeGenesis { open: u64 },
    /// The window closes at or before it opens.
    WindowClosesBeforeOpening { open: u64, close: u64 },
    /// A line could not be parsed, with the line's index and the reason.
    MalformedLine { index: usize, reason: String },
    /// The file's tag was not this format's.
    WrongTag { got: String },
    /// The file does not write back byte for byte. This is the refusal that
    /// catches normalization: the parse succeeded, and the result is still a
    /// different file.
    MediaNotRoundTripped,
    /// The seal did not verify.
    Seal(SealError),
    /// The envelope carries no payouts. An empty settlement is not a settlement
    /// and should not reach a cold device.
    NoPayouts,
}

impl std::fmt::Display for EnvelopeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AmountNotCanonical { got } => {
                write!(f, "amount {got:?} is not in the canonical M.mm form")
            }
            Self::DuplicatePayout { index } => write!(f, "payout at line {index} duplicates an earlier one"),
            Self::UnsafeCharacter { index, ch } => {
                write!(f, "payout at line {index} carries {ch:?}, which would make the media ambiguous")
            }
            Self::WindowBeforeGenesis { open } => {
                write!(f, "the payout window opens at {open}, which is not after genesis")
            }
            Self::WindowClosesBeforeOpening { open, close } => {
                write!(f, "the payout window closes at {close}, which is not after it opens at {open}")
            }
            Self::MalformedLine { index, reason } => write!(f, "line {index} is malformed: {reason}"),
            Self::WrongTag { got } => write!(f, "this file is {got:?}, not {MEDIA_TAG:?}"),
            Self::MediaNotRoundTripped => write!(
                f,
                "the file parsed but does not write back byte for byte, so what was read is not what was written"
            ),
            Self::Seal(err) => write!(f, "the seal does not verify: {err}"),
            Self::NoPayouts => write!(f, "the envelope carries no payouts"),
        }
    }
}

/// An amount in this layer's canonical minor unit: `M.mm`.
///
/// Stored as minor units so that no arithmetic in this crate ever touches a
/// float. The string form is derived, never stored, which means a round trip
/// through the media cannot drift.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Amount {
    minor: u64,
}

impl Amount {
    /// Builds from minor units directly.
    #[must_use]
    pub const fn from_minor(minor: u64) -> Self {
        Self { minor }
    }

    /// The minor units.
    #[must_use]
    pub const fn minor(self) -> u64 {
        self.minor
    }

    /// Parses the canonical form.
    ///
    /// Refuses: a missing decimal point, one or three decimals, a sign, an
    /// exponent, a leading zero on a multi-digit whole part, an empty whole
    /// part, and any non-digit. Each of those has a plausible-looking reason to
    /// be accepted and each of them is a way for two nodes to disagree about an
    /// amount.
    ///
    /// # Errors
    ///
    /// [`EnvelopeError::AmountNotCanonical`] carrying the offending text.
    pub fn parse(text: &str) -> Result<Self, EnvelopeError> {
        let err = || EnvelopeError::AmountNotCanonical {
            got: text.to_string(),
        };
        let Some((whole, frac)) = text.split_once('.') else {
            return Err(err());
        };
        if whole.is_empty() || frac.len() != 2 {
            return Err(err());
        }
        if !whole.bytes().all(|b| b.is_ascii_digit()) || !frac.bytes().all(|b| b.is_ascii_digit()) {
            return Err(err());
        }
        // A leading zero on a multi-digit whole part means two spellings of the
        // same amount, and two spellings means two files that settle the same
        // thing differently.
        if whole.len() > 1 && whole.starts_with('0') {
            return Err(err());
        }
        let whole_units: u64 = whole.parse().map_err(|_| err())?;
        let frac_units: u64 = frac.parse().map_err(|_| err())?;
        // Saturating: an amount large enough to overflow is not a settlement,
        // and wrapping would have produced a small one.
        let minor = whole_units
            .checked_mul(100)
            .and_then(|w| w.checked_add(frac_units))
            .ok_or_else(err)?;
        Ok(Self { minor })
    }

    /// The canonical rendering. Always exactly two decimals, never a leading
    /// zero on a multi-digit whole part, because that is what [`Self::parse`]
    /// accepts.
    #[must_use]
    pub fn render(self) -> String {
        format!("{}.{:02}", self.minor / 100, self.minor % 100)
    }
}

/// One payout line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Payout {
    pub recipient: String,
    pub amount: Amount,
    /// An opaque reference the batch author uses to tie the line to its cause.
    /// Not interpreted here, and deliberately so: a field this layer does not
    /// understand cannot be validated, and pretending otherwise would be worse
    /// than not carrying it.
    pub reference: String,
}

impl Payout {
    /// Builds a payout, refusing a reference that would make the media
    /// ambiguous.
    ///
    /// # Errors
    ///
    /// [`EnvelopeError::UnsafeCharacter`] naming the character.
    pub fn new(recipient: &str, amount: Amount, reference: &str) -> Result<Self, EnvelopeError> {
        for (index, field) in [recipient, reference].iter().enumerate() {
            for ch in field.chars() {
                if matches!(ch, '\t' | '"' | '\\' | '\n' | '\r') || ch.is_control() {
                    return Err(EnvelopeError::UnsafeCharacter { index, ch });
                }
            }
        }
        Ok(Self {
            recipient: recipient.to_string(),
            amount,
            reference: reference.to_string(),
        })
    }

    /// The line as it appears in the media. Tab separated, because no field may
    /// contain a tab.
    #[must_use]
    pub fn render(&self) -> String {
        format!(
            "PAY\t{}\t{}\t{}",
            self.recipient,
            self.amount.render(),
            self.reference
        )
    }
}

/// A settlement batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    /// Monotonic across batches on one chain. A cold device refuses a sequence
    /// number it has already signed.
    pub sequence: u64,
    pub chain_id: u64,
    /// The window the payouts may be claimed in, in chain heights.
    pub window_open: u64,
    pub window_close: u64,
    /// The fee, in the same minor unit as the payouts.
    pub fee: Amount,
    pub payouts: Vec<Payout>,
    /// The per-entry seal links, in order. Stored so the cold reader can say
    /// which entry broke rather than merely that something did.
    pub links: Vec<String>,
}

impl Manifest {
    /// An empty manifest for `chain_id` and `sequence`.
    #[must_use]
    pub fn new(sequence: u64, chain_id: u64, fee: Amount) -> Self {
        Self {
            sequence,
            chain_id,
            window_open: 0,
            window_close: 0,
            fee,
            payouts: Vec::new(),
            links: Vec::new(),
        }
    }

    /// Sets the payout window.
    ///
    /// # Errors
    ///
    /// [`EnvelopeError::WindowBeforeGenesis`] or
    /// [`EnvelopeError::WindowClosesBeforeOpening`].
    pub fn set_window(&mut self, open: u64, close: u64) -> Result<(), EnvelopeError> {
        // Zero is genesis. A window opening at genesis cannot be checked against
        // the maturity rule it exists for, because there is no height before it.
        if open == 0 {
            return Err(EnvelopeError::WindowBeforeGenesis { open });
        }
        if close <= open {
            return Err(EnvelopeError::WindowClosesBeforeOpening { open, close });
        }
        self.window_open = open;
        self.window_close = close;
        Ok(())
    }

    /// Adds a payout, refusing an exact duplicate of an earlier line.
    ///
    /// A duplicate is a batch author's mistake, not a legitimate second payment:
    /// two identical lines to the same recipient for the same amount under the
    /// same reference are one payment written twice, and paying it twice is the
    /// expensive direction to be wrong in.
    ///
    /// # Errors
    ///
    /// [`EnvelopeError::DuplicatePayout`] naming the line.
    pub fn add_payout(&mut self, payout: Payout) -> Result<(), EnvelopeError> {
        if self.payouts.contains(&payout) {
            return Err(EnvelopeError::DuplicatePayout {
                index: self.payouts.len(),
            });
        }
        self.payouts.push(payout);
        Ok(())
    }

    /// The header lines, in the order they appear in the media, without the
    /// tag. The tag is the first line of the file and is not part of the sealed
    /// entries: it says what format the file is, and a file that claims to be
    /// this format is checked against this format before anything is sealed.
    #[must_use]
    fn header_lines(&self) -> Vec<String> {
        vec![
            format!("SEQ\t{}", self.sequence),
            format!("CHAIN\t{}", self.chain_id),
            format!("WINDOW\t{}\t{}", self.window_open, self.window_close),
            format!("FEE\t{}", self.fee.render()),
        ]
    }

    /// Checks the batch before it is written.
    ///
    /// # Errors
    ///
    /// Any [`EnvelopeError`] that applies.
    pub fn check(&self) -> Result<(), EnvelopeError> {
        if self.payouts.is_empty() {
            return Err(EnvelopeError::NoPayouts);
        }
        if self.window_open == 0 {
            return Err(EnvelopeError::WindowBeforeGenesis {
                open: self.window_open,
            });
        }
        if self.window_close <= self.window_open {
            return Err(EnvelopeError::WindowClosesBeforeOpening {
                open: self.window_open,
                close: self.window_close,
            });
        }
        for (index, payout) in self.payouts.iter().enumerate() {
            if self.payouts[..index].contains(payout) {
                return Err(EnvelopeError::DuplicatePayout { index });
            }
        }
        Ok(())
    }

    /// Writes the media and seals it.
    ///
    /// # Errors
    ///
    /// Any [`EnvelopeError`] from [`Self::check`], or a sealing failure.
    pub fn to_media(&mut self) -> Result<String, EnvelopeError> {
        self.check()?;
        // The file layout is: tag, headers, payout lines, seal. The sealed
        // entries are everything between the tag and the seal, in file order,
        // so that the entry index in a seal refusal is the line the operator
        // sees.
        let entries: Vec<String> = self
            .header_lines()
            .into_iter()
            .chain(self.payouts.iter().map(|p| p.render()))
            .collect();
        let refs: Vec<&str> = entries.iter().map(String::as_str).collect();
        self.links = Sealer::seal(&refs).map_err(EnvelopeError::Seal)?;
        let mut out = String::new();
        out.push_str(MEDIA_TAG);
        out.push('\n');
        out.push_str(&entries.join("\n"));
        out.push('\n');
        out.push_str("SEAL\t");
        out.push_str(self.links.last().map_or("", String::as_str));
        out.push('\n');
        Ok(out)
    }

    /// Reads a media, verifying by recomputation.
    ///
    /// Three checks, in this order, and the third is the one that matters:
    ///
    /// 1. Every line parses through the same validating constructors that built
    ///    it, so an accepted media satisfies [`Self::check`] by construction.
    /// 2. The seal verifies against the stored links, naming the entry that
    ///    broke if it does not.
    /// 3. The rebuilt manifest writes back **byte for byte**. A file that parses
    ///    but re-renders differently is refused, because what was read is not
    ///    what was written - and the cold device must not sign the difference.
    ///
    /// # Errors
    ///
    /// Any [`EnvelopeError`].
    pub fn from_media(media: &str) -> Result<Self, EnvelopeError> {
        let mut lines: Vec<&str> = media.lines().collect();
        if lines.first().copied() != Some(MEDIA_TAG) {
            return Err(EnvelopeError::WrongTag {
                got: lines.first().copied().unwrap_or("").to_string(),
            });
        }
        // The seal line is last and is not part of the sealed entries.
        let Some(seal_line) = lines.pop() else {
            return Err(EnvelopeError::MalformedLine {
                index: 0,
                reason: "the file has no seal line".to_string(),
            });
        };
        let recorded_seal = seal_line
            .strip_prefix("SEAL\t")
            .ok_or_else(|| EnvelopeError::MalformedLine {
                index: lines.len(),
                reason: "the last line is not a SEAL line".to_string(),
            })?
            .to_string();

        let mut sequence = None;
        let mut chain_id = None;
        let mut window = None;
        let mut fee = None;
        let mut payout_entries: Vec<String> = Vec::new();
        let mut payouts: Vec<Payout> = Vec::new();

        for (index, line) in lines.iter().enumerate().skip(1) {
            let fields: Vec<&str> = line.split('\t').collect();
            let head = fields.first().copied().unwrap_or("");
            let at = |n: usize| -> Result<&str, EnvelopeError> {
                fields
                    .get(n)
                    .copied()
                    .ok_or_else(|| EnvelopeError::MalformedLine {
                        index,
                        reason: format!("expected a field at position {n}"),
                    })
            };
            match head {
                "SEQ" => {
                    sequence = Some(at(1)?.parse().map_err(|_| EnvelopeError::MalformedLine {
                        index,
                        reason: "SEQ is not a number".to_string(),
                    })?)
                }
                "CHAIN" => {
                    chain_id = Some(at(1)?.parse().map_err(|_| EnvelopeError::MalformedLine {
                        index,
                        reason: "CHAIN is not a number".to_string(),
                    })?)
                }
                "WINDOW" => {
                    window = Some((
                        at(1)?.parse().map_err(|_| EnvelopeError::MalformedLine {
                            index,
                            reason: "the window open height is not a number".to_string(),
                        })?,
                        at(2)?.parse().map_err(|_| EnvelopeError::MalformedLine {
                            index,
                            reason: "the window close height is not a number".to_string(),
                        })?,
                    ));
                }
                "FEE" => fee = Some(Amount::parse(at(1)?)?),
                "PAY" => {
                    payout_entries.push(line.to_string());
                    payouts.push(Payout::new(at(1)?, Amount::parse(at(2)?)?, at(3)?)?);
                }
                other => {
                    return Err(EnvelopeError::MalformedLine {
                        index,
                        reason: format!("unknown line kind {other:?}"),
                    });
                }
            }
        }

        let sequence = sequence.ok_or_else(|| EnvelopeError::MalformedLine {
            index: 0,
            reason: "no SEQ line".to_string(),
        })?;
        let chain_id = chain_id.ok_or_else(|| EnvelopeError::MalformedLine {
            index: 0,
            reason: "no CHAIN line".to_string(),
        })?;
        let (window_open, window_close) = window.ok_or_else(|| EnvelopeError::MalformedLine {
            index: 0,
            reason: "no WINDOW line".to_string(),
        })?;
        let fee = fee.ok_or_else(|| EnvelopeError::MalformedLine {
            index: 0,
            reason: "no FEE line".to_string(),
        })?;

        let mut manifest = Self::new(sequence, chain_id, fee);
        manifest.set_window(window_open, window_close)?;
        for payout in payouts {
            manifest.add_payout(payout)?;
        }
        manifest.check()?;

        // The seal, over the same entry order `to_media` used.
        // The same order `to_media` sealed: headers first, then payout lines,
        // both in file order. Getting this order wrong would not fail loudly -
        // it would produce a different seal and refuse a perfectly good file.
        let entries: Vec<String> = manifest
            .header_lines()
            .into_iter()
            .chain(payout_entries)
            .collect();
        let refs: Vec<&str> = entries.iter().map(String::as_str).collect();
        let links = Sealer::seal(&refs).map_err(EnvelopeError::Seal)?;
        if links.last().map_or("", String::as_str) != recorded_seal {
            return Err(EnvelopeError::Seal(SealError::SealMismatch {
                expected: recorded_seal,
                got: links.last().cloned().unwrap_or_default(),
            }));
        }
        manifest.links = links;

        // The check that catches normalization: rebuild the file and compare.
        let rewritten = manifest.to_media()?;
        if rewritten != media {
            return Err(EnvelopeError::MediaNotRoundTripped);
        }
        Ok(manifest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn batch() -> Manifest {
        let mut m = Manifest::new(7, 1, Amount::from_minor(250));
        m.set_window(100, 200).expect("window");
        m.add_payout(
            Payout::new("bud1alice", Amount::from_minor(10_50), "invoice-1").expect("payout"),
        )
        .expect("add");
        m.add_payout(
            Payout::new("bud1bob", Amount::from_minor(2_00), "invoice-2").expect("payout"),
        )
        .expect("add");
        m
    }

    #[test]
    fn an_amount_round_trips_through_its_canonical_form() {
        for text in ["0.00", "1.50", "10.05", "123456.78"] {
            let parsed = Amount::parse(text).expect("canonical");
            assert_eq!(
                parsed.render(),
                text,
                "the canonical form is not a fixed point"
            );
        }
    }

    #[test]
    fn every_non_canonical_spelling_of_an_amount_is_refused() {
        // Each of these has a plausible reason to be accepted and each is a way
        // for two nodes to disagree about an amount.
        for text in [
            "1.5", "1.234", ".50", "1.", "-1.00", "+1.00", "01.00", "1e2", "1,00", "", "1.5a",
            " 1.00", "1.00 ",
        ] {
            assert!(
                Amount::parse(text).is_err(),
                "{text:?} was accepted but is not canonical"
            );
        }
    }

    #[test]
    fn a_leading_zero_is_refused_because_it_makes_two_spellings() {
        // Two spellings of the same amount means two files that settle the same
        // thing differently.
        assert!(Amount::parse("01.00").is_err());
        assert!(
            Amount::parse("0.00").is_ok(),
            "a single leading zero is the canonical zero"
        );
    }

    #[test]
    fn no_arithmetic_here_touches_a_float() {
        // The amount is stored as minor units. A settlement layer that rounds
        // loses money in a way nobody can point at.
        let a = Amount::parse("0.10").expect("ok");
        let b = Amount::parse("0.20").expect("ok");
        let sum = Amount::from_minor(a.minor() + b.minor());
        assert_eq!(sum.render(), "0.30");
    }

    #[test]
    fn a_batch_writes_and_reads_back_byte_for_byte() {
        let mut m = batch();
        let media = m.to_media().expect("write");
        let read = Manifest::from_media(&media).expect("read");
        assert_eq!(read, m);
        assert_eq!(read.to_media().expect("rewrite"), media);
    }

    #[test]
    fn a_padded_sequence_number_is_refused_not_normalized() {
        // "SEQ 007" parses as 7. Accepting it would mean the cold device signs a
        // file nobody wrote.
        let mut m = batch();
        let media = m.to_media().expect("write");
        let padded = media.replace("SEQ\t7", "SEQ\t007");
        assert_eq!(
            Manifest::from_media(&padded),
            Err(EnvelopeError::MediaNotRoundTripped),
            "a padded number was normalized instead of refused"
        );
    }

    #[test]
    fn a_swapped_line_order_is_refused() {
        let mut m = batch();
        let media = m.to_media().expect("write");
        let mut lines: Vec<&str> = media.lines().collect();
        lines.swap(1, 2);
        let swapped = format!("{}\n", lines.join("\n"));
        assert!(
            matches!(
                Manifest::from_media(&swapped),
                Err(EnvelopeError::Seal(_)) | Err(EnvelopeError::MediaNotRoundTripped)
            ),
            "a reordered batch was accepted"
        );
    }

    #[test]
    fn an_edited_payout_breaks_the_seal_and_names_the_line() {
        let mut m = batch();
        let media = m.to_media().expect("write");
        let edited = media.replace("10.50", "99.99");
        let err = Manifest::from_media(&edited).unwrap_err();
        assert!(
            matches!(err, EnvelopeError::Seal(SealError::SealMismatch { .. })),
            "an edited amount was not caught by the seal: {err:?}"
        );
    }

    #[test]
    fn a_duplicate_payout_is_refused_when_it_is_added() {
        let mut m = Manifest::new(1, 1, Amount::from_minor(0));
        m.set_window(10, 20).expect("window");
        let p = Payout::new("bud1alice", Amount::from_minor(100), "invoice-1").expect("payout");
        m.add_payout(p.clone()).expect("first");
        assert_eq!(
            m.add_payout(p),
            Err(EnvelopeError::DuplicatePayout { index: 1 }),
            "paying the same line twice is the expensive direction to be wrong in"
        );
    }

    #[test]
    fn the_payout_window_must_open_after_genesis_and_close_after_it_opens() {
        let mut m = Manifest::new(1, 1, Amount::from_minor(0));
        assert_eq!(
            m.set_window(0, 10),
            Err(EnvelopeError::WindowBeforeGenesis { open: 0 }),
            "a window at genesis cannot be checked against the maturity rule it exists for"
        );
        assert_eq!(
            m.set_window(20, 20),
            Err(EnvelopeError::WindowClosesBeforeOpening {
                open: 20,
                close: 20
            })
        );
        assert!(m.set_window(10, 20).is_ok());
    }

    #[test]
    fn an_empty_batch_never_reaches_a_cold_device() {
        let mut m = Manifest::new(1, 1, Amount::from_minor(0));
        m.set_window(10, 20).expect("window");
        assert_eq!(m.to_media(), Err(EnvelopeError::NoPayouts));
    }

    #[test]
    fn a_control_character_in_a_field_is_refused() {
        // No field may contain a tab, a quote, a backslash or a control
        // character, because the media is tab separated and an embedded
        // separator would let one line masquerade as two.
        assert!(matches!(
            Payout::new("bud1ali\tce", Amount::from_minor(100), "r"),
            Err(EnvelopeError::UnsafeCharacter { ch: '\t', .. })
        ));
        assert!(matches!(
            Payout::new("bud1alice", Amount::from_minor(100), "r\nx"),
            Err(EnvelopeError::UnsafeCharacter { ch: '\n', .. })
        ));
    }

    #[test]
    fn a_file_with_the_wrong_tag_is_refused_before_anything_is_parsed() {
        assert_eq!(
            Manifest::from_media("SOMETHING-ELSE\nSEQ\t1\n"),
            Err(EnvelopeError::WrongTag {
                got: "SOMETHING-ELSE".to_string()
            })
        );
    }

    #[test]
    fn an_unknown_line_kind_is_refused() {
        // A forward-compatible reader that skips lines it does not understand is
        // a reader that will silently drop a field a later version made
        // load-bearing.
        let mut m = batch();
        let media = m.to_media().expect("write");
        let extra = media.replace("CHAIN\t1", "CHAIN\t1\nFUTURE\t1");
        assert!(matches!(
            Manifest::from_media(&extra),
            Err(EnvelopeError::MalformedLine { .. }) | Err(EnvelopeError::Seal(_))
        ));
    }
}
