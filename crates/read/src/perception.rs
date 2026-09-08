//! # lubot-read::perception - what Lubot reads, and how much of it
//!
//! The education report fixes the modality set and its ceilings; this module
//! is where they live in code, so the rule is measured against, not quoted.
//!
//! * The set is exactly [`PerceptionKind`]: Text, Image, Audio, Video.
//!   There is **no generating variant** - the enum is the interface, so a
//!   generation surface cannot exist as a runtime flag.
//! * Every kind has a fixed ceiling in its own unit (bytes, pixels,
//!   milliseconds, frames). [`check_units`] is the single admission check;
//!   an unknown kind is a refusal, and an over-ceiling unit count is a
//!   refusal.
//! * Lubot's primary weight is text (code + data analysis). The other three
//!   modalities serve one task only: read it, interpret it, answer in
//!   Markdown. No generation task appears anywhere in the corpus.

/// A modality Lubot may be asked to read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerceptionKind {
    Text,
    Image,
    Audio,
    Video,
}

impl PerceptionKind {
    /// Parse the wire tag (1=text, 2=image, 3=audio, 4=video, matching the
    /// chain's perception tags). Anything else is a refusal - fail-closed.
    pub fn parse(tag: u32) -> Option<Self> {
        match tag {
            1 => Some(PerceptionKind::Text),
            2 => Some(PerceptionKind::Image),
            3 => Some(PerceptionKind::Audio),
            4 => Some(PerceptionKind::Video),
            _ => None,
        }
    }

    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            PerceptionKind::Text => "text",
            PerceptionKind::Image => "image",
            PerceptionKind::Audio => "audio",
            PerceptionKind::Video => "video",
        }
    }

    /// The fixed ceiling for this modality, in its own unit.
    #[must_use]
    pub fn ceiling_units(self) -> u64 {
        match self {
            PerceptionKind::Text => MAX_TEXT_BYTES,
            PerceptionKind::Image => MAX_IMAGE_PIXELS,
            PerceptionKind::Audio => MAX_AUDIO_MS,
            PerceptionKind::Video => MAX_VIDEO_FRAMES,
        }
    }
}

/// Text: bytes.
pub const MAX_TEXT_BYTES: u64 = 1_048_576;
/// Image: pixels.
pub const MAX_IMAGE_PIXELS: u64 = 16_777_216;
/// Audio: milliseconds (60 minutes).
pub const MAX_AUDIO_MS: u64 = 3_600_000;
/// Video: frames.
pub const MAX_VIDEO_FRAMES: u64 = 4096;

/// Why a perception request was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerceptionRefusal {
    UnknownKind(u32),
    OverCeiling {
        kind: PerceptionKind,
        units: u64,
        ceiling: u64,
    },
}

impl PerceptionRefusal {
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            PerceptionRefusal::UnknownKind(_) => "unknown-kind",
            PerceptionRefusal::OverCeiling { .. } => "over-ceiling",
        }
    }
}

/// The single admission check for a perception request. Unknown kind and
/// over-ceiling units are both refusals; there is no "clamp" answer.
pub fn check_units(kind_tag: u32, units: u64) -> Result<PerceptionKind, PerceptionRefusal> {
    let Some(kind) = PerceptionKind::parse(kind_tag) else {
        return Err(PerceptionRefusal::UnknownKind(kind_tag));
    };
    let ceiling = kind.ceiling_units();
    if units > ceiling {
        return Err(PerceptionRefusal::OverCeiling {
            kind,
            units,
            ceiling,
        });
    }
    Ok(kind)
}

/// Whether the tag is one of the four known kinds ("the set is closed").
#[must_use]
pub fn is_known(tag: u32) -> bool {
    PerceptionKind::parse(tag).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_set_is_closed_and_maps_the_chain_tags() {
        assert_eq!(PerceptionKind::parse(1), Some(PerceptionKind::Text));
        assert_eq!(PerceptionKind::parse(2), Some(PerceptionKind::Image));
        assert_eq!(PerceptionKind::parse(3), Some(PerceptionKind::Audio));
        assert_eq!(PerceptionKind::parse(4), Some(PerceptionKind::Video));
        assert_eq!(PerceptionKind::parse(0), None);
        assert_eq!(PerceptionKind::parse(5), None);
        assert_eq!(PerceptionKind::parse(u32::MAX), None);
    }

    #[test]
    fn ceilings_are_the_fixed_numbers() {
        assert_eq!(MAX_TEXT_BYTES, 1_048_576);
        assert_eq!(MAX_IMAGE_PIXELS, 16_777_216);
        assert_eq!(MAX_AUDIO_MS, 3_600_000);
        assert_eq!(MAX_VIDEO_FRAMES, 4096);
        assert_eq!(PerceptionKind::Text.ceiling_units(), MAX_TEXT_BYTES);
        assert_eq!(PerceptionKind::Image.ceiling_units(), MAX_IMAGE_PIXELS);
        assert_eq!(PerceptionKind::Audio.ceiling_units(), MAX_AUDIO_MS);
        assert_eq!(PerceptionKind::Video.ceiling_units(), MAX_VIDEO_FRAMES);
    }

    #[test]
    fn within_ceiling_is_admitted_exactly_at_the_boundary() {
        assert_eq!(check_units(1, 1_048_576), Ok(PerceptionKind::Text));
        assert_eq!(check_units(2, 16_777_216), Ok(PerceptionKind::Image));
        assert_eq!(check_units(3, 3_600_000), Ok(PerceptionKind::Audio));
        assert_eq!(check_units(4, 4096), Ok(PerceptionKind::Video));
    }

    #[test]
    fn over_ceiling_is_refused_not_clamped() {
        assert_eq!(
            check_units(1, 1_048_577),
            Err(PerceptionRefusal::OverCeiling {
                kind: PerceptionKind::Text,
                units: 1_048_577,
                ceiling: 1_048_576,
            })
        );
        assert_eq!(
            check_units(4, 4097),
            Err(PerceptionRefusal::OverCeiling {
                kind: PerceptionKind::Video,
                units: 4097,
                ceiling: 4096,
            })
        );
    }

    #[test]
    fn unknown_tag_is_refused_even_with_zero_units() {
        assert_eq!(check_units(9, 0), Err(PerceptionRefusal::UnknownKind(9)));
    }
}
