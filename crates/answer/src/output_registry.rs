#![forbid(unsafe_code)]
//! # output registry - Aşama 9'un Lubot tarafı (kapalı devre)
//!
//! A finalized output is sealed here, and sealing is a gate, not a stamp:
//! the Markdown schema validator runs **before** the record is built, and a
//! failure is a rejection - the output is never downgraded to the nearest
//! format, because there is no nearest format in the rule.
//!
//! The report's closed loop finalizes with `ai_output_to_nft` (an
//! `"ai-inference"`-tagged NFT) and `register_data_asset` (a Pollen
//! DataAsset). Those two calls are chain-side and are **not** among the
//! seven fixed RPC methods of Aşama 10, so Lubot's client surface does not
//! contain them. What Lubot does contain is the handoff record: content id
//! (SHA-256 of the bytes), digest, tag, kind and the moment, ready to be
//! carried to the node by the data plane. That boundary is recorded in
//! TRAINING.md as scope, not papered over with a client that would violate
//! the fixed surface.

use lubot_grant::Seconds;
use lubot_read::{output_schema, sha256_hex};

/// The NFT tag the report fixes for finalized inference outputs.
pub const OUTPUT_TAG: &str = "ai-inference";

/// One finalized output, ready for the chain-side registration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputRecord {
    /// SHA-256 of the finalized bytes; also the content id they carry.
    pub content_id: String,
    pub digest: String,
    /// The Pollen asset id, once `register_data_asset` has run at the node;
    /// `None` while pending is an honest state, not a missing value.
    pub asset_id: Option<String>,
    pub tag: &'static str,
    pub kind_label: &'static str,
    pub at: Seconds,
}

/// Seal a finalized output: schema validation first, rejection on failure.
///
/// # Errors
/// The schema rejection - the output is not registered in any form.
pub fn finalize_output(
    markdown: &str,
    kind_label: &'static str,
    at: Seconds,
) -> Result<OutputRecord, String> {
    output_schema::validate_markdown_output(markdown.as_bytes())
        .map_err(|e| format!("output rejected by schema: {e}"))?;
    let digest = sha256_hex(markdown.as_bytes());
    Ok(OutputRecord {
        content_id: digest.clone(),
        digest,
        asset_id: None,
        tag: OUTPUT_TAG,
        kind_label,
        at,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Answer;

    #[test]
    fn a_validated_answer_seals_a_record_with_its_own_digest() {
        let answer = Answer::NotFound;
        let md = answer.render_markdown().expect("not-found renders valid");
        let record = finalize_output(&md, "not-found", 42).expect("a schema-valid document seals");
        assert_eq!(record.tag, OUTPUT_TAG);
        assert_eq!(record.content_id, record.digest);
        assert_eq!(record.at, 42);
        assert!(record.asset_id.is_none());
    }

    #[test]
    fn an_invalid_document_is_rejected_not_downgraded() {
        // Heading skip: # then ### without a level-2 heading in between.
        let bad = "# Title\n\n### Sub\n";
        let err = finalize_output(bad, "grounded", 1).expect_err("must be rejected");
        assert!(err.contains("rejected by schema"), "{err}");
    }
}
