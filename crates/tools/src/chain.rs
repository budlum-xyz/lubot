//! # lubot-tools::chain - reading the chain's analysis records
//!
//! The training core scans chain analysis records (`AiInferenceOutcome`) as
//! well as the codebase. This module is the reading side of that: it builds
//! JSON-RPC requests a node understands, parses the responses, and converts
//! an outcome into the exact one-line format the corpus builder consumes
//! (the corpus builder's chain-record line format).
//!
//! The transport stays out of this crate: a request string in, a response
//! string in, a structured record out. A reader that talks HTTP is testable
//! only with a server; a reader that turns strings into records is testable
//! with a fixture, and the fixtures are what say the format is right.

use serde_json::Value;

/// One chain analysis outcome, as the reading core needs it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainOutcome {
    pub request_id: String,
    pub output: String,
    pub verifier: Option<String>,
    pub model_id: Option<String>,
    pub asset_id: Option<String>,
    pub content_id: Option<String>,
}

/// Build a JSON-RPC 2.0 request body (no transport here).
pub fn jsonrpc_request(method: &str, params: Vec<Value>) -> Result<String, String> {
    if !is_allowed_method(method) {
        return Err(format!("disallowed RPC method `{method}`"));
    }
    Ok(serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params,
    })
    .to_string())
}

/// Parse a `bud_aiGetOutcome` response. Missing output text is a refusal:
/// the collector must never guess what a finalized outcome said.
pub fn parse_get_outcome(response: &str) -> Result<ChainOutcome, String> {
    let value: Value =
        serde_json::from_str(response).map_err(|e| format!("not valid JSON-RPC: {e}"))?;
    let error = value.get("error");
    if error.is_some() {
        return Err(format!("RPC error: {error:?}"));
    }
    let result = value.get("result").ok_or("response has no result object")?;
    let request_id = result
        .get("request_id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let output = result
        .get("output")
        .or_else(|| result.get("output_text"))
        .and_then(Value::as_str)
        .ok_or("outcome carries no output text; refusing to guess")?
        .trim()
        .to_string();
    if output.is_empty() {
        return Err("outcome output text is empty; refusing to guess".into());
    }
    Ok(ChainOutcome {
        request_id,
        output,
        verifier: result
            .get("verifier")
            .and_then(Value::as_str)
            .map(str::to_string),
        model_id: result
            .get("model_id")
            .and_then(Value::as_str)
            .map(str::to_string),
        asset_id: result
            .get("asset_id")
            .and_then(Value::as_str)
            .map(str::to_string),
        content_id: result
            .get("content_id")
            .and_then(Value::as_str)
            .map(str::to_string),
    })
}

/// The one-line corpus record format the builder consumes. Every field the
/// pipeline needs is present; provenance that the node did not carry is
/// marked absent rather than invented.
#[must_use]
pub fn to_corpus_line(outcome: &ChainOutcome, asset_id: Option<&str>) -> String {
    let mut record = serde_json::Map::new();
    record.insert("record_type".into(), Value::String("outcome".into()));
    record.insert(
        "request_id".into(),
        Value::String(outcome.request_id.clone()),
    );
    record.insert("text".into(), Value::String(outcome.output.clone()));
    let asset = outcome.asset_id.as_deref().or(asset_id);
    if let Some(asset) = asset {
        record.insert("asset_id".into(), Value::String(asset.to_string()));
    }
    if let Some(content) = &outcome.content_id {
        record.insert("content_id".into(), Value::String(content.clone()));
    }
    if let Some(verifier) = &outcome.verifier {
        record.insert("verifier".into(), Value::String(verifier.clone()));
    }
    if let Some(model) = &outcome.model_id {
        record.insert("model_id".into(), Value::String(model.clone()));
    }
    Value::Object(record).to_string()
}

/// One inference request record, as the reading core needs it (Aşama 5).
///
/// `effort_hash` is the hashed effort ceiling (Aşama 7): a request whose
/// effort tag is outside the `0.5x`-`10.0x` range is refused at parse time,
/// so a request the executor could not have admitted is not a request the
/// corpus may read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AiInferenceRequest {
    pub request_id: String,
    pub model_hash: String,
    pub effort_tag: String,
    pub effort_hash: String,
    pub content_id: String,
    pub modality_tag: u32,
    pub declared_units: u64,
}

/// One inference result record, as the reading core needs it (Aşama 5/11).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AiInferenceResult {
    pub request_id: String,
    pub content_id: String,
    pub agreeing: usize,
    pub threshold: u64,
}

fn str_field(value: &Value, name: &str) -> Result<String, String> {
    value
        .get(name)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .ok_or_else(|| format!("missing field `{name}`"))
}

fn uint_field(value: &Value, name: &str) -> Result<u64, String> {
    value
        .get(name)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("missing field `{name}`"))
}

/// Parse a JSON request record; refuses anything unreadable, an unknown
/// modality tag, a declaration over the modality ceiling, or an effort tag
/// outside the report's range.
///
/// # Errors
/// The first rule the record breaks.
pub fn parse_request(record: &str) -> Result<AiInferenceRequest, String> {
    let value: Value = serde_json::from_str(record).map_err(|e| format!("request: {e}"))?;
    let request = AiInferenceRequest {
        request_id: str_field(&value, "request_id")?,
        model_hash: str_field(&value, "model_hash")?,
        effort_tag: str_field(&value, "effort_tag")?,
        effort_hash: str_field(&value, "effort_hash")?,
        content_id: str_field(&value, "content_id")?,
        modality_tag: uint_field(&value, "modality_tag")? as u32,
        declared_units: uint_field(&value, "declared_units")?,
    };
    // The tier is hashed into the request; a record whose hash does not
    // match its own tag is a record that was tampered with or mis-built.
    let derived = crate::operator::effort_hash(&request.effort_tag)?;
    if derived != request.effort_hash {
        return Err("request refused: effort_hash does not match effort_tag".to_string());
    }
    lubot_read::perception::check_units(request.modality_tag, request.declared_units)
        .map_err(|why| format!("request refused: {}", why.label()))?;
    Ok(request)
}

/// Parse a JSON result record.
///
/// # Errors
/// Missing or empty fields.
pub fn parse_result(record: &str) -> Result<AiInferenceResult, String> {
    let value: Value = serde_json::from_str(record).map_err(|e| format!("result: {e}"))?;
    Ok(AiInferenceResult {
        request_id: str_field(&value, "request_id")?,
        content_id: str_field(&value, "content_id")?,
        agreeing: uint_field(&value, "agreeing")? as usize,
        threshold: uint_field(&value, "threshold")?,
    })
}

/// The only chain methods Lubot may call. The set is registered in
/// `training/rpc-seti.json`; the gate keeps the file and this constant in
/// agreement, so an extension is a reviewable change in both places - and
/// the report's seven stay mandatory.
pub const ALLOWED_METHODS: [&str; 8] = [
    "bud_aiGetModel",
    "bud_aiRegisterModel",
    "bud_aiSubmitRequest",
    "bud_aiSubmitResult",
    "bud_aiGetOutcome",
    "bud_aiGetActiveVerifiers",
    "bud_aiInferenceStats",
];

/// Is this method inside the registered set?
#[must_use]
pub fn is_allowed_method(method: &str) -> bool {
    ALLOWED_METHODS.contains(&method)
}

/// K5 rule: the production threshold for consuming a high-stakes coding
/// output. A single-operator result is NOT consumed while verification is
/// attestation-only (Aşama 11); the transition value is 2 until the zkVM
/// content proof is live for Lubot's model class, then 1.
pub const OPERATOR_THRESHOLD: u64 = 2;

/// Whether a result with this many agreeing verifiers may be consumed.
/// Threshold 0 is refused by construction; the single-operator case is
/// refused while the attestation-only transition stands.
pub fn consumes(agreeing: usize, threshold: u64) -> bool {
    if threshold == 0 {
        return false;
    }
    (agreeing as u64) >= threshold
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_bodies_name_the_method_and_params() {
        let body = jsonrpc_request("bud_aiGetOutcome", vec![Value::String("0x01".into())])
            .expect("valid method");
        assert!(body.contains("\"bud_aiGetOutcome\""));
        assert!(body.contains("0x01"));
        assert!(body.contains("\"jsonrpc\":\"2.0\""));
    }

    #[test]
    fn parse_get_outcome_extracts_the_fields() {
        let response = r#"{"jsonrpc":"2.0","id":1,"result":{"request_id":"0x01","output":"finalized.","verifier":"0xOP","model_id":"0xMODEL","asset_id":"0xAA","content_id":"0xBB"}}"#;
        let parsed = parse_get_outcome(response).expect("parses");
        assert_eq!(parsed.request_id, "0x01");
        assert_eq!(parsed.output, "finalized.");
        assert_eq!(parsed.verifier.as_deref(), Some("0xOP"));
        assert_eq!(parsed.model_id.as_deref(), Some("0xMODEL"));
        assert_eq!(parsed.asset_id.as_deref(), Some("0xAA"));
        assert_eq!(parsed.content_id.as_deref(), Some("0xBB"));
    }

    #[test]
    fn missing_output_text_is_a_refusal_not_a_guess() {
        let response = r#"{"jsonrpc":"2.0","id":1,"result":{"request_id":"0x02"}}"#;
        let err = parse_get_outcome(response).unwrap_err();
        assert!(err.contains("refusing to guess"));
    }

    #[test]
    fn an_rpc_error_is_propagated_named() {
        let response =
            r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32601,"message":"no such method"}}"#;
        let err = parse_get_outcome(response).unwrap_err();
        assert!(err.contains("RPC error"));
    }

    #[test]
    fn the_surface_refuses_the_unregistered() {
        for method in ALLOWED_METHODS {
            assert!(is_allowed_method(method));
        }
        assert!(!is_allowed_method("bud_aiDisputeSlash"));
        assert!(!is_allowed_method("bud_pollenGetTrainingGrants"));
        assert!(!is_allowed_method("unrelated"));
    }

    #[test]
    fn the_extended_surface_is_the_registered_set() {
        for method in ALLOWED_METHODS {
            assert!(is_allowed_method(method));
        }
        assert!(
            !is_allowed_method("bud_aiGetCeilings"),
            "the ceilings flow was removed as callerless; a method with no parser must not be allowed"
        );
        // The report's seven stay mandatory even with the extension.
        for method in [
            "bud_aiGetModel",
            "bud_aiRegisterModel",
            "bud_aiSubmitRequest",
            "bud_aiSubmitResult",
            "bud_aiGetOutcome",
            "bud_aiGetActiveVerifiers",
            "bud_aiInferenceStats",
        ] {
            assert!(ALLOWED_METHODS.contains(&method), "{method} lost");
        }
        assert!(!is_allowed_method("bud_aiDisputeSlash"));
    }

    #[test]
    fn single_operator_results_are_not_consumed_while_attestation_only() {
        // Transition stands (K5): threshold is 2, one agreeing verifier is a
        // refusal even though it equals nothing a threshold of 1 would stop.
        assert!(!consumes(1, OPERATOR_THRESHOLD));
        assert!(consumes(2, OPERATOR_THRESHOLD));
        // A threshold of zero is a configuration error, not a door.
        assert!(!consumes(100, 0));
    }

    #[test]
    fn request_parse_enforces_effort_hash_and_modality_ceiling() {
        let good = r#"{"request_id":"0x01","model_hash":"0xM","effort_tag":"2.0x","effort_hash":"H2","content_id":"0xC","modality_tag":1,"declared_units":100}"#;
        // Correct hash for 2.0x:
        let good = good.replace(
            "H2",
            &super::super::operator::effort_hash("2.0x").expect("ok"),
        );
        let parsed = parse_request(&good).expect("a well-formed request parses");
        assert_eq!(parsed.effort_tag, "2.0x");
        assert_eq!(parsed.modality_tag, 1);
        // A tag whose hash does not match is refused.
        let tampered = good.replace("2.0x", "4.0x");
        assert!(parse_request(&tampered).is_err());
        // An over-ceiling declaration is refused.
        let over = good.replace("\"declared_units\":100", "\"declared_units\":99999999999");
        assert!(parse_request(&over).is_err());
        // An unknown modality tag is refused even with zero units.
        let unknown = good.replace("\"modality_tag\":1", "\"modality_tag\":9");
        assert!(parse_request(&unknown).is_err());
    }

    #[test]
    fn result_parse_reads_the_agreement_numbers() {
        let line = r#"{"request_id":"0x01","content_id":"0xC","agreeing":2,"threshold":2}"#;
        let result = parse_result(line).expect("a well-formed result parses");
        assert_eq!(result.agreeing, 2);
        assert!(consumes(result.agreeing, result.threshold));
        assert!(!consumes(1, result.threshold));
        let incomplete = r#"{"request_id":"0x01","content_id":"0xC"}"#;
        assert!(parse_result(incomplete).is_err());
    }

    #[test]
    fn corpus_line_keeps_the_dump_field_names() {
        let outcome = ChainOutcome {
            request_id: "0x01".into(),
            output: "finalized.".into(),
            verifier: Some("0xOP".into()),
            model_id: None,
            asset_id: None,
            content_id: None,
        };
        let line = to_corpus_line(&outcome, Some("0xAA"));
        let value: Value = serde_json::from_str(&line).expect("valid json");
        assert_eq!(value["record_type"], "outcome");
        assert_eq!(value["asset_id"], "0xAA");
        assert_eq!(value["text"], "finalized.");
        assert!(value.get("content_id").is_none());
    }
    #[test]
    fn jsonrpc_request_rejects_disallowed_method() {
        assert!(jsonrpc_request("bud_aiDisputeSlash", vec![]).is_err());
        assert!(jsonrpc_request("unknownMethod", vec![]).is_err());
    }
}
