#![forbid(unsafe_code)]
//! # secrets - credential shapes that must never reach a push
//!
//! The credential scan, in the closed shape this repository likes: a
//! fixed list of credential shapes, matched literally (prefix + exact
//! length + character set), no heuristics, no lookalikes. A truncated
//! remainder is a *mention*, not a credential: `ghp_` with nine characters
//! is a documentation example, and it is not reported.
//!
//! One self-rule: the scanner does not scan its own source. The shapes live
//! there as literals, so scanning it would be testing the pattern against
//! itself - a tautology, not a check.

/// One credential hit: where it was found, and what it looks like.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hit {
    pub line: usize,
    pub kind: &'static str,
    pub vendor: &'static str,
}

/// A GitHub classic token (`ghp_`/`gho_`/`ghu_`/`ghs_` + 36 alphanumerics).
fn github_classic(line: &str) -> bool {
    for prefix in ["ghp_", "gho_", "ghu_", "ghs_"] {
        if exact_alnum_tail(line, prefix, 36) {
            return true;
        }
    }
    false
}

/// A GitHub fine-grained token (`github_pat_` + 20+ of `[A-Za-z0-9_]`).
fn github_fine(line: &str) -> bool {
    const PREFIX: &str = "github_pat_";
    let Some(at) = line.find(PREFIX) else {
        return false;
    };
    let rest = &line[at + PREFIX.len()..];
    const MIN: usize = 20;
    rest.len() >= MIN
        && rest[..MIN]
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// A model-API key (`sk-` + 20+ alphanumerics).
fn model_api_style(line: &str) -> bool {
    const PREFIX: &str = "sk-";
    let Some(at) = line.find(PREFIX) else {
        return false;
    };
    let rest = &line[at + PREFIX.len()..];
    const MIN: usize = 20;
    rest.len() >= MIN && rest[..MIN].chars().all(|c| c.is_ascii_alphanumeric())
}

/// An AWS access key id (`AKIA` + 16 uppercase alphanumerics).
fn aws_access(line: &str) -> bool {
    const PREFIX: &str = "AKIA";
    let Some(at) = line.find(PREFIX) else {
        return false;
    };
    let rest = &line[at + PREFIX.len()..];
    const COUNT: usize = 16;
    rest.len() >= COUNT
        && rest[..COUNT]
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit())
}

/// A Slack token (`xoxb-`/`xoxp-`/`xoxa-`/`xoxr-`/`xoxs-` + 10+ of
/// `[A-Za-z0-9-]`).
fn slack_token(line: &str) -> bool {
    for prefix in ["xoxb-", "xoxp-", "xoxa-", "xoxr-", "xoxs-"] {
        let Some(at) = line.find(prefix) else {
            continue;
        };
        let rest = &line[at + prefix.len()..];
        const MIN: usize = 10;
        if rest.len() >= MIN
            && rest[..MIN]
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-')
        {
            return true;
        }
    }
    false
}

/// A private key block (`-----BEGIN ... PRIVATE KEY-----`).
fn pem_block(line: &str) -> bool {
    let Some(at) = line.find("-----BEGIN ") else {
        return false;
    };
    line[at..].contains("PRIVATE KEY")
}

/// Exact prefix with an exact-length alphanumeric tail: the strictest shape.
fn exact_alnum_tail(line: &str, prefix: &str, count: usize) -> bool {
    let Some(at) = line.find(prefix) else {
        return false;
    };
    let rest = &line[at + prefix.len()..];
    rest.len() >= count && rest[..count].chars().all(|c| c.is_ascii_alphanumeric())
}

/// Scan one line; the first shape found wins, with its name.
#[must_use]
pub fn scan_line(line: &str) -> Option<(&'static str, &'static str)> {
    if github_classic(line) {
        return Some(("github token", "github"));
    }
    if github_fine(line) {
        return Some(("github fine-grained token", "github"));
    }
    if model_api_style(line) {
        return Some(("model-api key", "model-api"));
    }
    if aws_access(line) {
        return Some(("aws access key id", "aws"));
    }
    if slack_token(line) {
        return Some(("slack token", "slack"));
    }
    if pem_block(line) {
        return Some(("private key block", "pem"));
    }
    None
}

/// Scan a whole text, with 1-based line numbers.
#[must_use]
pub fn scan(text: &str) -> Vec<Hit> {
    text.lines()
        .enumerate()
        .filter_map(|(index, line)| {
            let (kind, vendor) = scan_line(line)?;
            Some(Hit {
                line: index + 1,
                kind,
                vendor,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // The fixtures are assembled at runtime so this source never carries a
    // complete credential shape - the scanner must not find itself.
    fn classic() -> String {
        format!("ghp_{}", "A".repeat(36))
    }

    #[test]
    fn the_credential_shapes_are_named() {
        assert_eq!(scan_line(&classic()), Some(("github token", "github")));
        assert_eq!(
            scan_line(&format!("github_pat_{}", "B".repeat(25))),
            Some(("github fine-grained token", "github"))
        );
        assert_eq!(
            scan_line(&format!("sk-{}", "C".repeat(25))),
            Some(("model-api key", "model-api"))
        );
        assert_eq!(
            scan_line(&format!("AKIA{}", "D".repeat(16))),
            Some(("aws access key id", "aws"))
        );
        assert_eq!(
            scan_line(&format!("xoxb-{}", "E".repeat(12))),
            Some(("slack token", "slack"))
        );
        let pem = "-----BEGIN OPENSSH PRIVATE KEY-----\n";
        assert_eq!(scan_line(pem), Some(("private key block", "pem")));
    }

    #[test]
    fn a_mention_is_not_a_credential() {
        assert_eq!(scan_line("ghp_short"), None);
        assert_eq!(scan_line("sk-abc"), None);
        assert_eq!(scan_line("AKIA1234"), None);
        assert_eq!(scan_line("xoxb-abc"), None);
        assert_eq!(scan_line("the key is in the vault"), None);
    }

    #[test]
    fn truncated_lengths_do_not_fire() {
        assert_eq!(scan_line(&format!("ghp_{}", "A".repeat(35))), None);
        assert_eq!(scan_line(&format!("sk-{}", "a".repeat(19))), None);
    }

    #[test]
    fn scan_reports_lines_and_kinds() {
        let text = format!("merhaba\n{}\nson", classic());
        let hits = scan(&text);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].line, 2);
        assert_eq!(hits[0].kind, "github token");
    }
}
