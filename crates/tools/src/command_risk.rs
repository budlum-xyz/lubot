#![forbid(unsafe_code)]
//! # command_risk - deterministic command classification
//!
//! The workspace command-review power, encoded as syntax: shapes that are
//! destructive or history-rewriting, with one line of reason each. The
//! classifier only reports; the caller decides what a refusal costs. The
//! pattern set is closed and matched literally (no shell evaluation, no
//! execution) - classification is a reading, not a sandbox.

/// Non-destructive command shapes reported for their effect.
pub const OVERWRITE_PATTERNS: [(&str, &str); 5] = [
    (":(){", "fork bomb"),
    ("dd if=/dev/zero of=/dev/", "device overwrite"),
    ("mkfs.", "device format"),
    ("git push --force", "history rewrite"),
    ("git push -f", "history rewrite"),
];

/// Command tokens that never name a file target: flags and mode words.
fn is_flag(token: &str) -> bool {
    matches!(
        token,
        "-r" | "-f" | "-rf" | "-fr" | "--" | "--recursive" | "--force"
    )
}

/// The reason for the first risky shape in `line`, if any.
#[must_use]
pub fn classify(line: &str) -> Option<&'static str> {
    for (pattern, reason) in OVERWRITE_PATTERNS {
        if line.contains(pattern) {
            return Some(reason);
        }
    }
    // `rm` is parsed by argument, so `rm -rf /tmp` is never reported as a
    // root wipe: only an exact root `/` target is. The command word may sit
    // behind a prefix such as `sudo`.
    let mut words: Vec<&str> = line.split_whitespace().collect();
    let position = words.iter().position(|w| *w == "rm" || *w == "rmdir")?;
    if words[position] == "rmdir" {
        return None;
    }
    for token in words.drain(position + 1..) {
        if is_flag(token) {
            continue;
        }
        return match token {
            "/" => Some("destructive root wipe"),
            "~" => Some("destructive home wipe"),
            "." => Some("destructive cwd wipe"),
            _ if token.starts_with("~/") => Some("destructive home wipe"),
            _ if token.starts_with("./") => Some("destructive cwd wipe"),
            _ => None,
        };
    }
    None
}

/// Every risky line in `text`, as (1-based line number, reason).
#[must_use]
pub fn classify_lines(text: &str) -> Vec<(usize, &'static str)> {
    text.lines()
        .enumerate()
        .filter_map(|(index, line)| classify(line).map(|reason| (index + 1, reason)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_destructive_shapes_are_named() {
        assert_eq!(
            classify("sudo rm -rf / --no-preserve-root"),
            Some("destructive root wipe")
        );
        assert_eq!(classify("rm -rf ~/secrets"), Some("destructive home wipe"));
        assert_eq!(classify("rm -rf ./target"), Some("destructive cwd wipe"));
        assert_eq!(
            classify("git push --force origin main"),
            Some("history rewrite")
        );
        assert_eq!(classify("git push -f origin usl"), Some("history rewrite"));
        assert_eq!(classify(":(){ :|:& };:"), Some("fork bomb"));
        assert_eq!(
            classify("dd if=/dev/zero of=/dev/sda bs=4M"),
            Some("device overwrite")
        );
        assert_eq!(classify("mkfs.ext4 /dev/sdb1"), Some("device format"));
    }

    #[test]
    fn an_absolute_path_under_root_is_not_a_root_wipe() {
        assert_eq!(classify("rm -rf /tmp/build"), None);
        assert_eq!(classify("rm -rf -- /var/log"), None);
    }

    #[test]
    fn a_benign_invocation_is_not_reported() {
        assert_eq!(classify("git push origin usl"), None);
        assert_eq!(classify("rm draft.txt"), None);
        assert_eq!(classify("cargo test --workspace"), None);
        assert_eq!(classify("cargo clippy"), None);
    }

    #[test]
    fn the_lines_report_keeps_positions() {
        let text = "cargo test\ngit push -f main\ncargo clippy\n";
        let hits = classify_lines(text);
        assert_eq!(hits, vec![(2, "history rewrite")]);
    }
}
