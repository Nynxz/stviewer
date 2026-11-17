//! Make untrusted strings (metadata keys/values, tensor names, dtypes) safe to
//! print. A malicious file can embed terminal escape sequences (retitle the
//! window, clear or hide text, write the clipboard via OSC 52, …) or Unicode
//! bidi overrides that visually reorder text. Human-readable output escapes
//! them; JSON output keeps the exact data but never emits them raw.

use std::borrow::Cow;

/// Characters that can change terminal state or visually reorder text:
/// C0/C1 controls and DEL, plus Unicode bidi formatting characters.
pub fn is_unsafe(c: char) -> bool {
    c.is_control()
        || matches!(
            c,
            '\u{061C}' | '\u{200E}' | '\u{200F}' | '\u{202A}'..='\u{202E}' | '\u{2066}'..='\u{2069}'
        )
}

/// Escape unsafe characters as `\u{1b}`-style text. With `keep_layout`,
/// newlines and tabs pass through (for multi-line values).
pub fn clean(s: &str, keep_layout: bool) -> Cow<'_, str> {
    let bad = |c: char| is_unsafe(c) && !(keep_layout && matches!(c, '\n' | '\t'));
    if !s.chars().any(bad) {
        return Cow::Borrowed(s);
    }
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        if bad(c) {
            out.push_str(&format!("\\u{{{:x}}}", c as u32));
        } else {
            out.push(c);
        }
    }
    Cow::Owned(out)
}

/// For serialized JSON: escape C1 controls and bidi characters as `\uXXXX`.
/// serde_json already escapes C0; these can only occur inside strings, so the
/// result is still valid JSON that decodes to exactly the same data.
pub fn json_safe(s: &str) -> Cow<'_, str> {
    if !s.chars().any(|c| !c.is_ascii() && is_unsafe(c)) {
        return Cow::Borrowed(s);
    }
    let mut out = String::with_capacity(s.len() + 16);
    for c in s.chars() {
        if !c.is_ascii() && is_unsafe(c) {
            out.push_str(&format!("\\u{:04x}", c as u32)); // all such chars are in the BMP
        } else {
            out.push(c);
        }
    }
    Cow::Owned(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escapes_terminal_sequences() {
        assert_eq!(clean("\u{1b}]0;pwned\u{7}", false), "\\u{1b}]0;pwned\\u{7}");
        assert_eq!(clean("a\u{202e}b", false), "a\\u{202e}b");
        assert_eq!(clean("\u{9b}31m", false), "\\u{9b}31m");
        assert_eq!(clean("l1\nl2\t", true), "l1\nl2\t");
        assert_eq!(clean("l1\nl2", false), "l1\\u{a}l2");
        assert!(matches!(clean("plain 日本語", false), Cow::Borrowed(_)));
    }

    #[test]
    fn json_stays_valid_and_lossless() {
        let original = "x\u{9b}y\u{202e}z\u{1b}";
        let text = serde_json::to_string(&original).unwrap();
        let safe = json_safe(&text);
        assert!(!safe.chars().any(is_unsafe));
        let back: String = serde_json::from_str(&safe).unwrap();
        assert_eq!(back, original);
    }
}
