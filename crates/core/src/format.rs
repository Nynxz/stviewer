//! Human-readable formatting shared by the CLI and GUI.

/// `1536` -> `"1.50 KiB"`.
pub fn bytes(n: u64) -> String {
    const UNITS: [&str; 6] = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
    if n < 1024 {
        return format!("{n} B");
    }
    let mut v = n as f64;
    let mut i = 0;
    while v >= 1024.0 && i < UNITS.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    format!("{v:.2} {}", UNITS[i])
}

/// `1_234_567_890` -> `"1.23B"`.
pub fn count(n: u64) -> String {
    let (div, suffix) = match n {
        0..1_000 => return n.to_string(),
        1_000..1_000_000 => (1e3, "K"),
        1_000_000..1_000_000_000 => (1e6, "M"),
        1_000_000_000..1_000_000_000_000 => (1e9, "B"),
        _ => (1e12, "T"),
    };
    format!("{:.2}{suffix}", n as f64 / div)
}

/// `[4096, 320]` -> `"[4096, 320]"`.
pub fn shape(s: &[u64]) -> String {
    let inner: Vec<_> = s.iter().map(u64::to_string).collect();
    format!("[{}]", inner.join(", "))
}

/// Case-insensitive name filter: `*` and `?` wildcards if present, else substring.
pub fn name_matches(name: &str, pattern: &str) -> bool {
    let name = name.to_lowercase();
    let pattern = pattern.to_lowercase();
    if !pattern.contains(['*', '?']) {
        return name.contains(&pattern);
    }
    wildcard(name.as_bytes(), pattern.as_bytes())
}

fn wildcard(s: &[u8], p: &[u8]) -> bool {
    let (mut si, mut pi) = (0, 0);
    let mut star: Option<(usize, usize)> = None;
    while si < s.len() {
        match p.get(pi) {
            Some(b'*') => {
                star = Some((pi, si));
                pi += 1;
            }
            Some(&c) if c == b'?' || c == s[si] => {
                si += 1;
                pi += 1;
            }
            _ => match star {
                Some((sp, ss)) => {
                    pi = sp + 1;
                    si = ss + 1;
                    star = Some((sp, ss + 1));
                }
                None => return false,
            },
        }
    }
    p[pi..].iter().all(|&c| c == b'*')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats() {
        assert_eq!(bytes(512), "512 B");
        assert_eq!(bytes(1536), "1.50 KiB");
        assert_eq!(count(999), "999");
        assert_eq!(count(1_234_567_890), "1.23B");
    }

    #[test]
    fn matching() {
        assert!(name_matches("model.layers.0.attn.q_proj.weight", "q_proj"));
        assert!(name_matches("lora_unet_down.alpha", "lora_*.alpha"));
        assert!(!name_matches("lora_unet_down.weight", "lora_*.alpha"));
        assert!(name_matches("abc", "a?c"));
    }
}
