//! Hostile-input tests: the parser must reject or safely accept anything,
//! never panic, overflow, hang, or allocate based on untrusted sizes.
//! (Tests build with overflow checks on, so any arithmetic overflow fails here.)

use std::time::{Duration, Instant};
use stv_core::{Error, MAX_HEADER_LEN, SafeTensorsHeader, read_header_from, serialize_header};

fn file(header: &str, data_len: usize) -> Vec<u8> {
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.resize(out.len() + data_len, 0);
    out
}

fn parse(bytes: &[u8]) -> Result<SafeTensorsHeader, Error> {
    read_header_from(&mut &bytes[..], bytes.len() as u64)
}

/// Everything a consumer (CLI/GUI) does with a parsed header must be safe too.
fn exercise(h: &SafeTensorsHeader) {
    let _ = h.total_params();
    let _ = h.dtype_breakdown();
    let _ = serialize_header(&h.metadata, &h.tensors);
    let mut covered = 0u64;
    for t in &h.tensors {
        let _ = t.numel();
        assert!(t.data_offsets.1 <= h.data_len(), "tensor past end of data");
        covered += t.byte_len();
    }
    assert_eq!(covered, h.data_len(), "tensors don't tile the data section");
    for (k, v) in h.metadata_strings() {
        let _ = (k.len(), v.len());
    }
}

#[test]
fn length_prefix_extremes() {
    for n in [
        0u64,
        1,
        7,
        MAX_HEADER_LEN,
        MAX_HEADER_LEN + 1,
        u64::MAX / 2,
        u64::MAX - 7,
        u64::MAX,
    ] {
        let mut bytes = n.to_le_bytes().to_vec();
        bytes.extend_from_slice(b"{}");
        let r = parse(&bytes);
        assert!(r.is_err(), "length {n} accepted");
    }
}

#[test]
fn truncated_files() {
    for len in 0..8 {
        assert!(matches!(parse(&vec![0u8; len]), Err(Error::TooSmall(_))));
    }
}

#[test]
fn size_claims_never_drive_allocation() {
    // Header claims 100 MiB but the file is tiny: must fail on the size check
    // before allocating the buffer.
    let mut bytes = MAX_HEADER_LEN.to_le_bytes().to_vec();
    bytes.extend_from_slice(b"{}");
    assert!(matches!(parse(&bytes), Err(Error::HeaderPastEof { .. })));
}

#[test]
fn shape_overflow_is_rejected() {
    let cases = [
        // Product overflows u64.
        r#"{"t":{"dtype":"F32","shape":[4294967296,4294967296],"data_offsets":[0,4]}}"#,
        // numel fits, but numel * bits overflows.
        r#"{"t":{"dtype":"F64","shape":[4611686018427387904],"data_offsets":[0,8]}}"#,
        // Unknown dtype: size can't be checked, but numel must still be bounded.
        r#"{"a":{"dtype":"X9","shape":[4294967296,2147483648],"data_offsets":[0,1]},"b":{"dtype":"X9","shape":[4294967296,2147483648],"data_offsets":[1,2]}}"#,
    ];
    for h in cases {
        // Give each case exactly as many data bytes as its offsets claim, so it
        // is the overflow (not a layout mismatch) that has to reject it.
        let data_len = if h.contains("[1,2]") {
            2
        } else if h.contains("[0,8]") {
            8
        } else {
            4
        };
        let r = parse(&file(h, data_len));
        match r {
            Err(_) => {}
            Ok(parsed) => {
                exercise(&parsed);
                panic!("accepted: {h}");
            }
        }
    }
}

#[test]
fn unknown_dtypes_cannot_inflate_parameter_counts() {
    // Many unknown-dtype tensors each claiming a huge element count in 1 byte.
    let mut h = String::from("{");
    for i in 0..64 {
        if i > 0 {
            h.push(',');
        }
        // 2^62 elements each: 64 of them sum past u64::MAX.
        h.push_str(&format!(
            r#""t{i}":{{"dtype":"Q1","shape":[4611686018427387904],"data_offsets":[{i},{}]}}"#,
            i + 1
        ));
    }
    h.push('}');
    if let Ok(parsed) = parse(&file(&h, 64)) {
        exercise(&parsed);
    }
}

#[test]
fn bad_offsets_and_layouts() {
    let cases = [
        r#"{"t":{"dtype":"U8","shape":[4],"data_offsets":[0,99999999999999999999]}}"#,
        r#"{"t":{"dtype":"U8","shape":[4],"data_offsets":[-4,0]}}"#,
        r#"{"t":{"dtype":"U8","shape":[4],"data_offsets":[4.0,8.0]}}"#,
        r#"{"t":{"dtype":"U8","shape":[4],"data_offsets":[8,4]}}"#,
        r#"{"t":{"dtype":"U8","shape":[4],"data_offsets":[0,4,8]}}"#,
        r#"{"t":{"dtype":"U8","shape":[4],"data_offsets":"0,4"}}"#,
        r#"{"t":{"dtype":"U8","shape":[-4],"data_offsets":[0,4]}}"#,
        r#"{"t":{"dtype":"U8","shape":[2.5],"data_offsets":[0,4]}}"#,
        r#"{"t":{"dtype":7,"shape":[4],"data_offsets":[0,4]}}"#,
        r#"{"t":{"shape":[4],"data_offsets":[0,4]}}"#,
        r#"{"t":[1,2,3]}"#,
        // Overlap.
        r#"{"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"b":{"dtype":"U8","shape":[4],"data_offsets":[2,6]}}"#,
        // Gap.
        r#"{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"b":{"dtype":"U8","shape":[2],"data_offsets":[4,6]}}"#,
        // Points past the data section.
        r#"{"t":{"dtype":"U8","shape":[16],"data_offsets":[0,16]}}"#,
        // Duplicate name: the first entry is silently replaced, leaving a gap.
        r#"{"t":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"t":{"dtype":"U8","shape":[4],"data_offsets":[4,8]}}"#,
        // A tensor-shaped object posing as metadata.
        r#"{"__metadata__":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}"#,
        r#"{"__metadata__":"x"}"#,
        r#"{"__metadata__":{"k":{"nested":"obj"}}}"#,
        r#"[]"#,
        r#""str""#,
        r#"null"#,
        "",
        "   ",
        "{",
    ];
    for h in cases {
        assert!(parse(&file(h, 8)).is_err(), "accepted: {h}");
    }
}

#[test]
fn non_utf8_and_binary_headers() {
    let mut bytes = 6u64.to_le_bytes().to_vec();
    bytes.extend_from_slice(&[b'{', 0xff, 0xfe, b'"', 0x00, b'}']);
    assert!(parse(&bytes).is_err());
}

#[test]
fn deep_nesting_does_not_overflow_the_stack() {
    let deep = format!(
        r#"{{"__metadata__":{}"x"{}}}"#,
        "[".repeat(100_000),
        "]".repeat(100_000)
    );
    assert!(parse(&file(&deep, 0)).is_err());
    let deep_obj = format!(
        r#"{{"t":{}1{}}}"#,
        r#"{"a":"#.repeat(100_000),
        "}".repeat(100_000)
    );
    assert!(parse(&file(&deep_obj, 0)).is_err());
}

#[test]
fn many_tensors_parse_in_linear_time() {
    // 200k zero-sized tensors at the same offset is valid; it must not go quadratic.
    let mut h = String::from("{");
    for i in 0..200_000 {
        if i > 0 {
            h.push(',');
        }
        h.push_str(&format!(
            r#""t{i}":{{"dtype":"F32","shape":[0],"data_offsets":[0,0]}}"#
        ));
    }
    h.push('}');
    let start = Instant::now();
    let parsed = parse(&file(&h, 0)).unwrap();
    exercise(&parsed);
    assert_eq!(parsed.tensors.len(), 200_000);
    assert!(
        start.elapsed() < Duration::from_secs(10),
        "took {:?}",
        start.elapsed()
    );
}

#[test]
fn control_characters_are_preserved_as_data() {
    // Escape sequences in names/values are data; the parser must not choke or
    // alter them (sanitising is the display layer's job).
    let h = r#"{"__metadata__":{"\u001b]0;pwned\u0007":"\u001b[2J\u202e"},"\u001b[31mt":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#;
    let parsed = parse(&file(h, 1)).unwrap();
    assert!(parsed.metadata.contains_key("\u{1b}]0;pwned\u{7}"));
    assert_eq!(parsed.tensors[0].name, "\u{1b}[31mt");
}

/// Small xorshift PRNG so the fuzz test is deterministic and dependency-free.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

#[test]
fn fuzz_mutated_files_never_panic() {
    let seeds = [
        r#"{"__metadata__":{"format":"pt","k":"v"},"a.weight":{"dtype":"F16","shape":[2,2],"data_offsets":[0,8]},"a.bias":{"dtype":"F32","shape":[2],"data_offsets":[8,16]},"s":{"dtype":"F4","shape":[3],"data_offsets":[16,18]}}"#,
        r#"{"x":{"dtype":"BF16","shape":[],"data_offsets":[0,2]},"y":{"dtype":"Q7","shape":[5],"data_offsets":[2,4]}}"#,
    ];
    let tokens: &[&[u8]] = &[
        b"18446744073709551615",
        b"18446744073709551616",
        b"-1",
        b"0",
        b"1e308",
        b"9999999999999",
        b"[",
        b"]",
        b"{",
        b"}",
        b",",
        b":",
        b"\"",
        b"\\u0000",
        b"\"__metadata__\"",
        b"null",
        b"[0,0]",
        b"\"F8_E4M3\"",
        b"\"Q9\"",
        b"[4294967296,4294967296]",
    ];
    let mut rng = Rng(0x5eed_1234_abcd_ef01);
    let iterations = if cfg!(debug_assertions) {
        30_000
    } else {
        200_000
    };
    let mut accepted = 0;
    for i in 0..iterations {
        let seed = seeds[i % seeds.len()];
        let mut header = seed.as_bytes().to_vec();
        for _ in 0..1 + rng.below(4) {
            let pos = rng.below(header.len() + 1);
            match rng.below(4) {
                0 if !header.is_empty() => {
                    let at = pos.min(header.len() - 1);
                    header[at] = rng.next() as u8;
                }
                1 if pos < header.len() => {
                    let end = (pos + 1 + rng.below(8)).min(header.len());
                    header.drain(pos..end);
                }
                _ => {
                    let t = tokens[rng.below(tokens.len())];
                    header.splice(pos..pos, t.iter().copied());
                }
            }
        }
        let data_len = rng.below(24);
        // Sometimes lie about the header length.
        let claimed = match rng.below(8) {
            0 => rng.next(),
            1 => (header.len() as u64).wrapping_add(rng.below(16) as u64),
            _ => header.len() as u64,
        };
        let mut bytes = claimed.to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        bytes.resize(bytes.len() + data_len, 0xAB);
        if let Ok(h) = parse(&bytes) {
            exercise(&h);
            accepted += 1;
        }
    }
    // Sanity: the fuzzer should exercise both the accept and reject paths.
    assert!(
        accepted > 0 && accepted < iterations,
        "accepted {accepted}/{iterations}"
    );
}
