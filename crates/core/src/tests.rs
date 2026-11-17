use super::*;
use std::io::Write as _;

/// Build a safetensors file from a raw JSON header string and data bytes.
fn raw_file(header: &str, data: &[u8]) -> Vec<u8> {
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(data);
    out
}

fn parse(bytes: &[u8]) -> Result<SafeTensorsHeader> {
    read_header_from(&mut &bytes[..], bytes.len() as u64)
}

const GOOD: &str = r#"{"__metadata__":{"format":"pt","b":"2"},"w":{"dtype":"F16","shape":[2,2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[2],"data_offsets":[8,16]}}"#;

#[test]
fn parses_valid_file() {
    let h = parse(&raw_file(GOOD, &[0u8; 16])).unwrap();
    assert_eq!(h.tensors.len(), 2);
    assert_eq!(h.tensors[0].name, "w");
    assert_eq!(h.total_params(), 6);
    assert_eq!(
        h.metadata_strings().collect::<Vec<_>>(),
        [("format", "pt"), ("b", "2")]
    );
}

#[test]
fn rejects_oversized_header() {
    let mut bytes = (MAX_HEADER_LEN + 1).to_le_bytes().to_vec();
    bytes.extend_from_slice(b"{}");
    assert!(matches!(parse(&bytes), Err(Error::HeaderTooLarge(_))));
}

#[test]
fn rejects_header_past_eof() {
    let mut bytes = 1000u64.to_le_bytes().to_vec();
    bytes.extend_from_slice(b"{}");
    assert!(matches!(parse(&bytes), Err(Error::HeaderPastEof { .. })));
}

#[test]
fn rejects_size_mismatch() {
    let h = r#"{"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,8]}}"#;
    assert!(matches!(
        parse(&raw_file(h, &[0u8; 8])),
        Err(Error::BadTensor { .. })
    ));
}

#[test]
fn rejects_gaps_and_trailing_data() {
    let h = r#"{"w":{"dtype":"U8","shape":[4],"data_offsets":[4,8]}}"#;
    assert!(matches!(
        parse(&raw_file(h, &[0u8; 8])),
        Err(Error::Layout(_))
    ));
    let h = r#"{"w":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}"#;
    assert!(matches!(
        parse(&raw_file(h, &[0u8; 8])),
        Err(Error::Layout(_))
    ));
}

#[test]
fn rejects_non_string_metadata() {
    let h = r#"{"__metadata__":{"n":1}}"#;
    assert!(matches!(
        parse(&raw_file(h, &[])),
        Err(Error::BadMetadata(_))
    ));
}

#[test]
fn unknown_dtype_is_a_warning() {
    let h = r#"{"w":{"dtype":"F2_FUTURE","shape":[4],"data_offsets":[0,1]}}"#;
    let parsed = parse(&raw_file(h, &[0u8; 1])).unwrap();
    assert_eq!(parsed.warnings.len(), 1);
}

#[test]
fn serialized_header_is_aligned() {
    let h = parse(&raw_file(GOOD, &[0u8; 16])).unwrap();
    let bytes = serialize_header(&h.metadata, &h.tensors);
    assert_eq!((8 + bytes.len()) % 8, 0);
}

#[test]
fn rewrite_roundtrip_preserves_data() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("m.safetensors");
    let data: Vec<u8> = (0..16).collect();
    std::fs::File::create(&path)
        .unwrap()
        .write_all(&raw_file(GOOD, &data))
        .unwrap();

    let mut meta = Metadata::new();
    meta.insert("hello".into(), "world".into());
    let new = write_with_metadata(&path, &path, &meta, &WriteOptions { backup: true }).unwrap();

    assert_eq!(
        new.metadata_strings().collect::<Vec<_>>(),
        [("hello", "world")]
    );
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[new.data_start() as usize..], &data[..]);
    // Backup still holds the original.
    let bak = parse(&std::fs::read(dir.path().join("m.safetensors.bak")).unwrap()).unwrap();
    assert_eq!(bak.metadata.len(), 2);
}
