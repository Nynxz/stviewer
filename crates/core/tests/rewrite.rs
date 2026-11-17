//! End-to-end tests for metadata rewriting, cross-checked against Hugging
//! Face's reference `safetensors` crate: fixtures are written by it, and every
//! file we produce must load in it with identical tensors.

use safetensors::tensor::{Dtype as RefDtype, SafeTensors, TensorView, serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use stv_core::{Error, Metadata, WriteOptions, read_header, write_with_metadata};

/// Deterministic, non-trivial tensor bytes (so a misplaced offset can't pass by luck).
fn bytes(seed: u8, len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
        .collect()
}

/// Write a reference-crate file with mixed dtypes into `dir`.
fn fixture(dir: &Path, meta: &[(&str, &str)]) -> PathBuf {
    let a = bytes(1, 4 * 3 * 2); // F32 [3,2]
    let b = bytes(2, 2 * 5); // BF16 [5]
    let c = bytes(3, 7); // U8 [7]
    let d = bytes(4, 4); // F32 scalar
    let tensors = vec![
        (
            "layers.0.weight",
            TensorView::new(RefDtype::F32, vec![3, 2], &a).unwrap(),
        ),
        (
            "layers.0.bias",
            TensorView::new(RefDtype::BF16, vec![5], &b).unwrap(),
        ),
        ("embed", TensorView::new(RefDtype::U8, vec![7], &c).unwrap()),
        ("scale", TensorView::new(RefDtype::F32, vec![], &d).unwrap()),
    ];
    let info = (!meta.is_empty()).then(|| {
        meta.iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect::<HashMap<_, _>>()
    });
    let path = dir.join("model.safetensors");
    fs::write(&path, serialize(tensors, info).unwrap()).unwrap();
    path
}

fn meta(pairs: &[(&str, &str)]) -> Metadata {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), Value::String(v.to_string())))
        .collect()
}

/// name -> (dtype, shape, data)
type Tensors = HashMap<String, (String, Vec<usize>, Vec<u8>)>;

/// Load with the reference crate; return (metadata, tensors).
fn reference_load(path: &Path) -> (HashMap<String, String>, Tensors) {
    let buf = fs::read(path).unwrap();
    let (_, m) = SafeTensors::read_metadata(&buf).expect("reference crate rejected our file");
    let st = SafeTensors::deserialize(&buf).expect("reference crate rejected our file");
    let tensors = st
        .tensors()
        .into_iter()
        .map(|(n, t)| {
            (
                n,
                (
                    format!("{:?}", t.dtype()),
                    t.shape().to_vec(),
                    t.data().to_vec(),
                ),
            )
        })
        .collect();
    (m.metadata().clone().unwrap_or_default(), tensors)
}

fn no_temp_files(dir: &Path) -> bool {
    fs::read_dir(dir).unwrap().all(|e| {
        !e.unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with(".stv-")
    })
}

#[test]
fn reads_reference_files() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(dir.path(), &[("format", "pt")]);
    let h = read_header(&path).unwrap();
    assert_eq!(h.tensors.len(), 4);
    assert_eq!(h.metadata_strings().collect::<Vec<_>>(), [("format", "pt")]);
    assert_eq!(h.total_params(), 6 + 5 + 7 + 1);
}

#[test]
fn set_update_and_remove_keys_in_place() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(dir.path(), &[("format", "pt"), ("old", "x")]);
    let (_, before) = reference_load(&path);

    let new = meta(&[("format", "safetensors"), ("added", "yes")]);
    write_with_metadata(&path, &path, &new, &WriteOptions::default()).unwrap();

    let (m, after) = reference_load(&path);
    assert_eq!(m.len(), 2);
    assert_eq!(m["format"], "safetensors");
    assert_eq!(m["added"], "yes");
    assert!(!m.contains_key("old"));
    assert_eq!(before, after, "tensors changed");
    assert!(no_temp_files(dir.path()));
}

#[test]
fn key_order_is_preserved() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(dir.path(), &[]);
    let new = meta(&[("z", "1"), ("a", "2"), ("m", "3")]);
    let h = write_with_metadata(&path, &path, &new, &WriteOptions::default()).unwrap();
    assert_eq!(h.metadata.keys().collect::<Vec<_>>(), ["z", "a", "m"]);
    // Tensor order in the header is untouched too.
    let names: Vec<_> = read_header(&path)
        .unwrap()
        .tensors
        .into_iter()
        .map(|t| t.name)
        .collect();
    let orig: Vec<_> = read_header(fixture(tempfile::tempdir().unwrap().path(), &[]))
        .unwrap()
        .tensors
        .into_iter()
        .map(|t| t.name)
        .collect();
    assert_eq!(names, orig);
}

#[test]
fn clearing_metadata_drops_the_key_entirely() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(dir.path(), &[("a", "1"), ("b", "2")]);
    write_with_metadata(&path, &path, &Metadata::new(), &WriteOptions::default()).unwrap();
    let raw = fs::read(&path).unwrap();
    let n = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
    let header = std::str::from_utf8(&raw[8..8 + n]).unwrap();
    assert!(
        !header.contains("__metadata__"),
        "empty __metadata__ left behind: {header}"
    );
    assert!(reference_load(&path).0.is_empty());
}

#[test]
fn awkward_values_survive_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(dir.path(), &[]);
    let big = "x".repeat(2 * 1024 * 1024);
    let json = r#"{"nested": {"list": [1, 2, 3]}, "quote": "\"hi\""}"#;
    let cases = [
        ("unicode", "日本語 · émoji 🚀 · Ελληνικά"),
        ("newlines", "line 1\nline 2\r\n\ttabbed"),
        ("quotes", r#"she said "hi" \ back\slash"#),
        ("json", json),
        ("empty", ""),
        ("big", big.as_str()),
        ("ключ 🔑", "non-ascii key"),
    ];
    write_with_metadata(&path, &path, &meta(&cases), &WriteOptions::default()).unwrap();
    let (m, _) = reference_load(&path);
    for (k, v) in cases {
        assert_eq!(m[k], v, "value for {k:?} changed");
    }
}

#[test]
fn output_path_leaves_source_untouched() {
    let dir = tempfile::tempdir().unwrap();
    let src = fixture(dir.path(), &[("keep", "me")]);
    let original = fs::read(&src).unwrap();
    let dst = dir.path().join("out.safetensors");
    write_with_metadata(&src, &dst, &meta(&[("new", "1")]), &WriteOptions::default()).unwrap();

    assert_eq!(fs::read(&src).unwrap(), original);
    let (m, t_dst) = reference_load(&dst);
    assert_eq!(m.len(), 1);
    assert_eq!(t_dst, reference_load(&src).1);
}

#[test]
fn backup_holds_exact_original_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(dir.path(), &[("v", "1")]);
    let original = fs::read(&path).unwrap();
    let opts = WriteOptions { backup: true };
    write_with_metadata(&path, &path, &meta(&[("v", "2")]), &opts).unwrap();
    assert_eq!(
        fs::read(dir.path().join("model.safetensors.bak")).unwrap(),
        original
    );
    // A second save refreshes the backup to the then-current file.
    let second = fs::read(&path).unwrap();
    write_with_metadata(&path, &path, &meta(&[("v", "3")]), &opts).unwrap();
    assert_eq!(
        fs::read(dir.path().join("model.safetensors.bak")).unwrap(),
        second
    );
    assert_eq!(reference_load(&path).0["v"], "3");
}

#[test]
fn rewriting_is_deterministic() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(dir.path(), &[]);
    let m = meta(&[("a", "1"), ("b", "2")]);
    write_with_metadata(&path, &path, &m, &WriteOptions::default()).unwrap();
    let first = fs::read(&path).unwrap();
    write_with_metadata(&path, &path, &m, &WriteOptions::default()).unwrap();
    assert_eq!(fs::read(&path).unwrap(), first);
}

#[test]
fn header_is_padded_to_eight_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(dir.path(), &[]);
    for len in 0..16 {
        let h = write_with_metadata(
            &path,
            &path,
            &meta(&[("k", &"v".repeat(len))]),
            &WriteOptions::default(),
        )
        .unwrap();
        assert_eq!(h.data_start() % 8, 0, "unaligned with value len {len}");
    }
}

#[test]
fn invalid_source_fails_without_side_effects() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("bad.safetensors");
    fs::write(&src, b"\x10\0\0\0\0\0\0\0not json at all!").unwrap();
    let dst = dir.path().join("out.safetensors");
    let err = write_with_metadata(&src, &dst, &meta(&[("a", "b")]), &WriteOptions::default());
    assert!(matches!(err, Err(Error::Json(_))), "{err:?}");
    assert!(!dst.exists());
    assert!(no_temp_files(dir.path()));
}

#[test]
fn non_string_metadata_is_rejected_before_writing() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(dir.path(), &[("a", "1")]);
    let original = fs::read(&path).unwrap();
    let mut m = Metadata::new();
    m.insert("n".into(), Value::from(5));
    assert!(write_with_metadata(&path, &path, &m, &WriteOptions::default()).is_err());
    assert_eq!(fs::read(&path).unwrap(), original);
    assert!(no_temp_files(dir.path()));
}

#[test]
fn oversized_header_is_refused_before_writing() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(dir.path(), &[]);
    let original = fs::read(&path).unwrap();
    let huge = "x".repeat(stv_core::MAX_HEADER_LEN as usize + 1);
    let err = write_with_metadata(
        &path,
        &path,
        &meta(&[("huge", &huge)]),
        &WriteOptions::default(),
    );
    assert!(matches!(err, Err(Error::HeaderTooLarge(_))), "{err:?}");
    assert_eq!(fs::read(&path).unwrap(), original);
    assert!(no_temp_files(dir.path()));
}

#[cfg(unix)]
#[test]
fn permissions_are_preserved() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(dir.path(), &[]);
    fs::set_permissions(&path, fs::Permissions::from_mode(0o640)).unwrap();
    write_with_metadata(&path, &path, &meta(&[("a", "1")]), &WriteOptions::default()).unwrap();
    assert_eq!(
        fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o640
    );
}

#[cfg(unix)]
#[test]
fn writing_over_a_symlink_replaces_the_link_not_its_target() {
    let dir = tempfile::tempdir().unwrap();
    let src = fixture(dir.path(), &[]);
    let victim = dir.path().join("victim.txt");
    fs::write(&victim, "do not touch").unwrap();
    let link = dir.path().join("link.safetensors");
    std::os::unix::fs::symlink(&victim, &link).unwrap();
    write_with_metadata(&src, &link, &meta(&[("a", "1")]), &WriteOptions::default()).unwrap();
    assert_eq!(fs::read_to_string(&victim).unwrap(), "do not touch");
    assert!(
        !fs::symlink_metadata(&link)
            .unwrap()
            .file_type()
            .is_symlink()
    );
}
