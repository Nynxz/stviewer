//! Integration tests: run the real `stv` binary against files on disk.

use serde_json::{Value, json};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

const STV: &str = env!("CARGO_BIN_EXE_stv");

/// Write a valid file with 3 tensors and the given metadata; returns its path.
fn fixture(dir: &Path, name: &str, meta: Value) -> PathBuf {
    let mut header = json!({
        "a.weight": {"dtype": "F16", "shape": [2, 2], "data_offsets": [0, 8]},
        "a.bias": {"dtype": "F32", "shape": [2], "data_offsets": [8, 16]},
        "b.weight": {"dtype": "BF16", "shape": [4], "data_offsets": [16, 24]},
    });
    if meta.as_object().is_some_and(|m| !m.is_empty()) {
        header["__metadata__"] = meta;
    }
    let mut h = serde_json::to_vec(&header).unwrap();
    while !(8 + h.len()).is_multiple_of(8) {
        h.push(b' ');
    }
    let mut bytes = (h.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&h);
    bytes.extend((0..24u8).map(|i| i.wrapping_mul(37).wrapping_add(5)));
    let path = dir.join(name);
    fs::write(&path, bytes).unwrap();
    path
}

/// The tensor data section of a file (everything after the header).
fn data_section(path: &Path) -> Vec<u8> {
    let b = fs::read(path).unwrap();
    let n = u64::from_le_bytes(b[..8].try_into().unwrap()) as usize;
    b[8 + n..].to_vec()
}

fn stv(args: &[&str]) -> Output {
    Command::new(STV).args(args).output().unwrap()
}

fn stv_stdin(args: &[&str], input: &str) -> Output {
    let mut child = Command::new(STV)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    child.wait_with_output().unwrap()
}

#[track_caller]
fn ok(o: &Output) -> String {
    assert!(
        o.status.success(),
        "stv failed: {}\nstderr: {}",
        o.status,
        String::from_utf8_lossy(&o.stderr)
    );
    String::from_utf8(o.stdout.clone()).unwrap()
}

fn meta_json(path: &Path) -> Value {
    serde_json::from_str(&ok(&stv(&["meta", path.to_str().unwrap(), "--json"]))).unwrap()
}

fn p(path: &Path) -> &str {
    path.to_str().unwrap()
}

#[test]
fn show_json_summary() {
    let d = tempfile::tempdir().unwrap();
    let f = fixture(d.path(), "m.safetensors", json!({"format": "pt"}));
    let v: Value = serde_json::from_str(&ok(&stv(&["show", p(&f), "--json"]))).unwrap();
    assert_eq!(v["tensor_count"], 3);
    assert_eq!(v["total_params"], 4 + 2 + 4);
    assert_eq!(v["dtypes"]["F16"]["tensors"], 1);
    assert_eq!(v["metadata"]["format"], "pt");
}

#[test]
fn show_multiple_files_gives_array() {
    let d = tempfile::tempdir().unwrap();
    let a = fixture(d.path(), "a.safetensors", json!({}));
    let b = fixture(d.path(), "b.safetensors", json!({}));
    let v: Value = serde_json::from_str(&ok(&stv(&["show", p(&a), p(&b), "--json"]))).unwrap();
    assert_eq!(v.as_array().unwrap().len(), 2);
}

#[test]
fn meta_get_single_key() {
    let d = tempfile::tempdir().unwrap();
    let f = fixture(
        d.path(),
        "m.safetensors",
        json!({"dim": "8", "multi": "a\nb"}),
    );
    assert_eq!(ok(&stv(&["meta", p(&f), "dim"])), "8\n");
    assert_eq!(ok(&stv(&["meta", p(&f), "multi"])), "a\nb\n");
    let missing = stv(&["meta", p(&f), "nope"]);
    assert_eq!(missing.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&missing.stderr).contains("not found"));
}

#[test]
fn tensors_filter_and_json() {
    let d = tempfile::tempdir().unwrap();
    let f = fixture(d.path(), "m.safetensors", json!({}));
    let v: Value =
        serde_json::from_str(&ok(&stv(&["tensors", p(&f), "-f", "*.weight", "--json"]))).unwrap();
    let names: Vec<_> = v
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["a.weight", "b.weight"]);
    let table = ok(&stv(&["tensors", p(&f), "-f", "bias"]));
    assert!(table.contains("a.bias") && !table.contains("a.weight"));
}

#[test]
fn set_adds_and_updates_keys() {
    let d = tempfile::tempdir().unwrap();
    let f = fixture(
        d.path(),
        "m.safetensors",
        json!({"keep": "1", "change": "old"}),
    );
    let data = data_section(&f);
    ok(&stv(&["set", p(&f), "change=new", "added=x=y", "empty="]));
    assert_eq!(
        meta_json(&f),
        json!({"keep": "1", "change": "new", "added": "x=y", "empty": ""})
    );
    assert_eq!(data_section(&f), data, "tensor data changed");
}

#[test]
fn set_rejects_malformed_pairs() {
    let d = tempfile::tempdir().unwrap();
    let f = fixture(d.path(), "m.safetensors", json!({}));
    let original = fs::read(&f).unwrap();
    for bad in ["novalue", "=value"] {
        let o = stv(&["set", p(&f), bad]);
        assert!(!o.status.success(), "{bad} accepted");
    }
    assert_eq!(fs::read(&f).unwrap(), original);
}

#[test]
fn rm_removes_keys() {
    let d = tempfile::tempdir().unwrap();
    let f = fixture(
        d.path(),
        "m.safetensors",
        json!({"a": "1", "b": "2", "c": "3"}),
    );
    ok(&stv(&["rm", p(&f), "a", "c", "not-there"]));
    assert_eq!(meta_json(&f), json!({"b": "2"}));
}

#[test]
fn rm_strict_fails_without_writing() {
    let d = tempfile::tempdir().unwrap();
    let f = fixture(d.path(), "m.safetensors", json!({"a": "1"}));
    let original = fs::read(&f).unwrap();
    let o = stv(&["rm", p(&f), "a", "missing", "--strict"]);
    assert!(!o.status.success());
    assert_eq!(fs::read(&f).unwrap(), original);
}

#[test]
fn clean_strips_metadata_keeping_selected_keys() {
    let d = tempfile::tempdir().unwrap();
    let a = fixture(
        d.path(),
        "a.safetensors",
        json!({"title": "A", "secret": "x", "path": "/home/me"}),
    );
    let b = fixture(d.path(), "b.safetensors", json!({"secret": "y"}));
    ok(&stv(&["clean", p(&a), p(&b), "--keep", "title"]));
    assert_eq!(meta_json(&a), json!({"title": "A"}));
    assert_eq!(meta_json(&b), json!({}));
}

#[test]
fn clean_output_requires_single_file() {
    let d = tempfile::tempdir().unwrap();
    let a = fixture(d.path(), "a.safetensors", json!({"k": "v"}));
    let b = fixture(d.path(), "b.safetensors", json!({"k": "v"}));
    let o = stv(&[
        "clean",
        p(&a),
        p(&b),
        "-o",
        p(&d.path().join("out.safetensors")),
    ]);
    assert!(!o.status.success());
    assert_eq!(meta_json(&a), json!({"k": "v"}));
}

#[test]
fn import_replaces_or_merges() {
    let d = tempfile::tempdir().unwrap();
    let f = fixture(
        d.path(),
        "m.safetensors",
        json!({"old": "1", "shared": "old"}),
    );
    ok(&stv_stdin(
        &["import", p(&f), "-", "--merge"],
        r#"{"shared": "new", "n": 5, "obj": {"a": 1}}"#,
    ));
    assert_eq!(
        meta_json(&f),
        json!({"old": "1", "shared": "new", "n": "5", "obj": "{\"a\":1}"}),
        "merge failed or non-strings not stringified"
    );
    let src = d.path().join("meta.json");
    fs::write(&src, r#"{"only": "this"}"#).unwrap();
    ok(&stv(&["import", p(&f), p(&src)]));
    assert_eq!(meta_json(&f), json!({"only": "this"}));
}

#[test]
fn import_rejects_non_objects() {
    let d = tempfile::tempdir().unwrap();
    let f = fixture(d.path(), "m.safetensors", json!({"a": "1"}));
    for bad in ["[1,2]", "\"str\"", "not json"] {
        assert!(
            !stv_stdin(&["import", p(&f), "-"], bad).status.success(),
            "{bad} accepted"
        );
    }
    assert_eq!(meta_json(&f), json!({"a": "1"}));
}

#[test]
fn metadata_pipes_between_files() {
    let d = tempfile::tempdir().unwrap();
    let a = fixture(
        d.path(),
        "a.safetensors",
        json!({"title": "from a", "x": "1"}),
    );
    let b = fixture(d.path(), "b.safetensors", json!({}));
    let exported = ok(&stv(&["meta", p(&a), "--json"]));
    ok(&stv_stdin(&["import", p(&b), "-"], &exported));
    assert_eq!(meta_json(&b), meta_json(&a));
}

#[test]
fn dry_run_writes_nothing() {
    let d = tempfile::tempdir().unwrap();
    let f = fixture(d.path(), "m.safetensors", json!({"a": "1"}));
    let original = fs::read(&f).unwrap();
    for args in [
        vec!["set", p(&f), "a=2", "-n"],
        vec!["rm", p(&f), "a", "-n"],
        vec!["clean", p(&f), "-n"],
    ] {
        let o = stv(&args);
        ok(&o);
        assert!(String::from_utf8_lossy(&o.stderr).contains("dry run"));
        assert_eq!(
            fs::read(&f).unwrap(),
            original,
            "{args:?} modified the file"
        );
    }
}

#[test]
fn output_flag_leaves_source_untouched() {
    let d = tempfile::tempdir().unwrap();
    let f = fixture(d.path(), "m.safetensors", json!({"a": "1"}));
    let original = fs::read(&f).unwrap();
    let out = d.path().join("out.safetensors");
    ok(&stv(&["set", p(&f), "a=2", "-o", p(&out)]));
    assert_eq!(fs::read(&f).unwrap(), original);
    assert_eq!(meta_json(&out), json!({"a": "2"}));
    assert_eq!(data_section(&out), data_section(&f));
}

#[test]
fn backup_keeps_original() {
    let d = tempfile::tempdir().unwrap();
    let f = fixture(d.path(), "m.safetensors", json!({"a": "1"}));
    let original = fs::read(&f).unwrap();
    ok(&stv(&["set", p(&f), "a=2", "--backup"]));
    assert_eq!(
        fs::read(d.path().join("m.safetensors.bak")).unwrap(),
        original
    );
}

#[test]
fn unchanged_metadata_is_not_rewritten() {
    let d = tempfile::tempdir().unwrap();
    let f = fixture(d.path(), "m.safetensors", json!({"a": "1"}));
    let before = fs::metadata(&f).unwrap().modified().unwrap();
    let o = stv(&["set", p(&f), "a=1"]);
    ok(&o);
    assert!(String::from_utf8_lossy(&o.stderr).contains("no changes"));
    assert_eq!(fs::metadata(&f).unwrap().modified().unwrap(), before);
}

#[test]
fn invalid_files_fail_cleanly() {
    let d = tempfile::tempdir().unwrap();
    let cases: [(&str, &[u8]); 4] = [
        ("empty", b""),
        ("short", b"\x01\x02"),
        ("huge-len", b"\xff\xff\xff\xff\xff\xff\xff\xff{}"),
        ("garbage", b"\x08\0\0\0\0\0\0\0not json"),
    ];
    for (name, bytes) in cases {
        let f = d.path().join(name);
        fs::write(&f, bytes).unwrap();
        for cmd in [vec!["show", p(&f)], vec!["set", p(&f), "a=1"]] {
            let o = stv(&cmd);
            assert_eq!(o.status.code(), Some(1), "{name} {cmd:?}");
            let err = String::from_utf8_lossy(&o.stderr);
            assert!(
                err.starts_with("stv: ") && !err.contains("panicked"),
                "{name}: {err}"
            );
        }
        assert_eq!(fs::read(&f).unwrap(), bytes, "{name} was modified");
    }
    assert!(
        !stv(&["show", p(&d.path().join("missing"))])
            .status
            .success()
    );
}

// ---------------------------------------------------------------------------
// Hostile content: nothing from the file may reach the terminal unescaped.
// ---------------------------------------------------------------------------

const EVIL_KEY: &str = "\u{1b}]0;pwned\u{7}title";
const EVIL_VALUE: &str = "\u{1b}[2J\u{1b}]52;c;Y3VybCBldmlsfHNo\u{7}\u{9b}31mred\u{202e}txt.exe";

fn evil_file(dir: &Path) -> PathBuf {
    let header = json!({
        "__metadata__": {EVIL_KEY: EVIL_VALUE},
        "\u{1b}[31mname\u{202e}": {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]},
        "x": {"dtype": "\u{1b}[5mQ\u{9b}", "shape": [4], "data_offsets": [4, 8]},
    });
    let h = serde_json::to_vec(&header).unwrap();
    let mut bytes = (h.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&h);
    bytes.extend_from_slice(&[0u8; 8]);
    let f = dir.join("evil.safetensors");
    fs::write(&f, bytes).unwrap();
    f
}

#[track_caller]
fn assert_terminal_safe(label: &str, bytes: &[u8]) {
    let s = String::from_utf8_lossy(bytes);
    for c in s.chars() {
        let allowed = matches!(c, '\n' | '\t');
        let dangerous = (c.is_control() && !allowed)
            || matches!(c, '\u{202A}'..='\u{202E}' | '\u{2066}'..='\u{2069}' | '\u{200E}' | '\u{200F}' | '\u{061C}');
        assert!(!dangerous, "{label}: raw {c:?} reached output:\n{s}");
    }
}

#[test]
fn hostile_strings_are_escaped_in_human_output() {
    let d = tempfile::tempdir().unwrap();
    let f = evil_file(d.path());
    let cases: Vec<Vec<&str>> = vec![
        vec!["show", p(&f)],
        vec!["meta", p(&f)],
        vec!["meta", p(&f), "--full"],
        vec!["tensors", p(&f)],
        vec!["set", p(&f), "-n", "a=1"],
        vec!["clean", p(&f), "-n"],
    ];
    for args in cases {
        let o = stv(&args);
        ok(&o);
        assert_terminal_safe(&format!("{args:?} stdout"), &o.stdout);
        assert_terminal_safe(&format!("{args:?} stderr"), &o.stderr);
    }
}

#[test]
fn hostile_strings_in_json_are_escaped_but_lossless() {
    let d = tempfile::tempdir().unwrap();
    let f = evil_file(d.path());
    for args in [
        vec!["meta", p(&f), "--json"],
        vec!["show", p(&f), "--json"],
        vec!["tensors", p(&f), "--json"],
    ] {
        let o = stv(&args);
        assert_terminal_safe(&format!("{args:?}"), &o.stdout);
        let _: Value = serde_json::from_slice(&o.stdout).expect("output is not valid JSON");
    }
    let m = meta_json(&f);
    assert_eq!(m[EVIL_KEY], EVIL_VALUE, "JSON escaping lost data");
}

#[test]
fn raw_value_is_exact_when_piped() {
    // `stv meta FILE KEY` is for scripts: piped output is byte-exact.
    // (When stdout is a terminal it is escaped instead.)
    let d = tempfile::tempdir().unwrap();
    let f = evil_file(d.path());
    assert_eq!(
        ok(&stv(&["meta", p(&f), EVIL_KEY])),
        format!("{EVIL_VALUE}\n")
    );
}

#[test]
fn errors_about_hostile_names_are_escaped() {
    let d = tempfile::tempdir().unwrap();
    let header = r#"{"\u001b]0;x\u0007bad":{"dtype":"U8","shape":[9],"data_offsets":[0,4]}}"#;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&[0; 4]);
    let f = d.path().join("bad.safetensors");
    fs::write(&f, bytes).unwrap();
    let o = stv(&["show", p(&f)]);
    assert!(!o.status.success());
    assert_terminal_safe("error message", &o.stderr);
}
