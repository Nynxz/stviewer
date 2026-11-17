//! An opened file plus the user's in-progress edits and view state.

use serde_json::Value;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::path::PathBuf;
use stv_core::{Metadata, SafeTensorsHeader, format, read_header};

#[derive(Clone, Copy, PartialEq)]
pub enum SortKey {
    Name,
    Dtype,
    Params,
    Size,
}

#[derive(Clone, Copy, PartialEq)]
pub enum TensorView {
    Tree,
    List,
}

/// A node in the dotted-name hierarchy (`model.layers.0.mlp.weight`).
pub struct Node {
    pub segment: String,
    pub path: String,
    pub children: Vec<Node>,
    pub tensor: Option<usize>,
    pub params: u64,
    pub count: usize,
}

pub struct Doc {
    pub path: PathBuf,
    pub header: SafeTensorsHeader,
    /// Editable copy of `__metadata__`, in file order.
    pub meta: Vec<(String, String)>,
    pub selected: Option<usize>,
    /// A large value the user chose to edit despite the size warning.
    pub edit_large: Option<usize>,
    pub meta_filter: String,

    pub tensor_filter: String,
    pub tensor_view: TensorView,
    pub sort: (SortKey, bool),
    /// Indices into `header.tensors`, filtered and sorted (cached).
    pub rows: Vec<usize>,
    pub tree: Vec<Node>,
    /// `(prefix, tensors, params)` groups for the overview, largest first.
    pub structure: Vec<(String, usize, u64)>,
    cache_key: Option<(String, SortKey, bool)>,
}

impl Doc {
    pub fn load(path: PathBuf) -> Result<Self, String> {
        let header = read_header(&path).map_err(|e| e.to_string())?;
        let mut doc = Doc {
            path,
            header,
            meta: Vec::new(),
            selected: None,
            edit_large: None,
            meta_filter: String::new(),
            tensor_filter: String::new(),
            tensor_view: TensorView::Tree,
            sort: (SortKey::Name, true),
            rows: Vec::new(),
            tree: Vec::new(),
            structure: Vec::new(),
            cache_key: None,
        };
        doc.structure = structure(&doc.header);
        doc.reset_meta();
        Ok(doc)
    }

    pub fn reset_meta(&mut self) {
        self.meta = self
            .header
            .metadata_strings()
            .map(|(k, v)| (k.to_owned(), v.to_owned()))
            .collect();
        self.selected = self.selected.filter(|&i| i < self.meta.len());
        self.edit_large = None;
    }

    /// Replace the header after a save, keeping view state.
    pub fn reload(&mut self, path: PathBuf, header: SafeTensorsHeader) {
        self.path = path;
        self.header = header;
        self.structure = structure(&self.header);
        self.cache_key = None;
        self.reset_meta();
    }

    pub fn is_dirty(&self) -> bool {
        self.meta.len() != self.header.metadata.len()
            || self
                .meta
                .iter()
                .zip(self.header.metadata_strings())
                .any(|((k, v), (ok, ov))| k != ok || v != ov)
    }

    /// Has this key/value pair changed relative to the file on disk?
    pub fn entry_state(&self, i: usize) -> EntryState {
        let (k, v) = &self.meta[i];
        match self.header.metadata.get(k).and_then(Value::as_str) {
            None => EntryState::New,
            Some(orig) if orig != v => EntryState::Modified,
            Some(_) => EntryState::Unchanged,
        }
    }

    /// Validated metadata ready to write, or a message explaining what's wrong.
    pub fn edited_metadata(&self) -> Result<Metadata, String> {
        let mut out = Metadata::new();
        for (k, v) in &self.meta {
            if k.is_empty() {
                return Err("a metadata key is empty".into());
            }
            if out.insert(k.clone(), Value::String(v.clone())).is_some() {
                return Err(format!("duplicate metadata key {k:?}"));
            }
        }
        Ok(out)
    }

    pub fn file_name(&self) -> String {
        self.path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| self.path.display().to_string())
    }

    /// Recompute filtered/sorted rows and the tree when inputs change.
    pub fn refresh(&mut self) {
        let key = (self.tensor_filter.clone(), self.sort.0, self.sort.1);
        if self.cache_key.as_ref() == Some(&key) {
            return;
        }
        let tensors = &self.header.tensors;
        let f = self.tensor_filter.trim();
        let mut rows: Vec<usize> = (0..tensors.len())
            .filter(|&i| f.is_empty() || format::name_matches(&tensors[i].name, f))
            .collect();
        let (by, asc) = self.sort;
        rows.sort_by(|&a, &b| {
            let (a, b) = (&tensors[a], &tensors[b]);
            let o = match by {
                SortKey::Name => natural_cmp(&a.name, &b.name),
                SortKey::Dtype => a
                    .dtype
                    .cmp(&b.dtype)
                    .then_with(|| natural_cmp(&a.name, &b.name)),
                SortKey::Params => a.numel().cmp(&b.numel()),
                SortKey::Size => a.byte_len().cmp(&b.byte_len()),
            };
            if asc { o } else { o.reverse() }
        });
        self.tree = build_tree(&self.header, &rows);
        self.rows = rows;
        self.cache_key = Some(key);
    }
}

/// Group tensors by name prefix for the overview's structure card.
/// Picks the shallowest depth that yields at least 3 groups (max depth 3).
fn structure(header: &SafeTensorsHeader) -> Vec<(String, usize, u64)> {
    let mut best = Vec::new();
    for depth in 1..=3 {
        let mut groups: Vec<(String, usize, u64)> = Vec::new();
        let mut index: HashMap<String, usize> = HashMap::new();
        for t in &header.tensors {
            let prefix: Vec<&str> = t.name.split('.').take(depth).collect();
            let prefix = prefix.join(".");
            match index.get(&prefix) {
                Some(&g) => {
                    groups[g].1 += 1;
                    groups[g].2 += t.numel();
                }
                None => {
                    index.insert(prefix.clone(), groups.len());
                    groups.push((prefix, 1, t.numel()));
                }
            }
        }
        best = groups;
        if best.len() >= 3 {
            break;
        }
    }
    best.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| natural_cmp(&a.0, &b.0)));
    best
}

#[derive(PartialEq, Clone, Copy)]
pub enum EntryState {
    Unchanged,
    Modified,
    New,
}

fn build_tree(header: &SafeTensorsHeader, rows: &[usize]) -> Vec<Node> {
    /// Build-time node with a hash index of group children. A linear scan here
    /// went quadratic: a file with 1M sibling tensors would hang the GUI.
    struct Building {
        node: Node,
        groups: HashMap<String, usize>,
        kids: Vec<Building>,
    }
    fn leaf_or_group(segment: &str, parent_path: &str) -> Building {
        let path = if parent_path.is_empty() {
            segment.to_owned()
        } else {
            format!("{parent_path}.{segment}")
        };
        Building {
            node: Node {
                segment: segment.to_owned(),
                path,
                children: Vec::new(),
                tensor: None,
                params: 0,
                count: 0,
            },
            groups: HashMap::new(),
            kids: Vec::new(),
        }
    }
    fn finish(b: Building) -> Node {
        let mut node = b.node;
        node.children = b.kids.into_iter().map(finish).collect();
        node
    }

    let mut root = leaf_or_group("", "");
    for &i in rows {
        let t = &header.tensors[i];
        let numel = t.numel();
        let parts: Vec<&str> = t.name.split('.').collect();
        let mut cur = &mut root;
        for (depth, seg) in parts.iter().enumerate() {
            cur.node.params = cur.node.params.saturating_add(numel);
            cur.node.count += 1;
            let is_leaf = depth + 1 == parts.len();
            let idx = match (is_leaf, cur.groups.get(*seg)) {
                (false, Some(&g)) => g,
                _ => {
                    let child = leaf_or_group(seg, &cur.node.path);
                    cur.kids.push(child);
                    let idx = cur.kids.len() - 1;
                    if !is_leaf {
                        cur.groups.insert((*seg).to_owned(), idx);
                    }
                    idx
                }
            };
            cur = &mut cur.kids[idx];
        }
        cur.node.params = cur.node.params.saturating_add(numel);
        cur.node.count += 1;
        cur.node.tensor = Some(i);
    }
    let mut children: Vec<Node> = root.kids.into_iter().map(finish).collect();
    sort_tree(&mut children);
    children
}

fn sort_tree(nodes: &mut [Node]) {
    nodes.sort_by(|a, b| {
        // Groups before leaves, then natural order.
        a.tensor
            .is_some()
            .cmp(&b.tensor.is_some())
            .then_with(|| natural_cmp(&a.segment, &b.segment))
    });
    for n in nodes {
        sort_tree(&mut n.children);
    }
}

/// Compare strings treating runs of digits as numbers (`layers.2` < `layers.10`).
pub fn natural_cmp(a: &str, b: &str) -> Ordering {
    let (mut a, mut b) = (a.as_bytes(), b.as_bytes());
    loop {
        match (a.first(), b.first()) {
            (None, None) => return Ordering::Equal,
            (None, _) => return Ordering::Less,
            (_, None) => return Ordering::Greater,
            (Some(x), Some(y)) if x.is_ascii_digit() && y.is_ascii_digit() => {
                let na = a.iter().take_while(|c| c.is_ascii_digit()).count();
                let nb = b.iter().take_while(|c| c.is_ascii_digit()).count();
                let (da, db) = (trim_zeros(&a[..na]), trim_zeros(&b[..nb]));
                let o = da.len().cmp(&db.len()).then_with(|| da.cmp(db));
                if o != Ordering::Equal {
                    return o;
                }
                a = &a[na..];
                b = &b[nb..];
            }
            (Some(x), Some(y)) => {
                if x != y {
                    return x.cmp(y);
                }
                a = &a[1..];
                b = &b[1..];
            }
        }
    }
}

fn trim_zeros(d: &[u8]) -> &[u8] {
    let n = d
        .iter()
        .take_while(|&&c| c == b'0')
        .count()
        .min(d.len().saturating_sub(1));
    &d[n..]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Ordering::*;
    use std::time::{Duration, Instant};
    use stv_core::{WriteOptions, write_with_metadata};

    #[test]
    fn natural() {
        assert_eq!(natural_cmp("layers.2.w", "layers.10.w"), Less);
        assert_eq!(natural_cmp("a10", "a010"), Equal);
        assert_eq!(natural_cmp("b", "a"), Greater);
    }

    /// Write a file whose tensors are 1-byte U8 scalars with the given names.
    fn fixture(dir: &std::path::Path, names: &[String], meta: &[(&str, &str)]) -> PathBuf {
        let mut root = serde_json::Map::new();
        if !meta.is_empty() {
            let m: serde_json::Map<_, _> = meta
                .iter()
                .map(|(k, v)| (k.to_string(), Value::from(*v)))
                .collect();
            root.insert("__metadata__".into(), Value::Object(m));
        }
        for (i, n) in names.iter().enumerate() {
            root.insert(
                n.clone(),
                serde_json::json!({"dtype": "U8", "shape": [1], "data_offsets": [i, i + 1]}),
            );
        }
        let h = serde_json::to_vec(&Value::Object(root)).unwrap();
        let mut bytes = (h.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&h);
        bytes.resize(bytes.len() + names.len(), 7);
        let path = dir.join("m.safetensors");
        std::fs::write(&path, bytes).unwrap();
        path
    }

    fn names(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn dirty_tracking_and_entry_states() {
        let d = tempfile::tempdir().unwrap();
        let mut doc =
            Doc::load(fixture(d.path(), &names(&["t"]), &[("a", "1"), ("b", "2")])).unwrap();
        assert!(!doc.is_dirty());

        doc.meta[0].1 = "changed".into();
        assert!(doc.is_dirty());
        assert!(doc.entry_state(0) == EntryState::Modified);
        assert!(doc.entry_state(1) == EntryState::Unchanged);

        doc.meta.push(("c".into(), "3".into()));
        assert!(doc.entry_state(2) == EntryState::New);

        doc.reset_meta();
        assert!(!doc.is_dirty());

        // Reordering keys counts as a change (order is preserved on save).
        doc.meta.swap(0, 1);
        assert!(doc.is_dirty());
        doc.reset_meta();
        // Editing back to the original value is not dirty.
        doc.meta[0].1 = "x".into();
        doc.meta[0].1 = "1".into();
        assert!(!doc.is_dirty());
    }

    #[test]
    fn invalid_edits_are_refused() {
        let d = tempfile::tempdir().unwrap();
        let mut doc = Doc::load(fixture(d.path(), &names(&["t"]), &[("a", "1")])).unwrap();
        doc.meta.push((String::new(), "v".into()));
        assert!(doc.edited_metadata().unwrap_err().contains("empty"));
        doc.meta.pop();
        doc.meta.push(("a".into(), "dup".into()));
        assert!(doc.edited_metadata().unwrap_err().contains("duplicate"));
    }

    #[test]
    fn gui_edit_save_reload_roundtrip() {
        // The same path the Save button takes: edit -> validate -> write -> reload.
        let d = tempfile::tempdir().unwrap();
        let path = fixture(d.path(), &names(&["t"]), &[("keep", "1"), ("drop", "2")]);
        let mut doc = Doc::load(path.clone()).unwrap();
        doc.meta.retain(|(k, _)| k != "drop");
        doc.meta[0].1 = "edited".into();
        doc.meta.push(("added".into(), "日本語\nline".into()));
        let meta = doc.edited_metadata().unwrap();
        let header = write_with_metadata(&path, &path, &meta, &WriteOptions::default()).unwrap();
        doc.reload(path.clone(), header);

        assert!(!doc.is_dirty());
        let fresh = Doc::load(path).unwrap();
        assert_eq!(
            fresh.meta,
            vec![
                ("keep".into(), "edited".into()),
                ("added".into(), "日本語\nline".into())
            ]
        );
    }

    #[test]
    fn tree_groups_by_dotted_prefix_in_natural_order() {
        let d = tempfile::tempdir().unwrap();
        let n = names(&[
            "m.layers.10.w",
            "m.layers.2.w",
            "m.layers.2.b",
            "m.norm",
            "head",
        ]);
        let mut doc = Doc::load(fixture(d.path(), &n, &[])).unwrap();
        doc.refresh();
        let top: Vec<_> = doc.tree.iter().map(|n| n.segment.as_str()).collect();
        assert_eq!(top, ["m", "head"], "groups sort before leaves");
        let m = &doc.tree[0];
        assert_eq!(m.count, 4);
        let layers = &m.children[0];
        assert_eq!(layers.segment, "layers");
        let idx: Vec<_> = layers.children.iter().map(|n| n.segment.as_str()).collect();
        assert_eq!(idx, ["2", "10"]);
        assert_eq!(layers.children[0].count, 2);
        assert_eq!(layers.children[0].path, "m.layers.2");

        doc.tensor_filter = "*.layers.2.*".into();
        doc.refresh();
        assert_eq!(doc.rows.len(), 2);
        assert_eq!(doc.tree[0].count, 2);
    }

    #[test]
    fn tree_handles_a_name_that_is_both_leaf_and_group() {
        let d = tempfile::tempdir().unwrap();
        let mut doc = Doc::load(fixture(d.path(), &names(&["a", "a.b"]), &[])).unwrap();
        doc.refresh();
        assert_eq!(doc.tree.len(), 2);
        assert_eq!(doc.tree.iter().map(|n| n.count).sum::<usize>(), 2);
    }

    #[test]
    fn huge_flat_files_stay_fast() {
        // A hostile file with 300k sibling tensors must not hang the GUI
        // (the original tree builder was quadratic here).
        let d = tempfile::tempdir().unwrap();
        let n: Vec<String> = (0..300_000).map(|i| format!("w{i}")).collect();
        let start = Instant::now();
        let mut doc = Doc::load(fixture(d.path(), &n, &[])).unwrap();
        doc.refresh();
        assert_eq!(doc.tree.len(), 300_000);
        doc.sort = (SortKey::Size, false);
        doc.refresh();
        assert!(
            start.elapsed() < Duration::from_secs(20),
            "took {:?}",
            start.elapsed()
        );
    }
}
