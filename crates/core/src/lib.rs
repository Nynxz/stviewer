//! Header-only reader and metadata editor for `.safetensors` files.
//!
//! File layout:
//!
//! ```text
//! [u64 LE: header length N][N bytes JSON header][tensor data ...]
//! ```
//!
//! This crate only ever reads the first `8 + N` bytes when inspecting a file.
//! Tensor bytes are never interpreted; when rewriting (to change metadata)
//! they are streamed verbatim from the source into the new file.

mod dtype;
pub mod format;
mod write;

pub use dtype::Dtype;
pub use write::{WriteOptions, write_with_metadata};

use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

/// Key in the header that holds free-form `string -> string` metadata.
pub const METADATA_KEY: &str = "__metadata__";

/// Refuse headers larger than this (matches the reference implementation).
pub const MAX_HEADER_LEN: u64 = 100 * 1024 * 1024;

/// Ordered `string -> string` metadata map (preserves file order).
pub type Metadata = Map<String, Value>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("file is too small to be a safetensors file ({0} bytes)")]
    TooSmall(u64),
    #[error("header length {0} exceeds the {MAX_HEADER_LEN} byte limit")]
    HeaderTooLarge(u64),
    #[error("header length {header} runs past end of file ({file} bytes)")]
    HeaderPastEof { header: u64, file: u64 },
    #[error("header is not valid UTF-8 JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("header must be a JSON object")]
    NotObject,
    #[error("__metadata__ must be an object of string values (key {0:?} is not a string)")]
    BadMetadata(String),
    #[error("tensor {name:?}: {reason}")]
    BadTensor { name: String, reason: String },
    #[error("tensor data is not contiguous: {0}")]
    Layout(String),
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    /// Byte range relative to the start of the data section.
    pub data_offsets: (u64, u64),
}

impl TensorInfo {
    /// Number of elements (product of the shape; 1 for scalars).
    pub fn numel(&self) -> u64 {
        self.shape.iter().product()
    }

    pub fn byte_len(&self) -> u64 {
        self.data_offsets.1 - self.data_offsets.0
    }
}

/// Parsed header of a safetensors file.
#[derive(Debug, Clone)]
pub struct SafeTensorsHeader {
    pub metadata: Metadata,
    /// Tensors in the order they appear in the header.
    pub tensors: Vec<TensorInfo>,
    /// Length of the JSON header in bytes (the `N` above).
    pub header_len: u64,
    /// Total file size in bytes.
    pub file_len: u64,
    /// Non-fatal issues found while validating (e.g. unknown dtypes).
    pub warnings: Vec<String>,
}

impl SafeTensorsHeader {
    /// Byte offset where tensor data begins.
    pub fn data_start(&self) -> u64 {
        8 + self.header_len
    }

    pub fn data_len(&self) -> u64 {
        self.file_len - self.data_start()
    }

    pub fn total_params(&self) -> u64 {
        // Saturating as defence in depth; parsing already bounds every numel by
        // the tensor's byte length.
        self.tensors
            .iter()
            .fold(0u64, |acc, t| acc.saturating_add(t.numel()))
    }

    /// Per-dtype `(tensor count, parameter count)`.
    pub fn dtype_breakdown(&self) -> BTreeMap<Dtype, (usize, u64)> {
        let mut out: BTreeMap<Dtype, (usize, u64)> = BTreeMap::new();
        for t in &self.tensors {
            let e = out.entry(t.dtype.clone()).or_default();
            e.0 += 1;
            e.1 = e.1.saturating_add(t.numel());
        }
        out
    }

    /// Metadata as plain strings (values are guaranteed strings after parsing).
    pub fn metadata_strings(&self) -> impl Iterator<Item = (&str, &str)> {
        self.metadata
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str().unwrap_or_default()))
    }
}

/// Read and validate the header of the file at `path`.
pub fn read_header(path: impl AsRef<Path>) -> Result<SafeTensorsHeader> {
    let mut f = File::open(path)?;
    let file_len = f.metadata()?.len();
    read_header_from(&mut f, file_len)
}

/// Read and validate a header from any reader positioned at the start of the file.
pub fn read_header_from<R: Read>(r: &mut R, file_len: u64) -> Result<SafeTensorsHeader> {
    if file_len < 8 {
        return Err(Error::TooSmall(file_len));
    }
    let mut len_buf = [0u8; 8];
    r.read_exact(&mut len_buf)?;
    let header_len = u64::from_le_bytes(len_buf);
    if header_len > MAX_HEADER_LEN {
        return Err(Error::HeaderTooLarge(header_len));
    }
    if 8 + header_len > file_len {
        return Err(Error::HeaderPastEof {
            header: header_len,
            file: file_len,
        });
    }

    let mut buf = vec![0u8; header_len as usize];
    r.read_exact(&mut buf)?;
    let root: Value = serde_json::from_slice(&buf)?;
    let Value::Object(root) = root else {
        return Err(Error::NotObject);
    };

    parse_root(root, header_len, file_len)
}

fn parse_root(
    root: Map<String, Value>,
    header_len: u64,
    file_len: u64,
) -> Result<SafeTensorsHeader> {
    let data_len = file_len - 8 - header_len;
    let mut metadata = Metadata::new();
    let mut tensors = Vec::with_capacity(root.len());
    let mut warnings = Vec::new();

    for (name, value) in root {
        if name == METADATA_KEY {
            let Value::Object(m) = value else {
                return Err(Error::BadMetadata(METADATA_KEY.into()));
            };
            for (k, v) in &m {
                if !v.is_string() {
                    return Err(Error::BadMetadata(k.clone()));
                }
            }
            metadata = m;
            continue;
        }
        tensors.push(parse_tensor(name, value, &mut warnings)?);
    }

    validate_layout(&tensors, data_len)?;
    Ok(SafeTensorsHeader {
        metadata,
        tensors,
        header_len,
        file_len,
        warnings,
    })
}

fn parse_tensor(name: String, value: Value, warnings: &mut Vec<String>) -> Result<TensorInfo> {
    let bad = |reason: &str| Error::BadTensor {
        name: name.clone(),
        reason: reason.into(),
    };

    let Value::Object(obj) = value else {
        return Err(bad("entry is not an object"));
    };
    let dtype = obj
        .get("dtype")
        .and_then(Value::as_str)
        .map(Dtype::parse)
        .ok_or_else(|| bad("missing or non-string \"dtype\""))?;
    let shape = obj
        .get("shape")
        .and_then(Value::as_array)
        .ok_or_else(|| bad("missing \"shape\" array"))?
        .iter()
        .map(|d| {
            d.as_u64()
                .ok_or_else(|| bad("shape contains a non-integer"))
        })
        .collect::<Result<Vec<_>>>()?;
    let offs = obj
        .get("data_offsets")
        .and_then(Value::as_array)
        .filter(|a| a.len() == 2)
        .ok_or_else(|| bad("\"data_offsets\" must be [begin, end]"))?;
    let (Some(begin), Some(end)) = (offs[0].as_u64(), offs[1].as_u64()) else {
        return Err(bad("data_offsets must be non-negative integers"));
    };
    if end < begin {
        return Err(bad("data_offsets end is before begin"));
    }

    let numel = shape
        .iter()
        .try_fold(1u64, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| bad("shape overflows u64"))?;
    match dtype.bits() {
        Some(bits) => {
            let expected = numel
                .checked_mul(bits)
                .map(|b| b.div_ceil(8))
                .ok_or_else(|| bad("byte size overflows u64"))?;
            if expected != end - begin {
                return Err(bad(&format!(
                    "shape {shape:?} x {dtype} needs {expected} bytes but data_offsets span {}",
                    end - begin
                )));
            }
        }
        None => {
            // Can't verify the exact size, but every element takes at least one
            // bit. This bounds element counts by the file size, so totals can't
            // overflow (or claim absurd parameter counts).
            if numel > (end - begin).saturating_mul(8) {
                return Err(bad(&format!(
                    "shape {shape:?} has {numel} elements but only {} bytes of data",
                    end - begin
                )));
            }
            warnings.push(format!(
                "tensor {name:?} has unknown dtype {:?}; size not checked",
                dtype.as_str()
            ));
        }
    }

    Ok(TensorInfo {
        name,
        dtype,
        shape,
        data_offsets: (begin, end),
    })
}

/// Tensors must tile the data section exactly: no gaps, no overlaps, nothing past the end.
fn validate_layout(tensors: &[TensorInfo], data_len: u64) -> Result<()> {
    let mut ranges: Vec<_> = tensors.iter().map(|t| (t.data_offsets, &t.name)).collect();
    ranges.sort_by_key(|r| r.0);
    let mut cursor = 0u64;
    for ((begin, end), name) in ranges {
        if begin != cursor {
            return Err(Error::Layout(format!(
                "tensor {name:?} starts at {begin}, expected {cursor} (gap or overlap)"
            )));
        }
        cursor = end;
    }
    if cursor != data_len {
        return Err(Error::Layout(format!(
            "tensors cover {cursor} bytes but the data section is {data_len} bytes"
        )));
    }
    Ok(())
}

/// Serialize a header back to JSON bytes, padded with spaces to an 8-byte boundary.
pub fn serialize_header(metadata: &Metadata, tensors: &[TensorInfo]) -> Vec<u8> {
    let mut root = Map::new();
    if !metadata.is_empty() {
        root.insert(METADATA_KEY.into(), Value::Object(metadata.clone()));
    }
    for t in tensors {
        let mut o = Map::new();
        o.insert("dtype".into(), t.dtype.as_str().into());
        o.insert("shape".into(), t.shape.clone().into());
        o.insert(
            "data_offsets".into(),
            vec![t.data_offsets.0, t.data_offsets.1].into(),
        );
        root.insert(t.name.clone(), Value::Object(o));
    }
    let mut bytes =
        serde_json::to_vec(&Value::Object(root)).expect("serializing a Value cannot fail");
    let pad = (8 - (8 + bytes.len()) % 8) % 8;
    bytes.resize(bytes.len() + pad, b' ');
    bytes
}

#[cfg(test)]
mod tests;
