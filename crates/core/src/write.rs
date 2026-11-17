use crate::{Metadata, Result, SafeTensorsHeader, serialize_header};
use std::fs::{self, File};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Default)]
pub struct WriteOptions {
    /// When overwriting an existing file, keep the original as `<name>.bak`.
    pub backup: bool,
}

/// Write a copy of `src` with its metadata replaced by `metadata` to `dst`.
///
/// `dst` may equal `src`. The new file is written to a temporary file in the
/// destination directory, flushed to disk, and then atomically renamed into
/// place, so a crash or full disk never leaves a half-written model behind.
///
/// Tensor data is streamed byte-for-byte and never interpreted.
pub fn write_with_metadata(
    src: impl AsRef<Path>,
    dst: impl AsRef<Path>,
    metadata: &Metadata,
    opts: &WriteOptions,
) -> Result<SafeTensorsHeader> {
    let src = src.as_ref();
    let dst = dst.as_ref();

    let mut input = File::open(src)?;
    let file_len = input.metadata()?.len();
    let header = crate::read_header_from(&mut input, file_len)?;

    for v in metadata.values() {
        if !v.is_string() {
            return Err(crate::Error::BadMetadata("<new metadata>".into()));
        }
    }

    let new_header = serialize_header(metadata, &header.tensors);
    // Never produce a file that readers (including ours) would refuse to open.
    if new_header.len() as u64 > crate::MAX_HEADER_LEN {
        return Err(crate::Error::HeaderTooLarge(new_header.len() as u64));
    }
    let dir = parent_dir(dst);
    let mut tmp = tempfile::Builder::new()
        .prefix(".stv-")
        .suffix(".tmp")
        .tempfile_in(&dir)?;

    {
        let mut out = BufWriter::with_capacity(1 << 20, tmp.as_file_mut());
        out.write_all(&(new_header.len() as u64).to_le_bytes())?;
        out.write_all(&new_header)?;
        input.seek(SeekFrom::Start(header.data_start()))?;
        let copied = io::copy(&mut (&mut input).take(header.data_len()), &mut out)?;
        if copied != header.data_len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "source file shrank while copying",
            )
            .into());
        }
        out.flush()?;
    }
    tmp.as_file().sync_all()?;
    // Close the source before renaming over it: Windows can refuse to replace
    // a file that still has an open handle.
    drop(input);

    // Preserve the original file's permissions where possible.
    if let Ok(meta) = fs::metadata(if dst.exists() { dst } else { src }) {
        let _ = fs::set_permissions(tmp.path(), meta.permissions());
    }

    if opts.backup && dst.exists() {
        // A hard link keeps the old inode alive under the .bak name for free;
        // the rename below then swaps the new file in. Fall back to a copy on
        // filesystems without hard links (e.g. FAT/exFAT).
        let bak = backup_path(dst);
        let _ = fs::remove_file(&bak);
        if fs::hard_link(dst, &bak).is_err() {
            fs::copy(dst, &bak)?;
        }
    }
    tmp.persist(dst).map_err(|e| e.error)?;

    crate::read_header(dst)
}

fn parent_dir(p: &Path) -> PathBuf {
    match p.parent() {
        Some(d) if !d.as_os_str().is_empty() => d.to_path_buf(),
        _ => PathBuf::from("."),
    }
}

fn backup_path(p: &Path) -> PathBuf {
    let mut s = p.as_os_str().to_owned();
    s.push(".bak");
    PathBuf::from(s)
}
