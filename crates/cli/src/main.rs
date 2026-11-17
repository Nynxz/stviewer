mod term;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use serde_json::{Value, json};
use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use stv_core::{
    Metadata, SafeTensorsHeader, WriteOptions, format, read_header, write_with_metadata,
};

/// Inspect and edit .safetensors metadata. Never loads or executes tensor data.
#[derive(Parser)]
#[command(name = "stv", version, about)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Summary: size, parameter count, dtypes, metadata
    Show {
        #[arg(required = true)]
        files: Vec<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Print metadata (all keys, or the raw value of one key)
    Meta {
        file: PathBuf,
        /// Print only this key's raw value (exit 1 if missing)
        key: Option<String>,
        #[arg(long)]
        json: bool,
        /// Don't truncate long values in terminal output
        #[arg(long)]
        full: bool,
    },
    /// List tensors with dtype, shape and size
    Tensors {
        file: PathBuf,
        /// Substring, or wildcard pattern with * and ?
        #[arg(short, long)]
        filter: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Set one or more metadata keys: stv set FILE key=value ...
    Set {
        file: PathBuf,
        #[arg(required = true, value_parser = parse_kv)]
        pairs: Vec<(String, String)>,
        #[command(flatten)]
        out: OutArgs,
    },
    /// Remove metadata keys
    Rm {
        file: PathBuf,
        #[arg(required = true)]
        keys: Vec<String>,
        /// Fail if a key does not exist
        #[arg(long)]
        strict: bool,
        #[command(flatten)]
        out: OutArgs,
    },
    /// Strip all metadata (optionally keeping some keys)
    Clean {
        #[arg(required = true)]
        files: Vec<PathBuf>,
        /// Key to keep (repeatable)
        #[arg(short, long = "keep")]
        keep: Vec<String>,
        #[command(flatten)]
        out: OutArgs,
    },
    /// Replace metadata from a JSON object file, or '-' for stdin
    Import {
        file: PathBuf,
        source: PathBuf,
        /// Merge into existing metadata instead of replacing it
        #[arg(long)]
        merge: bool,
        #[command(flatten)]
        out: OutArgs,
    },
}

#[derive(Args)]
struct OutArgs {
    /// Write to this path instead of modifying the file in place
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Keep the original as <file>.bak when modifying in place
    #[arg(long)]
    backup: bool,
    /// Show what would change without writing anything
    #[arg(short = 'n', long)]
    dry_run: bool,
}

fn parse_kv(s: &str) -> Result<(String, String), String> {
    let (k, v) = s
        .split_once('=')
        .ok_or_else(|| format!("expected key=value, got {s:?}"))?;
    if k.is_empty() {
        return Err("key must not be empty".into());
    }
    Ok((k.to_owned(), v.to_owned()))
}

fn main() -> ExitCode {
    // Exit quietly when piped into e.g. `head` instead of panicking on EPIPE.
    #[cfg(unix)]
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
    match run(Cli::parse()) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("stv: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn open(path: &Path) -> Result<SafeTensorsHeader> {
    let h = read_header(path).with_context(|| format!("{}", path.display()))?;
    for w in &h.warnings {
        eprintln!("stv: warning: {}: {w}", path.display());
    }
    Ok(h)
}

fn run(cli: Cli) -> Result<ExitCode> {
    match cli.cmd {
        Cmd::Show { files, json } => {
            let mut all = Vec::new();
            for (i, f) in files.iter().enumerate() {
                let h = open(f)?;
                if json {
                    all.push(summary_json(f, &h));
                } else {
                    if i > 0 {
                        println!();
                    }
                    print_summary(f, &h);
                }
            }
            if json {
                let v = if all.len() == 1 {
                    all.remove(0)
                } else {
                    Value::Array(all)
                };
                print_json(&v)?;
            }
        }
        Cmd::Meta {
            file,
            key,
            json,
            full,
        } => {
            let h = open(&file)?;
            if let Some(key) = key {
                let Some(v) = h.metadata.get(&key) else {
                    eprintln!("stv: key {key:?} not found");
                    return Ok(ExitCode::FAILURE);
                };
                if json {
                    print_json(v)?;
                } else {
                    // Exact bytes for scripts; escaped if it would hit a terminal.
                    let v = v.as_str().unwrap_or_default();
                    if std::io::stdout().is_terminal() {
                        println!("{}", term::clean(v, true));
                    } else {
                        println!("{v}");
                    }
                }
            } else if json {
                print_json(&Value::Object(h.metadata))?;
            } else {
                let truncate = !full && std::io::stdout().is_terminal();
                for (k, v) in h.metadata_strings() {
                    println!(
                        "{}: {}",
                        term::clean(k, false),
                        if truncate {
                            ellipsize(v, 160)
                        } else {
                            term::clean(v, true).into_owned()
                        }
                    );
                }
            }
        }
        Cmd::Tensors { file, filter, json } => {
            let h = open(&file)?;
            let rows = h.tensors.iter().filter(|t| {
                filter
                    .as_deref()
                    .is_none_or(|p| format::name_matches(&t.name, p))
            });
            if json {
                let v: Vec<Value> = rows
                    .map(|t| {
                        json!({
                            "name": t.name, "dtype": t.dtype.as_str(), "shape": t.shape,
                            "params": t.numel(), "bytes": t.byte_len(),
                            "data_offsets": [t.data_offsets.0, t.data_offsets.1],
                        })
                    })
                    .collect();
                print_json(&Value::Array(v))?;
            } else {
                let rows: Vec<_> = rows
                    .map(|t| {
                        (
                            term::clean(&t.name, false),
                            term::clean(t.dtype.as_str(), false),
                            t,
                        )
                    })
                    .collect();
                let w = rows
                    .iter()
                    .map(|r| r.0.chars().count())
                    .max()
                    .unwrap_or(4)
                    .max(4);
                let mut out = std::io::stdout().lock();
                writeln!(out, "{:w$}  {:8}  {:>10}  SHAPE", "NAME", "DTYPE", "SIZE")?;
                for (name, dtype, t) in rows {
                    writeln!(
                        out,
                        "{:w$}  {:8}  {:>10}  {}",
                        name,
                        dtype,
                        format::bytes(t.byte_len()),
                        format::shape(&t.shape)
                    )?;
                }
            }
        }
        Cmd::Set { file, pairs, out } => {
            edit(&file, &out, |m| {
                for (k, v) in pairs {
                    m.insert(k, Value::String(v));
                }
                Ok(())
            })?;
        }
        Cmd::Rm {
            file,
            keys,
            strict,
            out,
        } => {
            edit(&file, &out, |m| {
                for k in &keys {
                    if m.shift_remove(k).is_none() && strict {
                        bail!("key {k:?} not found");
                    }
                }
                Ok(())
            })?;
        }
        Cmd::Clean { files, keep, out } => {
            if out.output.is_some() && files.len() > 1 {
                bail!("--output can only be used with a single file");
            }
            for f in &files {
                edit(f, &out, |m| {
                    m.retain(|k, _| keep.contains(k));
                    Ok(())
                })?;
            }
        }
        Cmd::Import {
            file,
            source,
            merge,
            out,
        } => {
            let text = if source.as_os_str() == "-" {
                let mut s = String::new();
                std::io::stdin().read_to_string(&mut s)?;
                s
            } else {
                std::fs::read_to_string(&source).with_context(|| format!("{}", source.display()))?
            };
            let Value::Object(incoming) = serde_json::from_str(&text).context("parsing JSON")?
            else {
                bail!("expected a JSON object of string values");
            };
            // Non-string values are stringified, since the format only allows strings.
            let incoming: Metadata = incoming
                .into_iter()
                .map(|(k, v)| match v {
                    Value::String(_) => (k, v),
                    other => (k, Value::String(other.to_string())),
                })
                .collect();
            edit(&file, &out, |m| {
                if !merge {
                    m.clear();
                }
                m.extend(incoming);
                Ok(())
            })?;
        }
    }
    Ok(ExitCode::SUCCESS)
}

/// Apply `f` to the file's metadata and write the result (atomically).
fn edit(file: &Path, out: &OutArgs, f: impl FnOnce(&mut Metadata) -> Result<()>) -> Result<()> {
    let h = open(file)?;
    let mut meta = h.metadata.clone();
    f(&mut meta)?;

    let dst = out.output.as_deref().unwrap_or(file);
    let changed = meta != h.metadata || out.output.is_some();
    print_diff(&h.metadata, &meta);
    if out.dry_run {
        eprintln!("{}: dry run, nothing written", file.display());
        return Ok(());
    }
    if !changed {
        eprintln!("{}: no changes", file.display());
        return Ok(());
    }
    write_with_metadata(file, dst, &meta, &WriteOptions { backup: out.backup })
        .with_context(|| format!("writing {}", dst.display()))?;
    eprintln!("{}: wrote {} metadata keys", dst.display(), meta.len());
    Ok(())
}

fn print_diff(old: &Metadata, new: &Metadata) {
    for (k, v) in old {
        match new.get(k) {
            None => eprintln!("- {}", term::clean(k, false)),
            Some(nv) if nv != v => eprintln!(
                "~ {}: {}",
                term::clean(k, false),
                ellipsize(nv.as_str().unwrap_or_default(), 80)
            ),
            _ => {}
        }
    }
    for (k, v) in new {
        if !old.contains_key(k) {
            eprintln!(
                "+ {}: {}",
                term::clean(k, false),
                ellipsize(v.as_str().unwrap_or_default(), 80)
            );
        }
    }
}

fn print_summary(path: &Path, h: &SafeTensorsHeader) {
    println!("{}", path.display());
    println!("  file size   {}", format::bytes(h.file_len));
    println!("  header      {}", format::bytes(h.header_len));
    println!("  tensors     {}", h.tensors.len());
    println!(
        "  parameters  {} ({})",
        format::count(h.total_params()),
        h.total_params()
    );
    for (dtype, (n, params)) in h.dtype_breakdown() {
        println!(
            "    {:8} {n:>6} tensors  {:>10} params",
            term::clean(dtype.as_str(), false),
            format::count(params)
        );
    }
    if h.metadata.is_empty() {
        println!("  metadata    (none)");
    } else {
        println!("  metadata    {} keys", h.metadata.len());
        let truncate = std::io::stdout().is_terminal();
        for (k, v) in h.metadata_strings() {
            println!(
                "    {}: {}",
                term::clean(k, false),
                if truncate {
                    ellipsize(v, 100)
                } else {
                    term::clean(v, true).into_owned()
                }
            );
        }
    }
}

fn summary_json(path: &Path, h: &SafeTensorsHeader) -> Value {
    let dtypes: serde_json::Map<String, Value> = h
        .dtype_breakdown()
        .into_iter()
        .map(|(d, (n, p))| (d.as_str().to_owned(), json!({"tensors": n, "params": p})))
        .collect();
    json!({
        "file": path.display().to_string(),
        "file_size": h.file_len,
        "header_size": h.header_len,
        "tensor_count": h.tensors.len(),
        "total_params": h.total_params(),
        "dtypes": dtypes,
        "metadata": h.metadata,
        "warnings": h.warnings,
    })
}

fn print_json(v: &Value) -> Result<()> {
    let text = serde_json::to_string_pretty(v)?;
    let mut out = std::io::stdout().lock();
    writeln!(out, "{}", term::json_safe(&text))?;
    Ok(())
}

/// Single-line preview: whitespace controls collapsed, other unsafe
/// characters escaped, cut at `max` chars.
fn ellipsize(s: &str, max: usize) -> String {
    let spaced: String = s
        .chars()
        .map(|c| {
            if matches!(c, '\n' | '\r' | '\t') {
                ' '
            } else {
                c
            }
        })
        .collect();
    let flat = term::clean(&spaced, false).into_owned();
    if flat.chars().count() <= max {
        flat
    } else {
        let cut: String = flat.chars().take(max).collect();
        format!("{cut}… ({} chars)", s.chars().count())
    }
}
