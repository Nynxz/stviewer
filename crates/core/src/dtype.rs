use std::fmt;

/// Element type of a tensor as declared in the header.
///
/// Unknown dtypes (e.g. from a newer spec revision) are preserved as
/// [`Dtype::Other`] rather than rejected, so the file can still be viewed
/// and its metadata edited without loss.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Dtype {
    Bool,
    U8,
    I8,
    F8E5M2,
    F8E4M3,
    F8E8M0,
    F6E2M3,
    F6E3M2,
    F4,
    I16,
    U16,
    F16,
    BF16,
    I32,
    U32,
    F32,
    I64,
    U64,
    F64,
    C64,
    Other(String),
}

impl Dtype {
    pub fn parse(s: &str) -> Self {
        match s {
            "BOOL" => Self::Bool,
            "U8" => Self::U8,
            "I8" => Self::I8,
            "F8_E5M2" => Self::F8E5M2,
            "F8_E4M3" => Self::F8E4M3,
            "F8_E8M0" => Self::F8E8M0,
            "F6_E2M3" => Self::F6E2M3,
            "F6_E3M2" => Self::F6E3M2,
            "F4" => Self::F4,
            "I16" => Self::I16,
            "U16" => Self::U16,
            "F16" => Self::F16,
            "BF16" => Self::BF16,
            "I32" => Self::I32,
            "U32" => Self::U32,
            "F32" => Self::F32,
            "I64" => Self::I64,
            "U64" => Self::U64,
            "F64" => Self::F64,
            "C64" => Self::C64,
            other => Self::Other(other.to_owned()),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::Bool => "BOOL",
            Self::U8 => "U8",
            Self::I8 => "I8",
            Self::F8E5M2 => "F8_E5M2",
            Self::F8E4M3 => "F8_E4M3",
            Self::F8E8M0 => "F8_E8M0",
            Self::F6E2M3 => "F6_E2M3",
            Self::F6E3M2 => "F6_E3M2",
            Self::F4 => "F4",
            Self::I16 => "I16",
            Self::U16 => "U16",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::I32 => "I32",
            Self::U32 => "U32",
            Self::F32 => "F32",
            Self::I64 => "I64",
            Self::U64 => "U64",
            Self::F64 => "F64",
            Self::C64 => "C64",
            Self::Other(s) => s,
        }
    }

    /// Size of one element in bits, or `None` for unknown dtypes.
    pub fn bits(&self) -> Option<u64> {
        Some(match self {
            Self::F4 => 4,
            Self::F6E2M3 | Self::F6E3M2 => 6,
            Self::Bool | Self::U8 | Self::I8 | Self::F8E5M2 | Self::F8E4M3 | Self::F8E8M0 => 8,
            Self::I16 | Self::U16 | Self::F16 | Self::BF16 => 16,
            Self::I32 | Self::U32 | Self::F32 => 32,
            Self::I64 | Self::U64 | Self::F64 | Self::C64 => 64,
            Self::Other(_) => return None,
        })
    }
}

impl fmt::Display for Dtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
