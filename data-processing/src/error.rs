use std::path::PathBuf;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum PgnError {
    #[error("IO error reading '{path}': {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Invalid argument: {0}")]
    Argument(String),
}

impl PgnError {
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}
