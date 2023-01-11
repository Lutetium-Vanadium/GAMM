use std::{
    io,
    path::{Path, PathBuf},
};

use serde::Deserialize;

use crate::common;

#[derive(Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum Bin {
    Single,
    Intra,
    Inter,
    Parallel,
    All,
}

#[derive(Deserialize, Debug)]
pub struct Config {
    #[serde(default = "defaults::x")]
    pub x: PathBuf,
    #[serde(default = "defaults::y")]
    pub y: PathBuf,
    #[serde(default = "defaults::l")]
    pub l: usize,
    #[serde(default = "defaults::beta")]
    pub beta: common::Float,
    #[serde(default = "common::hardware_concurrency")]
    pub t: usize,
    #[serde(default = "defaults::bin")]
    pub bin: Bin,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            x: defaults::x(),
            y: defaults::y(),
            l: defaults::l(),
            beta: defaults::beta(),
            t: common::hardware_concurrency(),
            bin: defaults::bin(),
        }
    }
}

impl Config {
    pub fn from_file(file: &Path) -> io::Result<Option<Self>> {
        Ok(toml::from_str(&std::fs::read_to_string(file)?).ok())
    }
}

// Serde defaults take in a function to call for the default, it cannot have default by value
mod defaults {
    use crate::common::Float;
    use std::path::PathBuf;

    pub(super) fn x() -> PathBuf {
        PathBuf::from("./matrices/x.dat")
    }

    pub(super) fn y() -> PathBuf {
        PathBuf::from("./matrices/y.dat")
    }

    pub(super) fn l() -> usize {
        400
    }

    pub(super) fn beta() -> Float {
        28.0
    }

    pub(super) fn bin() -> super::Bin {
        std::env::var("GAMM_BIN")
            // It is unfortunate, but to use serde to deserialize the GAMM_BIN from the
            // environment, variable, we need to wrap it in quotes
            .map(|mut s| {
                s.insert(0, '"');
                s.push('"');
                toml::from_str::<super::Bin>(&s).ok()
            })
            .ok()
            .flatten()
            .unwrap_or(super::Bin::All)
    }
}
