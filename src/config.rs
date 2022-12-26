use std::{
    io,
    path::{Path, PathBuf},
};

use serde::Deserialize;

use crate::common;

#[derive(Default, Deserialize)]
struct ConfigOptional {
    x: Option<PathBuf>,
    y: Option<PathBuf>,
    l: Option<usize>,
    beta: Option<f32>,
    t: Option<usize>,
    bin: Option<String>,
}

#[derive(Debug)]
pub struct Config {
    pub x: PathBuf,
    pub y: PathBuf,
    pub l: usize,
    pub beta: f32,
    pub t: usize,
    pub bin: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self::from_optional(Default::default())
    }
}

impl Config {
    fn from_optional(optional: ConfigOptional) -> Self {
        Config {
            x: optional
                .x
                .unwrap_or_else(|| PathBuf::from("./baseline/x.dat")),
            y: optional
                .y
                .unwrap_or_else(|| PathBuf::from("./baseline/y.dat")),
            l: optional.l.unwrap_or(400),
            beta: optional.beta.unwrap_or(28.0),
            t: optional.t.unwrap_or_else(common::hardware_concurrency),
            bin: optional.bin,
        }
    }

    pub fn from_file(file: &Path) -> io::Result<Option<Self>> {
        Ok(toml::from_str(&std::fs::read_to_string(file)?)
            .map(Self::from_optional)
            .ok())
    }
}
