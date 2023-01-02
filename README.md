# GAMM

## Getting started

First make sure you have rust installed. Currently, a nightly version of
`cargo` is required due to use of unstable features.

If you are using `rustup`, the run:
```shell
rustup override set nightly
```

Due to the large sizes of the matrices, this project also uses
[`git-lfs`](https://git-lfs.com) to store the matrices. Please make sure
you have `git-lfs` installed.

## Running the binaries

To run the program:
```shell
cargo r --release
```

If no `bin` is specified in the configuration file, then all AMM
implementations are run. Otherwise, only the implementation specified by
`bin` will be run along the full matrix multiplication to calculate the
error. See also [configuration](#configuration-file) for the available
bins.

The program tries to measure the energy taken by each AMM using the
[Jetson energy meter](https://docs.nvidia.com/jetson/archives/r34.1/DeveloperGuide/text/SD/PlatformPowerAndPerformance/JetsonOrinNxSeriesAndJetsonAgxOrinSeries.html#jetson-agx-orin-series).
If able to, the program outputs the total energy consumed by each of the
AMMs.

## Environment Variables

If the `HARDWARE_CONCURRENCY` environment variable is set to some
number, that is used as the default hardware concurrency. If the
configuration file _does not_ specify a value for `t`, then
`HARDWARE_CONCURRENCY` will be used.

If the `MEASURE_LOOP_MM` environment variable is set, then it will also
measure the time taken to perform full matrix multiplication using a
handwritten unoptimized loop.

If the `ENERGY_READINGS_DIR` environment variable is set, then the
detailed energy readings taken by the energy meter will be written to
the directory specified by it. Each file will be located at
`ENERGY_READINGS_DIR/<name>.csv`.
> The program will overwrite the file if it exists. Check that old
> readings if still required are moved to a different path.

## Configuration file

The following options can be specified for the configuration file, with
their default options:
```toml
# file path of the matrices
x = "./matrices/x.dat"
y = "./matrices/y.dat"

# Reduced column size -- integer >= 0
l = 400

# Beta value used B-AMM -- floating point number
beta = 28.0

# The number of threads to use -- integer > 0
t = "<detected hardware concurrency>"

# The AMM function to run when running the program -- intra/inter/single
bin = "<not set by default>"
```

The available options for bin are `intra`, `inter` and `single`.

The configuration must be in a `.toml` file and can be passed as the
last argument when running a binary.

For example, if you want to run with the configuration given in
`./config.toml`:
```shell
cargo r --release -- ./config.toml
```
> Note the `--` is needed to delimit the arguments being sent to `cargo`
> versus the binary after it has been compiled.

## Structure of the project

All of the main implementation is available in the `gamm` library (entry
point is [`src/lib.rs`](./src/lib.rs). The runnable program is located
in [`src/main.rs`](./src/main.rs).

Within the `gamm` library the following modules are present:

- [`common`](./src/common.rs) -- this contains helper functions used
  throughout the code base.

- [`config`](./src/config.rs) -- this contains the code required for
  loading the configuration file.

- [`energy_meter`](./src/energy_meter.rs) -- this contains the
  implementation of the energy meter. Currently only the
  `JetsonEnergyMeter` is available.
 
- [`svd`](./src/svd/mod.rs) -- this contains the implementation of JTS.
  There are three implementations present:
  - Sequential JTS
  - Simple parallel JTS
  - Group parallel JTS

- [`bamm`](./src/bamm.rs) -- this contains the implementation of the
  reduction part of BCooccurring-AMM. It is used by `inter`, `intra` and
  `single` to implement the different types of parallelism in the AMM.
 
- [`inter`](./src/inter.rs)/[`intra`](./src/intra.rs)/[`single`](./src/single.rs) -- 
  these contain the approximate matrix multiplication implementations
  for inter-parallelism, intra-parallelism and no parallelism
  respectively. These use the JTS based implementation.

- [`libsvd`](./src/libsvd) -- this contains two sub-modules (`single` and `multi`) which
  contain single-threaded and multi-threaded approximate matrix
  multiplication implementation using the `nalgebra` SVD implementation.
