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

There are 5 binaries that you can run:

- `gamm` -- this runs all the ways to compute the AMM and outputs the
  error and time taken for each one of them

- `single` -- this no-parallelism AMM and outputs the error and time
  taken for each one of them

- `intra` -- this intra-parallelism AMM and outputs the error and time
  taken for each one of them

- `inter` -- this inter-parallelism AMM and outputs the error and time
  taken for each one of them

- `profile` -- this runs the binary specified in the `config` with an
  energy meter.

To run a particular binary:
```shell
cargo r --release --bin <bin>
```

If running the `gamm` binary, `--bin gamm` is not required.

## Environment Variables

If the `HARDWARE_CONCURRENCY` environment variable is set to some
number, that is used as the default hardware concurrency. If the
configuration file _does not_ specify a value for `t`, then
`HARDWARE_CONCURRENCY` will be used.

When running the `gamm` binary, if the `MEASURE_LOOP_MM` environment
variable is set, then it will also measure the time taken to perform
full matrix multiplication using a handwritten unoptimized loop.

When running the `profile` binary, if the `WRITE_ENERGY_READINGS`
environment variable is set, then the detailed energy readings taken by
the energy meter will be written to the file specified by it.

## Configuration file

The following options can be specified for the configuration file, with
their default options:
```toml
x = "./matrices/x.dat"
y = "./matrices/y.dat"
l = 400
beta = 28.0
t = <detected hardware concurrency>
bin = <not set by default>
```

The available options for bin are `intra`, `inter` and `single`.

The configuration must be in a `.toml` file and can be passed as the
last argument when running a binary.

For example, if you want to run `profile` with the configuration given
in `./config.toml`:
```shell
cargo r --release --bin profile -- ./config.toml
```
> Note the `--` is needed to delimit the arguments being sent to `cargo`
> versus the binary after it has been compiled.

## Structure of the project

All of the main implementation is available in the `gamm` library (entry
point is `src/lib.rs`).

The binaries are available in `src/bin/` with the exception of the
`gamm` binary which is in `src/main.rs`.

Within the `gamm` library the following modules are present:

- `common` -- this contains helper functions used throughout the
  code base.

- `config` -- this contains the code required for loading the
  configuration file.

- `energy_meter` -- this contains the implementation of the energy
  meter. Currently only the `JetsonEnergyMeter` is available.
 
- `svd` -- this contains the implementation of JTS. There are three
  implementations present:
  - Sequential JTS
  - Simple parallel JTS
  - Group parallel JTS

- `bamm` -- this contains the implementation of the reduction part of
  BCooccurring-AMM. It is used by `inter`, `intra` and `single` to
  implement the different types of parallelism in the AMM.
 
- `inter`/`intra`/`single` -- these contain the approximate matrix
  multiplication implementations for inter-parallelism,
  intra-parallelism and no parallelism respectively. These use the JTS
  based implementation.

- `libsvd` -- this contains two sub-modules (`single` and `multi`) which
  contain single-threaded and multi-threaded approximate matrix
  multiplication implementation using the `nalgebra` SVD implementation.
