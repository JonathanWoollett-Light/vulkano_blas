# vulkano_blas

- [Cargo](https://crates.io/crates/vulkano_blas)

## Crate status

Currently due to the limitations of Vulkano very few BLAS operations can be implemented ([my issue on GitHub concerning this](https://github.com/vulkano-rs/vulkano/issues/1395)).

All shaders I am certain I will implement but cannot implement at the moment due to this limitation are marked as deprecated.

I hope perhaps this crate while at the moment extremely limited might spike some interest and motiviation in the topic.

## Installation

1. [Install Vulkano](https://github.com/vulkano-rs/vulkano#setup)
2. Add `vulkano_blas = "0.1"` to `Cargo.toml`.

## Docs

For some reason (I do not understand) cargo cannot build the docs automatically, as such to see docs you will need to download this repo and run `cargo rustdoc --open` in the directory. 
