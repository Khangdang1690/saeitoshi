//! EleutherAI sparsify loader. Layered `layers.{N}/` directory layout,
//! tensor name remap from `encoder.weight`/`encoder.bias` to our canonical
//! `W_enc`/`b_enc`.
//!
//! Implementation lands in M5.
