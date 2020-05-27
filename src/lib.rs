//! Cogent is a library to cosntruct and train neural networks.
//!
//! The goal of Cogent is to provide a simple library for usage of nueral networks.
//! ## Crate Status
//! While Cogent has come quite a way it is still very early in its development, as such things will be changing all the time.
//!
//! Things may change massively from version to version.

pub use crate::activations::*;
mod activations;

pub use crate::costs::*;
mod costs;

mod layer;

pub use crate::neural_network::{NeuralNetwork,Layer};
mod neural_network;

pub use crate::setter_enums::*;
mod setter_enums;

mod trainer;