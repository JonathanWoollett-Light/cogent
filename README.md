# Cogent

 - [Cargo](https://crates.io/crates/cogent)
 - [Documentation](https://docs.rs/cogent/)

## Introduction

Cogent is a very basic library for training basic neural networks for classification tasks.
It is designed to be as simple as feasible.
Hyperparameters in neural network training can be set automatically, so why not?
Ideally you could simply do:
```rust
let net = NeuralNetwork::Train(&data);
```
This is the most basic and not quite there yet implementation of that idea.

Training a network to classify MNIST:
```rust
// Setup
// ----------
// 784-Sigmoid->100-Softmax->10
let mut neural_network = NeuralNetwork::new(784,&[
    Layer::new(800,Activation::ReLU),
    Layer::new(10,Activation::Softmax)
]);

// Setting training and testing data
// `get_mnist_dataset(bool)` simply gets MNIST data in format of `Vec<(Vec<f32>,usize)>` where each entry is an example (tuple.0=input and tuple.1=class).
// The boolean specifies if it is the MNIST testing data (`true`) or training data (`false`).

let training_data:Vec<(Vec<f32>,usize)> = get_mnist_dataset(false);
let testing_data:Vec<(Vec<f32>,usize)> = get_mnist_dataset(true);

// Execution
// ----------
// Trains until no notable accuracy improvements are being made over a number of iterations.
// By default this would end training if 0.5% accuracy improvement was not seen over 6 iterations (often referred to as 'epochs').

neural_network.train(&training_data)
    .evaluation_data(EvaluationData::Actual(&testing_data))
    .l2(0.1f32)
    .go();

// `.evaluation_data(...)` sets the evaluation data. 
// If evaluation data is not set it will simply shuffle and split off a random group from training data to be evaluation data.
// In the case of MNIST where training and evaluation datasets are given seperately, it makes sense to set it as such.
// `.l2(...)` sets the lambda value for L2 regularisation. 

// Evaluation
// ----------
let evaluation:(f32,u32) = neural_network.evaluate(&testing_data,None); // (cost,example correctly classified)
```

While a huge amount of my work has gone into making this and learning the basics of neural networks along the way, I am immensely (and I cannot stress this enough) amateur in inumerable ways.

If you find any issues I would really appreciate if you could let me know (and possibly suggest any solutions).

## Features

 - GPU compute using [ArrayFire Rust Bindings](https://github.com/arrayfire/arrayfire-rust)
 - Optimisers: Stochastic gradient descent.
 - Layers: Dense
 - Activations: Softmax, Sigmoid, ReLU.
 - Loss functions: Mean sqaured error, Cross entropy.
 - Regularisation techniques: L2, Dropout.



## Installation

1. [Setup ArrayFire Rust bindings](https://github.com/arrayfire/arrayfire-rust#use-from-cratesio--) (Ignore step 4).
2. Add `cogent = "^0.2"` to `Cargo.toml`.

## TODO

Task types:

 - :ballot_box_with_check: = Completable
 - :repeat: = Perpetual
 
Tasks:

1. :ballot_box_with_check: Convolutional layers.
2. :ballot_box_with_check: Meticulous testing and validation.
2. :ballot_box_with_check: Automatic net creation and layer setting from given dataset.
3. :repeat: Improve defaults.
4. :repeat: Learn more.

Please note that things may not be developed inorder, it is only my estimation.
