# Cogent

## Introduction

Cogent is a very basic library for training basic neural networks for classification tasks.
It is designed to be as simple as feasible.
Hyperparameters in neural network training can be set automitcally, so why not?
Ideally you could simply do:
> let net = NeuralNetwork::Train(&data)
This is the most basic and not quite there yet implementation of that idea.

Training a network to classify MNIST:
```
// Setup
let mut neural_network = NeuralNetwork::new(784,&[
    Layer::new(100,Activation::Sigmoid),
    Layer::new(10,Activation::Softmax)
]);
// Sets training and testing data

// `get_mnist_dataset(bool)` simply gets MNIST data in format of `Vec<(Vec<f32>,usize)>` where each entry is an example and tuple.0=input and tuple.1=class.
// The boolean specifiers if it is the MNIST testing data (`true`) or training data (`false`).

let training_data:Vec<(Vec<f32>,usize)> = get_mnist_dataset(false);
let testing_data:Vec<(Vec<f32>,usize)> = get_mnist_dataset(true);

// Execution

// Trains until no notable accuracy improvements are being made over a number of iterations.
// By default this would end training if 0.5% accuracy improvement was not seen over 6 iterations (often referred to as 'epochs').

neural_network.train(&training_data,10) // `10`=number of classes
    .evaluation_data(EvaluationData::Actual(&testing_data))
    .go();

// Evaluation

let evaluation:(f32,u32) = neural_network.evaluate(&testing_data,10); // (cost,accuracy)
```

While a huge amount of my work has gone into making this and learning the basics of neural networks along the way, I am immensely (and I cannot stress this enough) amateur in the field.

If you find any issues I would really appreciate if you could let me know (and possible suggest any solutions).
