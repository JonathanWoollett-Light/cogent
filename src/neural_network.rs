use crate::activations::Activation;
use crate::costs::Cost;
use crate::layer::{DenseLayer, DropoutLayer};
use crate::setter_enums::*;
use crate::trainer::Trainer;
use serde::{Deserialize, Serialize};

use arrayfire::{
    af_print, cols, constant, device_mem_info, diag_extract, div, eq, imax, print_gen, set_col,
    sum, sum_all, sum_by_key, transpose, Array, Dim4,
};

use itertools::izip;

use rand::{thread_rng, Rng};

use ndarray::{ArrayView2, ArrayViewMut2, Axis};

use crossterm::{cursor, QueueableCommand};

use std::{
    collections::HashMap,
    fs::{create_dir, remove_dir_all, File},
    io::{stdout, Read, Write},
    path::Path,
    time::Instant,
};

// Default percentage of training data to set as evaluation data (0.1=5%).
const DEFAULT_EVALUTATION_DATA: f32 = 0.05f32;
// Default percentage of size of training data to set batch size (0.01=1%).
const DEFAULT_BATCH_SIZE: f32 = 0.01f32;
// Default learning rate.
const DEFAULT_LEARNING_RATE: f32 = 0.1f32;
// Default interval in iterations before early stopping.
// early stopping = default early stopping * (size of examples / number of examples) Iterations
const DEFAULT_EARLY_STOPPING: f32 = 500f32;
// Default percentage minimum positive accuracy change required to prevent early stopping or learning rate decay (0.005=0.5%).
const DEFAULT_EVALUATION_MIN_CHANGE: f32 = 0.001f32;
// Default amount to decay learning rate after period of un-notable (what word should I use here?) change.
// `new learning rate = learning rate decay * old learning rate`
const DEFAULT_LEARNING_RATE_DECAY: f32 = 0.5f32;
// Default interval in iterations before learning rate decay.
// interval = default learning rate interval * (size of examples / number of examples) iterations.
const DEFAULT_LEARNING_RATE_INTERVAL: f32 = 200f32;

/// Specifies layers to cosntruct neural net.
pub enum Layer {
    Dropout(f32),
    Dense(u64, Activation),
}
// Specifies layers within neural net.
pub enum InnerLayer {
    Dropout(DropoutLayer),
    Dense(DenseLayer),
}

/// The fundamental neural network struct.
///
/// All other types are ancillary to this structure.
pub struct NeuralNetwork {
    // Inputs to network.
    inputs: u64,
    // Activations of layers.
    layers: Vec<InnerLayer>,
}
impl<'a> NeuralNetwork {
    /// Constructs network of given layers.
    ///
    /// Returns constructed network.
    /// ```
    /// use cogent::{NeuralNetwork,Layer,Activation};
    ///
    /// let mut net = NeuralNetwork::new(2,&[
    ///     Layer::Dense(3,Activation::Sigmoid),
    ///     Layer::Dense(2,Activation::Softmax)
    /// ]);
    /// ```
    pub fn new(mut inputs: u64, layers: &[Layer]) -> NeuralNetwork {
        NeuralNetwork::new_checks(inputs, layers);

        // Necessary variable to use mutable `inputs` to nicely specify right layer sizes.
        let net_inputs = inputs;

        // Sets holder for neural net layers
        let mut inner_layers: Vec<InnerLayer> = Vec::with_capacity(layers.len());

        // Sets iterator across given layers data
        let mut layers_iter = layers.iter();
        let layer_1 = layers_iter.next().unwrap();

        // Constructs first non-input layer
        if let &Layer::Dense(size, activation) = layer_1 {
            inner_layers.push(InnerLayer::Dense(DenseLayer::new(inputs, size, activation)));
            inputs = size;
        } else if let &Layer::Dropout(p) = layer_1 {
            inner_layers.push(InnerLayer::Dropout(DropoutLayer::new(p)));
        }

        // Constructs other layers
        for layer in layers_iter {
            if let &Layer::Dense(size, activation) = layer {
                inner_layers.push(InnerLayer::Dense(DenseLayer::new(inputs, size, activation)));
                inputs = size;
            } else if let &Layer::Dropout(p) = layer {
                inner_layers.push(InnerLayer::Dropout(DropoutLayer::new(p)));
            }
        }

        // Constructs and returns neural network
        return NeuralNetwork {
            inputs: net_inputs,
            layers: inner_layers,
        };
    }
    /// Constructs network of given layers with all weights and biases set to given value.
    /// IMPORTANT: This function seems to cause issues in training and HAS NOT been properly tested, I DO NOT recommend you use this.
    pub fn new_constant(mut inputs: u64, layers: &[Layer], val: f32) -> NeuralNetwork {
        NeuralNetwork::new_checks(inputs, layers);

        // Neccessary variable to use mutable `inputs` to nicely specify right layer sizes.
        let net_inputs = inputs;

        // Sets holder for neural net layers
        let mut inner_layers: Vec<InnerLayer> = Vec::with_capacity(layers.len());

        // Sets iterator across given layers data
        let mut layers_iter = layers.iter();
        let layer_1 = layers_iter.next().unwrap();

        // Constructs first non-input layer
        if let &Layer::Dense(size, activation) = layer_1 {
            inner_layers.push(InnerLayer::Dense(DenseLayer::new_constant(
                inputs, size, activation, val,
            )));
            inputs = size;
        } else if let &Layer::Dropout(p) = layer_1 {
            inner_layers.push(InnerLayer::Dropout(DropoutLayer::new(p)));
        }

        // Constructs other layers
        for layer in layers_iter {
            if let &Layer::Dense(size, activation) = layer {
                inner_layers.push(InnerLayer::Dense(DenseLayer::new_constant(
                    inputs, size, activation, val,
                )));
                inputs = size;
            } else if let &Layer::Dropout(p) = layer {
                inner_layers.push(InnerLayer::Dropout(DropoutLayer::new(p)));
            }
        }

        // Constructs and returns neural network
        return NeuralNetwork {
            inputs: net_inputs,
            layers: inner_layers,
        };
    }
    // Checks that given data to construct neural network from valid.
    fn new_checks(inputs: u64, layers: &[Layer]) {
        // Checks network contains output layer
        if layers.len() == 0 {
            panic!("Requires output layer (layers.len() must be >0).");
        }
        // Checks inputs != 0
        if inputs == 0 {
            panic!("Input size must be >0.");
        }
        // Chekcs last layer is not a dropout layer
        if let Layer::Dropout(_) = layers[layers.len() - 1] {
            panic!("Last layer cannot be a dropout layer.");
        }
    }
    /// Sets activation of layer specified by index (excluding input layer).
    /// ```
    /// use cogent::{NeuralNetwork,Layer,Activation};
    ///
    /// // Net (2 -Sigmoid-> 3 -Sigmoid-> 2)
    /// let mut net = NeuralNetwork::new(2,&[
    ///     Layer::Dense(3,Activation::Sigmoid),
    ///     Layer::Dense(2,Activation::Sigmoid)
    /// ]);
    ///
    /// net.activation(1,Activation::Softmax); // Changes activation of output layer.
    /// // Net will now be (2 -Sigmoid-> 3 -Softmax-> 2)
    /// ```
    pub fn activation(&mut self, index: usize, activation: Activation) {
        // Checks lyaer exists
        if index >= self.layers.len() {
            panic!(
                "Layer {} does not exist. 0 <= given index < {}",
                index,
                self.layers.len()
            );
        }
        // Checks layer has activation function
        if let InnerLayer::Dense(dense_layer) = &mut self.layers[index] {
            dense_layer.activation = activation;
        } else {
            panic!("Layer {} does not have an activation function.", index);
        }
    }
    // TODO Maybe renmae this to 'forepropagate'?
    /// Runs a batch of examples through the network.
    ///
    /// Returns classes.
    pub fn run(&mut self, input: &ndarray::Array2<f32>) -> Vec<usize> {
        if input.len_of(Axis(1)) as u64 != self.inputs {
            panic!(
                "Given data inputs don't match network inputs ({}!={})",
                input.len_of(Axis(1)),
                self.inputs
            );
        }

        // // Converts 2d vec to array for input
        // let in_vec: Vec<f32> = inputs.iter().flat_map(|x| x.clone()).collect();
        // let input: Array<f32> = Array::<f32>::new(
        //     &in_vec,
        //     Dim4::new(&[example_len as u64, in_len as u64, 1, 1]),
        // );

        // Converts `ndarray::Array2` to `arrayfire::Array`
        let dims = Dim4::new(&[
            input.len_of(Axis(1)) as u64,
            input.len_of(Axis(0)) as u64,
            1,
            1,
        ]);
        let input = arrayfire::Array::new(&input.as_slice().unwrap(), dims);

        // Forepropagates
        let output = self.inner_run(&input);
        // Computes classes of each example
        let classes = arrayfire::imax(&output, 0).1;

        // Converts classes array to classes vec
        let classes_vec: Vec<u32> = NeuralNetwork::to_vec(&classes);

        // Returns classes vec casted from `Vec<u32>` to `Vec<usize>`
        return classes_vec.into_iter().map(|x| x as usize).collect();
    }
    fn to_vec<T: arrayfire::HasAfEnum + Default + Clone>(array: &arrayfire::Array<T>) -> Vec<T> {
        let mut vec = vec![T::default(); array.elements()];
        array.host(&mut vec);
        return vec;
    }
    /// Runs a batch of examples through the network.
    ///
    /// Returns output.
    pub fn inner_run(&mut self, inputs: &Array<f32>) -> Array<f32> {
        // Number of examples in input.
        let examples = inputs.dims().get()[1];
        let ones = &constant(1f32, Dim4::new(&[1, examples, 1, 1]));

        // Forepropagates.
        let mut activation = inputs.clone(); // Sets input layer
        for layer in self.layers.iter_mut() {
            activation = match layer {
                InnerLayer::Dropout(dropout_layer) => {
                    dropout_layer.forepropagate(&activation, ones)
                }
                InnerLayer::Dense(dense_layer) => dense_layer.forepropagate(&activation, &ones).0,
            };
        }

        // Returns activation of last layer.
        return activation;
    }
    /// Begins setting hyperparameters for training.
    ///
    /// Returns `Trainer` struct used to specify hyperparameters
    ///
    /// Training a network to learn an XOR gate:
    /// ```
    /// use ndarray::{Array2,array};
    /// use cogent::{
    ///     NeuralNetwork,Layer,
    ///     Activation,
    ///     EvaluationData
    /// };
    ///
    /// // Sets network
    /// let mut net = NeuralNetwork::new(2,&[
    ///     Layer::Dense(3,Activation::Sigmoid),
    ///     Layer::Dense(2,Activation::Softmax)
    /// ]);
    /// // Sets data
    /// // 0=false,  1=true.
    /// let mut data:Array2<f32> = array![[0.,0.],[1.,0.],[0.,1.],[1.,1.]];
    /// let mut labels:Array2<usize> = array![[0],[1],[1],[0]];
    ///
    /// // Trains network
    /// net.train(&mut data.clone(),&mut labels.clone()) // `.clone()` neccessary to satisfy borrow checker concerning later immutable borrow as evaluation data.
    ///     .learning_rate(2f32)
    ///     .evaluation_data(EvaluationData::Actual(&data,&labels)) // Use testing data as evaluation data.
    /// .go();
    /// ```
    pub fn train(
        &'a mut self,
        data: &'a mut ndarray::Array2<f32>,
        labels: &'a mut ndarray::Array2<usize>,
    ) -> Trainer<'a> {
        self.check_dataset(data, labels);

        let number_of_examples = data.len_of(Axis(1));
        let data_inputs = data.len_of(Axis(0));
        let multiplier: f32 = data_inputs as f32 / number_of_examples as f32;

        let early_stopping_condition: u32 = (DEFAULT_EARLY_STOPPING * multiplier).ceil() as u32;
        let learning_rate_interval: u32 =
            (DEFAULT_LEARNING_RATE_INTERVAL * multiplier).ceil() as u32;

        // TODO Do this better
        let batch_size: usize = if number_of_examples < 100usize {
            number_of_examples
        } else {
            let batch_holder: f32 = DEFAULT_BATCH_SIZE * number_of_examples as f32;
            if batch_holder < 100f32 {
                100usize
            } else {
                batch_holder.ceil() as usize
            }
        };

        return Trainer {
            training_data: data,
            training_labels: labels,
            evaluation_dataset: EvaluationData::Percent(DEFAULT_EVALUTATION_DATA),
            cost: Cost::Crossentropy,
            halt_condition: None,
            log_interval: None,
            batch_size: batch_size,
            learning_rate: DEFAULT_LEARNING_RATE,
            l2: None,
            early_stopping_condition: MeasuredCondition::Iteration(early_stopping_condition),
            evaluation_min_change: Proportion::Percent(DEFAULT_EVALUATION_MIN_CHANGE),
            learning_rate_decay: DEFAULT_LEARNING_RATE_DECAY,
            learning_rate_interval: MeasuredCondition::Iteration(learning_rate_interval),
            checkpoint_interval: None,
            name: None,
            tracking: false,
            neural_network: self,
        };
    }
    /// Checks a dataset has an equal number of example and labels and fits the network.
    ///
    /// This is called whenever you give a dataset to the library, you do not need to call this yourself.
    ///
    /// For example this is called when you pass a dataset to `.train(..)`.
    pub fn check_dataset(&self, data: &ndarray::Array2<f32>, labels: &ndarray::Array2<usize>) {
        // Checks data matches labels.
        let number_of_examples = data.len_of(Axis(0));
        if number_of_examples != labels.len_of(Axis(0)) {
            panic!(
                "Number of examples ({}) does not match number of labels ({}).",
                number_of_examples,
                labels.len_of(Axis(0))
            );
        }

        // Checks all examples fit the neural network.
        let data_inputs = data.len_of(Axis(1));
        if data_inputs != self.inputs as usize {
            panic!(
                "Input size of examples ({}) does not match input size of network ({}).",
                data_inputs, self.inputs
            );
        }

        // Gets number of network outputs
        let net_outs = match &self.layers[self.layers.len() - 1] {
            InnerLayer::Dense(dense_layer) => dense_layer.biases.dims().get()[0] as usize,
            _ => panic!("Last layer is somehow a dropout layer, this should not be possible"),
        };
        for (index, label) in labels.axis_iter(Axis(0)).enumerate() {
            if label[0] > net_outs {
                panic!(
                    "Label of example {} ({}) exceeds network outputs ({}).",
                    index, label[0], net_outs
                );
            }
        }
    }

    // TODO Name this better
    /// Runs training.
    ///
    /// In most cases you shouldn't call this, instead call `.train()` then call the functions to set the hyperparameters, then call `.go()` (which calls this).
    ///
    /// Using this function directly is ugly. Would not recommend.
    pub fn inner_train(
        &mut self,
        mut training_data: ArrayViewMut2<f32>,
        mut training_labels: ArrayViewMut2<usize>,
        evaluation_data: ArrayView2<f32>,
        evaluation_labels: ArrayView2<usize>,
        cost: &Cost,
        halt_condition: Option<HaltCondition>,
        log_interval: Option<MeasuredCondition>,
        batch_size: usize,
        intial_learning_rate: f32,
        l2: Option<f32>,
        early_stopping_n: MeasuredCondition,
        evaluation_min_change: Proportion,
        learning_rate_decay: f32,
        learning_rate_interval: MeasuredCondition,
        checkpoint_interval: Option<MeasuredCondition>,
        name: Option<&str>,
        tracking: bool,
    ) -> () {
        if let Some(_) = checkpoint_interval {
            if !Path::new("checkpoints").exists() {
                // Create folder
                create_dir("checkpoints").unwrap();
            }
            if let Some(folder) = name {
                let path = format!("checkpoints/{}", folder);
                // If folder exists, empty it.
                if Path::new(&path).exists() {
                    remove_dir_all(&path).unwrap(); // Delete folder
                }
                create_dir(&path).unwrap(); // Create folder
            }
        }

        let mut learning_rate: f32 = intial_learning_rate;

        let mut stdout = stdout(); // Handle for standard output for this process.

        let start_instant = Instant::now(); // Beginning instant to compute duration of training.
        let mut iterations_elapsed = 0u32; // Iteration counter of training.

        let mut best_accuracy_iteration = 0u32; // Iteration of best accuracy.
        let mut best_accuracy_instant = Instant::now(); // Instant of best accuracy.
        let mut best_accuracy = 0u32; // Value of best accuracy.

        // Sets array of evaluation data.
        let matrix_evaluation_data = self.matrixify(&evaluation_data, &evaluation_labels);

        // Computes intial evaluation.
        let starting_evaluation =
            self.inner_evaluate(&matrix_evaluation_data, &evaluation_labels, cost);

        // If `log_interval` has been defined, print intial evaluation.
        if let Some(_) = log_interval {
            stdout.write(format!("Iteration: {}, Time: {}, Cost: {:.5}, Classified: {}/{} ({:.3}%), Learning rate: {}\n",
                iterations_elapsed,
                NeuralNetwork::time(start_instant),
                starting_evaluation.0,
                starting_evaluation.1,evaluation_data.len_of(Axis(0)),
                (starting_evaluation.1 as f32)/(evaluation_data.len_of(Axis(0)) as f32) * 100f32,
                learning_rate
            ).as_bytes()).unwrap();
        }

        // TODO Can we only define these if we need them?
        let mut last_checkpointed_instant = Instant::now();
        let mut last_logged_instant = Instant::now();

        //panic!("got here");

        // Backpropgation loop
        // ------------------------------------------------
        loop {
            // TODO Can `matrixify` and `batch_chunks` be combined in this use case to be more efficient?
            // Sets array of training data.
            let training_data_matrix =
                self.matrixify(&training_data.view(), &training_labels.view());

            // Split training data into batchs.
            let batches = batch_chunks(&training_data_matrix, batch_size);

            // Iterates across batches running backpropagation.
            //  If `tracking` output backpropagation percentage progress.
            if tracking {
                let mut percentage: f32 = 0f32;
                stdout.queue(cursor::SavePosition).unwrap();
                let backprop_start_instant = Instant::now();
                let percent_change: f32 =
                    100f32 * batch_size as f32 / training_data_matrix.0.dims().get()[1] as f32;

                for batch in batches {
                    stdout
                        .write(format!("Backpropagating: {:.2}%", percentage).as_bytes())
                        .unwrap();
                    percentage += percent_change;
                    stdout.queue(cursor::RestorePosition).unwrap();
                    stdout.flush().unwrap();

                    // Runs backpropagation
                    self.backpropagate(
                        &batch,
                        learning_rate,
                        cost,
                        l2,
                        training_data.len_of(Axis(0)),
                    );
                }
                stdout
                    .write(
                        format!(
                            "Backpropagated: {}\n",
                            NeuralNetwork::time(backprop_start_instant)
                        )
                        .as_bytes(),
                    )
                    .unwrap();
            } else {
                for batch in batches {
                    // Runs backpropagation
                    self.backpropagate(
                        &batch,
                        learning_rate,
                        cost,
                        l2,
                        training_data.len_of(Axis(0)),
                    );
                }
            }
            iterations_elapsed += 1;

            // Computes iteration evaluation.
            let evaluation = self.inner_evaluate(&matrix_evaluation_data, &evaluation_labels, cost);

            // If `checkpoint_interval` number of iterations or length of duration passed, export weights  (`connections`) and biases (`biases`) to file.
            match checkpoint_interval {
                Some(MeasuredCondition::Iteration(iteration_interval)) => {
                    if iterations_elapsed % iteration_interval == 0 {
                        checkpoint(self, iterations_elapsed.to_string(), name);
                    }
                }
                Some(MeasuredCondition::Duration(duration_interval)) => {
                    if last_checkpointed_instant.elapsed() >= duration_interval {
                        checkpoint(self, NeuralNetwork::time(start_instant), name);
                        last_checkpointed_instant = Instant::now();
                    }
                }
                _ => {}
            }
            // If `log_interval` number of iterations or length of duration passed, print evaluation of network.
            match log_interval {
                // TODO Reduce code duplication here
                Some(MeasuredCondition::Iteration(iteration_interval)) => {
                    if iterations_elapsed % iteration_interval == 0 {
                        log_fn(
                            &mut stdout,
                            iterations_elapsed,
                            start_instant,
                            learning_rate,
                            evaluation,
                            evaluation_data.len_of(Axis(0)),
                        );
                    }
                }
                Some(MeasuredCondition::Duration(duration_interval)) => {
                    if last_logged_instant.elapsed() >= duration_interval {
                        log_fn(
                            &mut stdout,
                            iterations_elapsed,
                            start_instant,
                            learning_rate,
                            evaluation,
                            evaluation_data.len_of(Axis(0)),
                        );
                        last_logged_instant = Instant::now();
                    }
                }
                _ => {}
            }

            // If 100% accuracy, halt.
            if evaluation.1 as usize == evaluation_data.len_of(Axis(0)) {
                break;
            }

            // If `halt_condition` number of iterations occured, duration passed or accuracy acheived, halt training.
            match halt_condition {
                Some(HaltCondition::Iteration(iteration)) => {
                    if iterations_elapsed == iteration {
                        break;
                    }
                }
                Some(HaltCondition::Duration(duration)) => {
                    if start_instant.elapsed() > duration {
                        break;
                    }
                }
                Some(HaltCondition::Accuracy(accuracy)) => {
                    if evaluation.1 >= (evaluation_data.len_of(Axis(0)) as f32 * accuracy) as u32 {
                        break;
                    }
                }
                _ => {}
            }

            // TODO Reduce code duplication here
            // If change in evaluation more than `evaluation_min_change` update `best_accuracy`,`best_accuracy_iteration` and `best_accuracy_instant`.
            match evaluation_min_change {
                Proportion::Percent(percent) => {
                    if (evaluation.1 as f32 / evaluation_data.len_of(Axis(0)) as f32)
                        > (best_accuracy as f32 / evaluation_data.len_of(Axis(0)) as f32) + percent
                    {
                        best_accuracy = evaluation.1;
                        best_accuracy_iteration = iterations_elapsed;
                        best_accuracy_instant = Instant::now();
                    }
                }
                Proportion::Scalar(scalar) => {
                    if evaluation.1 > best_accuracy + scalar {
                        best_accuracy = evaluation.1;
                        best_accuracy_iteration = iterations_elapsed;
                        best_accuracy_instant = Instant::now();
                    }
                }
            }

            // If `early_stopping_n` number of iterations or length of duration passed, without improvement in accuracy (`evaluation.1`), halt training. (early_stopping_n<=halt_condition)
            match early_stopping_n {
                MeasuredCondition::Iteration(stopping_iteration) => {
                    if iterations_elapsed - best_accuracy_iteration == stopping_iteration {
                        println!("---------------\nEarly stoppage!\n---------------");
                        break;
                    }
                }
                MeasuredCondition::Duration(stopping_duration) => {
                    if best_accuracy_instant.elapsed() >= stopping_duration {
                        println!("---------------\nEarly stoppage!\n---------------");
                        break;
                    }
                }
            }

            // If `learning_rate_interval` number of iterations or length of duration passed, without improvement in accuracy (`evaluation.1`), reduce learning rate. (learning_rate_interval<early_stopping_n<=halt_condition)
            match learning_rate_interval {
                MeasuredCondition::Iteration(interval_iteration) => {
                    if iterations_elapsed - best_accuracy_iteration == interval_iteration {
                        learning_rate *= learning_rate_decay
                    }
                }
                MeasuredCondition::Duration(interval_duration) => {
                    if best_accuracy_instant.elapsed() >= interval_duration {
                        learning_rate *= learning_rate_decay
                    }
                }
            }

            // Shuffles training data
            // Training data has been shuffled when it is intially passed to this function, so don't need to shuffle on the 1st itereation.
            shuffle_dataset(&mut training_data, &mut training_labels);
        }

        // Compute and print final evaluation.
        // ------------------------------------------------
        let evaluation = self.inner_evaluate(&matrix_evaluation_data, &evaluation_labels, cost);
        let new_percent = (evaluation.1 as f32) / (evaluation_data.len_of(Axis(0)) as f32) * 100f32;
        let starting_percent =
            (starting_evaluation.1 as f32) / (evaluation_data.len_of(Axis(0)) as f32) * 100f32;
        println!();
        println!("Cost: {:.4} -> {:.4}", starting_evaluation.0, evaluation.0);
        println!(
            "Classified: {} ({:.2}%) -> {} ({:.2}%)",
            starting_evaluation.1, starting_percent, evaluation.1, new_percent
        );
        println!("Cost: {:.4}", evaluation.0 - starting_evaluation.0);
        println!(
            "Classified: +{} (+{:.3}%)",
            evaluation.1 - starting_evaluation.1,
            new_percent - starting_percent
        );
        println!(
            "Time: {}, Iterations: {}",
            NeuralNetwork::time(start_instant),
            iterations_elapsed
        );
        println!();

        // Prints evaluation of network
        fn log_fn(
            stdout: &mut std::io::Stdout,
            iterations_elapsed: u32,
            start_instant: Instant,
            learning_rate: f32,
            evaluation: (f32, u32),
            eval_len: usize,
        ) -> () {
            stdout.write(format!("Iteration: {}, Time: {}, Cost: {:.5}, Classified: {}/{} ({:.3}%), Learning rate: {}\n",
                iterations_elapsed,
                NeuralNetwork::time(start_instant),
                evaluation.0,
                evaluation.1,eval_len,
                (evaluation.1 as f32)/(eval_len as f32) * 100f32,
                learning_rate
            ).as_bytes()).unwrap();
        }
        // TODO This doesn't seem to require any more memory, look into that.
        // Splits data into chunks of examples.
        fn batch_chunks(
            data: &(Array<f32>, Array<f32>),
            batch_size: usize,
        ) -> Vec<(Array<f32>, Array<f32>)> {
            // Number of examples in dataset
            let examples = data.0.dims().get()[1];

            // Number of batches
            let batches = (examples as f32 / batch_size as f32).ceil() as usize;

            // vec containg array input and out for each batch
            let mut chunks: Vec<(Array<f32>, Array<f32>)> = Vec::with_capacity(batches);

            // Iterate over batches setting inputs and outputs
            for i in 0..batches - 1 {
                let batch_indx: usize = i * batch_size;
                let in_batch: Array<f32> = cols(
                    &data.0,
                    batch_indx as u64,
                    (batch_indx + batch_size - 1) as u64,
                );
                let out_batch: Array<f32> = cols(
                    &data.1,
                    batch_indx as u64,
                    (batch_indx + batch_size - 1) as u64,
                );

                chunks.push((in_batch, out_batch));
            }
            // Since length of final batch may be less than `batch_size`, set final batch out of loop.
            let batch_indx: usize = (batches - 1) * batch_size;
            let in_batch: Array<f32> = cols(&data.0, batch_indx as u64, examples - 1);
            let out_batch: Array<f32> = cols(&data.1, batch_indx as u64, examples - 1);
            chunks.push((in_batch, out_batch));

            return chunks;
        }
        // Outputs a checkpoint file.
        fn checkpoint(net: &NeuralNetwork, marker: String, name: Option<&str>) {
            if let Some(folder) = name {
                net.export(&format!("checkpoints/{}/{}", folder, marker));
            } else {
                net.export(&format!("checkpoints/{}", marker));
            }
        }
    }
    // Runs batch backpropgation.
    fn backpropagate(
        &mut self,
        (net_input, target): &(Array<f32>, Array<f32>),
        learning_rate: f32,
        cost: &Cost,
        l2: Option<f32>,
        training_set_length: usize,
    ) {
        // Feeds forward
        // --------------

        let examples = net_input.dims().get()[1];
        let ones = &constant(1f32, Dim4::new(&[1, examples, 1, 1]));

        // Represents activations and weighted outputs of layers.
        //  For element i we have the activation of layer i and the weighted inputs of layer i+1.
        //  All layers have activations, but not all layers have useful weighted inputs (.e.g dropout), this is why we use `Option<..>`
        let mut layer_outs: Vec<(Array<f32>, Option<Array<f32>>)> =
            Vec::with_capacity(self.layers.len());

        // TODO Name this better
        // Sets input layer activation
        let mut input = net_input.clone();

        for layer in self.layers.iter_mut() {
            let (a, z) = match layer {
                InnerLayer::Dropout(dropout_layer) => {
                    (dropout_layer.forepropagate(&input, ones), None)
                }
                InnerLayer::Dense(dense_layer) => {
                    let (a, z) = dense_layer.forepropagate(&input, &ones);
                    (a, Some(z))
                }
            };
            layer_outs.push((input, z));
            input = a;
            //NeuralNetwork::mem_info("Forepropagated layer",false);
        }
        layer_outs.push((input, None));

        //NeuralNetwork::mem_info("Forepropagated",false);

        //println!("step size: {:.4}mb",arrayfire::get_mem_step_size() as f32 / (1024f32*1024f32));

        //panic!("panic after foreprop");

        // Backpropagates
        // --------------

        let mut out_iter = layer_outs.into_iter().rev();
        let l_iter = self.layers.iter_mut().rev();

        let last_activation = &out_iter.next().unwrap().0;

        // ∇(a)C
        let mut partial_error = cost.derivative(target, last_activation);

        for (layer, (a, z)) in izip!(l_iter, out_iter) {
            // w(i)^T dot δ(i)
            // Error of layer i matrix multiplied by transposition of weights connections layer i-1 to layer i.
            partial_error = match layer {
                InnerLayer::Dropout(dropout_layer) => dropout_layer.backpropagate(&partial_error),
                InnerLayer::Dense(dense_layer) => dense_layer.backpropagate(
                    &partial_error,
                    &z.unwrap(),
                    &a,
                    learning_rate,
                    l2,
                    training_set_length,
                ),
            };
            //NeuralNetwork::mem_info("Backpropagated layer",false);
        }
    }

    // For debug purposes
    #[allow(dead_code)]
    fn mem_info(msg: &str, bytes: bool) {
        let mem_info = device_mem_info();
        println!(
            "{} : {:.4}mb | {:.4}mb",
            msg,
            mem_info.0 as f32 / (1024f32 * 1024f32),
            mem_info.2 as f32 / (1024f32 * 1024f32),
        );
        println!("buffers: {} | {}", mem_info.1, mem_info.3);
        if bytes {
            println!("bytes: {} | {}", mem_info.0, mem_info.2);
        }
    }

    /// Evaluates dataset using network.
    ///
    /// Returns tuple: (Average cost across dataset, Number of examples correctly classified).
    /// ```
    /// # use ndarray::{Array2,array};
    /// # use cogent::{
    /// #     NeuralNetwork,Layer,
    /// #     Activation,
    /// #     EvaluationData
    /// # };
    /// #
    /// # let mut net = NeuralNetwork::new(2,&[
    /// #     Layer::Dense(3,Activation::Sigmoid),
    /// #     Layer::Dense(2,Activation::Softmax)
    /// # ]);
    /// #
    /// let mut data:Array2<f32> = array![[0.,0.],[1.,0.],[0.,1.],[1.,1.]];
    /// let mut labels:Array2<usize> = array![[0],[1],[1],[0]];
    /// #
    /// # net.train(&mut data.clone(),&mut labels.clone()) // `.clone()` neccessary to satisfy borrow checker concerning later immutable borrow as evaluation data.
    /// #    .learning_rate(2f32)
    /// #    .evaluation_data(EvaluationData::Actual(&data,&labels)) // Use testing data as evaluation data.
    /// # .go();
    /// // `net` is neural network trained to 100% accuracy to mimic an XOR gate.
    /// // Passing `None` for the cost uses the default cost function (crossentropy).
    /// let (cost,accuracy) = net.evaluate(&data,&labels,None);
    ///
    /// assert_eq!(accuracy,4);
    pub fn evaluate(
        &mut self,
        data: &ndarray::Array2<f32>,
        labels: &ndarray::Array2<usize>,
        cost: Option<&Cost>,
    ) -> (f32, u32) {
        if let Some(cost_function) = cost {
            return self.inner_evaluate(
                &self.matrixify(&data.view(), &labels.view()),
                &labels.view(),
                cost_function,
            );
        } else {
            return self.inner_evaluate(
                &self.matrixify(&data.view(), &labels.view()),
                &labels.view(),
                &Cost::Crossentropy,
            );
        }
    }
    // TODO Rewrite to accept `&ArrayViewMut2` and `&Array2` for `labels`
    /// Returns tuple: (Average cost across batch, Number of examples correctly classified).
    fn inner_evaluate(
        &mut self,
        (input, target): &(Array<f32>, Array<f32>),
        labels: &ArrayView2<usize>,
        cost: &Cost,
    ) -> (f32, u32) {
        // Forepropgatates input
        let output = self.inner_run(input);

        // Computes cost
        let cost: f32 = cost.run(target, &output);
        // Computes example output classes
        let output_classes = imax(&output, 0).1;

        // Sets array of target classes
        let target_classes: Vec<u32> = labels.axis_iter(Axis(0)).map(|x| x[0] as u32).collect();
        let number_of_examples = labels.len_of(Axis(0));
        let target_array = Array::<u32>::new(
            &target_classes,
            Dim4::new(&[1, number_of_examples as u64, 1, 1]),
        );

        // Gets number of correct classifications.
        let correct_classifications = eq(&output_classes, &target_array, false); // TODO Can this be a bitwise AND?
        let correct_classifications_numb: u32 = sum_all(&correct_classifications).0 as u32;

        // Returns average cost and number of examples correctly classified.
        return (
            cost / number_of_examples as f32,
            correct_classifications_numb,
        );
    }
    /// Returns tuple of: (Vector of class percentage accuracies, Percentage confusion matrix).
    /// ```
    /// # use ndarray::{Array2,array};
    /// # use cogent::{
    /// #     NeuralNetwork,Layer,
    /// #     Activation,
    /// #     EvaluationData
    /// # };
    /// #
    /// # let mut net = NeuralNetwork::new(2,&[
    /// #     Layer::Dense(3,Activation::Sigmoid),
    /// #     Layer::Dense(2,Activation::Softmax)
    /// # ]);
    /// #
    /// let mut data:Array2<f32> = array![[0.,0.],[1.,0.],[0.,1.],[1.,1.]];
    /// let mut labels:Array2<usize> = array![[0],[1],[1],[0]];
    /// #
    /// # net.train(&mut data.clone(),&mut labels.clone()) // `.clone()` neccessary to satisfy borrow checker concerning later immutable borrow for `analyze`.
    /// #    .learning_rate(2f32)
    /// #    .evaluation_data(EvaluationData::Actual(&data,&labels)) // Use testing data as evaluation data.
    /// # .go();
    /// // `net` is neural network trained to 100% accuracy to mimic an XOR gate.
    /// let (correct_vector,confusion_matrix) = net.analyze(&data,&labels);
    ///
    /// assert_eq!(correct_vector,vec![1f32,1f32]);
    /// assert_eq!(confusion_matrix,vec![[1f32,0f32],[0f32,1f32]]);
    /// ```
    // #[deprecated(
    //     note = "Not deprecated, just broken until ArrayFire update installer to match git (where issue has been reported and fixed)."
    // )]
    pub fn analyze(
        &mut self,
        data: &ndarray::Array2<f32>,
        labels: &ndarray::Array2<usize>,
    ) -> (Vec<f32>, Vec<Vec<f32>>) {
        // Gets number of network outputs
        let net_outs = match &self.layers[self.layers.len() - 1] {
            InnerLayer::Dense(dense_layer) => dense_layer.biases.dims().get()[0] as usize,
            _ => panic!("Last layer is somehow a dropout layer, this should not be possible"),
        };

        // Sorts by class labels
        let (sorted_data, sorted_labels) = counting_sort(data, labels, net_outs);

        let (input, classes) = matrixify_classes(&sorted_data, &sorted_labels);
        let outputs = self.inner_run(&input);

        let maxs: Array<f32> = arrayfire::max(&outputs, 0i32);

        let class_vectors: Array<bool> = eq(&outputs, &maxs, true);

        let confusion_matrix: Array<f32> =
            sum_by_key(&classes, &class_vectors, 1i32).1.cast::<f32>();

        let class_lengths: Array<f32> = sum(&confusion_matrix, 1i32); // Number of examples of each class

        let percent_confusion_matrix: Array<f32> = div(&confusion_matrix, &class_lengths, true); // Divides each row (example) by number of examples of that class.

        let dims = percent_confusion_matrix.dims();
        let mut flat_vec = vec![f32::default(); (dims.get()[0] * dims.get()[1]) as usize]; // dims.get()[0] == dims.get()[1]
                                                                                           // `x.host(...)` outputs in column-major order, calling `tranpose(x).host(...)` effectively outputs in row-major order.
        transpose(&percent_confusion_matrix, false).host(&mut flat_vec);
        let matrix_vec: Vec<Vec<f32>> = flat_vec
            .chunks(dims.get()[0] as usize)
            .map(|x| x.to_vec())
            .collect();

        // Gets diagonal from matrix, representing what percentage of examples where correctly identified as each class.
        let diag = diag_extract(&percent_confusion_matrix, 0i32);
        let mut diag_vec: Vec<f32> = vec![f32::default(); diag.dims().get()[0] as usize];
        diag.host(&mut diag_vec);

        return (diag_vec, matrix_vec);

        fn matrixify_classes(
            data: &ndarray::Array2<f32>,
            labels: &ndarray::Array2<usize>,
        ) -> (Array<f32>, Array<u32>) {
            let number_of_examples = data.len_of(Axis(0)) as u64;

            // Constructs input and output array
            let dims = Dim4::new(&[data.len_of(Axis(1)) as u64, number_of_examples, 1, 1]);
            let input = Array::new(&data.as_slice().unwrap(), dims);

            let labels_u32 = labels.mapv(|x| x as u32);
            let dims = Dim4::new(&[number_of_examples, 1, 1, 1]);
            let classes: Array<u32> = Array::<u32>::new(labels_u32.as_slice().unwrap(), dims);

            // Returns input and output array
            // Array(in,examples,1,1), Array(out,examples,1,1)
            return (input, classes);
        }
        fn counting_sort(
            data: &ndarray::Array2<f32>,
            labels: &ndarray::Array2<usize>,
            k: usize,
        ) -> (ndarray::Array2<f32>, ndarray::Array2<usize>) {
            let n = data.len_of(Axis(0)); // = labels.len_of(Axis(1))
            let mut count: Vec<usize> = vec![0usize; k];
            let mut output_vals: Vec<usize> = vec![0usize; n];

            for i in 0..n {
                let class = labels[[i, 0]];

                count[class] += 1usize;
                output_vals[i] = class;
            }
            for i in 1..count.len() {
                count[i] += count[i - 1];
            }

            let mut sorted_data = ndarray::Array2::from_elem(data.dim(), f32::default());
            let mut sorted_labels = ndarray::Array2::from_elem(labels.dim(), usize::default());

            for i in 0..n {
                set_row(i,count[output_vals[i]] - 1, data, &mut sorted_data);
                sorted_labels[[count[output_vals[i]] - 1, 0]] = labels[[i, 0]];
                count[output_vals[i]] -= 1;
            }

            return (sorted_data, sorted_labels);
        }
        // TODO Surely there must be a better way to do this? (Why is such a method not obvious in the ndarray docs?)
        fn set_row(from_index:usize,to_index: usize, from: &ndarray::Array2<f32>, to: &mut ndarray::Array2<f32>) {
            for i in 0..from.len_of(Axis(1)) {
                // TODO Double check `Axis(0)` (I mess it up a lot)
                to[[to_index,i]] = from[[from_index,i]];
            }
        }
    }

    /// Returns tuple of pretty strings of: (Vector of class percentage accuracies, Percentage confusion matrix).
    ///
    /// Example without dictionairy:
    /// ```
    /// # use ndarray::{Array2,array};
    /// # use cogent::{EvaluationData,MeasuredCondition,Activation,Layer,NeuralNetwork};
    /// #
    /// # let mut net = NeuralNetwork::new(2,&[
    /// #     Layer::Dense(3,Activation::Sigmoid),
    /// #     Layer::Dense(2,Activation::Softmax)
    /// # ]);
    /// #
    /// let mut data:Array2<f32> = array![[0.,0.],[1.,0.],[0.,1.],[1.,1.]];
    /// let mut labels:Array2<usize> = array![[0],[1],[1],[0]];
    /// 
    /// # net.train(&mut data.clone(),&mut labels.clone()) // `.clone()` neccessary to satisfy borrow checker concerning later immutable borrow for `analyze_string`.
    /// #    .learning_rate(2f32)
    /// #    .evaluation_data(EvaluationData::Actual(&data,&labels)) // Use testing data as evaluation data.
    /// # .go();
    /// #
    /// // `net` is neural network trained to 100% accuracy to mimic an XOR gate.
    /// let (correct_vector,confusion_matrix) = net.analyze_string(&data,&labels,2,None);
    ///
    /// let expected_vector:&str =
    /// "    0    1   
    ///   ┌           ┐
    /// % │ 1.00 1.00 │
    ///   └           ┘\n";
    /// assert_eq!(&correct_vector,expected_vector);
    ///
    /// let expected_matrix:&str =
    /// "%   0    1   
    ///   ┌           ┐
    /// 0 │ 1.00 0.00 │
    /// 1 │ 0.00 1.00 │
    ///   └           ┘\n";
    /// assert_eq!(&confusion_matrix,expected_matrix);
    /// ```
    /// Example with dictionairy:
    /// ```
    /// # use ndarray::{Array2,array};
    /// # use cogent::{EvaluationData,MeasuredCondition,Activation,Layer,NeuralNetwork};
    /// # use std::collections::HashMap;
    /// #
    /// # let mut net = NeuralNetwork::new(2,&[
    /// #     Layer::Dense(3,Activation::Sigmoid),
    /// #     Layer::Dense(2,Activation::Softmax)
    /// # ]);
    /// #
    /// let mut data:Array2<f32> = array![[0.,0.],[1.,0.],[0.,1.],[1.,1.]];
    /// let mut labels:Array2<usize> = array![[0],[1],[1],[0]];
    ///
    /// # net.train(&mut data.clone(),&mut labels.clone()) // `.clone()` neccessary to satisfy borrow checker concerning later immutable borrow for `analyze_string`.
    /// #    .learning_rate(2f32)
    /// #    .evaluation_data(EvaluationData::Actual(&data,&labels)) // Use testing data as evaluation data.
    /// # .go();
    /// #
    /// let mut dictionairy:HashMap<usize,&str> = HashMap::new();
    /// dictionairy.insert(0,"False");
    /// dictionairy.insert(1,"True");
    ///
    /// // `net` is neural network trained to 100% accuracy to mimic an XOR gate.
    /// let (correct_vector,confusion_matrix) = net.analyze_string(&data,&labels,2,Some(dictionairy));
    ///
    /// let expected_vector:&str =
    /// "     False True 
    ///   ┌              ┐
    /// % │  1.00  1.00  │
    ///   └              ┘\n";
    /// assert_eq!(&correct_vector,expected_vector);
    ///
    /// let expected_matrix:&str =
    /// "    %    False True 
    ///       ┌              ┐
    /// False │  1.00  0.00  │
    ///  True │  0.00  1.00  │
    ///       └              ┘\n";
    /// assert_eq!(&confusion_matrix,expected_matrix);
    /// ```
    // #[deprecated(
    //     note = "Not deprecated, just broken until ArrayFire update installer to match git (where issue has been reported and fixed)."
    // )]
    pub fn analyze_string(
        &mut self,
        data: &ndarray::Array2<f32>,
        labels: &ndarray::Array2<usize>,
        precision: usize,
        dict_opt: Option<HashMap<usize, &str>>,
    ) -> (String, String) {
        let (vector, matrix) = self.analyze(data, labels);

        let class_outs = match &self.layers[self.layers.len() - 1] {
            InnerLayer::Dense(dense_layer) => dense_layer.biases.dims().get()[0] as usize,
            _ => panic!("Last layer is somehow a dropout layer, this should not be possible"),
        };

        let classes: Vec<String> = if let Some(dictionary) = dict_opt {
            (0..class_outs)
                .map(
                    |class| {
                        if let Some(label) = dictionary.get(&class) {
                            String::from(*label)
                        } else {
                            format!("{}", class)
                        }
                    }, // TODO Do this conversion better
                )
                .collect()
        } else {
            (0..class_outs).map(|class| format!("{}", class)).collect() // TODO See above todo
        };

        let widest_class: usize = classes
            .iter()
            .fold(1usize, |max, x| std::cmp::max(max, x.chars().count()));
        let class_spacing: usize = std::cmp::max(precision + 2, widest_class);

        let vector_string = vector_string(&vector, &classes, precision, class_spacing);
        let matrix_string =
            matrix_string(&matrix, &classes, precision, widest_class, class_spacing);

        return (vector_string, matrix_string);

        fn vector_string(
            vector: &Vec<f32>,
            classes: &Vec<String>,
            precision: usize,
            spacing: usize,
        ) -> String {
            let mut string = String::new(); // TODO Change this to `::with_capacity();`

            let precision_width = precision + 2;
            let space_between_vals = spacing - precision_width + 1;
            let row_width = ((spacing + 1) * vector.len()) + space_between_vals;

            string.push_str(&format!("  {:1$}", "", space_between_vals));
            for class in classes {
                string.push_str(&format!(" {:1$}", class, spacing));
            }
            string.push_str("\n");
            string.push_str(&format!("{:1$}", "", 2));
            string.push_str(&format!("┌{:1$}┐\n", "", row_width));
            string.push_str(&format!("% │{:1$}", "", space_between_vals));
            for val in vector {
                string.push_str(&format!("{:.1$}", val, precision));
                string.push_str(&format!("{:1$}", "", space_between_vals))
            }
            string.push_str("│\n");
            string.push_str(&format!("{:1$}", "", 2));
            string.push_str(&format!("└{:1$}┘\n", "", row_width));

            return string;
        }
        fn matrix_string(
            matrix: &Vec<Vec<f32>>,
            classes: &Vec<String>,
            precision: usize,
            class_width: usize,
            spacing: usize,
        ) -> String {
            let mut string = String::new(); // TODO Change this to `::with_capacity();`
            let precision_width = precision + 2;
            let space_between_vals = spacing - precision_width + 1;
            let row_width = ((spacing + 1) * matrix[0].len()) + space_between_vals;

            string.push_str(&format!(
                "{:2$}% {:3$}",
                "",
                "",
                class_width - 1,
                space_between_vals
            ));

            for class in classes {
                string.push_str(&format!(" {:1$}", class, spacing));
            }
            string.push_str("\n");

            string.push_str(&format!("{:2$} ┌{:3$}┐\n", "", "", class_width, row_width));

            for i in 0..matrix.len() {
                string.push_str(&format!(
                    "{: >2$} │{:3$}",
                    classes[i], "", class_width, space_between_vals
                ));
                for val in matrix[i].iter() {
                    string.push_str(&format!("{:.1$}", val, precision));
                    string.push_str(&format!("{:1$}", "", space_between_vals))
                }
                string.push_str("│\n");
            }
            string.push_str(&format!("{:2$} └{:3$}┘\n", "", "", class_width, row_width));

            return string;
        }
    }
    // TODO Document this better
    // TODO Rewrite to accept `&ArrayView2`s and `&Array2`s
    // Convert ndarray arrays to arrayfire arrays.
    fn matrixify(
        &self,
        data: &ArrayView2<f32>,
        labels: &ArrayView2<usize>,
    ) -> (Array<f32>, Array<f32>) {
        // TODO Is there a better way to do either of these?
        // Flattens examples into `in_vec` and `out_vec`
        let net_outs = match &self.layers[self.layers.len() - 1] {
            InnerLayer::Dense(dense_layer) => dense_layer.biases.dims().get()[0] as usize,
            _ => panic!("Last layer is somehow a dropout layer, this should not be possible"),
        };

        let number_of_examples = data.len_of(Axis(0)) as u64;

        // Constructs input and output array
        let dims = Dim4::new(&[data.len_of(Axis(1)) as u64, number_of_examples, 1, 1]);
        let input = arrayfire::Array::new(&data.as_slice().unwrap(), dims);

        // Creates all possible target vecs to be cloned when needed.
        let mut target_vecs: Vec<Vec<f32>> = vec![vec!(0.; net_outs); net_outs];
        for i in 0..net_outs {
            target_vecs[i][i] = 1.;
        }

        let flat_labels: Vec<f32> = labels
            .axis_iter(Axis(0))
            .map(|label| target_vecs[label[0]].clone())
            .flatten()
            .collect();

        let target: Array<f32> = Array::<f32>::new(
            &flat_labels,
            Dim4::new(&[net_outs as u64, number_of_examples, 1, 1]),
        );

        // Returns input and output array
        // Array(in,examples,1,1), Array(out,examples,1,1)
        return (input, target);
    }
    // Returns Instant::elapsed() as hh:mm:ss string.
    fn time(instant: Instant) -> String {
        let mut seconds = instant.elapsed().as_secs();
        let hours = (seconds as f32 / 3600f32).floor();
        seconds = seconds % 3600;
        let minutes = (seconds as f32 / 60f32).floor();
        seconds = seconds % 60;
        let time = format!("{:#02}:{:#02}:{:#02}", hours, minutes, seconds);
        return time;
    }
    /// Exports neural network to `path.json`.
    /// ```ignore
    /// use cogent::{Activation,Layer,NeuralNetwork};
    ///
    /// let net = NeuralNetwork::new(2,&[
    ///     Layer::new(3,Activation::Sigmoid),
    ///     Layer::new(2,Activation::Softmax)
    /// ]);
    ///
    /// net.export("my_neural_network");
    /// ```
    pub fn export(&self, path: &str) {
        let mut layers: Vec<InnerLayerEnum> = Vec::with_capacity(self.layers.len() - 1);

        for layer in self.layers.iter() {
            layers.push(match layer {
                InnerLayer::Dropout(dropout_layer) => InnerLayerEnum::Dropout(dropout_layer.p),
                InnerLayer::Dense(dense_layer) => {
                    let mut bias_holder = vec![f32::default(); dense_layer.biases.elements()];
                    let mut weight_holder = vec![f32::default(); dense_layer.weights.elements()];
                    dense_layer.biases.host(&mut bias_holder);
                    dense_layer.weights.host(&mut weight_holder);
                    InnerLayerEnum::Dense(
                        dense_layer.activation,
                        *dense_layer.biases.dims().get(),
                        bias_holder,
                        *dense_layer.weights.dims().get(),
                        weight_holder,
                    )
                }
            });
        }

        let export_struct = ImportExportNet {
            inputs: self.inputs,
            layers,
        };

        let file = File::create(format!("{}.json", path));
        let serialized: String = serde_json::to_string(&export_struct).unwrap();
        file.unwrap().write_all(serialized.as_bytes()).unwrap();
    }
    /// Imports neural network from `path.json`.
    /// ```ignore
    /// use cogent::NeuralNetwork;
    /// let net = NeuralNetwork::import("my_neural_network");
    /// ```
    pub fn import(path: &str) -> NeuralNetwork {
        let file = File::open(format!("{}.json", path));
        let mut string_contents: String = String::new();
        file.unwrap().read_to_string(&mut string_contents).unwrap();
        let import_struct: ImportExportNet = serde_json::from_str(&string_contents).unwrap();

        let mut layers: Vec<InnerLayer> = Vec::with_capacity(import_struct.layers.len());

        for layer in import_struct.layers {
            layers.push(match layer {
                InnerLayerEnum::Dropout(p) => InnerLayer::Dropout(DropoutLayer::new(p)),
                InnerLayerEnum::Dense(activation, b_dims, biases, w_dims, weights) => {
                    InnerLayer::Dense(DenseLayer {
                        activation,
                        biases: Array::new(&biases, Dim4::new(&b_dims)),
                        weights: Array::new(&weights, Dim4::new(&w_dims)),
                    })
                }
            });
        }

        return NeuralNetwork {
            inputs: import_struct.inputs,
            layers,
        };
    }
}

/// Strcut used to import/export neural net.
#[derive(Serialize, Deserialize)]
struct ImportExportNet {
    inputs: u64,
    layers: Vec<InnerLayerEnum>,
}
// Defines layers for import/export struct.
#[derive(Serialize, Deserialize)]
enum InnerLayerEnum {
    Dropout(f32),
    Dense(Activation, [u64; 4], Vec<f32>, [u64; 4], Vec<f32>),
}

// TODO Can this be consended with `trainer::shuffle_dataset(..)`?
fn shuffle_dataset(data: &mut ArrayViewMut2<f32>, labels: &mut ArrayViewMut2<usize>) {
    let examples = data.len_of(Axis(0));
    let input_size = data.len_of(Axis(1));

    let mut data_slice = data.as_slice_mut().unwrap();
    let mut label_slice = labels.as_slice_mut().unwrap();

    for i in 0..examples - 1 {
        let new_index: usize = thread_rng().gen_range(i, examples);

        let (data_indx_1, data_indx_2) = (i * input_size, new_index * input_size);
        // TODO Can we swap slices better?
        for t in 0..input_size {
            swap(&mut data_slice, data_indx_1 + t, data_indx_2 + t);
        }
        swap(&mut label_slice, i, new_index);
    }

    fn swap<T: Copy>(list: &mut [T], a: usize, b: usize) {
        let temp = list[a];
        list[a] = list[b];
        list[b] = temp;
    }
}
