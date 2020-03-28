/// Core functionality of training a neural network.
pub mod core {
    use rand::{rngs::ThreadRng};
    use rand::prelude::SliceRandom;
    
    use std::time::{Duration,Instant};
    use itertools::izip;

    use scoped_threadpool::Pool;

    use ndarray::{Array2,Array1,ArrayD,Axis,ArrayView2};
    use ndarray_rand::{RandomExt,rand_distr::Uniform};

    use ndarray_einsum_beta::*;

    use std::io::{Read,Write, stdout};
    use crossterm::{QueueableCommand, cursor};

    use serde::{Serialize,Deserialize};
    
    use std::mem::swap;

    use std::fs::File;
    use std::fs;
    use std::path::Path;

    // Setting number of threads to use
    const THREAD_COUNT:usize = 12usize;

    use std::f32;

    // Default percentage of training data to set as evaluation data (0.1=10%).
    const DEFAULT_EVALUTATION_DATA:f32 = 0.1f32;
    // Default percentage of size of training data to set batch size (0.002=0.2%).
    const DEFAULT_BATCH_SIZE:f32 = 0.002f32;
    // Default learning rate.
    const DEFAULT_LEARNING_RATE:f32 = 0.1f32;
    // Default percentage size of training data to set regularization parameter (0.1=10%).
    const DEFAULT_LAMBDA:f32 = 0.1f32;
    // Default seconds to set duration for early stopping condition. 
    // early stopping = default early stopping * (size of examples / number of examples) seconds
    const DEFAULT_EARLY_STOPPING:f32 = 400f32;
    // Default percentage minimum positive accuracy change required to prevent early stopping or learning rate decay (0.005=0.5%).
    const DEFAULT_EVALUATION_MIN_CHANGE:f32 = 0.005f32;
    // Default amount to decay learning rate after period of un-notable (what word should I use here?) change.
    // `new learning rate = learning rate decay * old learning rate`
    const DEFAULT_LEARNING_RATE_DECAY:f32 = 0.5f32;
    // Default interval to go without notable importment before learning rate decay.
    // interval = default learning rate interval * (size of examples / number of examples) iterations.
    const DEFAULT_LEARNING_RATE_INTERVAL:f32 = 200f32;
    // ...
    const DEFAULT_MIN_LEARNING_RATE:f32 = 0.001f32;

    /// For setting `evaluation_data`.
    pub enum EvaluationData<'b> {
        Scaler(usize),
        Percent(f32),
        Actual(&'b Vec<(Vec<f32>,usize)>)
    }
    /// For setting a hyperparameter with measured intervals.
    #[derive(Clone,Copy)]
    pub enum MeasuredCondition {
        Iteration(u32),
        Duration(Duration)
    }
    /// For setting `halt_condition`.
    ///
    /// The training halt condition.
    #[derive(Clone,Copy)]
    pub enum HaltCondition {
        Iteration(u32),
        Duration(Duration),
        Accuracy(f32)
    }
    /// For setting a hyperparameter as a proportion of another.
    #[derive(Clone,Copy)]
    pub enum Proportion {
        Scaler(u32),
        Percent(f32),
    }
    
    /// To practicaly implement optional setting of training hyperparameters.
    pub struct Trainer<'a> {
        training_data: Vec<(Vec<f32>,usize)>,
        k:usize,
        evaluation_data: Vec<(Vec<f32>,usize)>,
        // Will halt after at a certain iteration, accuracy or duration.
        halt_condition: Option<HaltCondition>,
        // Can log after a certain number of iterations, a certain duration, or not at all.
        log_interval: Option<MeasuredCondition>,
        batch_size: usize, // TODO Maybe change `batch_size` to allow it to be set by a user as a % of their data
        // Reffered to as `ETA` in `NeuralNetwork`.
        learning_rate: f32, 
        // Regularization parameter
        lambda: f32,
        // Can stop after a lack of cost improvement over a certain number of iterations/durations, or not at all.
        early_stopping_condition: MeasuredCondition,
        // Minimum change required to log positive evaluation change.
        evaluation_min_change: Proportion, 
        // Amount to decrease learning rate by (less than 1)(`learning_rate` *= learning_rate_decay`).
        learning_rate_decay: f32, 
        // Time without notable improvement to wait until decreasing learning rate.
        learning_rate_interval: MeasuredCondition,
        // Duration/iterations between outputting neural network weights and biases to file.
        checkpoint_interval: Option<MeasuredCondition>,
        // Sets what to pretend to checkpoint files. Used to differentiate between nets when checkpointing multiple.
        name: Option<&'a str>,
        // Whether to print percantage progress in each iteration of backpropagation
        tracking: bool,
        // Minimum learning rate before adding new layer and resetting learning rate
        min_learning_rate: f32,
        neural_network: &'a mut NeuralNetwork
    }
    impl<'a> Trainer<'a> {
        /// Sets `evaluation_data`.
        /// 
        /// `evaluation_data` determines how to set the evaluation data.
        pub fn evaluation_data(&mut self, evaluation_data:EvaluationData) -> &mut Trainer<'a> {
            self.evaluation_data = match evaluation_data {
                EvaluationData::Scaler(scaler) => { self.training_data.split_off(self.training_data.len() - scaler) }
                EvaluationData::Percent(percent) => { self.training_data.split_off(self.training_data.len() - (self.training_data.len() as f32 * percent) as usize) }
                EvaluationData::Actual(actual) => { actual.clone() }
            };
            return self;
        }
        /// Sets `halt_condition`.
        /// 
        /// `halt_condition` sets after which Iteration/Duration or reached accuracy to stop training.
        pub fn halt_condition(&mut self, halt_condition:HaltCondition) -> &mut Trainer<'a> {
            self.halt_condition = Some(halt_condition);
            return self;
        }
        /// Sets `log_interval`.
        /// 
        /// `log_interval` sets some amount of Iterations/Duration to print the cost and accuracy of the neural net.
        pub fn log_interval(&mut self, log_interval:MeasuredCondition) -> &mut Trainer<'a> {
            self.log_interval = Some(log_interval);
            return self;
        }
        /// Sets `batch_size`.
        pub fn batch_size(&mut self, batch_size:Proportion) -> &mut Trainer<'a> {
            self.batch_size = match batch_size {
                Proportion::Percent(percent) => { (self.training_data.len() as f32 * percent) as usize },
                Proportion::Scaler(scaler) => { scaler as usize } 
            };
            return self;
        }
        /// Sets `learning_rate`.
        pub fn learning_rate(&mut self, learning_rate:f32) -> &mut Trainer<'a> {
            self.learning_rate = learning_rate;
            return self;
        }
        /// Sets `lamdba` (otherwise known as regulation parameter).
        /// 
        /// `lamdba` is the regularization paramter.
        pub fn lambda(&mut self, lambda:f32) -> &mut Trainer<'a> {
            self.lambda = lambda;
            return self;
        }
        /// Sets `early_stopping_condition`.
        /// 
        /// `early_stopping_condition` sets some amount of Iterations/Duration to stop after without notable cost improvement.
        pub fn early_stopping_condition(&mut self, early_stopping_condition:MeasuredCondition) -> &mut Trainer<'a> {
            self.early_stopping_condition = early_stopping_condition;
            return self;
        }
        /// Sets `evaluation_min_change`.
        /// 
        /// Minimum change required to log positive evaluation change.
        pub fn evaluation_min_change(&mut self, evaluation_min_change:Proportion) -> &mut Trainer<'a> {
            self.evaluation_min_change = evaluation_min_change;
            return self;
        }
        /// Sets `learning_rate_decay`.
        /// 
        /// `learning_rate_decay` is the mulipliers by which to decay the learning rate.
        pub fn learning_rate_decay(&mut self, learning_rate_decay:f32) -> &mut Trainer<'a> {
            self.learning_rate_decay = learning_rate_decay;
            return self;
        }
        /// Sets `learning_rate_interval`.
        pub fn learning_rate_interval(&mut self, learning_rate_interval:MeasuredCondition) -> &mut Trainer<'a> {
            self.learning_rate_interval = learning_rate_interval;
            return self;
        }
        /// Sets `checkpoint_interval`.
        /// 
        /// `checkpoint_interval` sets how often (if at all) to serialize and output neural network to .txt file.
        pub fn checkpoint_interval(&mut self, checkpoint_interval:MeasuredCondition) -> &mut Trainer<'a> {
            self.checkpoint_interval = Some(checkpoint_interval);
            return self;
        }
        /// Sets `name`
        /// 
        /// `name` sets what to pretend to checkpoint files. Used to differentiate between nets when checkpointing multiple.
        pub fn name(&mut self, name:&'a str) -> &mut Trainer<'a> {
            self.name=Some(name);
            return self;
        }
        /// Sets `tracking`.
        /// 
        /// `tracking` determines whether to output percentage progress during backpropgation.
        pub fn tracking(&mut self) -> &mut Trainer<'a> {
            self.tracking = true;
            return self;
        }
        /// Sets `min_learning_rate`
        /// 
        /// ...
        pub fn min_learning_rate(&mut self,min_learning_rate:f32) -> &mut Trainer<'a> {
            self.min_learning_rate = min_learning_rate;
            return self;
        }
        /// Begins training.
        pub fn go(&mut self) -> () {
            self.neural_network.train_details(
                &mut self.training_data,
                self.k,
                &self.evaluation_data,
                self.halt_condition,
                self.log_interval,
                self.batch_size,
                self.learning_rate,
                self.lambda,
                self.early_stopping_condition,
                self.evaluation_min_change,
                self.learning_rate_decay,
                self.learning_rate_interval,
                self.checkpoint_interval,
                self.name,
                self.tracking,
                self.min_learning_rate
            );
        }
    }
    /// Defines cost function of neural network.
    #[derive(Clone,Copy,Serialize,Deserialize)]
    pub enum Cost {
        /// Quadratic cost function.
        Quadratic,
        /// Crossentropy cost function.
        Crossentropy
    }
    impl Cost {
        /// Runs cost functions
        /// 
        /// y: Target out, a: Actual out
        fn run(&self,y:&Array2<f32>,a:&Array2<f32>) -> f32 {
            return match self {
                Self::Quadratic => { quadratic(y,a) },
                Self::Crossentropy => { cross_entropy(y,a) }
            };
            // Quadratic cost
            fn quadratic(y: &Array2<f32>, a: &Array2<f32>) -> f32 {
                (y - a).mapv(|x| x.powi(2)).sum() / (2f32*a.nrows() as f32)
            }
            // Cross entropy cost
            // TODO Need to double check this
            fn cross_entropy(y: &Array2<f32>, a: &Array2<f32>) -> f32 {
                let part1 = a.mapv(f32::ln) * y;
                let part2 = (1f32 - a).mapv(f32::ln) * (1f32 - y);
                let mut cost:f32 = (part1 + part2).sum();
                cost /= -(a.shape()[1] as f32);
                return cost;
            }
        }
        /// Derivative wrt layer output (∂C/∂a)
        /// 
        /// y: Target out, a: Actual out
        fn derivative(&self,y:&Array2<f32>,a:&Array2<f32>) -> Array2<f32> {
            return match self {
                Self::Quadratic => { a-y },
                Self::Crossentropy => { -1f32 * ((1f32/a)*y) + (1f32-y) * (1f32/(1f32-a)) } // (-1 * (y*(1/a))) + (1-y) * (1/(1-a))
            }
        }
    }
    /// Defines activations of layers in neural network.
    #[derive(Clone,Copy,Serialize,Deserialize)]
    pub enum Activation {
        /// Sigmoid activation functions.
        Sigmoid,
        /// Softmax activation function.
        Softmax,
        /// ReLU activation function.
        ReLU // Name it 'ReLU' or 'Relu'?
    }
    impl Activation {
        /// Computes activations given inputs.
        fn run(&self,z:&Array2<f32>) -> Array2<f32> {
            return match self {
                Self::Sigmoid => z.mapv(|x| Activation::sigmoid(x)),
                Self::Softmax => Activation::softmax(z),
                Self::ReLU => z.mapv(|x| Activation::relu(x)),
            };
        }
        // Derivative wrt layer input (∂a/∂z)
        fn derivative(&self,z:&Array2<f32>) -> Array2<f32> {
            // What should we name the derivative functions?
            return match self {
                Self::Sigmoid => z.mapv(|x| sigmoid_derivative(x)),
                Self::Softmax => softmax_derivative(z),
                Self::ReLU => z.mapv(|x| relu_derivative(x)),
            };

            // Derivative of sigmoid
            // s' = s(1-s)
            fn sigmoid_derivative(z:f32) -> f32 {
                let s:f32 = Activation::sigmoid(z);
                return s*(1f32-s);
            }
            // Derivative of softmax
            // e^z * (sum of other inputs e^input) / (sum of all inputs e^input)^2 = e^z * (exp_sum-e^z) / (exp_sum)^2
            fn softmax_derivative(z:&Array2<f32>) -> Array2<f32> {
                let mut derivatives:Array2<f32> = z.mapv(|x|x.exp());

                // Gets sum of each row
                let sums:Array1<f32> = derivatives.sum_axis(Axis(1));

                // Sets squared sum of each row
                let sqrd_sums:Array1<f32> = &sums * &sums;

                for (mut row,sum,sqrd_sum) in izip!(
                    derivatives.axis_iter_mut(Axis(0)),
                    sums.iter(),
                    sqrd_sums.iter()
                ) {
                    row.mapv_inplace(|val| (val*(sum-val))/sqrd_sum);
                }

                //panic!("testing stuff");
                return derivatives;
            }
            //Deritvative of ReLU
            // ReLU(z)/1 = if >0 1 else 0
            fn relu_derivative(z:f32) -> f32 {
                if z > 0f32 { 
                    return 1f32; 
                } else { 
                    return 0f32; 
                }
            }
        }
        // Applies sigmoid function
        fn sigmoid(x: f32) -> f32 {
            1f32 / (1f32 + (-x).exp())
        }
        // TODO Make this better
        // Applies softmax activation
        fn softmax(y: &Array2<f32>) -> Array2<f32> {
            let mut exp_matrix = y.clone();

            // Subtracts row max from all values.
            //  Allowing softmax to handle large values in y.
            // ------------------------------------------------
            // Get max value in each row (each example).
            let max_axis_vals = exp_matrix.fold_axis(Axis(1),0f32,|acc,x| (if acc > x { *acc } else { *x }));
            // Subtracts row max from every value in matrix.
            for i in 0..exp_matrix.nrows() {
                exp_matrix.row_mut(i).mapv_inplace(|x| x-max_axis_vals[i]); 
            }

            // Applies softmax
            // ------------------------------------------------
            // Apply e^(x) to every value in matrix
            exp_matrix.mapv_inplace(|x|x.exp());
            // Calculates sums of rows
            let sum = exp_matrix.sum_axis(Axis(1));
            // Divide every value in matrix by sum of its row
            for (sum,mut row) in izip!(sum.iter(),exp_matrix.axis_iter_mut(Axis(0))) {
                row.mapv_inplace(|x| x / sum);
            }
            return exp_matrix;
        }
        // Applies ReLU activation
        fn relu(x:f32) -> f32 {
            if x > 0f32 {
                return x;
            }
            else {
                return 0f32;
            }
        }
    }
    
    /// Used to specify layers to construct neural net.
    pub struct Layer {
        size: usize,
        activation: Activation,
    }
    impl Layer {
        /// Creates new layer
        pub fn new(size:usize,activation:Activation) -> Layer {
            Layer {size,activation}
        }
    }

    /// Neural network.
    #[derive(Serialize,Deserialize,Clone)]
    pub struct NeuralNetwork {
        // Inputs to network
        inputs: usize,
        // Layer biases
        biases: Vec<Array2<f32>>,
        // Connections between layers
        connections: Vec<Array2<f32>>,
        // Activations of layers
        layers: Vec<Activation>,
        // Cost function
        cost: Cost,
    }
    impl NeuralNetwork {
        /// Constructs network of given layers.
        /// 
        /// Returns constructed network.
        /// ```
        /// use cogent::core::{NeuralNetwork,Layer,Activation};
        /// 
        /// let mut net = NeuralNetwork::new(2,&[
        ///     Layer::new(3,Activation::Sigmoid),
        ///     Layer::new(2,Activation::Softmax)
        /// ],None);
        /// ```
        pub fn new(inputs:usize,layers: &[Layer],cost:Option<Cost>) -> NeuralNetwork {
            if layers.len() == 0 {
                panic!("Requires >1 layers");
            }
            if inputs == 0 {
                panic!("Input size must be >0");
            }
            for x in layers {
                if x.size < 1usize {
                    panic!("All layer sizes must be >0");
                }
            }

            let mut cost_function = Cost::Crossentropy;
            if let Some(function) = cost {
                cost_function = function;
            }

            let mut connections: Vec<Array2<f32>> = Vec::with_capacity(layers.len());
            let mut biases: Vec<Array2<f32>> = Vec::with_capacity(layers.len());

            let range = Uniform::new(-1f32, 1f32);
            // Sets connections between inputs and 1st hidden layer
            connections.push(Array2::random(
                (layers[0].size,inputs),range) / (inputs as f32).sqrt()
            );
            // Sets biases for 1st hidden layer
            biases.push(Array2::random((1,layers[0].size),range));
            // Sets connections and biases for all subsequent layers
            for i in 1..layers.len() {
                connections.push(
                    Array2::random((layers[i].size,layers[i-1].size),range)
                    / (layers[i-1].size as f32).sqrt()
                );
                biases.push(Array2::random((1,layers[i].size),range));
            }
            let layers:Vec<Activation> = layers.iter().map(|x|x.activation).collect();
            NeuralNetwork{ inputs, biases, connections, layers, cost:cost_function}
        }
        /// Sets activation of layer specified by index (excluding input layer).
        /// ```
        /// use cogent::core::{NeuralNetwork,Layer,Activation};
        /// 
        /// let mut net = NeuralNetwork::new(2,&[
        ///     Layer::new(3,Activation::Sigmoid),
        ///     Layer::new(2,Activation::Sigmoid)
        /// ],None);
        /// net.activation(1,Activation::Softmax); // Changes activation of output layer.
        /// ```
        pub fn activation(&mut self, index:usize, activation:Activation) {
            if index >= self.layers.len() {
                // TODO Make better panic message here.
                panic!("Layer {} does not exist. 0 <= given index < {}",index,self.layers.len()); 
            } 
            self.layers[index] = activation;
        }
        /// Runs batch of examples through network.
        /// 
        /// Returns outputs from batch of examples.
        /// ```
        /// use cogent::core::{NeuralNetwork,Layer,Activation};
        /// use ndarray::{Array2,array};
        /// 
        /// let mut net = NeuralNetwork::new(2,&[
        ///     Layer::new(3,Activation::Sigmoid),
        ///     Layer::new(2,Activation::Softmax)
        /// ],None);
        /// let input:Array2<f32> = array![
        ///     [0f32,0f32],
        ///     [1f32,0f32],
        ///     [0f32,1f32],
        ///     [1f32,1f32]
        /// ];
        /// let output:Array2<f32> = net.run(&input);
        /// ```
        pub fn run(&self, inputs:&Array2<f32>) -> Array2<f32> {
            let mut activations:Array2<f32> = inputs.clone();
            for i in 0..self.layers.len() {
                let weighted_inputs:Array2<f32> = activations.dot(&self.connections[i].t());
                let bias_matrix:Array2<f32> = Array2::ones((inputs.shape()[0],1)).dot(&self.biases[i]);
                let inputs = weighted_inputs + bias_matrix;
                activations = self.layers[i].run(&inputs);
            }
            return activations;
        }
        /// Begins setting hyperparameters for training.
        /// 
        /// Returns `Trainer` struct used to specify hyperparameters
        /// 
        /// Training a network to learn an XOR gate:
        /// ```
        /// use cogent::core::{NeuralNetwork,Layer,Activation,EvaluationData};
        /// 
        /// // Sets network
        /// let mut neural_network = NeuralNetwork::new(2,&[
        ///     Layer::new(3,Activation::Sigmoid),
        ///     Layer::new(2,Activation::Softmax)
        /// ],None);
        /// // Sets data
        /// // For output 0=false and 1=true.
        /// let data = vec![
        ///     (vec![0f32,0f32],0),
        ///     (vec![1f32,0f32],1),
        ///     (vec![0f32,1f32],1),
        ///     (vec![1f32,1f32],0)
        /// ];
        /// // Trains network
        /// neural_network.train(&data,2)
        ///     .learning_rate(2f32)
        ///     .evaluation_data(EvaluationData::Actual(&data)) // Use testing data as evaluation data.
        ///     .lambda(0f32)
        /// .go();
        /// ```
        pub fn train(&mut self,training_data:&Vec<(Vec<f32>,usize)>,k:usize) -> Trainer {
            // TODO Should we be helpful and do this check or not bother?
            // Checks all examples fit the neural network.
            for i in 0..training_data.len() {
                let example = &training_data[i];
                if example.0.len() != self.inputs {
                    panic!("Input size of example {} != size of input layer.",i);
                }
                else if k != self.biases[self.biases.len()-1].len() {
                    panic!("Output size of example {} != size of output layer.",i);
                }
            }

            let mut rng = rand::thread_rng();
            let mut temp_training_data = training_data.clone();
            temp_training_data.shuffle(&mut rng);
            let temp_evaluation_data = temp_training_data.split_off(training_data.len() - (training_data.len() as f32 * DEFAULT_EVALUTATION_DATA) as usize);

            let multiplier:f32 = training_data[0].0.len() as f32 / training_data.len() as f32;
            let early_stopping_condition:u32 = (DEFAULT_EARLY_STOPPING * multiplier).ceil() as u32;
            let learning_rate_interval:u32 = (DEFAULT_LEARNING_RATE_INTERVAL * multiplier).ceil() as u32;
            
            let batch_holder:f32 = DEFAULT_BATCH_SIZE * training_data.len() as f32;
            // TODO What should we use as min batch size here instead of `10f32`?
            let batch_size:usize = if batch_holder < 10f32 {
                training_data.len()
            }
            else {
                batch_holder.ceil() as usize
            };

            return Trainer {
                training_data: temp_training_data,
                k:k,
                evaluation_data: temp_evaluation_data,
                halt_condition: None,
                log_interval: None,
                batch_size: batch_size,
                learning_rate: DEFAULT_LEARNING_RATE,
                lambda: DEFAULT_LAMBDA,
                early_stopping_condition: MeasuredCondition::Iteration(early_stopping_condition),
                evaluation_min_change: Proportion::Percent(DEFAULT_EVALUATION_MIN_CHANGE),
                learning_rate_decay: DEFAULT_LEARNING_RATE_DECAY,
                learning_rate_interval: MeasuredCondition::Iteration(learning_rate_interval),
                checkpoint_interval: None,
                name: None,
                tracking: false,
                min_learning_rate: DEFAULT_MIN_LEARNING_RATE,
                neural_network:self
            };
        }

        // TODO Name this better
        // Runs training.
        fn train_details(&mut self,
            training_data: &mut [(Vec<f32>,usize)], // TODO Look into `&mut [(Vec<f32>,Vec<f32>)]` vs `&mut Vec<(Vec<f32>,Vec<f32>)>`
            k:usize,
            evaluation_data: &[(Vec<f32>,usize)],
            halt_condition: Option<HaltCondition>,
            log_interval: Option<MeasuredCondition>,
            batch_size: usize,
            intial_learning_rate: f32,
            lambda: f32,
            early_stopping_n: MeasuredCondition,
            evaluation_min_change: Proportion,
            learning_rate_decay: f32,
            learning_rate_interval: MeasuredCondition,
            checkpoint_interval: Option<MeasuredCondition>,
            name: Option<&str>,
            tracking:bool,
            min_learning_rate:f32
        ) -> (){
            if let Some(_) = checkpoint_interval {
                if !Path::new("checkpoints").exists() {
                    // Create folder
                    fs::create_dir("checkpoints").unwrap();
                }
                if let Some(folder) = name {
                    let path = format!("checkpoints/{}",folder);
                    // If folder exists, empty it.
                    if Path::new(&path).exists() {
                        fs::remove_dir_all(&path).unwrap();// Delete folder
                    }
                    fs::create_dir(&path).unwrap(); // Create folder
                }
            }

            let mut learning_rate:f32 = intial_learning_rate;

            let mut stdout = stdout(); // Handle for standard output for this process.
            let mut rng = rand::thread_rng(); // Random number generator.

            let start_instant = Instant::now(); // Beginning instant to compute duration of training.
            let mut iterations_elapsed = 0u32; // Iteration counter of training.
            
            let mut best_accuracy_iteration = 0u32;// Iteration of best accuracy.
            let mut best_accuracy_instant = Instant::now();// Instant of best accuracy.
            let mut best_accuracy = 0u32; // Value of best accuracy.

            let starting_evaluation = self.evaluate(evaluation_data,k); // Compute intial evaluation.

            // If `log_interval` has been defined, print intial evaluation.
            if let Some(_) = log_interval {
                stdout.write(format!("Iteration: {}, Time: {}, Cost: {:.5}, Classified: {}/{} ({:.3}%), Learning rate: {}\n",
                    iterations_elapsed,
                    NeuralNetwork::time(start_instant),
                    starting_evaluation.0,
                    starting_evaluation.1,evaluation_data.len(),
                    (starting_evaluation.1 as f32)/(evaluation_data.len() as f32) * 100f32,
                    learning_rate
                ).as_bytes()).unwrap();
            }

            // TODO Can we only define these if we need them?
            let mut last_checkpointed_instant = Instant::now();
            let mut last_logged_instant = Instant::now();

            training_data.shuffle(&mut rng);

            let mut inner_training_data =  NeuralNetwork::matrixify(training_data,k);

            // Backpropgation loop
            // ------------------------------------------------
            loop {
                shuffle_matrix_data(&mut rng,&mut inner_training_data);
                let batches = NeuralNetwork::batch_chunks(&inner_training_data,batch_size); // Split dataset into batches.
                // TODO Reduce code duplication here.
                // Runs backpropagation on all batches:
                //  If `tracking` output backpropagation percentage progress.

                if tracking {
                    let mut percentage:f32 = 0f32;
                    stdout.queue(cursor::SavePosition).unwrap();
                    let backprop_start_instant = Instant::now();
                    let percent_change:f32 = 100f32 * batch_size as f32 / inner_training_data.0.nrows() as f32;

                    for batch in batches {
                        stdout.write(format!("Backpropagating: {:.2}%",percentage).as_bytes()).unwrap();
                        percentage += percent_change;
                        stdout.queue(cursor::RestorePosition).unwrap();
                        stdout.flush().unwrap();

                        let (new_connections,new_biases) = self.update_batch(&batch,learning_rate,lambda,training_data.len() as f32);
                        self.connections = new_connections;
                        self.biases = new_biases;
                    }
                    stdout.write(format!("Backpropagated: {}\n",NeuralNetwork::time(backprop_start_instant)).as_bytes()).unwrap();
                }
                else {
                    for batch in batches {
                        let (new_connections,new_biases) = self.update_batch(&batch,learning_rate,lambda,training_data.len() as f32);
                        self.connections = new_connections;
                        self.biases = new_biases;
                    }
                }
                iterations_elapsed += 1;
                let evaluation = self.evaluate(evaluation_data,k);

                // If `checkpoint_interval` number of iterations or length of duration passed, export weights  (`connections`) and biases (`biases`) to file.
                match checkpoint_interval {// TODO Reduce code duplication here
                    Some(MeasuredCondition::Iteration(iteration_interval)) => if iterations_elapsed % iteration_interval == 0 {
                        if let Some(folder) = name {
                            self.export(&format!("checkpoints/{}/{}",folder,iterations_elapsed));
                        }
                        else {
                            self.export(&format!("checkpoints/{}",iterations_elapsed));
                        }
                    },
                    Some(MeasuredCondition::Duration(duration_interval)) => if last_checkpointed_instant.elapsed() >= duration_interval {
                        if let Some(folder) = name {
                            self.export(&format!("checkpoints/{}/{}",folder,NeuralNetwork::time(start_instant)));
                        }
                        else {
                            self.export(&format!("checkpoints/{}",NeuralNetwork::time(start_instant)));
                        }
                        last_checkpointed_instant = Instant::now();
                    },
                    _ => {},
                }

                // If `log_interval` number of iterations or length of duration passed, print evaluation of network.
                match log_interval {// TODO Reduce code duplication here
                    Some(MeasuredCondition::Iteration(iteration_interval)) => if iterations_elapsed % iteration_interval == 0 {
                        log_fn(&mut stdout,iterations_elapsed,start_instant,learning_rate,evaluation,evaluation_data.len()
                        );
                    },
                    Some(MeasuredCondition::Duration(duration_interval)) => if last_logged_instant.elapsed() >= duration_interval {
                        log_fn(&mut stdout,iterations_elapsed,start_instant,learning_rate,evaluation,evaluation_data.len()
                        );
                        last_logged_instant = Instant::now();
                    },
                    _ => {},
                }

                // If 100% accuracy, halt.
                if evaluation.1 as usize == evaluation_data.len() { break; }

                // If `halt_condition` number of iterations occured, duration passed or accuracy acheived, halt training.
                match halt_condition {
                    Some(HaltCondition::Iteration(iteration)) => if iterations_elapsed == iteration { break; },
                    Some(HaltCondition::Duration(duration)) => if start_instant.elapsed() > duration { break; },
                    Some(HaltCondition::Accuracy(accuracy)) => if evaluation.1 >= (evaluation_data.len() as f32 * accuracy) as u32  { break; },
                    _ => {},
                }

                // TODO Reduce code duplication here
                // If change in evaluation more than `evaluation_min_change` update `best_accuracy`,`best_accuracy_iteration` and `best_accuracy_instant`.
                match evaluation_min_change {
                    Proportion::Percent(percent) => if (evaluation.1 as f32 / evaluation_data.len() as f32) > (best_accuracy as f32 / evaluation_data.len() as f32) + percent {
                        best_accuracy = evaluation.1;
                        best_accuracy_iteration = iterations_elapsed;
                        best_accuracy_instant = Instant::now();
                    }
                    Proportion::Scaler(scaler) => if evaluation.1 > best_accuracy + scaler {
                        best_accuracy = evaluation.1;
                        best_accuracy_iteration = iterations_elapsed;
                        best_accuracy_instant = Instant::now();
                    }
                }

                // If `early_stopping_n` number of iterations or length of duration passed, without improvement in accuracy (`evaluation.1`), halt training. (early_stopping_n<=halt_condition)
                match early_stopping_n {
                    MeasuredCondition::Iteration(stopping_iteration) =>  if iterations_elapsed - best_accuracy_iteration == stopping_iteration { println!("---------------\nEarly stoppage!\n---------------"); break; },
                    MeasuredCondition::Duration(stopping_duration) => if best_accuracy_instant.elapsed() >= stopping_duration { println!("---------------\nEarly stoppage!\n---------------"); break; }
                }

                // If `learning_rate_interval` number of iterations or length of duration passed, without improvement in accuracy (`evaluation.1`), reduce learning rate. (learning_rate_interval<early_stopping_n<=halt_condition)
                match learning_rate_interval {
                    MeasuredCondition::Iteration(interval_iteration) =>  if iterations_elapsed - best_accuracy_iteration == interval_iteration { learning_rate *= learning_rate_decay },
                    MeasuredCondition::Duration(interval_duration) => if best_accuracy_instant.elapsed() >= interval_duration { learning_rate *= learning_rate_decay }
                }
                if learning_rate < min_learning_rate {
                    learning_rate = intial_learning_rate;
                    self.add_layer(Layer::new(
                        self.biases[self.biases.len()-2].ncols(),
                        self.layers[self.layers.len()-2]
                    ));
                }
            }

            // Compute and print final evaluation.
            // ------------------------------------------------
            let evaluation = self.evaluate(evaluation_data,k); 
            let new_percent = (evaluation.1 as f32)/(evaluation_data.len() as f32) * 100f32;
            let starting_percent = (starting_evaluation.1 as f32)/(evaluation_data.len() as f32) * 100f32;
            println!();
            println!("Cost: {:.4} -> {:.4}",starting_evaluation.0,evaluation.0);
            println!("Classified: {} ({:.2}%) -> {} ({:.2}%)",starting_evaluation.1,starting_percent,evaluation.1,new_percent);
            println!("Cost: {:.4}",evaluation.0-starting_evaluation.0);
            println!("Classified: +{} (+{:.3}%)",evaluation.1-starting_evaluation.1,new_percent - starting_percent);
            println!("Time: {}",NeuralNetwork::time(start_instant));
            println!();

            // Prints evaluation of network
            fn log_fn(
                stdout:&mut std::io::Stdout,
                iterations_elapsed:u32,
                start_instant:Instant,
                learning_rate:f32,
                evaluation: (f32,u32),
                eval_len: usize
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
            // Assumes `example.0.nrows()==example.1.nrows()`.
            // n = example.0.nrows() = example.1.nrows()
            // O(n+n/2) | O(1)
            fn shuffle_matrix_data(rng: &mut ThreadRng,example:&mut (Array2<f32>,Array2<f32>)) {
                let length = example.0.nrows();
                let mut indexs:Vec<usize> = (0usize..length).collect();
                indexs.shuffle(rng);

                for i in 0..length/2 {
                    if i <= indexs[i] {
                        swap(&mut example.0.row(i),&mut example.0.row(indexs[i]));
                        swap(&mut example.1.row(i),&mut example.1.row(indexs[i]));
                    }
                }
            }
        }
        // Runs batch through network to calculate weight and bias gradients.
        // Returns new weight and bias values.
        fn update_batch(&self, batch: &(ArrayView2<f32>,ArrayView2<f32>), eta: f32, lambda:f32, n:f32) -> (Vec<Array2<f32>>,Vec<Array2<f32>>) {
            
            // TODO Look into a better way to setup 'bias_nabla' and 'weight_nabla'
            // Copies structure of self.neurons and self.connections with values of 0f32
            let nabla_b_zeros:Vec<Array2<f32>> = self.biases.clone().iter().map(|x| x.map(|_| 0f32) ).collect();
            let nabla_w_zeros:Vec<Array2<f32>> = self.connections.clone().iter().map(|x| x.map(|_| 0f32) ).collect();

            let batch_len = batch.0.nrows();
            let chunk_lengths:usize = if batch_len < THREAD_COUNT {
                batch_len
            } else {
                (batch_len as f32 / THREAD_COUNT as f32).ceil() as usize
            };

            // TODO Move these 3 lines into a function (ideally generalize this function so it can be used in place of `batch_chunks` aswell).
            let input_chunks = batch.0.axis_chunks_iter(Axis(0),chunk_lengths);
            let output_chunks = batch.1.axis_chunks_iter(Axis(0),chunk_lengths);
            let chunks:Vec<(ArrayView2<f32>,ArrayView2<f32>)> = input_chunks.zip(output_chunks).collect();

            let mut pool = Pool::new(chunks.len() as u32);

            let mut out_nabla_b:Vec<Vec<Array2<f32>>> = vec!(nabla_b_zeros.clone();chunks.len());
            let mut out_nabla_w:Vec<Vec<Array2<f32>>> = vec!(nabla_w_zeros.clone();chunks.len());

            pool.scoped(|scope| {
                for (chunk,nabla_w,nabla_b) in izip!(chunks,&mut out_nabla_w,&mut out_nabla_b) {
                    scope.execute(move || {
                        let (mut delta_nabla_b,mut delta_nabla_w):(Vec<Array2<f32>>,Vec<Array2<f32>>) = self.backpropagate(&chunk);
                        
                        // TODO Why don't these work?
                        // *nabla_w = nabla_w.iter().zip(delta_nabla_w).map(|(x,y)| x + y).collect();
                        // *nabla_b = nabla_b.iter().zip(delta_nabla_b).map(|(x,y)| x + y).collect();
                        // Replacement code that does the above until I can figure out why the above doesn't work
                        for layer in 0..self.biases.len() {
                            delta_nabla_w[layer] = &nabla_w[layer] + &delta_nabla_w[layer].t(); // TODO Is this `.t()` neccessary?
                            delta_nabla_b[layer] = &nabla_b[layer] + &delta_nabla_b[layer];
                        }
                        *nabla_w = delta_nabla_w;
                        *nabla_b = delta_nabla_b;
                    });
                }
            });

            // TODO Look at turning these into iterator folds and maps, had issues trying it
            // Sums elements in `out_nabla_b` and `out_nabla_w`.
            let mut nabla_b:Vec<Array2<f32>> = nabla_b_zeros.clone();
            for example in &out_nabla_b {
                for i in 0..example.len() {
                    nabla_b[i] += &example[i]; // example[i] is layer i
                }
            }
            let mut nabla_w:Vec<Array2<f32>> = nabla_w_zeros.clone();
            for example in &out_nabla_w {
                for i in 0..example.len() {
                    nabla_w[i] += &example[i]; // example[i] is layer i
                }
            }

            // TODO Look into removing `.clone()`s here
            let return_connections:Vec<Array2<f32>> = self.connections.iter().zip(nabla_w).map(
                | (w,nw) | (1f32-eta*(lambda/n))*w - ((eta / batch_len as f32)) * nw
            ).collect();

            // TODO Make this work
            // let return_biases:Vec<Array1<f32>> = self.biases.iter().zip(nabla_b.iter()).map(
            //     | (b,nb) | b - ((eta / batch.len() as f32)) * nb
            // ).collect();
            // This code replicates functionality of above code, for the time being.
            let mut return_biases:Vec<Array2<f32>> = self.biases.clone();
            for i in 0..self.biases.len() {
                // TODO Improve this
                return_biases[i] = &self.biases[i] - &((eta / batch_len as f32)  * nabla_b[i].clone());
            }

            return (return_connections,return_biases);
        }
        // Runs backpropgation on chunk of batch.
        // Returns weight and bias partial derivatives (errors).
        fn backpropagate(&self, example:&(ArrayView2<f32>,ArrayView2<f32>)) -> (Vec<Array2<f32>>,Vec<Array2<f32>>) {

            // Feeds forward
            // --------------

            let number_of_examples = example.0.nrows(); // Number of examples (rows)
            let mut inputs:Vec<Array2<f32>> = Vec::with_capacity(self.biases.len()); // Name more intuitively
            let mut activations:Vec<Array2<f32>> = Vec::with_capacity(self.biases.len()+1);
            // TODO Is `.to_owned()` the best way to do this?
            activations.push(example.0.to_owned());
            for i in 0..self.layers.len() {
                let weighted_inputs = activations[i].dot(&self.connections[i].t());
                let bias_matrix:Array2<f32> = Array2::ones((number_of_examples,1)).dot(&self.biases[i]); // TODO consider precomputing these
                inputs.push(weighted_inputs + bias_matrix);
                activations.push(self.layers[i].run(&inputs[i]));
            }

            // Backpropagates
            // --------------

            let target = example.1.clone(); // TODO check we don't need '.clone' here
            let last_index = self.connections.len()-1; // = nabla_b.len()-1 = nabla_w.len()-1 = self.neurons.len()-2 = self.connections.len()-1
            
            
            // Gradients of biases and weights.
            let mut nabla_b:Vec<Array2<f32>> = Vec::with_capacity(self.biases.len());
            // TODO find way to make this ArrayD an Array3, ArrayD willl always have 3d imensions, just can't figure out caste.
            let mut nabla_w:Vec<ArrayD<f32>> = Vec::with_capacity(self.connections.len()); // this should really be 3d matrix instead of 'Vec<DMatrix<f32>>', its a bad workaround

            let last_layer = self.layers[self.layers.len()-1];
            // TODO Is `.to_owned()` a good solution here?

            let mut error:Array2<f32> = self.cost.derivative(&target.to_owned(),&activations[activations.len()-1]) * last_layer.derivative(&inputs[inputs.len()-1]);

            // Sets gradients in output layer
            nabla_b.insert(0,error.clone());
            let weight_errors = einsum("ai,aj->aji", &[&error, &activations[last_index]]).unwrap();
            nabla_w.insert(0,weight_errors);
            

            // self.layers.len()-1 -> 1 (inclusive)
            // (self.layers.len()=self.biases.len()=self.connections.len())
            for i in (1..self.layers.len()).rev() {
                // Calculates error
                // `mat1.dot(mat2)` performs matrix multiplication of `mat1` by `mat2`
                error = self.layers[i-1].derivative(&inputs[i-1]) *
                    error.dot(&self.connections[i]);

                // Sets gradients
                nabla_b.insert(0,error.clone());
                let ein_sum = einsum("ai,aj->aji", &[&error, &activations[i-1]]).unwrap();
                nabla_w.insert(0,ein_sum);
            }
            // Sum along columns (rows represent each example), push to `nabla_b_sum`.
            let nabla_b_sum = nabla_b.iter().map(|x| cast_1_to_2(x.sum_axis(Axis(0)))).collect();
            
            // Sums through layers (each layer is a matrix representing each example), casts to Arry2 then pushes to `nabla_w_sum`.
            let nabla_w_sum = nabla_w.iter().map(|x| cast_d_to_2(x.sum_axis(Axis(0)))).collect();
            
            // Returns gradients
            return (nabla_b_sum,nabla_w_sum);

            // TODO Improvement or replacement of both these cast functions needs to be done
            // TODO find way better way to cast from ArrayD to Array2, ArrayD willl always have 2 dimensions.
            // Casts shape from (n,k) to (n,k)
            fn cast_d_to_2(arrd:ArrayD<f32>) -> Array2<f32> {
                let shape = (arrd.shape()[0],arrd.shape()[1]);
                let mut arr2:Array2<f32> = Array2::zeros(shape);
                for i in 0..shape.0 {
                    for t in 0..shape.1 {
                        arr2[[i,t]]=arrd[[i,t]];
                    }
                }
                return arr2;
            }
            // Casts shape from (n) to (1,n)
            fn cast_1_to_2(arr1:Array1<f32>) -> Array2<f32> {
                let shape = (1,arr1.len());
                let mut arr2:Array2<f32> = Array2::zeros(shape);
                for i in 0..shape.1 {
                    arr2[[0,i]]=arr1[[i]];
                }
                return arr2;
            }
        }
        // Splits data into chunks of examples (rows).
        fn batch_chunks(data:&(Array2<f32>,Array2<f32>),batch_size:usize) -> Vec<(ArrayView2<f32>,ArrayView2<f32>)>{
            let input_chunks = data.0.axis_chunks_iter(Axis(0),batch_size);
            let output_chunks = data.1.axis_chunks_iter(Axis(0),batch_size);
            return input_chunks.zip(output_chunks).collect();
        }
        
        /// Inserts new layer before output layer in network.
        pub fn add_layer(&mut self, layer:Layer) {
            let prev_neurons:usize = if let Some(indx) = self.biases.len().checked_sub(2) {
                self.biases[indx].ncols()
            } else { 
                self.inputs
            };

            let range = Uniform::new(-1f32, 1f32);
            
            // Insert new layer
            self.layers.insert(self.layers.len()-1,layer.activation);
            self.connections.insert(self.connections.len()-1,
                Array2::random((layer.size,prev_neurons),range)
                / (prev_neurons as f32).sqrt()
            );
            self.biases.insert(self.biases.len()-1,Array2::random((1,layer.size),range));

            // Update output layer
            let connection_indx = self.connections.len()-1;
            let bias_indx = self.biases.len()-1;
            let outputs = self.biases[bias_indx].ncols();

            self.connections[connection_indx] = 
                Array2::random((outputs,layer.size),range)
                / (layer.size as f32).sqrt();

            self.biases[bias_indx] = Array2::random((1,outputs),range);
        }

        /// Returns tuple: (Average cost across batch, Number of examples correctly classified).
        pub fn evaluate(&self, test_data:&[(Vec<f32>,usize)],k:usize) -> (f32,u32) {
            let chunk_len:usize = if test_data.len() < THREAD_COUNT { 
                test_data.len() 
            } else {
                (test_data.len() as f32 / THREAD_COUNT as f32).ceil() as usize 
            };
            let chunks:Vec<_> = test_data.chunks(chunk_len).collect(); // Specify type further
            let mut pool = Pool::new(chunks.len() as u32);

            let mut cost_vec = vec!(0f32;chunks.len());
            let mut classified_vec = vec!(0u32;chunks.len());
            pool.scoped(|scope| {
                for (chunk,cost,classified) in izip!(chunks,&mut cost_vec,&mut classified_vec) {
                    scope.execute(move || {
                        let batch_tuple_matrix = NeuralNetwork::matrixify(&chunk,k);
                        let out = self.run(&batch_tuple_matrix.0);
                        let target = batch_tuple_matrix.1;
                        *cost = self.cost.run(&target,&out);

                        let output_class_indxs = max_output_indexs(&out); // Array1 of result classes.
                        let target_class_indxs:Vec<usize> = chunk.iter().map(|x|x.1).collect(); // Vec of target classes.
                        *classified = izip!(output_class_indxs.iter(),target_class_indxs.iter()).fold(0u32,|acc,(a,b)| {if a==b { acc+1u32 } else { acc }});
                    });
                }
            });
            // Sum costs and correctly classified
            let cost:f32 = cost_vec.iter().sum();
            let classified:u32 = classified_vec.iter().sum();

            return (cost / chunk_len as f32, classified);

            // Gets index of max value in each row (each row representing an example) of `Array2<f32>`
            fn max_output_indexs(matrix:&Array2<f32>) -> Array1<usize> {
                let examples = matrix.shape()[0];
                let mut max_indexs:Array1<usize> = Array1::zeros(examples);
                let mut max_values:Array1<f32> = Array1::zeros(examples);
                for i in 0..examples {
                    for t in 0..matrix.row(i).len() {
                        if matrix[[i,t]] > max_values[i] {
                            max_indexs[i] = t;
                            max_values[i] = matrix[[i,t]];
                        }
                    }
                }
                return max_indexs;
            }
        }
        /// Requires ordered test_data.
        /// 
        /// Returns tuple of: (List of correctly classified percentage for each class, Confusion matrix of percentages).
        pub fn evaluate_outputs(&self, test_data:&[(Vec<f32>,usize)],k:usize) -> (Array1<f32>,Array2<f32>) {
            let chunks:Vec<Array2<f32>> = class_chunks(test_data,k);
            let mut pool = Pool::new(chunks.len() as u32);

            let mut classifications:Vec<Array1<f32>> = vec!(Array1::zeros(k);k);
            pool.scoped(|scope| {
                for (chunk,classification) in izip!(chunks,&mut classifications) {
                    scope.execute(move || {
                        let results = self.run(&chunk);
                        let classes:Array2<u32> = set_nonmax_zero(&results);
                        let class_sums:Array1<u32> = classes.sum_axis(Axis(0)); // Number of examples classified as each class
                        let number_of_examples = chunk.len_of(Axis(0));
                        *classification = class_sums.mapv(|val| (val as f32 / number_of_examples as f32)); // Percentage of examples classified as each class
                    });
                }
            });
            let matrix:Array2<f32> = cast_array1s_to_array2(classifications,k);
            // TODO Is there a better way to set this?
            let diagonal:Array1<f32> = matrix.clone().into_diag();
            return (diagonal,matrix);

            // Splits `test_data` into chunks based on class.
            // This is the part which requires `test_data` to be sorted.
            fn class_chunks(test_data:&[(Vec<f32>,usize)],k:usize) -> Vec<Array2<f32>> {
                let mut chunks:Vec<Array2<f32>> = Vec::with_capacity(k);
                let mut slice = (0usize,0usize); // (lower bound,upper bound)
                loop {
                    slice.1+=1;
                    while test_data[slice.0].1 == test_data[slice.1+1].1 {
                        slice.1+=1;
                        if slice.1+1 == test_data.len() {
                            slice.1 += 1;
                            break;
                        }
                    } 
                    let chunk_holder = NeuralNetwork::matrixify_inputs(&test_data[slice.0..slice.1]);
                    chunks.push(chunk_holder);
                    
                    slice.0 = slice.1;
                    if chunks.len() == k { break };
                }

                // If `test_data` not sorted.
                if slice.1 != test_data.len() {
                    panic!("`evaluate outputs` requires given data to be sorted by output.");
                }
                return chunks;
            }
            
            // Sets all non-max values in row to 0 and max to 1 for each row in Array2.
            fn set_nonmax_zero(matrix:&Array2<f32>) -> Array2<u32> {
                let mut max_indx = 0usize;
                let mut zero_matrix:Array2<u32> = Array2::zeros((matrix.nrows(),matrix.ncols()));

                for i in 0..matrix.nrows() {
                    for t in 1..matrix.ncols() {
                        if matrix[[i,t]] > matrix[[i,max_indx]] {
                            max_indx=t;
                        }
                    }
                    zero_matrix[[i,max_indx]] = 1u32;
                    max_indx = 0usize;
                }
                return zero_matrix;
            }
        }

        // Given set of examples return Array2<f32> of inputs.
        fn matrixify_inputs(examples:&[(Vec<f32>,usize)]) -> Array2<f32> {
            let input_len = examples[0].0.len();
            let example_len = examples.len();
            let mut input_vec:Vec<f32> = Vec::with_capacity(example_len * input_len);
            for example in examples {
                input_vec.append(&mut example.0.clone());
            }
            let input_array:Array2<f32> = Array2::from_shape_vec((example_len,input_len),input_vec).unwrap();
            return input_array;
        }
        // Converts `[(Vec<f32>,usize)]` to `(Array2<f32>,Array2<f32>)`.
        fn matrixify(examples:&[(Vec<f32>,usize)],k:usize) -> (Array2<f32>,Array2<f32>) {
            let input_len = examples[0].0.len();
            let example_len = examples.len();

            let mut input_vec:Vec<f32> = Vec::with_capacity(example_len * input_len);
            let mut output_vec:Vec<f32> = Vec::with_capacity(example_len * k);
            for example in examples {
                // TODO Can I remove the `.clone()`s here?
                input_vec.append(&mut example.0.clone());
                let mut class_vec:Vec<f32> = vec!(0f32;k);
                class_vec[example.1] = 1f32;
                output_vec.append(&mut class_vec);
            }

            // TODO Look inot better way to do this
            let input:Array2<f32> = Array2::from_shape_vec((example_len,input_len),input_vec).unwrap();
            let output:Array2<f32>  = Array2::from_shape_vec((example_len,k),output_vec).unwrap();

            return (input,output);
        }

        // Returns Instant::elapsed() as hh:mm:ss string.
        fn time(instant:Instant) -> String {
            let mut seconds = instant.elapsed().as_secs();
            let hours = (seconds as f32 / 3600f32).floor();
            seconds = seconds % 3600;
            let minutes = (seconds as f32 / 60f32).floor();
            seconds = seconds % 60;
            let time = format!("{:#02}:{:#02}:{:#02}",hours,minutes,seconds);
            return time;
        }

        // TODO General improvement, specifically allow printing to variable accuracy.
        /// Returns pretty string of wieghts (`self.connections`) and biases (`self.biases`).
        pub fn print(&self) -> String {
            let mut prt_string:String = String::new();
            let max:usize = self.biases.iter().map(|x|x.shape()[1]).max().unwrap();
            let width = self.connections.len(); // == self.biases.len()

            for row in 0..max+2 {
                for t in 0..width {
                    let diff = (max - self.biases[t].shape()[1]) / 2;
                    let spacing = 6*self.connections[t].shape()[1];
                    if row == diff {
                        prt_string.push_str("  ");
                        prt_string.push_str(&format!("┌ {: <1$}┐","",spacing));
                        prt_string.push_str("   ");
                        prt_string.push_str("┌       ┐");
                        prt_string.push_str(" ");
                    }
                    else if row == self.biases[t].shape()[1] + diff + 1 {
                        prt_string.push_str("  ");
                        prt_string.push_str(&format!("└ {: <1$}┘","",spacing));
                        prt_string.push_str("   ");
                        prt_string.push_str("└       ┘");
                        prt_string.push_str(" ");
                    }
                    else if row < diff || row > self.biases[t].shape()[1] + diff + 1 {
                        prt_string.push_str("  ");
                        prt_string.push_str(&format!("  {: <1$} ","",spacing));
                        prt_string.push_str("   ");
                        prt_string.push_str("         ");
                        prt_string.push_str(" ");
                    }
                    else {
                        let inner_row = row-diff-1;
                        if self.biases[t].shape()[1] / 2 == inner_row { prt_string.push_str("* "); }
                        else { prt_string.push_str("  "); }
                        prt_string.push_str("│ ");
                        
                        for val in self.connections[t].row(inner_row) {
                            prt_string.push_str(&format!("{:+.2} ",val));
                            
                        }
                        if inner_row == self.biases[t].shape()[1] / 2 {
                            prt_string.push_str("│ + │ ");
                        } else { prt_string.push_str("│   │ "); }
                        
                        let val = self.biases[t][[0,inner_row]];
                        prt_string.push_str(&format!("{:+.2} ",val));
    
                        
                        prt_string.push_str("│ ");
                    }
                }
                prt_string.push_str("\n");
            }
            prt_string.push_str("\n");

            return prt_string;
        }
        /// Exports neural network to `path`.
        pub fn export(&self,path:&str) {
            let file = File::create(format!("{}.json",path));
            let serialized:String = serde_json::to_string(self).unwrap();
            file.unwrap().write_all(serialized.as_bytes()).unwrap();
        }
        /// Imports neural network from `path`.
        pub fn import(path:&str) -> NeuralNetwork {
            let file = File::open(format!("{}.json",path));
            let mut string_contents:String = String::new();
            file.unwrap().read_to_string(&mut string_contents).unwrap();
            let deserialized:NeuralNetwork = serde_json::from_str(&string_contents).unwrap();
            return deserialized;
        }
    }
    /// Returns `Array2<T>` from `Vec<Array1<T>>`
    pub fn cast_array1s_to_array2<T:Default+Copy>(vec:Vec<Array1<T>>,k:usize) -> Array2<T> {
        let mut arr2 = Array2::default((vec.len(),k));
        let k = vec[0].len();
        for i in 0..vec.len() {
            if vec[i].len() != k { panic!("Cannot convert `Vec<Array1<T>>` to `Array2<T>`. vec[{}].len() ({}) does not equal k ({})",i,vec[i].len(),k); }
            for t in 0..k {
                arr2[[i,t]] = vec[i][t];
            }
        }
        return arr2;
    }
}
/// Some uneccessary utility functions not fundementally linked to `core::NeuralNetwork`
pub mod utilities {
    extern crate ndarray;
    use ndarray::{Array2,Array3};
    use std::fmt::Display;
    // TODO: Generalise these pretty prints for an array of any number of dimensions.
    /// Returns pretty string of `Array2<T>`.
    /// ```
    /// use cogent::utilities::array2_prt;
    /// use ndarray::{array,Array2};
    /// 
    /// let array2:Array2<f32> = array![
    ///     [-4f32,-3f32,-2f32],
    ///     [-1f32,0f32,1f32],
    ///     [2f32,3f32,4f32]
    /// ];
    /// let prt:String = array2_prt(&array2);
    /// println!("{}",prt);
    /// let expect:&str = 
    /// "┌                ┐
    /// │ -4.0 -3.0 -2.0 │
    /// │ -1.0 +0.0 +1.0 │
    /// │ +2.0 +3.0 +4.0 │
    /// └                ┘
    ///       [3,3]\n";
    /// assert_eq!(&prt,expect);
    /// ```
    pub fn array2_prt<T:Display+Copy>(ndarray_param:&Array2<T>) -> String {
        let mut prt_string:String = String::new();
        let shape = ndarray_param.shape(); // shape[0],shape[1]=row,column
        let spacing = 5*shape[1];
        prt_string.push_str(&format!("┌ {: <1$}┐\n","",spacing));
        for row in 0..shape[0] {
            prt_string.push_str("│ ");
            for val in ndarray_param.row(row) {
                prt_string.push_str(&format!("{:+.1} ",val));
                
            }
            prt_string.push_str("│\n");
        }
        prt_string.push_str(&format!("└ {:<1$}┘\n","",spacing));
        prt_string.push_str(&format!("{:<1$}","",(spacing/2)-1));
        prt_string.push_str(&format!("[{},{}]\n",shape[0],shape[1]));

        return prt_string;
    }
    /// Returns pretty string of `Array3<T>`.
    /// ```
    /// use cogent::utilities::array3_prt;
    /// use ndarray::{array,Array3};
    /// 
    /// let array3:Array3<f32> = array![
    ///     [[-1.0f32,-1.1f32],[-1.2f32,-1.3f32]],
    ///     [[0.0f32,0.1f32],[0.2f32,0.3f32]],
    ///     [[1.0f32,1.1f32],[1.2f32,1.3f32]],
    /// ];
    /// let prt:String = array3_prt(&array3);
    /// println!("{}",prt);
    /// let expect:&str = 
    /// "┌                                         ┐
    /// │ ┌           ┐┌           ┐┌           ┐ │
    /// │ │ -1.0 -1.1 ││ +0.0 +0.1 ││ +1.0 +1.1 │ │
    /// │ │ -1.2 -1.3 ││ +0.2 +0.3 ││ +1.2 +1.3 │ │
    /// │ └           ┘└           ┘└           ┘ │
    /// └                                         ┘
    ///                   [3,2,2]\n";
    /// assert_eq!(&prt,expect);
    /// ```
    pub fn array3_prt<T:Display+Copy>(ndarray_param:&Array3<T>) -> String {
        let mut prt_string:String = String::new();
        let shape = ndarray_param.shape(); // shape[0],shape[1],shape[2]=layer,row,column
        let outer_spacing = (5*shape[0]*shape[2]) + (3*shape[0]) + 2;
        prt_string.push_str(&format!("┌{: <1$}┐\n","",outer_spacing));

        let inner_spacing = 5 * shape[2];

        prt_string.push_str("│ ");
        for _ in 0..shape[0] {
            prt_string.push_str(&format!("┌ {: <1$}┐","",inner_spacing));
            
        }
        prt_string.push_str(" │\n");

        for i in 0..shape[1] {
            prt_string.push_str("│ ");
            for t in 0..shape[0] {
                prt_string.push_str("│ ");
                for p in 0..shape[2] {
                    let val = ndarray_param[[t,i,p]];
                    prt_string.push_str(&format!("{:+.1} ",val));
                }
                prt_string.push_str("│");
            }
            prt_string.push_str(" │\n");
        }
        prt_string.push_str("│ ");
        for _ in 0..shape[0] {
            prt_string.push_str(&format!("└ {: <1$}┘","",inner_spacing));
        }
        prt_string.push_str(" │\n");

        prt_string.push_str(&format!("└{:<1$}┘\n","",outer_spacing));
        prt_string.push_str(&format!("{:<1$}","",(outer_spacing / 2) - 2));
        prt_string.push_str(&format!("[{},{},{}]\n",shape[0],shape[1],shape[2]));

        return prt_string;
    }
    /// Counting sort.
    /// 
    /// Implemented for use with `NeuralNetwork::evaluate_outputs`.
    /// 
    /// Counting sort implemented since typically classification datasets have high `n` vs low `k`.
    /// 
    /// Counting sort: `O(n+k)`.
    pub fn counting_sort(data:&[(Vec<f32>,usize)],k:usize) -> Vec<(Vec<f32>,usize)> {
        let mut count:Vec<usize> = vec!(0;k);

        for i in 0..data.len() {
            count[data[i].1] += 1;
        }
        for i in 1..count.len() {
            count[i] += count[i-1];
        }

        let input_size = data[0].0.len();
        let mut sorted_data:Vec<(Vec<f32>,usize)> = vec!((vec!(0f32;input_size),0usize);data.len());

        for i in 0..data.len() {
            sorted_data[count[data[i].1]-1] = data[i].clone();
            count[data[i].1] -= 1;
        }

        return sorted_data;
    }
}