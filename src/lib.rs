/// Core functionality of training a neural network.
pub mod core {

    use rand::prelude::SliceRandom;
    
    use std::time::{Duration,Instant};

    // use ndarray::{Array2,Array1,ArrayD,Axis,ArrayView2};
    // use ndarray_rand::{RandomExt,rand_distr::Uniform};
    // use ndarray_einsum_beta::*;

    use arrayfire::{
        Array,randu,Dim4,matmul,MatProp,constant,
        sigmoid,max,rows,exp,maxof,sum,pow,
        transpose,imax,eq,sum_all,log,diag_extract,sum_by_key,div
    };

    use std::io::{Read,Write, stdout};
    use crossterm::{QueueableCommand, cursor};

    use serde::{Serialize,Deserialize};

    use std::fs::File;
    use std::fs;
    use std::path::Path;

    use std::f32;

    // Default percentage of training data to set as evaluation data (0.1=10%).
    const DEFAULT_EVALUTATION_DATA:f32 = 0.1f32;
    // Default percentage of size of training data to set batch size (0.01=1%).
    const DEFAULT_BATCH_SIZE:f32 = 0.01f32;
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
        fn run(&self,y:&Array<f32>,a:&Array<f32>) -> f32 {
            return match self {
                Self::Quadratic => { quadratic(y,a) },
                Self::Crossentropy => { cross_entropy(y,a) }
            };
            // Quadratic cost
            fn quadratic(y: &Array<f32>, a: &Array<f32>) -> f32 {
                sum_all(&pow(&(y - a),&2,false)).0 as f32 / (2f32*a.dims().get()[0] as f32)
            }
            // Cross entropy cost
            // TODO Need to double check this
            fn cross_entropy(y: &Array<f32>, a: &Array<f32>) -> f32 {
                //let part1 = a.mapv(f32::ln) * y;
                let part1 = log(a) * y;

                //af_print!("part1:",part1);

                let part2 = log(&(1f32 - a)) * (1f32 - y);

                //af_print!("part2:",part2);
                //af_print!("part1+part2:",&part1+&part2);

                let mut cost:f32 = sum_all(&(part1+part2)).0 as f32;

                // println!("cost:{}",cost);
                // println!("a.dims().get()[1]:{}",a.dims().get()[0]);

                cost /= -(a.dims().get()[1] as f32);
                return cost;
            }
        }
        /// Derivative wrt layer output (∂C/∂a)
        /// 
        /// y: Target out, a: Actual out
        fn derivative(&self,y:&Array<f32>,a:&Array<f32>) -> Array<f32> {
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
        fn run(&self,z:&Array<f32>) -> Array<f32> {
            return match self {
                Self::Sigmoid => sigmoid(z),
                Self::Softmax => Activation::softmax(z),
                Self::ReLU => Activation::relu(z),
            };
        }
        // Derivative wrt layer input (∂a/∂z)
        fn derivative(&self,z:&Array<f32>) -> Array<f32> {
            // What should we name the derivative functions?
            return match self {
                Self::Sigmoid => sigmoid_derivative(z),
                Self::Softmax => softmax_derivative(z),
                Self::ReLU => relu_derivative(z),
            };

            // Derivative of sigmoid
            // s' = s(1-s)
            fn sigmoid_derivative(z:&Array<f32>) -> Array<f32> {
                let s = sigmoid(z);
                return s.clone()*(1f32-s); // TODO Can we remove the clone here?
            }
            // Derivative of softmax
            // e^z * (sum of other inputs e^input) / (sum of all inputs e^input)^2 = e^z * (exp_sum-e^z) / (exp_sum)^2
            fn softmax_derivative(z:&Array<f32>) -> Array<f32> {
                let derivatives = exp(z);
                // Gets sum of each row
                let sums = sum(&derivatives,1);
                // Sets squared sum of each row
                let sqrd_sums = pow(&sums,&2,false); // is this better than `&sums*&sums`?

                let ones = constant(1f32,Dim4::new(&[1,z.dims().get()[1],1,1]));

                let sums_matrix = matmul(&sums,&ones,MatProp::NONE,MatProp::NONE);

                let sums_sub = sums_matrix - &derivatives;

                // TODO Is it more efficient to do this matrix multiplication before or after squaring?
                let sqrd_sums_matrix = matmul(&sqrd_sums,&ones,MatProp::NONE,MatProp::NONE);

                let derivatives = derivatives * sums_sub / sqrd_sums_matrix;

                return derivatives;
            }
            //Deritvative of ReLU
            // ReLU(z)/1 = if >0 1 else 0
            fn relu_derivative(z:&Array<f32>) -> Array<f32> {
                // return Activation::relu(z) / z;
                // Follow code replaces the above line.
                // Above line replaced becuase it is prone to floating point error leading to f32:NAN.
                // Similar performance.
                let gt = arrayfire::gt(z,&0f32,false);
                return arrayfire::and(z,&gt,false);
            
                
            }
        }
        // TODO Make this better
        // Applies softmax activation
        fn softmax(y: &Array<f32>) -> Array<f32> {
            let ones = constant(1f32,Dim4::new(&[1,y.dims().get()[1],1,1]));
            // Subtracts row max from all values.
            //  Allowing softmax to handle large values in y.
            // ------------------------------------------------
            // Gets max values in each row
            let max_axis_vals = arrayfire::max(&y,1);
            let max_axis_vals_matrix = arrayfire::matmul(&max_axis_vals,&ones,arrayfire::MatProp::NONE,arrayfire::MatProp::NONE);
            let max_reduced = y - max_axis_vals_matrix;

            // Applies softmax
            // ------------------------------------------------
            // Apply e^(x) to every value in matrix
            let exp_matrix = arrayfire::exp(&max_reduced);
            // Calculates sums of rows
            let row_sums = arrayfire::sum(&exp_matrix,1);
            let row_sums_matrix = arrayfire::matmul(&row_sums,&ones,arrayfire::MatProp::NONE,arrayfire::MatProp::NONE);
            // Divides each value by row sum
            let softmax = exp_matrix / row_sums_matrix;

            return softmax;
        }
        // TODO Is this the best way to do this?
        // Applies ReLU activation
        fn relu(y: &Array<f32>) -> Array<f32> {
            let zeros = constant(0f32,y.dims());
            return maxof(y,&zeros,false);
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
    #[derive(Serialize,Deserialize)]
    struct ImportExportNet {
        inputs: usize,
        biases: Vec<Vec<f32>>,
        connections: Vec<(Vec<f32>,(u64,u64))>,
        layers: Vec<Activation>,
        cost: Cost,
    }
    /// Neural network.
    pub struct NeuralNetwork {
        // Inputs to network
        inputs: usize,
        // Layer biases
        biases: Vec<Array<f32>>,
        // Connections between layers
        connections: Vec<Array<f32>>,
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

            //let ones = constant(1f32,Dim4::new(&[numb_of_examples,1,1,1]));
            let mut connections: Vec<Array<f32>> = Vec::with_capacity(layers.len());
            let mut biases: Vec<Array<f32>> = Vec::with_capacity(layers.len());
            // Sets connections between inputs and 1st hidden layer
            connections.push(
                ((randu::<f32>(Dim4::new(&[inputs as u64,layers[0].size as u64, 1, 1])) * 2f32) - 1f32)
                / (inputs as f32).sqrt()
            );
            // connections.push(
            //     constant(0.5f32,Dim4::new(&[inputs as u64,layers[0].size as u64, 1, 1]))
            //     / (inputs as f32).sqrt()
            // );
            // Sets biases for 1st hidden layer
            biases.push((randu::<f32>(Dim4::new(&[1, layers[0].size as u64, 1, 1])) * 2f32) - 1f32);
            //biases.push(constant(0.5f32,Dim4::new(&[1, layers[0].size as u64, 1, 1])));
            // Sets connections and biases for all subsequent layers
            for i in 1..layers.len() {
                connections.push(
                    ((randu::<f32>(Dim4::new(&[layers[i-1].size as u64,layers[i].size as u64, 1, 1])) * 2f32) - 1f32)
                    / (layers[i-1].size as f32).sqrt()
                );
                // connections.push(
                //     constant(0.5f32,Dim4::new(&[layers[i-1].size as u64,layers[i].size as u64, 1, 1]))
                //     / (layers[i-1].size as f32).sqrt()
                // );
                biases.push((randu::<f32>(Dim4::new(&[1, layers[i].size as u64, 1, 1])) * 2f32) - 1f32);
                //biases.push(constant(0.5f32,Dim4::new(&[1, layers[i].size as u64, 1, 1])));
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
        /// let input:Array<f32> = array![
        ///     [0f32,0f32],
        ///     [1f32,0f32],
        ///     [0f32,1f32],
        ///     [1f32,1f32]
        /// ];
        /// let output:Array2<f32> = net.run(&input);
        /// ```
        pub fn run(&self, inputs:&Array<f32>) -> Array<f32> {
            let rows = inputs.dims().get()[0];
            let ones = constant(1f32,Dim4::new(&[rows,1,1,1]));
            let mut activations:Array<f32> = inputs.clone();
            //af_print!("activations",activations);
            for i in 0..self.layers.len() {
                //af_print!("self.connections[i]",self.connections[i]);
                let weighted_inputs:Array<f32> = matmul(&activations,&self.connections[i],MatProp::NONE,MatProp::NONE);
                //af_print!("weighted_inputs",weighted_inputs);
                let bias_matrix:Array<f32> = matmul(&ones,&self.biases[i],MatProp::NONE,MatProp::NONE);
                //af_print!("bias_matrix",bias_matrix);
                let inputs = weighted_inputs + bias_matrix;
                //af_print!("inputs",inputs);
                activations = self.layers[i].run(&inputs);
                //af_print!("activations",activations);
            }
            //panic!("finished run");
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
        /// neural_network.train(&data)
        ///     .learning_rate(2f32)
        ///     .evaluation_data(EvaluationData::Actual(&data)) // Use testing data as evaluation data.
        ///     .lambda(0f32)
        /// .go();
        /// ```
        pub fn train(&mut self,training_data:&Vec<(Vec<f32>,usize)>) -> Trainer {
            println!("got here 1");
            // TODO Should we be helpful and do this check or not bother?
            // Checks all examples fit the neural network.
            let out_len = self.biases[self.biases.len()-1].dims().get()[1];
            for i in 0..training_data.len() {
                let example = &training_data[i];
                if example.0.len() != self.inputs {
                    panic!("Input size of example {} != size of input layer.",i);
                }
                else if example.1 > out_len as usize {
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
            //println!("got here 2");
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

            //println!("got here 2.03");
            let starting_evaluation = self.evaluate(evaluation_data); // Compute intial evaluation.

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

            // self.export(&format!("checkpoints/1"));
            // panic!("checking first");

            // Backpropgation loop
            // ------------------------------------------------
            loop {
                // TODO Double check:
                //  I beleive it is cheaper to shuffle as slice and set as matrix each iteration vs
                //  setting as matrix once then shuffling as matrix each iteration.
                training_data.shuffle(&mut rng);
                let training_data_matrix = self.matrixify(training_data);

                let batches = NeuralNetwork::batch_chunks(&training_data_matrix,batch_size); // Split dataset into batches.

                // TODO Reduce code duplication here.
                // Runs backpropagation on all batches:
                //  If `tracking` output backpropagation percentage progress.
                if tracking {
                    let mut percentage:f32 = 0f32;
                    stdout.queue(cursor::SavePosition).unwrap();
                    let backprop_start_instant = Instant::now();
                    let percent_change:f32 = 100f32 * batch_size as f32 / training_data_matrix.0.dims().get()[0] as f32;

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
                    //println!("got here 2.3");
                    for batch in batches {
                        let (new_connections,new_biases) = self.update_batch(&batch,learning_rate,lambda,training_data.len() as f32);
                        self.connections = new_connections;
                        self.biases = new_biases;
                    }
                }
                iterations_elapsed += 1;


                //println!("pre eval");
                // for (b_layer,w_layer) in izip!(self.biases.iter(),self.connections.iter()) {
                //     af_print!("current w_layer",w_layer);
                //     af_print!("current b_layer",b_layer);
                // }
                // panic!("checked update");

                let evaluation = self.evaluate(evaluation_data);

                

                //println!("post eval");

                //println!("{} {}",evaluation.0,evaluation.1);
                //panic!("completed evaluation");

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
                        self.biases[self.biases.len()-2].dims().get()[1] as usize,
                        self.layers[self.layers.len()-2]
                    ));
                }
            }

            // Compute and print final evaluation.
            // ------------------------------------------------
            let evaluation = self.evaluate(evaluation_data); 
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
        }
        // Runs batch through network to calculate weight and bias gradients.
        // Returns new weight and bias values.
        fn update_batch(&self, batch: &(Array<f32>,Array<f32>), eta: f32, lambda:f32, n:f32) -> (Vec<Array<f32>>,Vec<Array<f32>>) {

            //println!("got here 3");

            let (nabla_b,nabla_w):(Vec<Array<f32>>,Vec<Array<f32>>) = self.backpropagate(&batch);
            //println!("got here 4");

            // for (b_layer,w_layer) in izip!(nabla_b.iter(),nabla_w.iter()) {
            //     af_print!("w_layer",w_layer);
            //     af_print!("b_layer",b_layer);
            // }
            // panic!("got values in update batch");

            let batch_len = batch.0.dims().get()[0];
            // TODO Look into removing `.clone()`s here
            let return_connections:Vec<Array<f32>> = self.connections.iter().zip(nabla_w).map(
                | (w,nw) | (1f32-eta*(lambda/n))*w - ((eta / batch_len as f32)) * nw
            ).collect();

            // TODO Make this work
            // let return_biases:Vec<Array1<f32>> = self.biases.iter().zip(nabla_b.iter()).map(
            //     | (b,nb) | b - ((eta / batch.len() as f32)) * nb
            // ).collect();
            // This code replicates functionality of above code, for the time being.
            let mut return_biases:Vec<Array<f32>> = self.biases.clone();
            for i in 0..self.biases.len() {
                return_biases[i] = &self.biases[i] - &((eta / batch_len as f32)  * nabla_b[i].clone());
            }

            // for (b_layer,w_layer) in izip!(self.biases.iter(),self.connections.iter()) {
            //     af_print!("old w_layer",w_layer);
            //     af_print!("old b_layer",b_layer);
            // }
            // for (b_layer,w_layer) in izip!(return_biases.iter(),return_connections.iter()) {
            //     af_print!("new w_layer",w_layer);
            //     af_print!("new b_layer",b_layer);
            // }
            // panic!("finished update batch");

            return (return_connections,return_biases);
        }
        // Runs backpropgation on chunk of batch.
        // Returns weight and bias partial derivatives (errors).
        fn backpropagate(&self, example:&(Array<f32>,Array<f32>)) -> (Vec<Array<f32>>,Vec<Array<f32>>) {
            // Feeds forward
            // --------------
            let numb_of_examples = example.0.dims().get()[0]; // Number of examples (rows)
            let mut inputs:Vec<Array<f32>> = Vec::with_capacity(self.biases.len()); // Name more intuitively
            let mut activations:Vec<Array<f32>> = Vec::with_capacity(self.biases.len()+1);
            // TODO Is `.to_owned()` the best way to do this?
            let ones = constant(1f32,Dim4::new(&[numb_of_examples,1,1,1]));
            activations.push(example.0.to_owned());
            for i in 0..self.layers.len() {
                let weighted_inputs:Array<f32> = matmul(&activations[i],&self.connections[i],MatProp::NONE,MatProp::NONE);
                let bias_matrix:Array<f32> = matmul(&ones,&self.biases[i],MatProp::NONE,MatProp::NONE);
                inputs.push(weighted_inputs + bias_matrix);
                activations.push(self.layers[i].run(&inputs[i]));
            }

            //println!("got here 3.5");

            // Backpropagates
            // --------------
            let target = example.1.clone(); // TODO check we don't need '.clone' here
            let last_index = self.connections.len()-1; // = nabla_b.len()-1 = nabla_w.len()-1 = self.neurons.len()-2 = self.connections.len()-1
            
            // Gradients of biases and weights.
            let mut nabla_b:Vec<Array<f32>> = Vec::with_capacity(self.biases.len());
            // TODO find way to make this ArrayD an Array3, ArrayD willl always have 3d imensions, just can't figure out caste.
            let mut nabla_w:Vec<Array<f32>> = Vec::with_capacity(self.connections.len()); // this should really be 3d matrix instead of 'Vec<DMatrix<f32>>', its a bad workaround

            let last_layer = self.layers[self.layers.len()-1];
            // TODO Is `.to_owned()` a good solution here?

            // af_print!("target",target);
            // af_print!("activations[activations.len()-1]",activations[activations.len()-1]);
            // af_print!("cost derivative",self.cost.derivative(&target.to_owned(),&activations[activations.len()-1]));
            // af_print!("inputs[inputs.len()-1]",inputs[inputs.len()-1]);
            // af_print!("last_layer.derivative(&inputs[inputs.len()-1])",last_layer.derivative(&inputs[inputs.len()-1]));
            
            let mut error:Array<f32> = self.cost.derivative(&target,&activations[activations.len()-1]) * last_layer.derivative(&inputs[inputs.len()-1]);
            // af_print!("error",error);
            // panic!("calculated initial error");

            //println!("got here 3.6");

            // Sets gradients in output layer
            nabla_b.insert(0,error.clone());
            //let weight_errors = einsum("ai,aj->aji", &[&error, &activations[last_index]]).unwrap();
            let weight_errors = calc_weight_errors(&error,&activations[last_index]);
            nabla_w.insert(0,weight_errors);
            
            //println!("got here 3.7");

            // self.layers.len()-1 -> 1 (inclusive)
            // (self.layers.len()=self.biases.len()=self.connections.len())
            for i in (1..self.layers.len()).rev() {
                // Calculates error
                // println!("3.71");
                // af_print!("self.layers[i-1].derivative(&inputs[i-1])",self.layers[i-1].derivative(&inputs[i-1]));
                // af_print!("error",error);
                // af_print!("self.connections[i]",self.connections[i]);
                // println!("3.72");
                error = self.layers[i-1].derivative(&inputs[i-1]) *
                    matmul(&error,&self.connections[i],MatProp::NONE,MatProp::TRANS);
                //println!("3.73");
                //af_print!("error",error);

                // Sets gradients
                nabla_b.insert(0,error.clone());
                //let ein_sum = einsum("ai,aj->aji", &[&error, &activations[i-1]]).unwrap();
                let weight_errors = calc_weight_errors(&error,&activations[i-1]);
                nabla_w.insert(0,weight_errors);
            }
            //println!("got here 3.8");
            // Sum along columns (rows represent each example), push to `nabla_b_sum`.
            //af_print!("nabla_b[1]",nabla_b[1]);
            
            let nabla_b_sum:Vec<Array<f32>> = nabla_b.iter().map(|x| sum(x,0)).collect();
            //af_print!("nabla_b_sum[1]",nabla_b_sum[1]);
            //panic!("testing");
            
            // Sums through layers (each layer is a matrix representing each example), casts to Arry2 then pushes to `nabla_w_sum`.
            //af_print!("nabla_w[1]",nabla_w[1]);
            let nabla_w_sum:Vec<Array<f32>> = nabla_w.iter().map(|x| sum(x,2)).collect();
            
            // for (b_layer,w_layer) in izip!(nabla_b_sum.iter(),nabla_w_sum.iter()) {
            //     af_print!("w_layer",w_layer);
            //     af_print!("b_layer",b_layer);
            // }
            // panic!("finished backprop");

            // Returns gradients
            return (nabla_b_sum,nabla_w_sum);

            fn calc_weight_errors(errors:&Array<f32>,activations:&Array<f32>) -> arrayfire::Array<f32> {
                let rows:u64 = errors.dims().get()[0];
                
                let er_width:u64 = errors.dims().get()[1];
                let act_width:u64 = activations.dims().get()[1];
                let dims = arrayfire::Dim4::new(&[act_width,er_width,rows,1]);
            
                let temp:arrayfire::Array<f32> = arrayfire::Array::<f32>::new_empty(dims);
            
                for i in 0..rows {
                    let holder = arrayfire::matmul(
                        &arrayfire::row(activations,i),
                        &arrayfire::row(errors,i),
                        arrayfire::MatProp::TRANS,
                        arrayfire::MatProp::NONE
                    );
                    arrayfire::set_slice(&temp,&holder,i); // TODO Why does this work? I don't think this should work.
                }
                return temp;
            }
        }
        // TODO Performance of this should be improved.
        // Splits data into chunks of examples (rows).
        fn batch_chunks(data:&(Array<f32>,Array<f32>),batch_size:usize) -> Vec<(Array<f32>,Array<f32>)>{
            let examples = data.0.dims().get()[0];
            let batches = (examples as f32 / batch_size as f32).ceil() as usize;

            let mut chunks:Vec<(Array<f32>,Array<f32>)> = Vec::with_capacity(batches);
            for i in 0..batches-1 {
                let batch_indx:usize = i * batch_size;
                let in_batch:Array<f32> = rows(&data.0,batch_indx as u64,(batch_indx+batch_size-1) as u64);
                let out_batch:Array<f32> = rows(&data.1,batch_indx as u64,(batch_indx+batch_size-1) as u64);
                chunks.push((in_batch,out_batch));
            }
            //println!("nearly");
            let batch_indx:usize = (batches-1) * batch_size;
            let in_batch:Array<f32> = rows(&data.0,batch_indx as u64,examples-1);
            let out_batch:Array<f32> = rows(&data.1,batch_indx as u64,examples-1);
            chunks.push((in_batch,out_batch));
            //println!("done");
            let mut total = 0usize;
            for chunk in chunks.iter() {
                total+=chunk.0.dims().get()[0] as usize;
            }
            //println!("{} = {}",total,examples);
            // TODO MAY NEED TO CHECK THIS

            return chunks;
        }
        
        /// Inserts new layer before output layer in network.
        pub fn add_layer(&mut self, layer:Layer) {
            let prev_neurons:u64 = if let Some(indx) = self.biases.len().checked_sub(2) {
                self.biases[indx].dims().get()[1]
            } else { 
                self.inputs as u64
            };

            // Insert new layer
            self.layers.insert(self.layers.len()-1,layer.activation);
            self.connections.insert(
                self.connections.len()-1,
                    ((randu::<f32>(Dim4::new(&[prev_neurons,layer.size as u64, 1, 1])) * 2f32) - 1f32)
                    / (prev_neurons as f32).sqrt()
            );
            self.biases.insert(self.biases.len()-1,(randu::<f32>(Dim4::new(&[1, layer.size as u64, 1, 1])) * 2f32) - 1f32);

            // Update output layer
            let connection_indx = self.connections.len()-1;
            let bias_indx = self.biases.len()-1;
            let outputs = self.biases[bias_indx].dims().get()[1];

            self.connections[connection_indx] =
                    ((randu::<f32>(Dim4::new(&[layer.size as u64,outputs, 1, 1])) * 2f32) - 1f32)
                    / (prev_neurons as f32).sqrt();

            self.biases[bias_indx] = (randu::<f32>(Dim4::new(&[1, outputs, 1, 1])) * 2f32) - 1f32;
        }

        /// Returns tuple: (Average cost across batch, Number of examples correctly classified).
        pub fn evaluate(&self, test_data:&[(Vec<f32>,usize)]) -> (f32,u32) {
            let (input,target) = self.matrixify(test_data);
            let output = self.run(&input);

            let cost:f32 = self.cost.run(&target,&output);

            //panic!("cost computed");

            //println!("done cost");
            let output_classes = imax(&output,1).1;
            let target_classes_vec:Vec<u32> = test_data.iter().map(|x|x.1 as u32).collect();
            let target_classes = Array::<u32>::new(&target_classes_vec,Dim4::new(&[test_data.len() as u64,1,1,1]));
            let correct_classifications = eq(&output_classes,&target_classes,false);
            let correct_classifications_numb:u32 = sum_all(&correct_classifications).0 as u32;

            
            //println!("finished evaluation");
            return (cost / test_data.len() as f32, correct_classifications_numb);
        }
        // TODO Name this better
        /// Returns tuple of: (List of correctly classified percentage for each class, Confusion matrix of percentages).
        pub fn evaluate_outputs(&self, test_data:&mut [(Vec<f32>,usize)]) -> (Vec<f32>,Vec<Vec<f32>>) {
            test_data.sort_by(|(_,a),(_,b)| a.cmp(b));

            let (input,classes) = matrixify_inputs(test_data);
            let outputs = self.run(&input);

            let maxs:Array<f32> = max(&outputs,1i32);
            let class_vectors:Array<bool> = eq(&outputs,&maxs,true);
            let confusion_matrix:Array<f32> = sum_by_key(&classes,&class_vectors,0i32).1.cast::<f32>();

            let class_lengths:Array<f32> = sum(&confusion_matrix,1i32);

            let percent_confusion_matrix:Array<f32> = div(&confusion_matrix,&class_lengths,true);

            let dims = percent_confusion_matrix.dims();
            let mut flat_vec = vec!(f32::default();(dims.get()[0]*dims.get()[1]) as usize); // dims.get()[0] == dims.get()[1]
            transpose(&percent_confusion_matrix,false).host(&mut flat_vec);
            let matrix_vec:Vec<Vec<f32>> = flat_vec.chunks(dims.get()[0] as usize).map(|x| x.to_vec()).collect();

            let diag = diag_extract(&percent_confusion_matrix,0i32);
            let mut diag_vec:Vec<f32> = vec!(f32::default();diag.dims().get()[0] as usize);
            
            diag.host(&mut diag_vec);

            return (diag_vec,matrix_vec);
            
            fn matrixify_inputs(examples:&[(Vec<f32>,usize)]) -> (Array<f32>,Array<u32>) {
                let in_len = examples[0].0.len();
                let example_len = examples.len();

                // Flattens examples into `in_vec` and `out_vec`
                let in_vec:Vec<f32> = examples.iter().flat_map(|(input,_)| input.clone() ).collect();
                let out_vec:Vec<u32> = examples.iter().map(|(_,class)| class.clone() as u32 ).collect();

                let input:Array<f32> = transpose(&Array::<f32>::new(&in_vec,Dim4::new(&[in_len as u64,example_len as u64,1,1])),false);
                let output:Array<u32> = Array::<u32>::new(&out_vec,Dim4::new(&[example_len as u64,1,1,1]));

                return (input,output);
            }
        }

        
        // Converts `[(Vec<f32>,usize)]` to `(Array2<f32>,Array2<f32>)`.
        fn matrixify(&self,examples:&[(Vec<f32>,usize)]) -> (Array<f32>,Array<f32>) {

            let in_len = examples[0].0.len();
            let out_len = self.biases[self.biases.len()-1].dims().get()[1] as usize;
            let example_len = examples.len();

            // TODO Is there a better way to do either of these?
            // Flattens examples into `in_vec` and `out_vec`
            let in_vec:Vec<f32> = examples.iter().flat_map(|(input,_)| input.clone()).collect();
            let out_vec:Vec<f32> = examples.iter().flat_map(|(_,class)| { 
                let mut vec = vec!(0f32;out_len);
                vec[*class]=1f32;
                return vec;
            }).collect();

            let input:Array<f32> = transpose(&Array::<f32>::new(&in_vec,Dim4::new(&[in_len as u64,example_len as u64,1,1])),false);
            let output:Array<f32> = transpose(&Array::<f32>::new(&out_vec,Dim4::new(&[out_len as u64,example_len as u64,1,1])),false);

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
        /// Exports neural network to `path`.
        pub fn export(&self,path:&str) {
            let mut biases:Vec<Vec<f32>> = Vec::with_capacity(self.biases.len());
            for i in 0..self.biases.len() {
                println!("bias:{}->dims:{}",i,self.biases[i].dims());
                let len = self.biases[i].dims().get()[1] as usize;
                let vec:Vec<f32> = vec!(f32::default();len);
                biases.push(vec);
                self.biases[i].host(&mut biases[i]);

                for val in biases[i].iter() {
                    if val.is_nan() {
                        println!("ffs bois: nan bias");
                        panic!("nan found");
                    }
                }
            }

            let mut weights:Vec<(Vec<f32>,(u64,u64))> = Vec::with_capacity(self.connections.len());
            for i in 0..self.connections.len() {
                println!("connection:{}->dims:{}",i,self.connections[i].dims());
                let dims = self.connections[i].dims();
                let inner_dims = dims.get();
                let vec:Vec<f32> = vec!(f32::default();(inner_dims[0]*inner_dims[1]) as usize);
                weights.push((vec,(inner_dims[0],inner_dims[1])));
                self.connections[i].host(&mut weights[i].0);

                for val in weights[i].0.iter() {
                    if val.is_nan() {
                        println!("ffs bois: nan weight");
                        panic!("nan found");
                    }
                }
            }

            let estruct = ImportExportNet{
                inputs:self.inputs,
                biases:biases,
                connections:weights,
                layers:self.layers.clone(),
                cost:self.cost
            };

            let file = File::create(format!("{}.json",path));
            let serialized:String = serde_json::to_string(&estruct).unwrap();
            file.unwrap().write_all(serialized.as_bytes()).unwrap();

            //panic!("checking export");
        }
        /// Imports neural network from `path`.
        pub fn import(path:&str) -> NeuralNetwork {
            let file = File::open(format!("{}.json",path));
            //println!("Does it exist: {}",Path::new("checkpoints/10.json").exists());
            let mut string_contents:String = String::new();
            file.unwrap().read_to_string(&mut string_contents).unwrap();
            //println!("got here?");
            let istruct:ImportExportNet = serde_json::from_str(&string_contents).unwrap();
            //println!("got here? 2");

            let mut biases:Vec<arrayfire::Array<f32>> = Vec::with_capacity(istruct.biases.len());
            for i in 0..istruct.biases.len() {
                let len = istruct.biases[i].len() as u64;
                let array = arrayfire::Array::<f32>::new(&istruct.biases[i],Dim4::new(&[1,len,1,1]));
                biases.push(array);
            }

            let mut weights:Vec<arrayfire::Array<f32>> = Vec::with_capacity(istruct.connections.len());
            for i in 0..istruct.connections.len() {
                let dims = Dim4::new(&[(istruct.connections[i].1).0,(istruct.connections[i].1).1,1,1]);
                let array = arrayfire::Array::<f32>::new(&istruct.connections[i].0,dims);
                weights.push(array);
            }

            return NeuralNetwork{
                inputs:istruct.inputs,
                biases:biases,
                connections:weights,
                layers:istruct.layers,
                cost:istruct.cost
            };
        }
    }
}
/// Some uneccessary utility functions not fundementally linked to `core::NeuralNetwork`
pub mod utilities {
    extern crate ndarray;
    use ndarray::{Array1,Array2,Array3};
    use std::fmt::Display;
    // TODO: Generalise these pretty prints for an array of any number of dimensions.
    /// Returns pretty string of `Array1<T>`.
    /// ```
    /// use cogent::utilities::array1_prt;
    /// use ndarray::{array,Array1};
    /// 
    /// let array1:Array1<f32> = array![-1.666f32,-1f32,-0.5f32,0.5f32,1f32,1.666f32];
    /// let prt:String = array1_prt(&array1);
    /// println!("{}",prt);
    /// let expect:&str = 
    /// "┌                               ┐
    /// │ -1.7 -1.0 -0.5 +0.5 +1.0 +1.7 │
    /// └                               ┘
    ///                [6]\n";
    /// assert_eq!(&prt,expect);
    /// ```
    pub fn array1_prt<T:Display+Copy>(vector:&Array1<T>) -> String {
        let mut prt_string:String = String::new();
        let spacing = 5*vector.len();
        prt_string.push_str(&format!("┌ {: <1$}┐\n","",spacing));
        prt_string.push_str("│ ");
        for val in vector {
            prt_string.push_str(&format!("{:+.1} ",val));
        }
        prt_string.push_str("│\n");
        prt_string.push_str(&format!("└ {:<1$}┘\n","",spacing));
        prt_string.push_str(&format!("{:<1$}","",(spacing/2)));
        prt_string.push_str(&format!("[{}]\n",vector.len()));

        return prt_string;
    }
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
    pub fn array2_prt<T:Display+Copy>(matrix:&Array2<T>) -> String {
        let mut prt_string:String = String::new();
        let spacing = 5*matrix.ncols();
        prt_string.push_str(&format!("┌ {: <1$}┐\n","",spacing));
        for row in 0..matrix.nrows() {
            prt_string.push_str("│ ");
            for val in matrix.row(row) {
                prt_string.push_str(&format!("{:+.1} ",val));
                
            }
            prt_string.push_str("│\n");
        }
        prt_string.push_str(&format!("└ {:<1$}┘\n","",spacing));
        prt_string.push_str(&format!("{:<1$}","",(spacing/2)-1));
        prt_string.push_str(&format!("[{},{}]\n",matrix.nrows(),matrix.ncols()));

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
    pub fn array3_prt<T:Display+Copy>(tensor:&Array3<T>) -> String {
        let mut prt_string:String = String::new();
        let shape = tensor.shape(); // shape[0],shape[1],shape[2]=layer,row,column
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
                    let val = tensor[[t,i,p]];
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
}