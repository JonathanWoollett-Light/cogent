#[allow(dead_code)]
mod core {
    use rand::prelude::SliceRandom;
    use std::time::{Duration,Instant};
    use itertools::izip;

    extern crate scoped_threadpool;
    use scoped_threadpool::Pool;

    extern crate ndarray;
    use ndarray::{Array2,Array1,Array3,ArrayD,Axis};
    use ndarray_rand::{RandomExt,rand_distr::Uniform};

    extern crate ndarray_einsum_beta;
    use ndarray_einsum_beta::*;

    use std::io::{Write, stdout};
    use crossterm::{QueueableCommand, cursor};

    use serde::{Serialize,Deserialize};
    use std::fmt;

    use std::fs::create_dir;
    use std::fs::File;
    use std::io::Read;
    use std::fs::read_dir;
    //Setting number of threads to use
    const THREAD_COUNT:usize = 12usize;

    use std::f32;

    //Defining euler's constant
    const E:f32 = 2.718281f32;

    const DEFAULT_EVALUTATION_DATA:f32 = 0.1f32; //`(x * examples.len() as f32) as usize` of `testing_data` is split_off into `evaluation_data`
    const DEFAULT_BATCH_SIZE:f32 = 0.002f32; //(x * examples.len() as f32).ceil() as usize. batch_size = x% of training data
    const DEFAULT_LEARNING_RATE:f32 = 0.1f32;
    const DEFAULT_LAMBDA:f32 = 0.1f32; // lambda = (x * examples.len() as f32). lambda = x% of training data. lambda = regularization parameter
    const DEFAULT_EARLY_STOPPING:f32 = 0.00005f32; // Duration::new(examples[0].0.len()*examples.len()*x,0). (MNIST is approx 4 mins)
    const DEFAULT_LEARNING_RATE_DECAY:f32 = 0.5f32;
    const DEFAULT_LEARNING_RATE_INTERVAL:u32 = 500u32; // Iteration(x * examples[0].0.len() / examples.len()). (MNIST is approx 6 iterations)

    pub enum EvaluationData {
        Scaler(usize),
        Percent(f32),
        Actual(Vec<(Vec<f32>,Vec<f32>)>)
    }
    #[derive(Clone,Copy)]
    pub enum MeasuredCondition {
        Iteration(u32),
        Duration(Duration)
    }
    #[derive(Clone,Copy)]
    pub enum HaltCondition {
        Iteration(u32),
        Duration(Duration),
        Accuracy(f32)
    }
    
    pub struct Trainer<'a> {
        training_data: Vec<(Vec<f32>,Vec<f32>)>,
        // TODO Since we never alter `evaluation_data` look into changing this into a reference
        evaluation_data: Vec<(Vec<f32>,Vec<f32>)>, 
        // Will halt after at a certain iteration, accuracy or duration.
        halt_condition: Option<HaltCondition>,
        // Can log after a certain number of iterations, a certain duration, or not at all.
        log_interval: Option<MeasuredCondition>,
        batch_size: usize, // TODO Maybe change `batch_size` to allow it to be set by a user as a % of their data
        learning_rate: f32, // Reffered to as `ETA` in `NeuralNetwork`.
        lambda: f32, // Regularization parameter
        // Can stop after no cost improvement over a certain number of iterations, a certain duration, or not at all.
        early_stopping_condition: MeasuredCondition,
        learning_rate_decay: f32,
        learning_rate_interval: MeasuredCondition,
        checkpoint_interval: Option<MeasuredCondition>,
        tracking: bool,
        neural_network: &'a mut NeuralNetwork
    }
    impl<'a> Trainer<'a> {
        pub fn evaluation_data(&mut self, evaluation_data:EvaluationData) -> &mut Trainer<'a> {
            self.evaluation_data = match evaluation_data {
                EvaluationData::Scaler(scaler) => { self.training_data.split_off(self.training_data.len() - scaler) }
                EvaluationData::Percent(percent) => { self.training_data.split_off(self.training_data.len() - (self.training_data.len() as f32 * percent) as usize) }
                EvaluationData::Actual(actual) => { actual }
            };
            return self;
        }
        pub fn halt_condition(&mut self, halt_condition:HaltCondition) -> &mut Trainer<'a> {
            self.halt_condition = Some(halt_condition);
            return self;
        }
        pub fn log_interval(&mut self, log_interval:MeasuredCondition) -> &mut Trainer<'a> {
            self.log_interval = Some(log_interval);
            return self;
        }
        pub fn batch_size(&mut self, batch_size:usize) -> &mut Trainer<'a> {
            self.batch_size = batch_size;
            return self;
        }
        pub fn learning_rate(&mut self, learning_rate:f32) -> &mut Trainer<'a> {
            self.learning_rate = learning_rate;
            return self;
        }
        pub fn lambda(&mut self, lambda:f32) -> &mut Trainer<'a> {
            self.lambda = lambda;
            return self;
        }
        pub fn early_stopping_condition(&mut self, early_stopping_condition:MeasuredCondition) -> &mut Trainer<'a> {
            self.early_stopping_condition = early_stopping_condition;
            return self;
        }
        pub fn learning_rate_decay(&mut self, learning_rate_decay:f32) -> &mut Trainer<'a> {
            self.learning_rate_decay = learning_rate_decay;
            return self;
        }
        pub fn learning_rate_interval(&mut self, learning_rate_interval:MeasuredCondition) -> &mut Trainer<'a> {
            self.learning_rate_interval = learning_rate_interval;
            return self;
        }
        pub fn checkpoint_interval(&mut self, checkpoint_interval:MeasuredCondition) -> &mut Trainer<'a> {
            self.checkpoint_interval = Some(checkpoint_interval);
            return self;
        }
        pub fn tracking(&mut self) -> &mut Trainer<'a> {
            self.tracking = true;
            return self;
        }
        // Begins training.
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
                self.learning_rate_decay,
                self.learning_rate_interval,
                self.checkpoint_interval,
                self.tracking
            );
        }
    }

    // Enum for each neuron activation type
    // For now we only have 1 activation type
    #[derive(Clone,Copy,Serialize,Deserialize)]
    pub enum Activation {
        Sigmoid,Softmax // TODO Fix softmax, it seems to suffer the exploding gradient problem.
    }
    impl Activation {
        fn run(&self,y:&Array2<f32>) -> Array2<f32> {
            return match self {
                Self::Sigmoid => y.mapv(|x| Activation::sigmoid(x)),
                Self::Softmax => Activation::softmax(y),
            };
        }
        fn derivative(&self,y:&Array2<f32>) -> Array2<f32> {
            return match self {
                Self::Sigmoid => y.mapv(|x| -> f32 { sigmoid(x) }),
                Self::Softmax => softmax(y),
            };
            // TODO Should we names things `x` like `sigmoid` or `x_derivative` like `sigmoid_derivative`?
            // TODO Why is this called `sigmoid prime` in some cases?
            // Derivative of sigmoid
            fn sigmoid(y:f32) -> f32 {
                return Activation::sigmoid(y) * (1f32 - Activation::sigmoid(y));
            }
            // Derivative of softmax
            fn softmax(y:&Array2<f32>) -> Array2<f32> {
                let mut softmax = Activation::softmax(y);
                softmax.mapv_inplace(|x|x*(1f32-x));
                return softmax;
            }
        }
        // Applies sigmoid function
        fn sigmoid(y: f32) -> f32 {
            1f32 / (1f32 + (-y).exp())
        }
        // Applies softmax activation
        // TODO Make this better
        fn softmax(y: &Array2<f32>) -> Array2<f32> {
            let mut exp_matrix = y.clone();

            // Subtracts row max from all values.
            //  Allowing softmax to handle large values in y.
            // ------------------------------------------------

            // Get max value in each row (each example)
            let mut max_axis_vals = exp_matrix.fold_axis(Axis(1),0f32,|acc,x| (if acc > x { *acc } else { *x }));
            // Subtracts row max from every value in matrix // TODO Improve this
            for i in 0..exp_matrix.shape()[0] {
                exp_matrix.row_mut(i).mapv_inplace(|x| x-max_axis_vals[i]); 
            }

            // Applies softmax
            // ------------------------------------------------

            // Apply e^(x) to every value in matrix
            exp_matrix.mapv_inplace(|x|x.exp());
            // Calculates sums of rows
            let sum = exp_matrix.sum_axis(Axis(1));
            // Divide every value in matrix by sum of its row
            for (sum,mut row) in izip!(sum.iter(),exp_matrix.axis_iter_mut(Axis(1))) {
                row.mapv_inplace(|x| x / sum);
            }
            return exp_matrix;
        }
    }
    #[derive(Serialize,Deserialize)]
    pub enum Cost {
        Quadratic,CrossEntropy
    }
    impl Cost {
        fn calc(&self,outputs: &Array2<f32>, targets: &Array2<f32>) -> Array1<f32>{
            return match self {
                Self::Quadratic => quadratic(outputs,targets),
                Self::CrossEntropy => quadratic(outputs,targets),//nTODO Need to add
            };
            fn quadratic(outputs: &Array2<f32>, targets: &Array2<f32>) -> Array1<f32> {
                (targets - outputs).mapv(|x| 0.5f32 * x.powi(2)).sum_axis(Axis(1))
            }
            // TODO This isn't done yet.
            fn cross_entropy(outputs: &Array2<f32>, targets: &Array2<f32>) -> Array2<f32> {
                outputs - targets
            }
        }
        fn derivative(&self,outputs: &Array2<f32>, targets: &Array2<f32>) -> Array2<f32>{
            return match self {
                Self::Quadratic => quadratic(outputs,targets),
                Self::CrossEntropy => quadratic(outputs,targets),
            };
            fn quadratic(outputs: &Array2<f32>, targets: &Array2<f32>) -> Array2<f32> {
                outputs - targets
            }
            // TODO Is this right?
            fn cross_entropy(outputs: &Array2<f32>, targets: &Array2<f32>) -> Array2<f32> {
                outputs.mapv(|x|x.ln()) * targets
            }
        }
    }
    // Struct for user specifiying layers in network
    pub struct Layer {
        size: usize,
        activation: Activation,
    }
    impl Layer {
        pub fn new(size:usize,activation:Activation) -> Layer {
            Layer {size,activation}
        }
    }

    // A stochastic/incremental descent neural network.
    // Implementing cross-entropy cost function and L2 regularization.
    #[derive(Serialize,Deserialize)]
    pub struct NeuralNetwork {
        inputs: usize, //TODO Remove this, add simply integer val for number of input neurons.
        biases: Vec<Array2<f32>>,
        connections: Vec<Array2<f32>>,
        layers: Vec<Activation>,
        cost: Cost
    }
    impl NeuralNetwork {
        // Constructs network of given layers
        // Returns constructed network.
        pub fn new(inputs:usize,layers: &[Layer],cost:Cost) -> NeuralNetwork {
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
            NeuralNetwork{ inputs, biases, connections, layers, cost}
        }
        // Runs batch of examples through network.
        // Returns outputs from batch of examples.
        pub fn run(&self, inputs:&Array2<f32>) -> Array2<f32> {
            let mut activations:Array2<f32> = inputs.clone(); // don't remove this clone though
            for i in 0..self.layers.len() {
                let weighted_inputs:Array2<f32> = activations.dot(&self.connections[i].t());
                let bias_matrix:Array2<f32> = Array2::ones((inputs.shape()[0],1)).dot(&self.biases[i]);
                let inputs = weighted_inputs + bias_matrix;
                activations = self.layers[i].run(&inputs);
            }
            return activations;
        }
        // Begin setting hyperparameters for training.
        // Returns internal `Trainer` struct used to specify hyperparameters
        pub fn train(&mut self,training_data:&Vec<(Vec<f32>,Vec<f32>)>) -> Trainer {
            let mut rng = rand::thread_rng();
            let mut temp_training_data = training_data.clone();
            temp_training_data.shuffle(&mut rng);
            let temp_evaluation_data = temp_training_data.split_off(training_data.len() - (training_data.len() as f32 * DEFAULT_EVALUTATION_DATA) as usize);

            let multiplier:f32 = training_data[0].0.len() as f32 * training_data.len() as f32;
            let early_stopping_condition = Duration::new((multiplier * DEFAULT_EARLY_STOPPING) as u64,0);
            let learning_rate_interval:u32 = (DEFAULT_LEARNING_RATE_INTERVAL as f32 * training_data[0].0.len() as f32 / training_data.len() as f32) as u32;
            
            
            return Trainer {
                training_data: temp_training_data,
                evaluation_data: temp_evaluation_data,
                halt_condition: None,
                log_interval: None,
                batch_size: (DEFAULT_BATCH_SIZE * training_data.len() as f32).ceil() as usize,
                learning_rate: DEFAULT_LEARNING_RATE,
                lambda: DEFAULT_LAMBDA,
                early_stopping_condition: MeasuredCondition::Duration(early_stopping_condition),
                learning_rate_decay: DEFAULT_LEARNING_RATE_DECAY,
                learning_rate_interval: MeasuredCondition::Iteration(learning_rate_interval),
                checkpoint_interval: None,
                tracking: false,
                neural_network:self
            };
        }

        // Rungs training.
        fn train_details(&mut self,
            training_data: &mut [(Vec<f32>,Vec<f32>)], // TODO Look into `&[(Vec<f32>,Vec<f32>)]` vs `&Vec<(Vec<f32>,Vec<f32>)>`
            evaluation_data: &[(Vec<f32>,Vec<f32>)],
            halt_condition: Option<HaltCondition>,
            log_interval: Option<MeasuredCondition>,
            batch_size: usize,
            mut learning_rate: f32,
            lambda: f32,
            early_stopping_n: MeasuredCondition,
            learning_rate_decay: f32,
            learning_rate_interval: MeasuredCondition,
            checkpoint_interval: Option<MeasuredCondition>,
            tracking:bool
        ) -> (){

            let mut stdout = stdout();

            let mut rng = rand::thread_rng();
            let start_instant = Instant::now();
            let mut iterations_elapsed = 0u32;
            
            let mut best_accuracy_iteration = 0u32;// Iteration of best accuracy
            let mut best_accuracy_instant = Instant::now();// Instant of best accuracy

            let mut best_accuracy = 0u32;

            let starting_evaluation = self.evaluate(evaluation_data);

            // Don't think we need this.
            // if let Some(_) = checkpoint_interval {
            //     create_dir("../checkpoints").unwrap();
            // }

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

            loop {
                let batch_start_instant = Instant::now();
                training_data.shuffle(&mut rng);
                let batches:Vec<_> = training_data.chunks(batch_size).collect();
                
                if tracking {
                    let mut percentage:f32 = 0f32;
                    stdout.queue(cursor::SavePosition).unwrap();
                    let backprop_start_instant = Instant::now();
                    let percent_change:f32 = 100f32 * batch_size as f32 / training_data.len() as f32;
                    for batch in batches {
                        stdout.write(format!("Backpropagating: {:.2}%",percentage).as_bytes()).unwrap();
                        percentage += percent_change;
                        stdout.queue(cursor::RestorePosition).unwrap();
                        stdout.flush().unwrap();

                        let (new_connections,new_biases) = self.update_batch(batch,learning_rate,lambda,training_data.len() as f32);
                        self.connections = new_connections;
                        self.biases = new_biases;
                    }
                    stdout.write(format!("Backpropagating: {}\n",NeuralNetwork::time(backprop_start_instant)).as_bytes()).unwrap();
                }
                else {
                    for batch in batches {
                        let (new_connections,new_biases) = self.update_batch(batch,learning_rate,lambda,training_data.len() as f32);
                        self.connections = new_connections;
                        self.biases = new_biases;
                    }
                }

                iterations_elapsed += 1;

                let evaluation = self.evaluate(evaluation_data);
                if evaluation.1 > best_accuracy { 
                    best_accuracy = evaluation.1;
                    best_accuracy_iteration = iterations_elapsed;
                    best_accuracy_instant = Instant::now();
                }
                
                match checkpoint_interval {// TODO Reduce code duplication here
                    Some(MeasuredCondition::Iteration(iteration_interval)) => if iterations_elapsed % iteration_interval == 0 {
                        self.export(&format!("{}",iterations_elapsed));
                    },
                    Some(MeasuredCondition::Duration(duration_interval)) => if last_checkpointed_instant.elapsed() >= duration_interval {
                        self.export(&format!("{}",NeuralNetwork::time(start_instant)));
                        last_checkpointed_instant = Instant::now();
                    },
                    _ => {},
                }

                match log_interval {// TODO Reduce code duplication here
                    Some(MeasuredCondition::Iteration(iteration_interval)) => if iterations_elapsed % iteration_interval == 0 {
                        log_fn(&mut stdout,self,iterations_elapsed,start_instant,learning_rate,evaluation,evaluation_data.len()
                        );
                    },
                    Some(MeasuredCondition::Duration(duration_interval)) => if last_logged_instant.elapsed() >= duration_interval {
                        log_fn(&mut stdout,self,iterations_elapsed,start_instant,learning_rate,evaluation,evaluation_data.len()
                        );
                        last_logged_instant = Instant::now();
                    },
                    _ => {},
                }

                match halt_condition {
                    Some(HaltCondition::Iteration(iteration)) => if iterations_elapsed == iteration { break; },
                    Some(HaltCondition::Duration(duration)) => if start_instant.elapsed() > duration { break; },
                    Some(HaltCondition::Accuracy(accuracy)) => if evaluation.1 as f32 / evaluation_data.len() as f32 > accuracy { break; },
                    _ => {},
                }

                match early_stopping_n {
                    MeasuredCondition::Iteration(stopping_iteration) =>  if iterations_elapsed - best_accuracy_iteration == stopping_iteration { println!("---------------\nEarly stoppage!\n---------------"); break; },
                    MeasuredCondition::Duration(stopping_duration) => if best_accuracy_instant.elapsed() >= stopping_duration { println!("---------------\nEarly stoppage!\n---------------"); break; }
                }

                match learning_rate_interval {
                    MeasuredCondition::Iteration(interval_iteration) =>  if iterations_elapsed - best_accuracy_iteration == interval_iteration { learning_rate *= learning_rate_decay },
                    MeasuredCondition::Duration(interval_duration) => if best_accuracy_instant.elapsed() >= interval_duration { learning_rate *= learning_rate_decay }
                }
            }
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

            fn get_batches(examples:&[(Vec<f32>,Vec<f32>)], batch_size: usize) -> Vec<&[(Vec<f32>,Vec<f32>)]> {
                let mut batches = Vec::new(); // TODO Look into if 'Vec::with_capacity(ceil(examples.len() / batch_size))' is more efficient
                
                let mut lower_bound = 0usize;
                let mut upper_bound = batch_size;

                while upper_bound < examples.len() {
                    batches.push(&examples[lower_bound..upper_bound]);
                    lower_bound = upper_bound;
                    // TODO Improve this to remove last unnecessary addition to 'upper_bound'
                    upper_bound += batch_size;

                    //println!("{}",batches.len() as f32 * batch_size as f32 / examples.len() as f32);
                }
                // Accounts for last batch possibly being under 'batch_size'
                batches.push(&examples[lower_bound..examples.len()]);

                return batches;
            }
            // TODO Do something better than thi
            fn log_fn(
                stdout:&mut std::io::Stdout,
                net:&NeuralNetwork,
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
        // Returns new weights and biases values.
        fn update_batch(&self, batch: &[(Vec<f32>, Vec<f32>)], eta: f32, lambda:f32, n:f32) -> (Vec<Array2<f32>>,Vec<Array2<f32>>) {
            
            // TODO Look into a better way to setup 'bias_nabla' and 'weight_nabla'
            // Copies structure of self.neurons and self.connections with values of 0f32
            let nabla_b_zeros:Vec<Array2<f32>> = self.biases.clone().iter().map(|x| x.map(|_| 0f32) ).collect();
            let nabla_w_zeros:Vec<Array2<f32>> = self.connections.clone().iter().map(|x| x.map(|_| 0f32) ).collect();


            let chunks_length:usize = if batch.len() < THREAD_COUNT { batch.len() } else { batch.len() / THREAD_COUNT };
            let chunks:Vec<_> = batch.chunks(chunks_length).collect(); // TODO Specify type further
            let mut pool = Pool::new(chunks.len() as u32);

            let mut out_nabla_b:Vec<Vec<Array2<f32>>> = vec!(nabla_b_zeros.clone();chunks.len());
            let mut out_nabla_w:Vec<Vec<Array2<f32>>> = vec!(nabla_w_zeros.clone();chunks.len());

            pool.scoped(|scope| {
                for (chunk,nabla_w,nabla_b) in izip!(chunks,&mut out_nabla_w,&mut out_nabla_b) {
                    scope.execute(move || {
                        let batch_tuple_matrix = NeuralNetwork::matrixify(chunk);
                        let (mut delta_nabla_b,mut delta_nabla_w):(Vec<Array2<f32>>,Vec<Array2<f32>>) = self.backpropagate(batch_tuple_matrix);
                        
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
                | (w,nw) | (1f32-eta*(lambda/n))*w - ((eta / batch.len() as f32)) * nw
            ).collect();

            // TODO Make this work
            // let return_biases:Vec<Array1<f32>> = self.biases.iter().zip(nabla_b.iter()).map(
            //     | (b,nb) | b - ((eta / batch.len() as f32)) * nb
            // ).collect();
            // This code replicates functionality of above code, for the time being.
            let mut return_biases:Vec<Array2<f32>> = self.biases.clone();
            for i in 0..self.biases.len() {
                // TODO Improve this
                return_biases[i] = &self.biases[i] - &((eta / batch.len() as f32)  * nabla_b[i].clone());
            }

            return (return_connections,return_biases);
        }
        // Runs backpropgation on chunk of batch.
        // Returns weight and bias partial derivatives (errors).
        fn backpropagate(&self, example:(Array2<f32>,Array2<f32>)) -> (Vec<Array2<f32>>,Vec<Array2<f32>>) {

            // Feeds forward
            // --------------

            let number_of_examples = example.0.shape()[0]; // Number of examples (rows)
            let mut inputs:Vec<Array2<f32>> = Vec::with_capacity(self.biases.len()); // Name more intuitively
            let mut activations:Vec<Array2<f32>> = Vec::with_capacity(self.biases.len()+1);
            activations.push(example.0);
            //println!("activations[{}]:{}",0,&activations[0].clone());
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
            // TODO:
            //  A few of these output error calculations can be greatly simplified (e.g. cross entropy sigmoid = outputs-target).
            //  Find a way to implement these simplifications.
            let mut error:Array2<f32> = self.cost.derivative(&activations[last_index+1],&target) * last_layer.derivative(&inputs[last_index]);

            // Sets gradients in output layer
            nabla_b.insert(0,error.clone());
            let weight_errors = einsum("ai,aj->aji", &[&error, &activations[last_index]]).unwrap();
            nabla_w.insert(0,weight_errors);
            

            // self.layers.len()-1 -> 1 (inclusive)
            // (self.layers.len()=self.biases.len()=self.connections.len())
            for i in (1..self.layers.len()).rev() {
                // Calculates error
                error = self.layers[i-1].derivative(&inputs[i-1]) *
                    error.dot(&self.connections[i]);

                // Sets gradients
                nabla_b.insert(0,error.clone());
                let ein_sum = einsum("ai,aj->aji", &[&error, &activations[i-1]]).unwrap();
                nabla_w.insert(0,ein_sum);
            }
            //println!("done that");

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
            fn quadratic_cost_derivative(outputs: &Array2<f32>, targets: &Array2<f32>) -> Array2<f32> {
                outputs - targets
            }
        }
        
        // Returns tuple (average cost, number of examples correctly classified)
        pub fn evaluate(&self, test_data:&[(Vec<f32>,Vec<f32>)]) -> (f32,u32) {

            let eval_start_instant = Instant::now();

            let chunks_length:usize = if test_data.len() < THREAD_COUNT { test_data.len() } else { test_data.len() / THREAD_COUNT };
            let chunks:Vec<_> = test_data.chunks(chunks_length).collect(); // Specify type further
            let mut pool = Pool::new(chunks.len() as u32);

            let mut cost_vec = vec!(0f32;chunks.len());
            let mut classified_vec = vec!(0u32;chunks.len());
            pool.scoped(|scope| {
                for (chunk,cost,classified) in izip!(chunks,&mut cost_vec,&mut classified_vec) {
                    scope.execute(move || {
                        let batch_tuple_matrix = NeuralNetwork::matrixify(&chunk);
                        let out = self.run(&batch_tuple_matrix.0);
                        let target = batch_tuple_matrix.1;
                        *cost = self.cost.calc(&out,&target).sum();

                        let mx_out_indxs = get_max_indexs(&out);
                        let mx_tar_indxs = get_max_indexs(&target);
                        *classified = izip!(mx_out_indxs.iter(),mx_tar_indxs.iter()).fold(0u32,|acc,(a,b)| {if a==b { acc+1u32 } else { acc }});
                    });
                }
            });
            // Sum costs and correctly classified
            let cost:f32 = cost_vec.iter().sum();
            let classified:u32 = classified_vec.iter().sum();


            return (cost / test_data.len() as f32, classified);

            
            // Returns vector of linear costs of each example.
            fn linear_cost(outputs: &Array2<f32>, targets: &Array2<f32>) -> Array1<f32> {
                (targets - outputs).mapv(f32::abs).sum_axis(Axis(1))
            }
            fn quadratic_cost(outputs: &Array2<f32>, targets: &Array2<f32>) -> Array1<f32> {
                (targets - outputs).mapv(|x| 0.5f32 * x.powi(2)).sum_axis(Axis(1))
            }
            // TODO
            // fn cross_entropy_cost(outputs: &Array2<f32>, targets: &Array2<f32>) -> Array1<f32> {
            //     ...
            // }
            // TODO Use iterator maps/folds etc.
            fn get_max_indexs(matrix:&Array2<f32>) -> Array1<usize> {
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
        // TODO Lot of stuff could be done to improve this function
        // Requires ordered test_data;
        // Returns tuple of:
        //  Array1<f32> of percentages of examples in each class correctly classified.
        //  Array2<f32> of confusion matrix with percentages.
        pub fn evaluate_outputs(&self, test_data:&[(Vec<f32>,Vec<f32>)]) -> (Array1<f32>,Array2<f32>) {
            let alphabet_size = test_data[0].1.len();
            let chunks = symbol_chunks(test_data,alphabet_size);
            let mut pool = Pool::new(chunks.len() as u32);

            let mut classifications:Vec<Array1<f32>> = vec!(Array1::zeros(alphabet_size);alphabet_size);
            pool.scoped(|scope| {
                for (chunk,classification) in izip!(chunks,&mut classifications) {
                    scope.execute(move || {
                        let results = self.run(&chunk.0);
                        let classes:Array2<u32> = set_nonmax_zero(&results);
                        let class_sums:Array1<u32> = classes.sum_axis(Axis(0));
                        let sum:f32 = class_sums.sum() as f32;
                        *classification = class_sums.mapv(|val| (val as f32 / sum));
                    });
                }
            });
            let matrix:Array2<f32> = cast_array1s_to_array2(classifications,alphabet_size);
            // TODO Getting `diagonal` could porbably be improved, look into that.
            let diagonal:Array1<f32> = matrix.clone().into_diag();
            return (diagonal,matrix);

            // TODO Lots needs to be done to improve this.
            // Returns Vec<(Array2<f32>,Array2<f32>)> with each tuple representing all examples of a character.
            fn symbol_chunks(test_data:&[(Vec<f32>,Vec<f32>)],alphabet_size:usize) -> Vec<(Array2<f32>,Array2<f32>)> {
                let mut chunks:Vec<(Array2<f32>,Array2<f32>)> = Vec::with_capacity(alphabet_size);
                let mut slice = (0usize,0usize); // (lower bound,upper bound)
                loop {
                    slice.1+=1;
                    while test_data[slice.1].1 == test_data[slice.1+1].1 {
                        slice.1+=1;
                        if slice.1+1 == test_data.len() {
                            slice.1 += 1;
                            break; 
                        }
                    } 
                    let chunk_holder = NeuralNetwork::matrixify(&test_data[slice.0..slice.1]);
                    chunks.push(chunk_holder);
                    if chunks.len() == alphabet_size { break };
                    slice.0 = slice.1;
                }
                return chunks;
            }
            // Sets all non-max values in row to 0 and max to 1 for each row in matrix.
            fn set_nonmax_zero(matrix:&Array2<f32>) -> Array2<u32> {
                let mut max_indx = 0usize;
                let shape = matrix.shape();
                let mut zero_matrix = Array2::zeros((shape[0],shape[1]));

                for i in 0..shape[0] {
                    for t in 1..shape[1] {
                        if matrix[[i,t]] > matrix[[i,max_indx]] {
                            max_indx=t;
                        }
                    }
                    zero_matrix[[i,max_indx]] = 1u32;
                    max_indx = 0usize;
                }
                return zero_matrix;
            }
            // TODO Need better way to caste than this
            // Returns Array2<_> casted from Vec<Array1<_>>
            fn cast_array1s_to_array2(vec:Vec<Array1<f32>>,alphabet_size:usize) -> Array2<f32> {
                let mut arr2 = Array2::default((alphabet_size,alphabet_size));
                for i in 0..vec.len() {
                    for t in 0..vec[i].shape()[0] {
                        arr2[[i,t]] = vec[i][[t]];
                    }
                }
                return arr2; 
            }
        }

        // Converts `[(Vec<f32>,Vec<f32>)]` to `(Array2<f32>,Array2<f32>)`.
        pub fn matrixify(examples:&[(Vec<f32>,Vec<f32>)]) -> (Array2<f32>,Array2<f32>) {
            //println!("began here");
            let input_len = examples[0].0.len();
            let output_len = examples[0].1.len();
            let example_len = examples.len();

            let mut input_vec:Vec<f32> = Vec::with_capacity(example_len * input_len);
            let mut output_vec:Vec<f32> = Vec::with_capacity(example_len * output_len);
            for example in examples {
                // TODO Remove the `.clone()`s here,
                input_vec.append(&mut example.0.clone());
                output_vec.append(&mut example.1.clone());
            }
            //println!("got here");
            // TODO Look inot better way to do this
            let input:Array2<f32> = Array2::from_shape_vec((example_len,input_len),input_vec).unwrap();
            let output:Array2<f32>  = Array2::from_shape_vec((example_len,output_len),output_vec).unwrap();
            //println!("finished here");
            return (input,output);
        }

        // Returns Instant::elapsed() as hh:mm:ss string
        fn time(instant:Instant) -> String {
            let mut seconds = instant.elapsed().as_secs();
            let hours = (seconds as f32 / 3600f32).floor();
            seconds = seconds % 3600;
            let minutes = (seconds as f32 / 60f32).floor();
            seconds = seconds % 60;
            let time = format!("{:#02}:{:#02}:{:#02}",hours,minutes,seconds);
            return time;
        }

        // TODO Make `f32_2d_prt` and `f32_3d_prt` be usable for any primitive datatype.
        // TODO Allow adjusting printing precision and customizing other flags (for `f32_2d_prt` and `f32_3d_prt`).
        // Nicely prints Array2<f32>
        pub fn f32_2d_prt(ndarray_param:&Array2<f32>) -> () {

            println!();
            let shape = ndarray_param.shape(); // shape[0],shape[1]=row,column
            let spacing = 5*shape[1];
            println!("┌ {: <1$}┐","",spacing);
            for row in 0..shape[0] {
                print!("│ ");
                for val in ndarray_param.row(row) {
                    if *val < 0f32 { print!("{:.1} ",val); }
                    else { print!("{:.2} ",val); }
                    
                }
                println!("│");
            }
            println!("└ {:<1$}┘","",spacing);
            print!("{:<1$}","",(spacing/2)-1);
            println!("[{},{}]",shape[0],shape[1]);
            println!();
        }
        // Nicely prints Array3<f32>
        pub fn f32_3d_prt(ndarray_param:&Array3<f32>) -> () {

            println!();
            let shape = ndarray_param.shape(); // shape[0],shape[1],shape[2]=layer,row,column
            let outer_spacing = (5*shape[0]*shape[2]) + (3*shape[0]) + 2;
            println!("┌{: <1$}┐","",outer_spacing);

            let inner_spacing = 5 * shape[2];

            print!("│ ");
            for _ in 0..shape[0] {
                print!("┌ {: <1$}┐","",inner_spacing);
                
            }
            print!(" │");

            println!();
            for i in 0..shape[1] {
                print!("│ ");
                for t in 0..shape[0] {
                    print!("│ ");
                    for p in 0..shape[2] {
                        let val = ndarray_param[[t,i,p]];
                        if val < 0f32 || val >= 10f32 { print!("{:.1} ",val); }
                        else { print!("{:.2} ",val); }
                    }
                    print!("│");
                }
                println!(" │");
            }

            print!("│ ");
            for _ in 0..shape[0] {
                print!("└ {: <1$}┘","",inner_spacing);
            }
            print!(" │");

            println!();
            println!("└{:<1$}┘","",outer_spacing);
            print!("{:<1$}","",(outer_spacing / 2) - 2);
            println!("[{},{},{}]",shape[0],shape[1],shape[2]);
            println!();
        }

        // TODO Improve this.
        // For use to sort a dataset before using `evaluate_outputs`.
        // Use counting sort since typically classification datasets have relatively high n compared to low k.
        pub fn counting_sort(test_data:&[(Vec<f32>,Vec<f32>)]) -> Vec<(Vec<f32>,Vec<f32>)> {
            let alphabet_size = test_data[0].1.len();
            let mut count:Vec<usize> = vec!(0usize;alphabet_size);
            let mut output_vals:Vec<usize> = vec!(0usize;test_data.len());

            for i in 0..test_data.len() {
                // TODO Put this in function.
                //  Had difficultly putting this in function.
                let mut one:usize = 0usize;
                for t in 0..alphabet_size {
                    if test_data[i].1[t] == 1f32 { 
                        one = t;
                        break;
                    }
                }

                count[one] += 1usize;
                output_vals[i] = one;
            }
            for i in 1..count.len() {
                count[i] += count[i-1];
            }

            let input_size = test_data[0].0.len();
            let mut sorted_data:Vec<(Vec<f32>,Vec<f32>)> = vec!((vec!(0f32;input_size),vec!(0f32;alphabet_size));test_data.len());

            for i in 0..test_data.len() {
                sorted_data[count[output_vals[i]]-1] = test_data[i].clone();
                count[output_vals[i]] -= 1;
            }

            return sorted_data;
        }
        
        // Prints the weights ands biases of the neural net.
        // TODO General improvement, specifically allow printing to variable accuracy.
        pub fn print_net(&self) -> () {
            let max:usize = self.biases.iter().map(|x|x.shape()[1]).max().unwrap();
            let width = self.connections.len(); // == self.biases.len()
            //println!("max:{}",max);
            for row in 0..max+2 {
                for t in 0..width {
                    //println!("\nrow:{}\n",self.biases[t].shape()[1]);
                    let diff = (max - self.biases[t].shape()[1]) / 2;
                    let spacing = 6*self.connections[t].shape()[1];
                    if row == diff {
                        print!("  ");
                        print!("┌ {: <1$}┐","",spacing);
                        print!("   ");
                        print!("┌       ┐");
                        print!(" ");
                    }
                    else if row == self.biases[t].shape()[1] + diff + 1 {
                        print!("  ");
                        print!("└ {: <1$}┘","",spacing);
                        print!("   ");
                        print!("└       ┘");
                        print!(" ");
                    }
                    else if row < diff || row > self.biases[t].shape()[1] + diff + 1 {
                        print!("  ");
                        print!("  {: <1$} ","",spacing);
                        print!("   ");
                        print!("         ");
                        print!(" ");
                    }
                    else {
                        let inner_row = row-diff-1;
                        if self.biases[t].shape()[1] / 2 == inner_row { print!("* "); }
                        else { print!("  "); }
                        print!("│ ");
                        
                        for val in self.connections[t].row(inner_row) {
                            print!("{:+.2} ",val);
                            
                        }
                        if inner_row == self.biases[t].shape()[1] / 2 {
                            print!("│ + │ ");
                        } else { print!("│   │ "); }
                        
                        let val = self.biases[t][[0,inner_row]];
                        print!("{:+.2} ",val);
    
                        
                        print!("│ ");
                    }
                }
                println!();
            }
            println!();
        }
        // Exports neural network to path.
        pub fn export(&self,path:&str) -> () {
            println!("here");
            let file = File::create(format!("/home/jonathan/Projects/rust_neural_network/checkpoints/{}",path));
            println!("can we get here?");
            let serialized:String = serde_json::to_string(self).unwrap();
            file.unwrap().write_all(serialized.as_bytes()).unwrap();
        }
        // Imports neural network from path.
        pub fn import(path:&str) -> NeuralNetwork {
            let file = File::open(format!("/home/jonathan/Projects/rust_neural_network/checkpoints/{}",path));
            let mut string_contents:String = String::new();
            file.unwrap().read_to_string(&mut string_contents).unwrap();
            let deserialized:NeuralNetwork = serde_json::from_str(&string_contents).unwrap();
            return deserialized;
        }
    }
}

// TODO Look into how to name tests
// TODO Look into using 'debug_assert's

#[cfg(test)]
mod tests {
    
    use std::fs::File;
    use std::time::{Instant,Duration};
    use crate::core::{EvaluationData,MeasuredCondition,Activation,Layer,Cost,NeuralNetwork};
    use std::io::Read;
    use std::io::prelude::*;
    use std::fs::OpenOptions;
    use std::io::stdout;
    use rand::prelude::SliceRandom;

    // TODO Figure out better name for this
    const TEST_RERUN_MULTIPLIER:u32 = 1; // Multiplies how many times we rerun tests (we rerun certain tests, due to random variation) (must be >= 0)
    // TODO Figure out better name for this
    const TESTING_MIN_ACCURACY:f32 = 0.90f32; // approx 10% min inaccuracy
    fn required_accuracy(test_data:&[(Vec<f32>,Vec<f32>)]) -> u32 {
        ((test_data.len() as f32) * TESTING_MIN_ACCURACY).ceil() as u32
    }
    fn export_result(test:&str,runs:u32,dataset_length:u32,total_time:u64,total_accuracy:u32,) -> () {
        let avg_time = (total_time / runs as u64) as f32 / 60f32;
        let avg_accuracy = total_accuracy / runs;
        let avg_accuracy_percent = 100f32 * avg_accuracy as f32 / dataset_length as f32;
        let file = OpenOptions::new().append(true).open("test_report.txt");
        let result = format!("{} : {} * {:.2} mins, {}%, {}/{}\n",test,runs,avg_time,avg_accuracy_percent,avg_accuracy,dataset_length);
        
        file.unwrap().write_all(result.as_bytes());
        //writeln!(file,&result_literal);
    }

    // Tests network to learn an XOR gate.
    #[test]
    fn train_xor_0() {
        let mut total_accuracy = 0u32;
        let mut total_time = 0u64;
        let runs = 10 * TEST_RERUN_MULTIPLIER;
        for _ in 0..runs {
            let start = Instant::now();
            //Setup
            let mut neural_network = NeuralNetwork::new(2,&[
                Layer::new(3,Activation::Sigmoid),
                Layer::new(2,Activation::Softmax)
            ],Cost::Quadratic);
            let training_data = vec![
                (vec![0f32,0f32],vec![0f32,1f32]),
                (vec![1f32,0f32],vec![1f32,0f32]),
                (vec![0f32,1f32],vec![1f32,0f32]),
                (vec![1f32,1f32],vec![0f32,1f32])
            ];
            let testing_data = training_data.clone();
            //Execution
            neural_network.train(&training_data)
                .halt_condition(MeasuredCondition::Iteration(8000u32))
                .early_stopping_condition(MeasuredCondition::Iteration(6000u32))
                .batch_size(4usize)
                .learning_rate(2f32)
                .learning_rate_interval(MeasuredCondition::Iteration(2000u32))
                .evaluation_data(crate::core::EvaluationData::Actual(testing_data.clone()))
                .lambda(0f32)
                .go();

            //Evaluation
            total_time += start.elapsed().as_secs();
            let evaluation = neural_network.evaluate(&testing_data);
            assert!(evaluation.1 >= required_accuracy(&testing_data));
            total_accuracy += evaluation.1;
        }
        export_result("train_xor_0",runs,4u32,total_time,total_accuracy);
    }
    #[test]
    fn train_xor_1() {
        let mut total_accuracy = 0u32;
        let mut total_time = 0u64;
        let runs = 10 * TEST_RERUN_MULTIPLIER;
        for _ in 0..runs {
            let start = Instant::now();
            //Setup
            let mut neural_network = NeuralNetwork::new(2,&[
                Layer::new(3,Activation::Sigmoid),
                Layer::new(2,Activation::Sigmoid)
            ],Cost::Quadratic);
            let training_data = vec![
                (vec![0f32,0f32],vec![0f32,1f32]),
                (vec![1f32,0f32],vec![1f32,0f32]),
                (vec![0f32,1f32],vec![1f32,0f32]),
                (vec![1f32,1f32],vec![0f32,1f32])
            ];
            let testing_data = training_data.clone();
            //Execution
            neural_network.train(&training_data)
                .halt_condition(MeasuredCondition::Iteration(8000u32))
                .early_stopping_condition(MeasuredCondition::Iteration(6000u32))
                .batch_size(4usize)
                .learning_rate(2f32)
                .learning_rate_interval(MeasuredCondition::Iteration(2000u32))
                .evaluation_data(crate::core::EvaluationData::Actual(testing_data.clone()))
                .lambda(0f32)
                .go();

            //Evaluation
            total_time += start.elapsed().as_secs();
            let evaluation = neural_network.evaluate(&testing_data);
            assert!(evaluation.1 >= required_accuracy(&testing_data));
            total_accuracy += evaluation.1;
        }
        export_result("train_xor_1",runs,4u32,total_time,total_accuracy);
    }
    // #[test]
    // fn train_xor_1() {
    //     let mut total_accuracy = 0u32;
    //     let mut total_time = 0u64;
    //     let runs = 10 * TEST_RERUN_MULTIPLIER;
    //     for _ in 0..runs {
    //         let start = Instant::now();
    //         //Setup
    //         let mut neural_network = NeuralNetwork::new(2,&[
    //             Layer::new(3,Activation::Sigmoid),
    //             Layer::new(4,Activation::Sigmoid),
    //             Layer::new(2,Activation::Sigmoid)
    //         ]);
    //         let training_data = vec![
    //             (vec![0f32,0f32],vec![0f32,1f32]),
    //             (vec![1f32,0f32],vec![1f32,0f32]),
    //             (vec![0f32,1f32],vec![1f32,0f32]),
    //             (vec![1f32,1f32],vec![0f32,1f32])
    //         ];
    //         let testing_data = training_data.clone();
    //         //Execution
    //         neural_network.train(&training_data)
    //             .halt_condition(crate::core::MeasuredCondition::Iteration(8000u32))
    //             .early_stopping_condition(MeasuredCondition::Iteration(6000u32))
    //             .batch_size(4usize)
    //             .learning_rate(2f32)
    //             .learning_rate_interval(MeasuredCondition::Iteration(1000u32))
    //             .evaluation_data(crate::core::EvaluationData::Actual(testing_data.clone()))
    //             .lambda(0f32)
    //             .log_interval(MeasuredCondition::Iteration(1000u32))
    //             .go();

    //         //Evaluation
    //         total_time += start.elapsed().as_secs();
    //         let evaluation = neural_network.evaluate(&testing_data);
    //         assert!(evaluation.1 >= required_accuracy(&testing_data));
    //         total_accuracy += evaluation.1;
    //     }
    //     export_result("train_xor_1",runs,4u32,total_time,total_accuracy);
    // }
    
    // Tests network to recognize handwritten digits of 28x28 pixels
    #[test]
    fn train_digits_0() {
        let mut total_accuracy = 0u32;
        let mut total_time = 0u64;
        let runs = TEST_RERUN_MULTIPLIER;
        for _ in 0..runs {
            let start = Instant::now();
            //Setup
            let mut neural_network = NeuralNetwork::new(784,&[
                Layer::new(100,Activation::Sigmoid),
                Layer::new(10,Activation::Sigmoid)
            ],Cost::Quadratic);
            let training_data = get_mnist_dataset(false);
            let testing_data = get_mnist_dataset(true);
            //Execution
            neural_network.train(&training_data)
                .evaluation_data(EvaluationData::Actual(testing_data.clone()))
                .go();
            //Evaluation
            total_time += start.elapsed().as_secs();
            

            let sorted_data = NeuralNetwork::counting_sort(&testing_data);
            println!("{:.?}",neural_network.evaluate_outputs(&sorted_data).0);

            let evaluation = neural_network.evaluate(&testing_data);
            assert!(evaluation.1 >= required_accuracy(&testing_data));
            total_accuracy += evaluation.1;
        }
        export_result("train_digits_0",runs,10000u32,total_time,total_accuracy);
    }
    #[test]
    fn train_digits_1() {
        let mut total_accuracy = 0u32;
        let mut total_time = 0u64;
        let runs = TEST_RERUN_MULTIPLIER;
        for _ in 0..runs {
            let start = Instant::now();
            //Setup
            let mut neural_network = NeuralNetwork::new(784,&[
                Layer::new(100,Activation::Sigmoid),
                Layer::new(10,Activation::Softmax)
            ],Cost::Quadratic);
            let training_data = get_mnist_dataset(false);
            let testing_data = get_mnist_dataset(true);
            //Execution
            neural_network.train(&training_data)
                .evaluation_data(EvaluationData::Actual(testing_data.clone()))
                .go();
            //Evaluation
            total_time += start.elapsed().as_secs();
            

            let sorted_data = NeuralNetwork::counting_sort(&testing_data);
            println!("{:.?}",neural_network.evaluate_outputs(&sorted_data).0);

            let evaluation = neural_network.evaluate(&testing_data);
            assert!(evaluation.1 >= required_accuracy(&testing_data));
            total_accuracy += evaluation.1;
        }
        export_result("train_digits_1",runs,10000u32,total_time,total_accuracy);
    }
    #[test]
    fn train_digits_2() {
        let mut total_accuracy = 0u32;
        let mut total_time = 0u64;
        let runs = TEST_RERUN_MULTIPLIER;
        for _ in 0..runs {
            let start = Instant::now();
            //Setup
            let mut neural_network = NeuralNetwork::new(784,&[
                Layer::new(100,Activation::Sigmoid),
                Layer::new(10,Activation::Sigmoid)
            ],Cost::CrossEntropy);
            let training_data = get_mnist_dataset(false);
            let testing_data = get_mnist_dataset(true);
            //Execution
            neural_network.train(&training_data)
                .evaluation_data(EvaluationData::Actual(testing_data.clone()))
                .go();
            //Evaluation
            total_time += start.elapsed().as_secs();
            

            let sorted_data = NeuralNetwork::counting_sort(&testing_data);
            println!("{:.?}",neural_network.evaluate_outputs(&sorted_data).0);

            let evaluation = neural_network.evaluate(&testing_data);
            assert!(evaluation.1 >= required_accuracy(&testing_data));
            total_accuracy += evaluation.1;
        }
        export_result("train_digits_0",runs,10000u32,total_time,total_accuracy);
    }
    #[test]
    fn train_digits_3() {
        let mut total_accuracy = 0u32;
        let mut total_time = 0u64;
        let runs = TEST_RERUN_MULTIPLIER;
        for _ in 0..runs {
            let start = Instant::now();
            //Setup
            let mut neural_network = NeuralNetwork::new(784,&[
                Layer::new(100,Activation::Sigmoid),
                Layer::new(10,Activation::Softmax)
            ],Cost::CrossEntropy);
            let training_data = get_mnist_dataset(false);
            let testing_data = get_mnist_dataset(true);
            //Execution
            neural_network.train(&training_data)
                .evaluation_data(EvaluationData::Actual(testing_data.clone()))
                .go();
            //Evaluation
            total_time += start.elapsed().as_secs();
            

            let sorted_data = NeuralNetwork::counting_sort(&testing_data);
            println!("{:.?}",neural_network.evaluate_outputs(&sorted_data).0);

            let evaluation = neural_network.evaluate(&testing_data);
            assert!(evaluation.1 >= required_accuracy(&testing_data));
            total_accuracy += evaluation.1;
        }
        export_result("train_digits_1",runs,10000u32,total_time,total_accuracy);
    }
    fn get_mnist_dataset(testing:bool) -> Vec<(Vec<f32>,Vec<f32>)> {
                
        let (images,labels) = if testing {
            (get_images("data/MNIST/t10k-images.idx3-ubyte"),get_labels("data/MNIST/t10k-labels.idx1-ubyte"))
        }
        else {
            (get_images("data/MNIST/train-images.idx3-ubyte"),get_labels("data/MNIST/train-labels.idx1-ubyte"))
        };

        let iterator = images.iter().zip(labels.iter());
        let mut examples = Vec::new();
        let set_output_layer = |label:u8| -> Vec<f32> { let mut temp = vec!(0f32;10); temp[label as usize] = 1f32; temp};
        for (image,label) in iterator {
            examples.push(
                (
                    image.clone(),
                    set_output_layer(*label)
                )
            );
        }
        return examples;

        fn get_labels(path:&str) -> Vec<u8> {
            let mut file = File::open(path).unwrap();
            let mut label_buffer = Vec::new();
            file.read_to_end(&mut label_buffer).expect("Couldn't read MNIST labels");

            // TODO Look into better ways to remove the 1st 7 elements
            return label_buffer.drain(8..).collect();
        }

        fn get_images(path:&str) -> Vec<Vec<f32>> {
            let mut file = File::open(path).unwrap();
            let mut image_buffer_u8 = Vec::new();
            file.read_to_end(&mut image_buffer_u8).expect("Couldn't read MNIST images");
            // Removes 1st 16 bytes of meta data
            image_buffer_u8 = image_buffer_u8.drain(16..).collect();

            // Converts from u8 to f32
            let mut image_buffer_f32 = Vec::new();
            for pixel in image_buffer_u8 {
                image_buffer_f32.push(pixel as f32 / 255f32);
            }

            // Splits buffer into vectors for each image
            let mut images_vector = Vec::new();
            for i in (0..image_buffer_f32.len() / (28 * 28)).rev() {
                images_vector.push(image_buffer_f32.split_off(i * 28 * 28));
            }
            // Does splitting in reverse order due to how '.split_off' works, so reverses back to original order.
            images_vector.reverse();
            return images_vector;
        }
    }
    
    // #[test]
    // fn train_full() {
    //     println!("test start");
    //     let runs = 1 * TEST_RERUN_MULTIPLIER;
    //     let mut total_accuracy = 0u32;
    //     let mut total_time = 0u64;
    //     for _ in 0..runs {
    //         let start = Instant::now();
    //         //Setup
    //         let training_data = get_combined("/home/jonathan/Projects/dataset_constructor/combined_dataset");
    //         let mut neural_network = NeuralNetwork::new(training_data[0].0.len(),&[
    //             Layer::new((training_data[0].0.len()+training_data[0].1.len())/2,Activation::Sigmoid),
    //             Layer::new(training_data[0].1.len(),Activation::Sigmoid)
    //         ]);
    //         //Execution#
    //         neural_network.train(&training_data)
    //             .learning_rate(0.05f32)
    //             .log_interval(MeasuredCondition::Iteration(1u32))
    //             .checkpoint_interval(MeasuredCondition::Duration(Duration::new(1800,0)))
    //             .halt_condition(MeasuredCondition::Duration(Duration::new(36000,0)))
    //         .go();
    //         //Evaluation
    //         total_time += start.elapsed().as_secs();
    //         let confusion_matrix = neural_network.evaluate_outputs(&training_data);

    //         NeuralNetwork::f32_2d_prt(&confusion_matrix);
    //         // assert!(evaluation.1 >= required_accuracy(&testing_data));
    //         // println!("train_full: accuracy: {}",evaluation.1);
    //         // println!();
    //         // total_accuracy += evaluation.1;
    //     }
    //     export_result("train_full",runs,450000,total_time,total_accuracy);
    //     assert!(false);
    // }
    // fn get_combined(path:&str) -> Vec<(Vec<f32>,Vec<f32>)> {
    //     let mut file = File::open(path).unwrap();
    //     let mut combined_buffer = Vec::new();
    //     file.read_to_end(&mut combined_buffer).expect("Couldn't read combined");

    //     let mut combined_vec = Vec::new();
    //     let multiplier:usize = (35*35)+1;
    //     let length = combined_buffer.len() / multiplier;
    //     println!("length: {}",length);
    //     let mut last_logged = Instant::now();
    //     for i in (0..length).rev() {
    //         let image_index = i * multiplier;
    //         let label_index = image_index + multiplier -1;

    //         let label:u8 = combined_buffer.split_off(label_index)[0]; // Array is only 1 element regardless
    //         let mut label_vec:Vec<f32> = vec!(0f32;95usize);
    //         label_vec[label as usize] = 1f32;

    //         let image:Vec<u8> = combined_buffer.split_off(image_index);
    //         let image_f32 = image.iter().map(|&x| x as f32).collect();

    //         combined_vec.push((image_f32,label_vec));

            
    //         if last_logged.elapsed().as_secs() > 5 {
    //             println!("{:.2}%",((length-i) as f32 / length as f32 *100f32));
    //             last_logged = Instant::now();
    //         }
    //     }
    //     return combined_vec;
    // }
}