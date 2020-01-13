#[allow(dead_code)]
mod core {
    extern crate nalgebra;
    use nalgebra::{DMatrix, DVector};
    use rand::prelude::SliceRandom;
    use std::time::{Duration,Instant};
    use itertools::izip;
    extern crate scoped_threadpool;
    use scoped_threadpool::Pool;

    const THREAD_COUNT:usize = 12usize;

    //Defining euler's constant
    const E:f32 = 2.718281f32;

    const DEFAULT_EVALUTATION_DATA:f32 = 0.1f32; //`(x * examples.len() as f32) as usize` of `testing_data` is split_off into `evaluation_data`
    const DEFAULT_HALT_CONDITION:f32 = 0.00002f32; // Duration::new(examples[0].0.len()*examples.len()*x,0). (MNIST is approx 15 mins)
    const DEFAULT_BATCH_SIZE:f32 = 0.002f32; //(x * examples.len() as f32).ceil() as usize. batch_size = x% of training data
    const DEFAULT_LEARNING_RATE:f32 = 0.1f32;
    const DEFAULT_LAMBDA:f32 = 0.1f32; // lambda = (x * examples.len() as f32). lambda = x% of training data. lambda = regularization parameter
    const DEFAULT_EARLY_STOPPING:f32 = 0.000005f32; // Duration::new(examples[0].0.len()*examples.len()*x,0). (MNIST is approx 4 mins)
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
    //use EvaluationData::{Actual,Scaler,Percent};
    
    pub struct Trainer<'a> {
        training_data: Vec<(Vec<f32>,Vec<f32>)>,
        // TODO Since we never alter `evaluation_data` look into changing this into a reference
        evaluation_data: Vec<(Vec<f32>,Vec<f32>)>, 
        // Will halt after at a certain iteration, accuracy or duration.
        halt_condition: MeasuredCondition,
        // Can log after a certain number of iterations, a certain duration, or not at all.
        log_interval: Option<MeasuredCondition>,
        batch_size: usize, // TODO Maybe change `batch_size` to allow it to be set by a user as a % of their data
        learning_rate: f32, // Reffered to as `ETA` in `NeuralNetwork`.
        lambda: f32, // Regularization parameter
        // Can stop after no cost improvement over a certain number of iterations, a certain duration, or not at all.
        early_stopping_condition: MeasuredCondition,
        learning_rate_decay: f32,
        learning_rate_interval: MeasuredCondition,
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
        pub fn halt_condition(&mut self, halt_condition:MeasuredCondition) -> &mut Trainer<'a> {
            self.halt_condition = halt_condition;
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
                self.learning_rate_interval
            );
        }
    }

    // A simple stochastic/incremental descent neural network.
    // Implementing cross-entropy cost function and L2 regularization.
    pub struct NeuralNetwork {
        neurons: Vec<DVector<f32>>,
        biases: Vec<DVector<f32>>,
        connections: Vec<DMatrix<f32>>
    }

    impl NeuralNetwork {

        // Constructs network of given layers
        pub fn new(layers: &[usize]) -> NeuralNetwork {
            if layers.len() < 2 {
                panic!("Requires >1 layers");
            }
            for &x in layers {
                if x < 1usize {
                    panic!("All layer sizes must be >0");
                }
            }
            let mut neurons: Vec<DVector<f32>> = Vec::with_capacity(layers.len());
            let mut connections: Vec<DMatrix<f32>> = Vec::with_capacity(layers.len() - 1);
            let mut biases: Vec<DVector<f32>> = Vec::with_capacity(layers.len() - 1);

            neurons.push(DVector::repeat(layers[0],0f32));
            for i in 1..layers.len() {
                neurons.push(DVector::repeat(layers[i],0f32));
                connections.push(DMatrix::new_random(layers[i],layers[i-1])/(layers[i-1] as f32).sqrt());//TODO Double check this is right
                biases.push(DVector::new_random(layers[i]));
            }
            NeuralNetwork{ neurons, biases, connections }
        }
        // Constructs and trains network for given dataset
        pub fn build(training_data:&Vec<(Vec<f32>,Vec<f32>)>) -> NeuralNetwork {
            println!("Building");
            let avg_size:usize = (((training_data[0].0.len() + training_data[0].1.len()) as f32 / 2f32) + 1f32) as usize;
            let layers:&[usize] = &[training_data[0].0.len(),avg_size,training_data[0].1.len()];
            let mut network = NeuralNetwork::new(layers);
            network.train(training_data).log_interval(MeasuredCondition::Duration(Duration::new(60,0))).go();
            println!("Built");
            return network;
        }

        // Feeds forward through network
        fn run(&self, inputs:&[f32]) -> DVector<f32> {

            let mut z = DVector::from_vec(inputs.to_vec());
            for i in 0..self.connections.len() {
                let a = (&self.connections[i] * z) + &self.biases[i];
                z = sigmoid_mapping(self,&a);
            }

            return z; // TODO Look into removing this

            // Returns new vector with sigmoid function applied component-wise
            fn sigmoid_mapping(net:&NeuralNetwork,y: &DVector<f32>) -> DVector<f32>{
                y.map(|x| -> f32 { net.sigmoid(x) })
            }
        }

        fn train_details(&mut self,
            training_data: &mut [(Vec<f32>,Vec<f32>)], // TODO Look into `&[(Vec<f32>,Vec<f32>)]` vs `&Vec<(Vec<f32>,Vec<f32>)>`
            evaluation_data: &[(Vec<f32>,Vec<f32>)],
            halt_condition: MeasuredCondition,
            log_interval: Option<MeasuredCondition>,
            batch_size: usize,
            mut learning_rate: f32,
            lambda: f32,
            early_stopping_n: MeasuredCondition,
            learning_rate_decay: f32,
            learning_rate_interval: MeasuredCondition
        ) -> () {

            let mut rng = rand::thread_rng();
            let start_instant = Instant::now();
            let mut iterations_elapsed = 0u32;
            
            let mut best_accuracy_iteration = 0u32;// Iteration of best accuracy
            let mut best_accuracy_instant = Instant::now();// Instant of best accuracy

            let mut best_accuracy = 0u32;
            println!("Evaluating");
            let eval_start_instant = Instant::now();
            let mut evaluation = self.evaluate(evaluation_data);
            println!("Evaluated: {:.3} mins",eval_start_instant.elapsed().as_secs() as f32 / 60f32);
            
            if let Some(_) = log_interval {
                println!("Iteration: {}, Time: {}, Cost: {:.7}, Classified: {}/{} ({:.4}%), Learning rate: {}",
                    iterations_elapsed,start_instant.elapsed().as_secs(),evaluation.0,
                    evaluation.1,evaluation_data.len(),
                    (evaluation.1 as f32)/(evaluation_data.len() as f32) * 100f32,
                    learning_rate
                );
            }

            let starting_evaluation = evaluation;
            let mut last_logged_instant = Instant::now();

            

            loop {
                match halt_condition {
                    MeasuredCondition::Iteration(iteration) => if iterations_elapsed == iteration { break; },
                    MeasuredCondition::Duration(duration) => if start_instant.elapsed() >= duration { break; },
                }

                //println!("Setting batches");
                //let batch_start_instant = Instant::now();
                training_data.shuffle(&mut rng);
                let batches = get_batches(training_data,batch_size);
                //println!("Set batches: {:.2} millis",batch_start_instant.elapsed().as_millis());

                //println!("Backpropagating");
                //let backprop_start_instant = Instant::now();
                let mut percentage:f32 = 0f32;
                //println!("{:.3} = {} / {}",batch_size as f32 / training_data.len() as f32,training_data.len(),batch_size);
                let percent_change:f32 = batch_size as f32 / training_data.len() as f32;
                for batch in batches {
                    println!("{:.3}%",percentage);
                    let (new_connections,new_biases) = self.update_batch(batch,learning_rate,lambda,training_data.len() as f32);
                    self.connections = new_connections;
                    self.biases = new_biases;
                    percentage += percent_change;
                }
                //println!("Backpropagated: {:.3} mins",backprop_start_instant.elapsed().as_secs() as f32 / 60f32);

                iterations_elapsed += 1;
                evaluation = self.evaluate(evaluation_data);

                if evaluation.1 > best_accuracy { 
                    best_accuracy = evaluation.1;
                    best_accuracy_iteration = iterations_elapsed;
                    best_accuracy_instant = Instant::now();
                }

                match log_interval {
                    Some(MeasuredCondition::Iteration(iteration_interval)) => if iterations_elapsed % iteration_interval == 0 { 
                        println!("Iteration: {}, Time: {}, Cost: {:.7}, Classified: {}/{} ({:.4}%), Learning rate: {}",
                            iterations_elapsed,start_instant.elapsed().as_secs(),
                            evaluation.0,evaluation.1,evaluation_data.len(),
                            (evaluation.1 as f32)/(evaluation_data.len() as f32) * 100f32,
                            learning_rate
                        );
                    },
                    Some(MeasuredCondition::Duration(duration_interval)) => if last_logged_instant.elapsed() >= duration_interval { 
                        println!("Iteration: {}, Time: {}, Cost: {:.7}, Classified: {}/{} ({:.4}%), Learning rate: {}",
                            iterations_elapsed,start_instant.elapsed().as_secs(),
                            evaluation.0,evaluation.1,evaluation_data.len(),
                            (evaluation.1 as f32)/(evaluation_data.len() as f32) * 100f32,
                            learning_rate
                        );
                        last_logged_instant = Instant::now();
                    },
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
            let new_percent = (evaluation.1 as f32)/(evaluation_data.len() as f32) * 100f32;
            let starting_percent = (starting_evaluation.1 as f32)/(evaluation_data.len() as f32) * 100f32;
            println!();
            println!("Cost: {:.7} -> {:.7}",starting_evaluation.0,evaluation.0);
            println!("Classified: {} ({:.4}%) -> {} ({:.4}%)",starting_evaluation.1,starting_percent,evaluation.1,new_percent);
            println!("Cost: {:.6}",evaluation.0-starting_evaluation.0);
            println!("Classified: +{} (+{:.4}%)",evaluation.1-starting_evaluation.1,new_percent - starting_percent);
            println!("Time: {}",start_instant.elapsed().as_secs());
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
        }
        pub fn train(&mut self,training_data:&Vec<(Vec<f32>,Vec<f32>)>) -> Trainer {
            let mut rng = rand::thread_rng();
            let mut temp_training_data = training_data.clone();
            temp_training_data.shuffle(&mut rng);
            let temp_evaluation_data = temp_training_data.split_off(training_data.len() - (training_data.len() as f32 * DEFAULT_EVALUTATION_DATA) as usize);

            let multiplier:f32 = training_data[0].0.len() as f32 * training_data.len() as f32;
            let halt_condition = Duration::new((multiplier * DEFAULT_HALT_CONDITION) as u64,0);
            let early_stopping_condition = Duration::new((multiplier * DEFAULT_EARLY_STOPPING) as u64,0);
            let learning_rate_interval:u32 = (DEFAULT_LEARNING_RATE_INTERVAL as f32 * training_data[0].0.len() as f32 / training_data.len() as f32) as u32;
            println!(
                "halt_condition: {:.2} mins ({:.2} hours), early_stopping_condition: {:.2} secs ({:.2} hours), learning_rate_interval: {} ints",
                halt_condition.as_secs() as f32 / 60f32,halt_condition.as_secs() as f32 / 3600f32,
                early_stopping_condition.as_secs() as f32 / 60f32,early_stopping_condition.as_secs() as f32 / 3600f32,
                learning_rate_interval
            );
            
            return Trainer {
                training_data: temp_training_data,
                evaluation_data: temp_evaluation_data,
                halt_condition: MeasuredCondition::Duration(halt_condition),
                log_interval: None,
                batch_size: (DEFAULT_BATCH_SIZE * training_data.len() as f32).ceil() as usize,
                learning_rate: DEFAULT_LEARNING_RATE,
                lambda: DEFAULT_LAMBDA,
                early_stopping_condition: MeasuredCondition::Duration(early_stopping_condition),
                learning_rate_decay: DEFAULT_LEARNING_RATE_DECAY,
                learning_rate_interval: MeasuredCondition::Iteration(learning_rate_interval),
                neural_network:self
            };
        }

        fn update_batch(&self, batch: &[(Vec<f32>, Vec<f32>)], eta: f32, lambda:f32, n:f32) -> (Vec<DMatrix<f32>>,Vec<DVector<f32>>) {
            // Copies structure of self.neurons and self.connections with values of 0f32
            // TODO Look into a better way to setup 'bias_nabla' and 'weight_nabla'
            // TODO Better understand what 'nabla' means
            let nabla_b_zeros:Vec<DVector<f32>> = self.biases.clone().iter().map(|x| x.map(|_y| -> f32 { 0f32 }) ).collect();
            let nabla_w_zeros:Vec<DMatrix<f32>> = self.connections.clone().iter().map(|x| x.map(|_y| -> f32 { 0f32 }) ).collect();

            
            let chunks_length:usize = if batch.len() < THREAD_COUNT { batch.len() } else { batch.len() / THREAD_COUNT };
            let chunks:Vec<_> = batch.chunks(chunks_length).collect(); // Specify type further
            let mut pool = Pool::new(chunks.len() as u32);

            let mut out_nabla_b:Vec<Vec<DVector<f32>>> = vec!(nabla_b_zeros.clone();chunks.len());
            let mut out_nabla_w:Vec<Vec<DMatrix<f32>>> = vec!(nabla_w_zeros.clone();chunks.len());

            pool.scoped(|scope| {
                for (chunk,nabla_w,nabla_b) in izip!(chunks,&mut out_nabla_w,&mut out_nabla_b) {
                    scope.execute(move || {
                        for example in chunk {
                            let (delta_nabla_w,delta_nabla_b):(Vec<DMatrix<f32>>,Vec<DVector<f32>>) = self.backpropagate(example);

                            *nabla_w = nabla_w.iter().zip(delta_nabla_w).map(|(x,y)| x + y).collect();
                            *nabla_b = nabla_b.iter().zip(delta_nabla_b).map(|(x,y)| x + y).collect();
                        }
                    });
                }
            });

            // TODO Look at turning these into iterator folds and maps, had issues trying it 1st time
            //      nabla_b is sum of out_nabla_b, nabla_w is sum of out_nabla_w
            let mut nabla_b = nabla_b_zeros.clone();
            for example in &out_nabla_b {
                for i in 0..example.len() {
                    nabla_b[i] += &example[i]; // example[i] is layer i
                }
            }
            let mut nabla_w = nabla_w_zeros.clone();
            for example in &out_nabla_w {
                for i in 0..example.len() {
                    nabla_w[i] += &example[i]; // example[i] is layer i
                }
            }

            // TODO Look into removing `.clone()`s here
            let return_connections:Vec<DMatrix<f32>> = self.connections.iter().zip(nabla_w).map(
                | (w,nw) | (1f32-eta*(lambda/n))*w - ((eta / batch.len() as f32)) * nw
            ).collect();
            let return_biases:Vec<DVector<f32>> = self.biases.iter().zip(nabla_b).map(
                | (b,nb) | b - ((eta / batch.len() as f32)) * nb
            ).collect();

            return (return_connections,return_biases);
        }
        
        // Runs backpropagation
        // Returns weight and bias gradients
        // TODO Implement fully matrix based approach for batches (run all examples in batch at once).
        //  This will likely require changing from using `nalgebra` to a library which supports n-dimensional arrays.
        //  This would be needed for gradients of weights for each example.
        fn backpropagate(&self, example:&(Vec<f32>,Vec<f32>)) -> (Vec<DMatrix<f32>>,Vec<DVector<f32>>) {
            
            // Feeds forward
            // --------------

            let mut zs = self.biases.clone(); // Name more intuitively
            let mut activations = self.neurons.clone();
            activations[0] = DVector::from_vec(example.0.to_vec());
            for i in 0..self.connections.len() {
                zs[i] = (&self.connections[i] * &activations[i])+ &self.biases[i];
                activations[i+1] = sigmoid_mapping(self,&zs[i]);
            }

            // Backpropagates
            // --------------

            let target = DVector::from_vec(example.1.clone());
            let last_index = self.connections.len()-1; // = nabla_b.len()-1 = nabla_w.len()-1 = self.neurons.len()-2 = self.connections.len()-1
            let mut error:DVector<f32> = cross_entropy_delta(&activations[last_index+1],&target);

            // Gradients of biases and weights.
            let mut nabla_b = self.biases.clone();
            let mut nabla_w = self.connections.clone();

            // Sets gradients in output layer
            nabla_b[last_index] = error.clone();
            nabla_w[last_index] = error.clone() * activations[last_index].transpose();
            // self.neurons.len()-2 -> 1 (inclusive)
            // With input layer, self.neurons.len()-1 is last nueron layer, but without, self.neurons.len()-2 is last neuron layer
            for i in (1..self.neurons.len()-1).rev() {
                // Calculates error
                error = sigmoid_prime_mapping(self,&zs[i-1]).component_mul(
                    &(self.connections[i].transpose() * error)
                );

                // Sets gradients
                nabla_b[i-1] = error.clone();
                nabla_w[i-1] = error.clone() * activations[i-1].transpose();// TODO Look into using `error.clone()` vs `&error` here
            }
            // Returns gradients
            return (nabla_w,nabla_b);

            // Returns new vector of `output-target`
            fn cross_entropy_delta(output:&DVector<f32>,target:&DVector<f32>) -> DVector<f32> {
                output - target
            }
            fn sigmoid_prime_mapping(net:&NeuralNetwork,y: &DVector<f32>) -> DVector<f32> {   
                y.map(|x| -> f32 { net.sigmoid(x) * (1f32 - net.sigmoid(x)) })
            }
            fn sigmoid_mapping(net:&NeuralNetwork,y: &DVector<f32>) -> DVector<f32>{
                y.map(|x| -> f32 { net.sigmoid(x) })
            }
        }

        // Returns tuple (average cost, number of examples correctly identified)
        pub fn evaluate(&self, test_data:&[(Vec<f32>,Vec<f32>)]) -> (f32,u32) {
            let chunks_length:usize = if test_data.len() < THREAD_COUNT { test_data.len() } else { test_data.len() / THREAD_COUNT };
            let chunks:Vec<_> = test_data.chunks(chunks_length).collect(); // Specify type further
            let mut pool = Pool::new(chunks.len() as u32);

            let mut cost_vec = vec!(0f32;chunks.len());
            let mut classified_vec = vec!(0u32;chunks.len());
            pool.scoped(|scope| {
                for (chunk,cost,classified) in izip!(chunks,&mut cost_vec,&mut classified_vec) {
                    // let last_logged = Instant::now();
                    scope.execute(move || {
                        for i in 0..chunk.len() {
                            let example = &chunk[i];
                            let out = self.run(&example.0);
                            let expected = DVector::from_vec(example.1.clone());
                            *cost += linear_cost(&out,&expected);// Adjust this to what ever cost function you would prefer to see
                            
                            if get_max_index(&out) == get_max_index(&expected) {
                                *classified += 1u32;
                            }
            
                            // if last_logged.elapsed().as_secs() > 10 {
                            //     println!("thread: {} {:.4}% ({}/{})",index,i as f32 / chunk.len() as f32,i,chunk.len());
                            //     last_logged = Instant::now();
                            // }
                        }
                    });
                }
            });
            //println!("cost_vec: {:.?}",cost_vec);
            let cost:f32 = cost_vec.iter().sum();
            let classified:u32 = classified_vec.iter().sum();
            //println!("cost: {:.3}",cost);

            return (cost / test_data.len() as f32, classified);

            // Returns index of max value
            fn get_max_index(vector:&DVector<f32>) -> usize{
                let mut max_index = 0usize;
                for i in 1..vector.len() {
                    if vector[i] > vector[max_index]  {
                        max_index = i;
                    }
                }
                return max_index;
            }

            fn linear_cost(outputs: &DVector<f32>, targets: &DVector<f32>) -> f32 {
                (targets - outputs).abs().sum()
            }
            //Returns average squared difference between `outputs` and `targets`
            fn quadratic_cost(outputs: &DVector<f32>, targets: &DVector<f32>) -> f32 {
                // TODO This could probably be 1 line, look into that
                let error_vector = targets - outputs;
                let cost_vector = error_vector.component_mul(&error_vector);
                return cost_vector.sum() / (2f32 * cost_vector.len() as f32);
            }
            fn cross_entropy_cost(outputs: &DVector<f32>, targets: &DVector<f32>) -> f32 {
                // TODO This could probably be 1 line, look into that
                let term1 = targets.component_mul(&ln_mapping(outputs));
                let temp = &DVector::repeat(targets.len(),1f32);
                let term2 = temp-targets;
                let term3 = temp-outputs;
                //print!("{}+{}*{}",&term1,&term2,&term3);
                return -0.5*(term1+term2.component_mul(&ln_mapping(&term3))).sum();

                fn ln_mapping(y: &DVector<f32>) -> DVector<f32> {   
                    return y.map(|x| -> f32 { x.log(E) })
                }
            }
        }
        
        // TODO Lot of stuff could be done to improve this function
        // Gets average output for every input
        pub fn evaluate_outputs(&self, test_data:&[(Vec<f32>,Vec<f32>)]) -> DMatrix<f32> {
            let input_len:usize = test_data[0].0.len();
            let output_len:usize = test_data[0].1.len();

            let mut similarities: Vec<DVector<f32>> = vec!(DVector::repeat(output_len,0f32);input_len);
            let mut count:DVector<f32> = DVector::repeat(input_len,0f32);

            for example in test_data {
                let index = example.0.iter().position(|&x|x==1f32).unwrap();
                similarities[index] += self.run(&example.0);
                count[index] += 1f32;
            }
            // TODO Look at turning this loop into a mapping
            for i in 0..similarities.len() {
                similarities[i] = similarities[i].map(|x| -> f32 { x / count[i] }); // TODO Do I need `similarities[i] = ` ?
            }
            return DMatrix::from_columns(&similarities);
        }
        
        fn sigmoid(&self,y: f32) -> f32 {
            1f32 / (1f32 + (-y).exp())
        }
    }
}

// TODO Look into how to name tests
// TODO Look into using 'debug_assert's

#[cfg(test)]
mod tests {
    
    extern crate nalgebra;
    use std::fs::File;
    use std::time::{Instant,Duration};
    use crate::core::{EvaluationData,MeasuredCondition,NeuralNetwork};
    use std::io;
    use std::fs::read_dir;
    use std::io::{Read,Result};
    use std::path::Path;
    use std::io::prelude::*;
    use std::io::IoSlice;
    use std::io::BufReader;
    use std::fs::OpenOptions;

    // TODO Figure out better name for this
    const TEST_RERUN_MULTIPLIER:u32 = 1; // Multiplies how many times we rerun tests (we rerun certain tests, due to random variation) (must be >= 0)
    // TODO Figure out better name for this
    const TESTING_MIN_ACCURACY:f32 = 0.925f32; // approx 5% min inaccuracy
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

    #[test]
    fn new() {
        crate::core::NeuralNetwork::new(&[2,3,1]);
    }
    #[test]
    #[should_panic(expected="Requires >1 layers")]
    fn new_few_layers() {
        crate::core::NeuralNetwork::new(&[2]);
    }
    #[test]
    #[should_panic(expected="All layer sizes must be >0")]
    fn new_small_layers_0() {
        crate::core::NeuralNetwork::new(&[0,3,1]);
    }
    #[test]
    #[should_panic(expected="All layer sizes must be >0")]
    fn new_small_layers_1() {
        crate::core::NeuralNetwork::new(&[2,0,1]);
    }
    #[test]
    #[should_panic(expected="All layer sizes must be >0")]
    fn new_small_layers_2() {
        crate::core::NeuralNetwork::new(&[2,3,0]);
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
            let mut neural_network = crate::core::NeuralNetwork::new(&[2,3,2]);
            let training_data = vec![
                (vec![0f32,0f32],vec![0f32,1f32]),
                (vec![1f32,0f32],vec![1f32,0f32]),
                (vec![0f32,1f32],vec![1f32,0f32]),
                (vec![1f32,1f32],vec![0f32,1f32])
            ];
            let testing_data = training_data.clone();
            //Execution
            neural_network.train(&training_data)
                .halt_condition(MeasuredCondition::Iteration(6000u32))
                .early_stopping_condition(MeasuredCondition::Iteration(5000u32))
                .batch_size(4usize)
                .learning_rate(2f32)
                .learning_rate_interval(MeasuredCondition::Iteration(2000u32))
                .evaluation_data(crate::core::EvaluationData::Actual(testing_data.clone()))
                .lambda(0f32)
                .log_interval(MeasuredCondition::Iteration(500u32))
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
        for _ in 0..(10 * TEST_RERUN_MULTIPLIER) {
            let start = Instant::now();
            //Setup
            let mut neural_network = crate::core::NeuralNetwork::new(&[2,3,4,2]);
            let training_data = vec![
                (vec![0f32,0f32],vec![0f32,1f32]),
                (vec![1f32,0f32],vec![1f32,0f32]),
                (vec![0f32,1f32],vec![1f32,0f32]),
                (vec![1f32,1f32],vec![0f32,1f32])
            ];
            let testing_data = training_data.clone();
            //Execution
            neural_network.train(&training_data)
                .halt_condition(crate::core::MeasuredCondition::Iteration(8000u32))
                .early_stopping_condition(MeasuredCondition::Iteration(6000u32))
                .batch_size(4usize)
                .learning_rate(2f32)
                .learning_rate_interval(MeasuredCondition::Iteration(1000u32))
                .evaluation_data(crate::core::EvaluationData::Actual(testing_data.clone()))
                .lambda(0f32)
                .log_interval(MeasuredCondition::Iteration(1000u32))
                .go();

            //Evaluation
            total_time += start.elapsed().as_secs();
            let evaluation = neural_network.evaluate(&testing_data);
            assert!(evaluation.1 >= required_accuracy(&testing_data));
            total_accuracy += evaluation.1;
        }
        export_result("train_xor_1",runs,4u32,total_time,total_accuracy);
    }

    //Tests network to recognize handwritten digits of 28x28 pixels
    #[test]
    fn train_digits_0() {
        let mut total_accuracy = 0u32;
        let mut total_time = 0u64;
        let runs = TEST_RERUN_MULTIPLIER;
        for _ in 0..runs {
            let start = Instant::now();
            //Setup
            let mut neural_network = NeuralNetwork::new(&[784,100,10]);
            let training_data = get_mnist_dataset(false);
            //Execution
            neural_network.train(&training_data).log_interval(MeasuredCondition::Duration(Duration::new(10,0))).go();
            //Evaluation
            total_time += start.elapsed().as_secs();
            let testing_data = get_mnist_dataset(true);
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
            let mut neural_network = NeuralNetwork::new(&[784,100,10]);
            let training_data = get_mnist_dataset(false);
            let testing_data = get_mnist_dataset(true);
            //Execution
            neural_network.train(&training_data)
                .halt_condition(MeasuredCondition::Iteration(30u32))
                .log_interval(MeasuredCondition::Iteration(1u32))
                .batch_size(10usize)
                .learning_rate(0.5f32)
                .evaluation_data(EvaluationData::Actual(testing_data.clone()))
                .lambda(5f32)
                .early_stopping_condition(MeasuredCondition::Iteration(10u32))
                .go();
            //Evaluation
            total_time += start.elapsed().as_secs();
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
            let training_data = get_mnist_dataset(false);
            //Execution
            let neural_network = NeuralNetwork::build(&training_data);
            //Evaluation
            total_time += start.elapsed().as_secs();
            let testing_data = get_mnist_dataset(true);
            let evaluation = neural_network.evaluate(&testing_data);
            assert!(evaluation.1 >= required_accuracy(&testing_data));
            total_accuracy += evaluation.1;
        }
        export_result("train_digits_2",runs,10000u32,total_time,total_accuracy);
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
    //     let mut total_accuracy = 0u32;
    //     for _ in 0..TEST_RERUN_MULTIPLIER {
    //         //Setup
    //         println!("Loading dataset");
    //         let training_data = get_combined("/home/jonathan/Projects/dataset_constructor/combined_dataset");
    //         println!("Loaded dataset");
    //         //Execution
    //         let mut neural_network = NeuralNetwork::build(&training_data);
    //         //Evaluation
    //         //let evaluation = neural_network.evaluate(&testing_data);

    //         //assert!(evaluation.1 >= required_accuracy(&testing_data));
    //         // println!("train_full: accuracy: {}",evaluation.1);
    //         // println!();
    //         // total_accuracy += evaluation.1;
    //     }
    //     println!("train_full: average accuracy: {}",total_accuracy / TEST_RERUN_MULTIPLIER);
    //     assert!(false);
    // }
    // fn get_combined(path:&str) -> Vec<(Vec<f32>,Vec<f32>)> {
    //     let mut file = File::open(path).unwrap();
    //     let mut combined_buffer = Vec::new();
    //     file.read_to_end(&mut combined_buffer).expect("Couldn't read combined");

    //     let mut combined_vec = Vec::new();
    //     let multiplier:usize = (45*45)+1;
    //     let length = combined_buffer.len() / ((45*45)+1);
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