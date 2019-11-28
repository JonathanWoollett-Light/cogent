#[allow(dead_code)]
mod core {
    extern crate nalgebra;
    use nalgebra::{DMatrix, DVector};
    use rand::prelude::SliceRandom;
    use std::time::{Duration,Instant};

    //Defining euler's constant
    const E:f64 = 2.7182818284f64;

    const DEFAULT_EVALUTATION_DATA:f64 = 0.1f64;//`(x * examples.len() as f64) as usize` of `testing_data` is split_off into `evaluation_data`
    const DEFAULT_HALT_CONDITION:u64 = 180u64;//Duration::new(x,0). x seconds
    const DEFAULT_BATCH_SIZE:f64 = 0.002f64;//(x * examples.len() as f64).ceil() as usize. batch_size = x% of training data
    const DEFAULT_LEARNING_RATE:f64 = 0.3f64;
    const DEFAULT_LAMBDA:f64 = 0.1f64;//lambda = (x * examples.len() as f64). lambda = x% of training data. lambda = regularization parameter

    pub enum EvaluationData {
        Scaler(usize),
        Percent(f64),
        Actual(Vec<(Vec<f64>,Vec<f64>)>)
    }

    #[derive(Clone,Copy)]
    pub enum MeasuredCondition {
        Iteration(u32),
        Duration(Duration)
    }
    //use EvaluationData::{Actual,Scaler,Percent};
    
    pub struct Trainer<'a> {
        training_data: Vec<(Vec<f64>,Vec<f64>)>,
        // TODO Since we never alter `evaluation_data` look into changing this into a reference
        evaluation_data: Vec<(Vec<f64>,Vec<f64>)>, 
        // Will halt after at a certain iteration, accuracy or duration.
        halt_condition: MeasuredCondition,
        // Can log after a certain number of iterations, a certain duration, or not at all.
        log_interval: Option<MeasuredCondition>,
        batch_size: usize, // TODO Maybe change `batch_size` to allow it to be set by a user as a % of their data
        learning_rate: f64, // Reffered to as `ETA` in `NeuralNetwork`.
        lambda: f64, // Regularization parameter
        // Can stop after no cost improvement over a certain number of iterations, a certain duration, or not at all.
        early_stopping_condition: Option<MeasuredCondition>,
        neural_network: &'a mut NeuralNetwork
    }

    impl<'a> Trainer<'a> {
        pub fn evaluation_data(&mut self, evaluation_data:EvaluationData) -> &mut Trainer<'a> {
            self.evaluation_data = match evaluation_data {
                EvaluationData::Scaler(scaler) => { self.training_data.split_off(self.training_data.len() - scaler) }
                EvaluationData::Percent(percent) => { self.training_data.split_off(self.training_data.len() - (self.training_data.len() as f64 * percent) as usize) }
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
        pub fn learning_rate(&mut self, learning_rate:f64) -> &mut Trainer<'a> {
            self.learning_rate = learning_rate;
            return self;
        }
        pub fn lambda(&mut self, lambda:f64) -> &mut Trainer<'a> {
            self.lambda = lambda;
            return self;
        }
        pub fn early_stopping_condition(&mut self, early_stopping_condition:MeasuredCondition) -> &mut Trainer<'a> {
            self.early_stopping_condition = Some(early_stopping_condition);
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
                self.early_stopping_condition
            );
        }
    }

    // A simple stochastic/incremental descent neural network.
    pub struct NeuralNetwork {
        neurons: Vec<DVector<f64>>,
        biases: Vec<DVector<f64>>,
        connections: Vec<DMatrix<f64>>
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
            let mut neurons: Vec<DVector<f64>> = Vec::with_capacity(layers.len());
            let mut connections: Vec<DMatrix<f64>> = Vec::with_capacity(layers.len() - 1);
            let mut biases: Vec<DVector<f64>> = Vec::with_capacity(layers.len() - 1);

            neurons.push(DVector::repeat(layers[0],0f64));
            for i in 1..layers.len() {
                neurons.push(DVector::repeat(layers[i],0f64));
                connections.push(DMatrix::new_random(layers[i],layers[i-1])/(layers[i-1] as f64).sqrt());//TODO Double check this is right
                biases.push(DVector::new_random(layers[i]));
                // connections.push(DMatrix::repeat(layers[i],layers[i-1],0.5f64));
                // biases.push(DVector::repeat(layers[i],0.5f64));
            }
            NeuralNetwork{ neurons, biases, connections }
        }
        // Constructs and trains network for given dataset
        pub fn build(training_data:&Vec<(Vec<f64>,Vec<f64>)>) -> NeuralNetwork {
            let avg_size:usize = (training_data[0].0.len() + training_data[0].1.len()) / 2;
            let layers:&[usize] = &[training_data[0].0.len(),avg_size,training_data[0].1.len()];
            let mut network = NeuralNetwork::new(layers);

            network.train(training_data).go();

            return network;
        }

        // Feeds forward through network
        pub fn run(&mut self, inputs:&[f64]) -> &DVector<f64> {

            if inputs.len() != self.neurons[0].len() {
                panic!("Wrong number of inputs: {} given, {} required",inputs.len(),self.neurons[0].len());
            }

            self.neurons[0] = DVector::from_vec(inputs.to_vec()); // TODO Look into improving this
            for i in 0..self.connections.len() {
                let temp = (&self.connections[i] * &self.neurons[i])+ &self.biases[i];
                self.neurons[i+1] = sigmoid_mapping(self,&temp);
            }

            return &self.neurons[self.neurons.len() - 1]; // TODO Look into removing this

            // Returns new vector with sigmoid function applied component-wise
            fn sigmoid_mapping(net:&NeuralNetwork,y: &DVector<f64>) -> DVector<f64>{
                y.map(|x| -> f64 { net.sigmoid(x) })
            }
        }

        fn train_details(&mut self,
            training_data: &mut [(Vec<f64>,Vec<f64>)], // TODO Look into `&[(Vec<f64>,Vec<f64>)]` vs `&Vec<(Vec<f64>,Vec<f64>)>`
            evaluation_data: &[(Vec<f64>,Vec<f64>)],
            halt_condition: MeasuredCondition,
            log_interval: Option<MeasuredCondition>,
            batch_size: usize,
            learning_rate: f64,
            lambda: f64,
            early_stopping_n: Option<MeasuredCondition>
        ) -> () {

            let mut rng = rand::thread_rng();
            let start_instant = Instant::now();
            let mut iterations_elapsed = 0u32;
            
            let mut best_accuracy_iteration = 0u32;// Iteration of best accuracy
            let mut best_accuracy_instant = Instant::now();// Instant of best accuracy

            let mut best_accuracy = 0u32;

            let mut evaluation = self.evaluate(evaluation_data);
            
            if let Some(_) = log_interval {
                println!("Iteration: {}, Time: {}s, Cost: {:.7}, Classified: {}/{} ({:.4}%)",iterations_elapsed,start_instant.elapsed().as_secs(),evaluation.0,evaluation.1,evaluation_data.len(), (evaluation.1 as f64)/(evaluation_data.len() as f64) * 100f64);
            }

            let starting_evaluation = evaluation;
            let mut last_logged_instant = Instant::now();

            loop {
                match halt_condition {
                    MeasuredCondition::Iteration(iteration) => if iterations_elapsed == iteration { break; },
                    MeasuredCondition::Duration(duration) => if start_instant.elapsed() >= duration { break; },
                }

                training_data.shuffle(&mut rng);
                let batches = get_batches(training_data,batch_size);

                for batch in batches {
                    self.update_batch(batch,learning_rate,lambda,training_data.len() as f64);
                }
                iterations_elapsed += 1;

                evaluation = self.evaluate(evaluation_data);

                if evaluation.1 > best_accuracy { 
                    best_accuracy = evaluation.1;
                    best_accuracy_iteration = iterations_elapsed;
                    best_accuracy_instant = Instant::now();
                }

                match log_interval {
                    Some(MeasuredCondition::Iteration(iteration_interval)) => if iterations_elapsed % iteration_interval == 0 { 
                         println!("Iteration: {}, Time: {}, Cost: {:.7}, Classified: {}/{} ({:.4}%)",iterations_elapsed,start_instant.elapsed().as_secs(),evaluation.0,evaluation.1,evaluation_data.len(), (evaluation.1 as f64)/(evaluation_data.len() as f64) * 100f64);
                    },
                    Some(MeasuredCondition::Duration(duration_interval)) => if last_logged_instant.elapsed() >= duration_interval { 
                        println!("Iteration: {}, Time: {}, Cost: {:.7}, Classified: {}/{} ({:.4}%)",iterations_elapsed,start_instant.elapsed().as_secs(),evaluation.0,evaluation.1,evaluation_data.len(), (evaluation.1 as f64)/(evaluation_data.len() as f64) * 100f64);
                        last_logged_instant = Instant::now();
                    },
                    _ => {},
                }

                match early_stopping_n {
                    Some(MeasuredCondition::Iteration(stopping_iteration)) =>  if iterations_elapsed - best_accuracy_iteration == stopping_iteration { println!("---------------\nEarly stoppage!\n---------------"); break; },
                    Some(MeasuredCondition::Duration(stopping_duration)) => if best_accuracy_instant.elapsed() >= stopping_duration { println!("---------------\nEarly stoppage!\n---------------"); break; },
                    _ => {},
                }
            }
            let new_percent = (evaluation.1 as f64)/(evaluation_data.len() as f64) * 100f64;
            let starting_percent = (starting_evaluation.1 as f64)/(evaluation_data.len() as f64) * 100f64;
            println!();
            println!("Cost: {:.7} -> {:.7}",starting_evaluation.0,evaluation.0);
            println!("Classified: {} ({:.4}%) -> {} ({:.4}%)",starting_evaluation.1,starting_percent,evaluation.1,new_percent);
            println!("Cost: {:.6}",evaluation.0-starting_evaluation.0);
            println!("Classified: +{} (+{:.4}%)",evaluation.1-starting_evaluation.1,new_percent - starting_percent);
            println!("Time: {}",start_instant.elapsed().as_secs());
            println!();

            fn get_batches(examples:&[(Vec<f64>,Vec<f64>)], batch_size: usize) -> Vec<&[(Vec<f64>,Vec<f64>)]> {
                let mut batches = Vec::new(); // TODO Look into if 'Vec::with_capacity(ceil(examples.len() / batch_size))' is more efficient

                let mut lower_bound = 0usize;
                let mut upper_bound = batch_size;
                while upper_bound < examples.len() {
                    batches.push(&examples[lower_bound..upper_bound]);
                    lower_bound = upper_bound;
                    // TODO Improve this to remove last unnecessary addition to 'upper_bound'
                    upper_bound += batch_size;
                }
                // Accounts for last batch possibly being under 'batch_size'
                batches.push(&examples[lower_bound..examples.len()]);
                batches
            }
        }
        pub fn train(&mut self,training_data:&Vec<(Vec<f64>,Vec<f64>)>) -> Trainer {
            let mut temp_training_data = training_data.clone();
            let temp_evaluation_data = temp_training_data.split_off(training_data.len() - (training_data.len() as f64 * DEFAULT_EVALUTATION_DATA) as usize);
            return Trainer {
                training_data: temp_training_data,
                evaluation_data: temp_evaluation_data,
                halt_condition: MeasuredCondition::Duration(Duration::new(DEFAULT_HALT_CONDITION,0)),
                log_interval: None,
                batch_size: (DEFAULT_BATCH_SIZE * training_data.len() as f64).ceil() as usize,
                learning_rate: DEFAULT_LEARNING_RATE,
                lambda: DEFAULT_LAMBDA,
                early_stopping_condition: None,
                neural_network:self
            };
        }

        fn update_batch(&mut self, batch: &[(Vec<f64>, Vec<f64>)], eta: f64, lambda:f64, n:f64) -> () {
            // Copies structure of self.neurons and self.connections with values of 0f64
            // TODO Look into a better way to setup 'bias_nabla' and 'weight_nabla'
            // TODO Better understand what 'nabla' means

            // Clones, `self.neurons` with every value set to `0f64`
            // TODO Look into changing this to clone `self.biases` to make following line obselete.
            let mut clone_holder_b:Vec<DVector<f64>> = self.neurons.clone().iter().map(|x| x.map(|_y| -> f64 { 0f64 }) ).collect();
            clone_holder_b.remove(0);
            // Clones, `self.connections` with every value set to `0f64`
            let clone_holder_w:Vec<DMatrix<f64>> = self.connections.clone().iter().map(|x| x.map(|_y| -> f64 { 0f64 }) ).collect();
            let mut nabla_b:Vec<DVector<f64>> = clone_holder_b.clone();
            let mut nabla_w:Vec<DMatrix<f64>> = clone_holder_w.clone();

            for example in batch {
                let (delta_nabla_w,delta_nabla_b):(Vec<DMatrix<f64>>,Vec<DVector<f64>>) =
                    self.backpropagate(example,clone_holder_w.clone(),clone_holder_b.clone());

                
                nabla_w = nabla_w.iter().zip(delta_nabla_w).map(|(x,y)| x + y).collect();
                nabla_b = nabla_b.iter().zip(delta_nabla_b).map(|(x,y)| x + y).collect();
            }

            
            self.connections = self.connections.iter().zip(nabla_w.clone()).map(
                | (w,nw) |
                    (1f64-eta*(lambda/n))*w - ((eta / batch.len() as f64)) * nw
            ).collect();
            self.biases = self.biases.iter().zip(nabla_b.clone()).map(
                | (b,nb) |
                    b - ((eta / batch.len() as f64)) * nb
            ).collect();
        }
        
        // Todo Make better name for 'zs' it is just all the neuron values without being put through the sigmoid function
        fn backpropagate(&mut self, example:&(Vec<f64>,Vec<f64>), mut nabla_w:Vec<DMatrix<f64>>, mut nabla_b:Vec<DVector<f64>>) -> (Vec<DMatrix<f64>>,Vec<DVector<f64>>) {

            let target = DVector::from_vec(example.1.clone());
            let last_index = self.connections.len()-1; // = nabla_b.len()-1 = nabla_w.len()-1 = self.neurons.len()-2 = self.connections.len()-1

            // TODO 
            let mut zs = nabla_b.clone();
            // Runs input through network
            self.neurons[0] = DVector::from_vec(example.0.to_vec());
            for i in 0..self.connections.len() {
                zs[i] = (&self.connections[i] * &self.neurons[i])+ &self.biases[i];
                self.neurons[i+1] = sigmoid_mapping(self,&zs[i]);
            }


            let mut delta:DVector<f64> = cross_entropy_delta(&self.neurons[last_index+1],&target);

            nabla_b[last_index] = delta.clone();
            nabla_w[last_index] = delta.clone() * self.neurons[last_index].transpose();

            for i in (1..self.neurons.len()-1).rev() {

                //println!("z[{}]: {}",(i as i32)-1-(zs.len() as i32),zs[i-1]);
                //println!("z siged: {}",self.sigmoid_prime_mapping(&zs[i-1]));

                delta = sigmoid_prime_mapping(self,&zs[i-1]).component_mul(
                    &(self.connections[i].transpose() * delta)
                );

                nabla_b[i-1] = delta.clone();
                // TODO Look into using `delta.clone()` vs `&delta` here
                nabla_w[i-1] = delta.clone() * self.neurons[i-1].transpose();
            }

            return (nabla_w,nabla_b);

            // Returns new vector of `output-target`
            fn cross_entropy_delta(output:&DVector<f64>,target:&DVector<f64>) -> DVector<f64> {
                output - target
            }
            fn sigmoid_prime_mapping(net:&NeuralNetwork,y: &DVector<f64>) -> DVector<f64> {   
                y.map(|x| -> f64 { net.sigmoid(x) * (1f64 - net.sigmoid(x)) })
            }
            fn sigmoid_mapping(net:&NeuralNetwork,y: &DVector<f64>) -> DVector<f64>{
                y.map(|x| -> f64 { net.sigmoid(x) })
            }
        }

        // Returns tuple (average cost, number of examples correctly identified)
        pub fn evaluate(&mut self, test_data:&[(Vec<f64>,Vec<f64>)]) -> (f64,u32) {
            let mut correctly_classified = 0u32;
            let mut return_cost = 0f64;
            for example in test_data {
                let out = self.run(&example.0);
                let expected = DVector::from_vec(example.1.clone());
                return_cost += my_simple_cost(out,&expected);// Adjust this to what ever cost function you would prefer to see
            
                if get_max_index(out) == get_max_index(&expected) {
                    correctly_classified += 1u32;
                }
            }
            return (return_cost / test_data.len() as f64, correctly_classified);

            // Returns index of max value
            fn get_max_index(vector:&DVector<f64>) -> usize{
                let mut max_index = 0usize;
                for i in 1..vector.len() {
                    if vector[i] > vector[max_index]  {
                        max_index = i;
                    }
                }
                return max_index;
            }

            fn my_simple_cost(outputs: &DVector<f64>, targets: &DVector<f64>) -> f64 {
                (targets - outputs).abs().sum()
            }
            //Returns average squared difference between `outputs` and `targets`
            fn quadratic_cost(outputs: &DVector<f64>, targets: &DVector<f64>) -> f64 {
                // TODO This could probably be 1 line, look into that
                let error_vector = targets - outputs;
                let cost_vector = error_vector.component_mul(&error_vector);
                return cost_vector.mean();
            }
            fn cross_entropy_cost(outputs: &DVector<f64>, targets: &DVector<f64>) -> f64 {
                // TODO This could probably be 1 line, look into that
                let term1 = targets.component_mul(&ln_mapping(outputs));
                let temp = &DVector::repeat(targets.len(),1f64);
                let term2 = temp-targets;
                let term3 = temp-outputs;
                //print!("{}+{}*{}",&term1,&term2,&term3);
                return -0.5*(term1+term2.component_mul(&ln_mapping(&term3))).sum();

                fn ln_mapping(y: &DVector<f64>) -> DVector<f64> {   
                    return y.map(|x| -> f64 { x.log(E) })
                }
            }
            // Regularized costs not included since this would require passing an addition parameter through `evaluation(...)` and the functions are functionalty useless anyway, only useful for learning.
        }
        
        fn sigmoid(&self,y: f64) -> f64 {
            1f64 / (1f64 + (-y).exp())
        }
    }
}

// TODO Look into how to name tests
// TODO Look into using 'debug_assert's

#[cfg(test)]
mod tests {
    
    extern crate nalgebra;
    use std::fs::File;
    use std::io::{Read};
    use std::time::Duration;
    use crate::core::{EvaluationData,MeasuredCondition,NeuralNetwork};

    // TODO Figure out better name for this
    const TEST_RERUN_MULTIPLIER:u32 = 1; // Multiplies how many times we rerun tests (we rerun certain tests, due to random variation) (must be >= 0)
    // TODO Figure out better name for this
    const TESTING_MIN_ACCURACY:f64 = 0.95f64; // approx 5% min inaccuracy
    fn required_accuracy(test_data:&[(Vec<f64>,Vec<f64>)]) -> u32 {
        ((test_data.len() as f64) * TESTING_MIN_ACCURACY).ceil() as u32
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

    #[test]
    fn run_0() {
        let mut neural_network = crate::core::NeuralNetwork::new(&[1,1]);
        assert_eq!(neural_network.run(&vec![1f64]).len(),1usize);
    }
    #[test]
    fn run_1() {
        let mut neural_network = crate::core::NeuralNetwork::new(&[2,3]);
        assert_eq!(neural_network.run(&vec![1f64,0f64]).len(),3usize);
    }
    #[test]
    fn run_2() {
        let mut neural_network = crate::core::NeuralNetwork::new(&[2,3,1]);
        assert_eq!(neural_network.run(&vec![1f64,0f64]).len(),1usize);
    }
    #[test]
    #[should_panic(expected="Wrong number of inputs: 1 given, 2 required")]
    fn run_inputs_wrong_0() {
        let mut neural_network = crate::core::NeuralNetwork::new(&[2,3,1]);
        neural_network.run(&vec![1f64]);
    }
    #[test]
    #[should_panic(expected="Wrong number of inputs: 3 given, 2 required")]
    fn run_inputs_wrong_1() {
        let mut neural_network = crate::core::NeuralNetwork::new(&[2,3,1]);
        neural_network.run(&vec![1f64,1f64,0f64]);
    }

    // Tests network to learn an XOR gate.
    #[test]
    fn train_xor_0() {
        let mut total_accuracy = 0u32;
        for _ in 0..(10 * TEST_RERUN_MULTIPLIER) {
            //Setup
            let mut neural_network = crate::core::NeuralNetwork::new(&[2,3,2]);
            let training_data = vec![
                (vec![0f64,0f64],vec![0f64,1f64]),
                (vec![1f64,0f64],vec![1f64,0f64]),
                (vec![0f64,1f64],vec![1f64,0f64]),
                (vec![1f64,1f64],vec![0f64,1f64])
            ];
            let testing_data = training_data.clone();
            //Execution
            neural_network.train(&training_data)
                .halt_condition(crate::core::MeasuredCondition::Iteration(4000u32))
                .batch_size(4usize)
                .learning_rate(2f64)
                .evaluation_data(crate::core::EvaluationData::Actual(testing_data.clone()))
                .lambda(0f64)
                .go();
            //Evaluation
            let evaluation = neural_network.evaluate(&testing_data);
            assert!(evaluation.1 >= required_accuracy(&testing_data));

            println!("train_xor_0: accuracy: {}",evaluation.1);
            println!();
            total_accuracy += evaluation.1;
        }
        println!("train_xor_0: average accuracy: {}",total_accuracy / (10 * TEST_RERUN_MULTIPLIER));
    }
    #[test]
    fn train_xor_1() {
        let mut total_accuracy = 0u32;
        for _ in 0..(10 * TEST_RERUN_MULTIPLIER) {
            //Setup
            let mut neural_network = crate::core::NeuralNetwork::new(&[2,3,4,2]);
            let training_data = vec![
                (vec![0f64,0f64],vec![0f64,1f64]),
                (vec![1f64,0f64],vec![1f64,0f64]),
                (vec![0f64,1f64],vec![1f64,0f64]),
                (vec![1f64,1f64],vec![0f64,1f64])
            ];
            let testing_data = training_data.clone();
            //Execution
            neural_network.train(&training_data)
                .halt_condition(crate::core::MeasuredCondition::Iteration(4000u32))
                .batch_size(4usize)
                .learning_rate(2f64)
                .evaluation_data(crate::core::EvaluationData::Actual(testing_data.clone()))
                .lambda(0f64)
                .go();
            //Evaluation
            let evaluation = neural_network.evaluate(&testing_data);
            assert!(evaluation.1 >= required_accuracy(&testing_data));

            println!("train_xor_1: accuracy: {}",evaluation.1);
            println!();
            total_accuracy += evaluation.1;
        }
        println!("train_xor_1: average accuracy: {}",total_accuracy / (10 * TEST_RERUN_MULTIPLIER));
    }

    // Tests network to recognize handwritten digits of 28x28 pixels
    #[test]
    fn train_digits_0() {
        let mut total_accuracy = 0u32;
        for _ in 0..TEST_RERUN_MULTIPLIER {
            //Setup
            let mut neural_network = NeuralNetwork::new(&[784,100,10]);
            let training_data = get_mnist_dataset(false);
            //Execution
            neural_network.train(&training_data).log_interval(MeasuredCondition::Duration(Duration::new(10,0))).go();
            //Evaluation
            let testing_data = get_mnist_dataset(true);
            let evaluation = neural_network.evaluate(&testing_data);
            assert!(evaluation.1 >= required_accuracy(&testing_data));

            println!("train_digits_0: accuracy: {}",evaluation.1);
            println!();
            total_accuracy += evaluation.1;
        }
        println!("train_digits_0: average accuracy: {}",total_accuracy / TEST_RERUN_MULTIPLIER);
    }
    #[test]
    fn train_digits_1() {
        let mut total_accuracy = 0u32;
        for _ in 0..TEST_RERUN_MULTIPLIER {
            //Setup
            let mut neural_network = NeuralNetwork::new(&[784,100,10]);
            let training_data = get_mnist_dataset(false);
            //Execution
            neural_network.train(&training_data)
                .halt_condition(MeasuredCondition::Iteration(30u32))
                .log_interval(MeasuredCondition::Iteration(1u32))
                .batch_size(10usize)
                .learning_rate(0.5f64)
                .evaluation_data(EvaluationData::Scaler(10000usize))
                .lambda(5f64)
                .early_stopping_condition(MeasuredCondition::Iteration(10u32))
                .go();
            //Evaluation
            let testing_data = get_mnist_dataset(true);
            let evaluation = neural_network.evaluate(&testing_data);
            assert!(evaluation.1 >= required_accuracy(&testing_data));

            println!("train_digits_1: accuracy: {}",evaluation.1);
            println!();
            total_accuracy += evaluation.1;
        }
        println!("train_digits_1: average accuracy: {}",total_accuracy / TEST_RERUN_MULTIPLIER);
    }
    // #[test]
    // fn train_digits_2() {
    //     let mut total_accuracy = 0u32;
    //     for _ in 0..TEST_RERUN_MULTIPLIER {
    //         //Setup
    //         let mut neural_network = crate::core::NeuralNetwork::new(&[784,200,100,50,10]);
    //         let training_data = get_mnist_dataset(false);
    //         //Execution
    //         neural_network.train(&training_data)
    //             .log_interval(crate::core::MeasuredCondition::Duration(Duration::new(10,0)))
    //             .halt_condition(crate::core::MeasuredCondition::Duration(Duration::new(240,0)))
    //             .learning_rate(0.1f64)
    //             .go();
    //         //Evaluation
    //         let testing_data = get_mnist_dataset(true);
    //         let evaluation = neural_network.evaluate(&testing_data);
    //         assert!(evaluation.1 >= required_accuracy(&testing_data));

    //         println!("train_digits_2: accuracy: {}",evaluation.1);
    //         println!();
    //         total_accuracy += evaluation.1;
    //     }
    //     println!("train_digits_2: average accuracy: {}",total_accuracy / TEST_RERUN_MULTIPLIER);
    // }
    #[test]
    fn train_digits_3() {
        let mut total_accuracy = 0u32;
        for _ in 0..TEST_RERUN_MULTIPLIER {
            //Setup
            let training_data = get_mnist_dataset(false);
            //Execution
            let mut neural_network = NeuralNetwork::build(&training_data);
            //Evaluation
            let testing_data = get_mnist_dataset(true);
            let evaluation = neural_network.evaluate(&testing_data);
            assert!(evaluation.1 >= required_accuracy(&testing_data));

            println!("train_digits_3: accuracy: {}",evaluation.1);
            println!();
            total_accuracy += evaluation.1;
        }
        println!("train_digits_3: average accuracy: {}",total_accuracy / TEST_RERUN_MULTIPLIER);
    }
    fn get_mnist_dataset(testing:bool) -> Vec<(Vec<f64>,Vec<f64>)> {
                
        let (images,labels) = if testing {
            (get_images("data/MNIST/t10k-images.idx3-ubyte"),get_labels("data/MNIST/t10k-labels.idx1-ubyte"))
        }
        else {
            (get_images("data/MNIST/train-images.idx3-ubyte"),get_labels("data/MNIST/train-labels.idx1-ubyte"))
        };

        let iterator = images.iter().zip(labels.iter());
        let mut examples = Vec::new();
        let set_output_layer = |label:u8| -> Vec<f64> { let mut temp = vec!(0f64;10); temp[label as usize] = 1f64; temp};
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

        fn get_images(path:&str) -> Vec<Vec<f64>> {
            let mut file = File::open(path).unwrap();
            let mut image_buffer_u8 = Vec::new();
            file.read_to_end(&mut image_buffer_u8).expect("Couldn't read MNIST images");
            // Removes 1st 16 bytes of meta data
            image_buffer_u8 = image_buffer_u8.drain(16..).collect();

            // Converts from u8 to f64
            let mut image_buffer_f64 = Vec::new();
            for pixel in image_buffer_u8 {
                image_buffer_f64.push(pixel as f64 / 255f64);
            }

            // Splits buffer into vectors for each image
            let mut images_vector = Vec::new();
            for i in (0..image_buffer_f64.len() / (28 * 28)).rev() {
                images_vector.push(image_buffer_f64.split_off(i * 28 * 28));
            }
            // Does splitting in reverse order due to how '.split_off' works, so reverses back to original order.
            images_vector.reverse();
            return images_vector;
        }
    }
    
}