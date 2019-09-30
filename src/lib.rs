#[allow(dead_code)]
mod core {
    extern crate nalgebra;
    use nalgebra::{DMatrix, DVector};
    use rand::prelude::SliceRandom;
    use std::fs::File;
    use std::io::Write;


    const E:f64 = 2.7182818284f64;
    // A simple stochastic/incremental descent neural network.
    pub struct NeuralNetwork {
        neurons: Vec<DVector<f64>>,
        biases: Vec<DVector<f64>>,
        connections: Vec<DMatrix<f64>>
    }

    impl NeuralNetwork {

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
                connections.push(DMatrix::new_random(layers[i],layers[i-1])/(layers[i-1] as f64).sqrt());
                biases.push(DVector::new_random(layers[i]));
                // connections.push(DMatrix::repeat(layers[i],layers[i-1],0.5f64));
                // biases.push(DVector::repeat(layers[i],0.5f64));
            }
            NeuralNetwork{ neurons, biases, connections }
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
        
        // Trains the network
        pub fn train(&mut self, examples:&mut [(Vec<f64>,Vec<f64>)], duration:i32, log_interval:i32, batch_size:usize, learning_rate:f64, test_data:&[(Vec<f64>,Vec<f64>)],lambda:f64) -> () {
            let mut rng = rand::thread_rng();
            let mut iterations_elapsed = 0i32;
            let starting_evaluation = self.evaluate(test_data);

            loop {
                if iterations_elapsed == duration { break; }

                if iterations_elapsed % log_interval == 0 && iterations_elapsed != 0 {
                    let evaluation = self.evaluate(test_data);
                    println!("Iteration: {}, Cost: {:.7}, Classified: {}/{} ({:.4}%)",iterations_elapsed,evaluation.0,evaluation.1,test_data.len(), (evaluation.1 as f64)/(test_data.len() as f64) * 100f64);
                }

                examples.shuffle(&mut rng);
                let batches = get_batches(examples,batch_size);

                //let mut counter = 0;//remove this after debugging
                for batch in batches {
                    //println!("batch:{}",counter);//remove this after debugging
                    self.update_batch(batch,learning_rate,lambda,examples.len() as f64);
                    // counter+=1;//remove this after debugging
                    // break;//remove this after debugging
                }

                iterations_elapsed += 1;
            }

            let evaluation = self.evaluate(test_data);
            let new_percent = (evaluation.1 as f64)/(test_data.len() as f64) * 100f64;
            let starting_percent = (starting_evaluation.1 as f64)/(test_data.len() as f64) * 100f64;
            println!("Iteration: {}, Cost: {:.7}, Classified: {}/{} ({:.4}%)",iterations_elapsed,evaluation.0,evaluation.1,test_data.len(),new_percent);
            println!("Cost: {:.7} -> {:.7}",starting_evaluation.0,evaluation.0);
            println!("Classified: {} ({:.4}%) -> {} ({:.4}%)",starting_evaluation.1,starting_percent,evaluation.1,new_percent);
            println!("Cost: {:.6}",evaluation.0-starting_evaluation.0);
            println!("Classified: +{} (+{:.4}%)",evaluation.1-starting_evaluation.1,new_percent - starting_percent);

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

        fn update_batch(&mut self, batch: &[(Vec<f64>, Vec<f64>)], eta: f64, lambda:f64, n:f64) -> () {
            // Copies structure of self.neurons and self.connections with values of 0f64
            // TODO Look into a better way to setup 'bias_nabla' and 'weight_nabla'
            // TODO Better understand what 'nabla' means

            let mut clone_holder_b:Vec<DVector<f64>> = self.neurons.clone().iter().map(|x| x.map(|_y| -> f64 { 0f64 }) ).collect();
            clone_holder_b.remove(0);
            let clone_holder_w:Vec<DMatrix<f64>> = self.connections.clone().iter().map(|x| x.map(|_y| -> f64 { 0f64 }) ).collect();
            let mut nabla_b:Vec<DVector<f64>> = clone_holder_b.clone();
            let mut nabla_w:Vec<DMatrix<f64>> = clone_holder_w.clone();

            for example in batch {
                let (delta_nabla_w,delta_nabla_b):(Vec<DMatrix<f64>>,Vec<DVector<f64>>) =
                    self.backpropagate(example,clone_holder_w.clone(),clone_holder_b.clone());

                // println!("delta_nabla_b:");//remove this after debugging
                // for dvec in &delta_nabla_b {//remove this after debugging
                //     print!("{}",&dvec);//remove this after debugging
                // }
                // break;//remove this after debugging
                
                // Sums values (matrices) in each index together
                nabla_w = nabla_w.iter().zip(delta_nabla_w).map(|(x,y)|x + y).collect();
                nabla_b = nabla_b.iter().zip(delta_nabla_b).map(|(x,y)| x + y).collect();
            }

            // println!("nabla_b:");//remove this after debugging
            // for dvec in &nabla_b {//remove this after debugging
            //     print!("{}",&dvec);//remove this after debugging
            // }//remove this after debugging

            // TODO Check if these lines could be done via matrix multiplication
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
                //println!("neurons[{}]: {}",i,&self.neurons[i].clone());
                //println!("biases[{}]: {}",i,&self.biases[i].clone());
                zs[i] = (&self.connections[i] * &self.neurons[i])+ &self.biases[i];
                //println!("zs[{}]: {}",i,zs[i]);
                self.neurons[i+1] = sigmoid_mapping(self,&zs[i]);
            }

            //print!("output:{}",&self.neurons[last_index+1]);

            let mut delta:DVector<f64> = cross_entropy_delta(&self.neurons[last_index+1],&target);

            //print!("delta out:{}",&delta.clone());

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
                return_cost += cross_entropy_cost(out,&expected);
            
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
    use std::time::{Instant};

    // TODO Figure out best name for this
    const TESTING_MIN_COST:f64 = 01f64; // approx 1% inaccuracy

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
    fn train_0() {
        let mut neural_network = crate::core::NeuralNetwork::new(&[2,3,4,2]);
        let mut examples = [
            (vec![0f64,0f64],vec![0f64,1f64]),
            (vec![1f64,0f64],vec![1f64,0f64]),
            (vec![0f64,1f64],vec![1f64,0f64]),
            (vec![1f64,1f64],vec![0f64,1f64])
        ];
        let test_data = examples.clone();
        neural_network.train(&mut examples,4000,400,4usize,2f64,&test_data,0f64);

        let evalutation = neural_network.evaluate(&examples);
        assert!(evalutation.0 < TESTING_MIN_COST);
        assert_eq!(evalutation.1,examples.len() as u32);
        //assert!(false);
    }

    // Tests network to recognize handwritten digits of 28x28 pixels
    #[test]
    fn train_1() {
        

        let mut neural_network = crate::core::NeuralNetwork::new(&[784,30,10]);

        let mut training_examples = get_examples(false);
        let testing_examples = get_examples(true);

        let start = Instant::now();

        neural_network.train(&mut training_examples, 30, 1, 10usize, 0.5f64, &testing_examples,0f64);

        println!("Time to train: {}", start.elapsed().as_millis());

        let evaluation = neural_network.evaluate(&testing_examples);
        // TODO This line and function is broken, takes ages.
        
        

        assert!(evaluation.0 < TESTING_MIN_COST);
        assert!(evaluation.1 > 9000u32);

        assert!(false);

        fn get_examples(testing:bool) -> Vec<(Vec<f64>,Vec<f64>)> {
            let (images,labels) = if testing {
                (
                    get_images("data/MNIST/t10k-images.idx3-ubyte"),
                    get_labels("data/MNIST/t10k-labels.idx1-ubyte")
                )
            } else {
                (
                    get_images("data/MNIST/train-images.idx3-ubyte"),
                    get_labels("data/MNIST/train-labels.idx1-ubyte")
                )
            };
            //images = images[0..10000].to_vec();// THIS FOR DEBUGGING
            //labels = labels[0..10000].to_vec();// THIS FOR DEBUGGING
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
            if !testing {
                examples.split_off(50000);
            }
            return examples;

            fn get_labels(path:&str) -> Vec<u8> {
                let mut file = File::open(path).unwrap();
                let mut label_buffer = Vec::new();
                file.read_to_end(&mut label_buffer).expect("Couldn't read MNIST labels");

                // TODO Look into better ways to remove the 1st 7 elements
                label_buffer.drain(8..).collect()
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
                images_vector
            }
        }
    }
}