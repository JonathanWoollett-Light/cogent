#[cfg(test)]
mod tests {
    use cogent::core::{HaltCondition,EvaluationData,MeasuredCondition,Activation,Layer,NeuralNetwork,Proportion};
    use std::io::Read;
    use std::fs::File;

    // Run with: `cargo test --release -- --test-threads=1` (You can set 1 higher if you have more VRAM)
    // Using more threads will likely overload vram and crash.

    // TODO Name this better
    const TEST_RERUN_MULTIPLIER:usize = 1usize; // Multiplies how many times we rerun tests (we rerun certain tests, due to random variation) (must be > 0)

    // TODO Name this better
    const TESTING_MIN_ACCURACY:f32 = 0.90f32; // 5% error min needed to pass tests
    // Returns `TESTING_MIN_ACCURACY` percentage as number of example in dataset.
    fn required_accuracy(test_data:&[(Vec<f32>,usize)]) -> u32 {
        ((test_data.len() as f32) * TESTING_MIN_ACCURACY).ceil() as u32
    }

    // Tests `NeuralNetwork::new` panics when no layers are set.
    #[test] // (2-)
    #[should_panic="Requires output layer (layers.len() must be >0)."]
    fn new_no_layers() { NeuralNetwork::new(2,&[]); }
    // Tests `NeuralNetwork::new` panics when inputs is set to 0.
    #[test] // (0-Sigmoid->1)
    #[should_panic="Input size must be >0."]
    fn new_0_input() { NeuralNetwork::new(0,&[Layer::Dense(1,Activation::Sigmoid)]); }

    // Tests `NeuralNetwork::new` panics when a layerers length is 0.
    // --------------
    #[test] // (784-ReLU->0-Sigmoid->100-Softmax->10)
    #[should_panic="All dense layer sizes must be >0."]
    fn new_small_layers_0() {
        NeuralNetwork::new(784,&[
            Layer::Dense(0,Activation::ReLU),
            Layer::Dense(100,Activation::Sigmoid),
            Layer::Dense(10,Activation::Softmax)
        ]);
    }
    #[test] // (784-ReLU->800-Sigmoid->0-Softmax->10)
    #[should_panic="All dense layer sizes must be >0."]
    fn new_small_layers_1() {
        NeuralNetwork::new(784,&[
            Layer::Dense(800,Activation::ReLU),
            Layer::Dense(0,Activation::Sigmoid),
            Layer::Dense(10,Activation::Softmax)
        ]);
    }
    #[test] // (784-ReLU->800-Sigmoid->100-Softmax->0)
    #[should_panic="All dense layer sizes must be >0."]
    fn new_small_layers_2() {
        NeuralNetwork::new(784,&[
            Layer::Dense(800,Activation::ReLU),
            Layer::Dense(100,Activation::Sigmoid),
            Layer::Dense(0,Activation::Softmax)
        ]);
    }

    // Tests changing activation of layer using out of range index.
    #[test]
    #[should_panic="Layer 2 does not exist. 0 <= given index < 2"]
    fn activation() {
        let mut net = NeuralNetwork::new(2,&[
            Layer::Dense(3,Activation::Sigmoid),
            Layer::Dense(2,Activation::Sigmoid)
        ]);
        net.activation(2,Activation::Softmax); // Changes activation of output layer.
    }

    // `train_xor_x` Tests network to learn an XOR gate.
    // --------------

    // (2-Sigmoid->3-Sigmoid->2)
    #[test]
    fn train_xor_0() {
        let runs = 10 * TEST_RERUN_MULTIPLIER;
        
        for _ in 0..runs {
            // Setup
            // ------------------------------------------------
            // Sets network
            let mut net = NeuralNetwork::new_constant(2,&[
                Layer::Dense(3,Activation::Sigmoid),
                Layer::Dense(2,Activation::Softmax)
            ],0.5f32);
            // Sets training and testing data
            let data = vec![
                (vec![0f32,0f32],0usize),
                (vec![1f32,0f32],1usize),
                (vec![0f32,1f32],1usize),
                (vec![1f32,1f32],0usize)
            ];

            // Execution
            // ------------------------------------------------
            net.train(&data)
                .learning_rate(2f32)
                .evaluation_data(EvaluationData::Actual(&data)) // Use testing data as evaluation data.
                .early_stopping_condition(MeasuredCondition::Iteration(3000))
                //.log_interval(MeasuredCondition::Iteration(50))
            .go();

            //panic!("do we get here? outerside training");

            // Evaluation
            // ------------------------------------------------
            let evaluation = net.evaluate(&data,None);
            assert!(evaluation.1 as usize == data.len());
        }
    }
    // (2-Softmax->3-Softmax->2)
    #[test]
    fn train_xor_1() {
        let runs = 10 * TEST_RERUN_MULTIPLIER;
        for _ in 0..runs {
            // Setup
            // ------------------------------------------------
            // Sets network
            let mut net = NeuralNetwork::new(2,&[
                Layer::Dense(3,Activation::Sigmoid),
                Layer::Dense(2,Activation::Sigmoid)
            ]);
            // Sets training and testing data
            let data = vec![
                (vec![0f32,0f32],0usize),
                (vec![1f32,0f32],1usize),
                (vec![0f32,1f32],1usize),
                (vec![1f32,1f32],0usize)
            ];

            // Execution
            // ------------------------------------------------
            net.train(&data)
                .learning_rate(2f32)
                .evaluation_data(EvaluationData::Actual(&data)) // Use testing data as evaluation data.
                .early_stopping_condition(MeasuredCondition::Iteration(3000))
            .go();

            // Evaluation
            // ------------------------------------------------
            let evaluation = net.evaluate(&data,None);
            assert!(evaluation.1 as usize == data.len());
        }
    }

    // ReLU doesn't seem to work with such small networks
    // My idea is that it effectively leads to 0 activations which lead to 0 gradients which stop it learning

    // (2-Sigmoid->3-Softmax->2)
    #[test]
    fn train_xor_3() {
        let runs = 10 * TEST_RERUN_MULTIPLIER;
        for _ in 0..runs {
            // Setup
            // ------------------------------------------------
            // Sets network
            let mut net = NeuralNetwork::new(2,&[
                Layer::Dense(3,Activation::Sigmoid),
                Layer::Dense(2,Activation::Softmax)
            ]);

            // Sets training and testing data
            let data = vec![
                (vec![0f32,0f32],0usize),
                (vec![1f32,0f32],1usize),
                (vec![0f32,1f32],1usize),
                (vec![1f32,1f32],0usize)
            ];

            // Execution
            // ------------------------------------------------
            net.train(&data)
                .learning_rate(2f32)
                .evaluation_data(EvaluationData::Actual(&data)) // Use testing data as evaluation data.
                .early_stopping_condition(MeasuredCondition::Iteration(3000))
                .log_interval(MeasuredCondition::Iteration(100))
            .go();

            // Evaluation
            // ------------------------------------------------
            let evaluation = net.evaluate(&data,None);
            assert!(evaluation.1 as usize == data.len());
        }
    }

    // `train_digits_x` Tests network to recognize handwritten digits of 28x28 pixels (MNIST dataset).
    // --------------
    #[test] // (784-ReLU->800-Softmax->10) with dropout
    fn train_digits_0() {
        let runs = TEST_RERUN_MULTIPLIER;
        for _ in 0..runs {
            // Setup
            // ------------------------------------------------
            let mut net = NeuralNetwork::new(784,&[
                Layer::Dense(1000,Activation::ReLU),
                Layer::Dropout(0.2),
                Layer::Dense(500,Activation::ReLU),
                Layer::Dropout(0.2),
                Layer::Dense(10,Activation::Softmax)
            ]);

            // Sets training and testing data
            let training_data = get_mnist_dataset(false);
            let testing_data = get_mnist_dataset(true);

            // Execution
            // ------------------------------------------------
            net.train(&training_data)
                .evaluation_data(EvaluationData::Actual(&testing_data)) // Use testing data as evaluation data.
                .halt_condition(HaltCondition::Accuracy(TESTING_MIN_ACCURACY))
                //.tracking().log_interval(MeasuredCondition::Iteration(1))
            .go();

            // Evaluation
            // ------------------------------------------------
            let evaluation = net.evaluate(&testing_data,None);
            assert!(evaluation.1 >= required_accuracy(&testing_data));
        }
    }
    #[test] // (784-ReLU->800-Softmax->10) with L2
    fn train_digits_1() {
        let runs = TEST_RERUN_MULTIPLIER;
        for _ in 0..runs {
            // Setup
            // ------------------------------------------------
            let mut net = NeuralNetwork::new(784,&[
                Layer::Dense(1000,Activation::ReLU),
                Layer::Dense(500,Activation::ReLU),
                Layer::Dense(10,Activation::Softmax)
            ]);

            // Sets training and testing data
            let training_data = get_mnist_dataset(false);
            let testing_data = get_mnist_dataset(true);

            // Execution
            // ------------------------------------------------
            net.train(&training_data)
                .evaluation_data(EvaluationData::Actual(&testing_data)) // Use testing data as evaluation data.
                .halt_condition(HaltCondition::Accuracy(TESTING_MIN_ACCURACY))
                //.tracking().log_interval(MeasuredCondition::Iteration(1))
                .l2(0.1f32)
            .go();

            // Evaluation
            // ------------------------------------------------
            let evaluation = net.evaluate(&testing_data,None);
            assert!(evaluation.1 >= required_accuracy(&testing_data));
        }
    }
    #[test] // (784-ReLU->800-Softmax->10) with L2
    fn train_digits_2() {
        let runs = TEST_RERUN_MULTIPLIER;
        for _ in 0..runs {
            // Setup
            // ------------------------------------------------
            let mut net = NeuralNetwork::new(784,&[
                Layer::Dense(1000,Activation::ReLU),
                Layer::Dense(500,Activation::Sigmoid),
                Layer::Dense(10,Activation::Softmax)
            ]);

            // Sets training and testing data
            let training_data = get_mnist_dataset(false);
            let testing_data = get_mnist_dataset(true);

            // Execution
            // ------------------------------------------------
            net.train(&training_data)
                .evaluation_data(EvaluationData::Actual(&testing_data)) // Use testing data as evaluation data.
                .halt_condition(HaltCondition::Accuracy(TESTING_MIN_ACCURACY))
                //.tracking().log_interval(MeasuredCondition::Iteration(1))
                .l2(0.1f32)
            .go();

            // Evaluation
            // ------------------------------------------------
            let evaluation = net.evaluate(&testing_data,None);
            assert!(evaluation.1 >= required_accuracy(&testing_data));
        }
    }
    // Gets MNIST dataset.
    fn get_mnist_dataset(testing:bool) -> Vec<(Vec<f32>,usize)> {
        // Gets testing dataset.
        let (images,labels) = if testing {
            (get_images("data/MNIST/t10k-images.idx3-ubyte"),get_labels("data/MNIST/t10k-labels.idx1-ubyte"))
        }
        // Gets training dataset.
        else {
            (get_images("data/MNIST/train-images.idx3-ubyte"),get_labels("data/MNIST/train-labels.idx1-ubyte"))
        };

        let iterator = images.iter().zip(labels.iter());
        let mut examples = Vec::new();
        for (image,label) in iterator {
            examples.push(
                (
                    image.clone(),
                    *label as usize
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
}