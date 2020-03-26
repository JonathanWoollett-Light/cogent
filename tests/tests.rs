#[cfg(test)]
mod tests {
    use cogent::core::{HaltCondition,EvaluationData,MeasuredCondition,Activation,Layer,NeuralNetwork};
    use std::io::Read;
    use std::fs::File;

    // TODO Figure out better name for this
    const TEST_RERUN_MULTIPLIER:u32 = 1; // Multiplies how many times we rerun tests (we rerun certain tests, due to random variation) (must be > 0)
    // TODO Figure out better name for this
    const TESTING_MIN_ACCURACY:f32 = 0.90f32; // approx 10% min inaccuracy
    // Returns `TESTING_MIN_ACCURACY` percentage as scaler as number of example in dataset.
    fn required_accuracy(test_data:&[(Vec<f32>,usize)]) -> u32 {
        ((test_data.len() as f32) * TESTING_MIN_ACCURACY).ceil() as u32
    }
    // Tests changing activation of layer using out of range index.
    #[test]
    #[should_panic="Layer 2 does not exist. 0 <= given index < 2"]
    fn activation() {
        
        let mut net = NeuralNetwork::new(2,&[
            Layer::new(3,Activation::Sigmoid),
            Layer::new(2,Activation::Sigmoid)
        ],None);
        net.activation(2,Activation::Softmax); // Changes activation of output layer.
    }
    // Tests network to learn an XOR gate.
    // Sigmoid
    #[test]
    fn train_xor_0() {
        let runs = 10 * TEST_RERUN_MULTIPLIER;
        
        for _ in 0..runs {
            // Setup
            // ------------------------------------------------
            // Sets network
            let mut neural_network = NeuralNetwork::new(2,&[
                Layer::new(3,Activation::Sigmoid),
                Layer::new(2,Activation::Sigmoid)
            ],None);
            // Sets training and testing data
            let data = vec![
                (vec![0f32,0f32],0usize),
                (vec![1f32,0f32],1usize),
                (vec![0f32,1f32],1usize),
                (vec![1f32,1f32],0usize)
            ];
            // Execution
            // ------------------------------------------------
            neural_network.train(&data,2)
                .learning_rate(2f32)
                .evaluation_data(EvaluationData::Actual(&data)) // Use testing data as evaluation data.
                .early_stopping_condition(MeasuredCondition::Iteration(3000))
                .lambda(0f32)
            .go();


            // Evaluation
            // ------------------------------------------------
            let evaluation = neural_network.evaluate(&data,2);
            assert!(evaluation.1 as usize == data.len());
        }
    }
    // Tests network to learn an XOR gate.
    // Softmax
    #[test]
    fn train_xor_1() {
        let runs = 10 * TEST_RERUN_MULTIPLIER;
        for _ in 0..runs {
            // Setup
            // ------------------------------------------------
            // Sets network
            let mut neural_network = NeuralNetwork::new(2,&[
                Layer::new(3,Activation::Sigmoid),
                Layer::new(2,Activation::Sigmoid)
            ],None);
            // Sets training and testing data
            let data = vec![
                (vec![0f32,0f32],0usize),
                (vec![1f32,0f32],1usize),
                (vec![0f32,1f32],1usize),
                (vec![1f32,1f32],0usize)
            ];

            // Execution
            // ------------------------------------------------
            neural_network.train(&data,2)
                .learning_rate(2f32)
                .evaluation_data(EvaluationData::Actual(&data)) // Use testing data as evaluation data.
                .early_stopping_condition(MeasuredCondition::Iteration(3000))
                .lambda(0f32)
            .go();

            // Evaluation
            // ------------------------------------------------
            let evaluation = neural_network.evaluate(&data,2);
            assert!(evaluation.1 as usize == data.len());
        }
    }

    // ReLU doesn't seem to work with such small networks
    // My idea is that it effectively leads to 0 activations which lead to 0 gradients which stop it learning

    // Tests network to learn an XOR gate.
    // Mixed (Sigmoid->Softmax)
    #[test]
    fn train_xor_3() {
        let runs = 10 * TEST_RERUN_MULTIPLIER;
        for _ in 0..runs {
            // Setup
            // ------------------------------------------------
            // Sets network
            let mut neural_network = NeuralNetwork::new(2,&[
                Layer::new(3,Activation::Sigmoid),
                Layer::new(2,Activation::Softmax)
            ],None);
            // Sets training and testing data
            let data = vec![
                (vec![0f32,0f32],0usize),
                (vec![1f32,0f32],1usize),
                (vec![0f32,1f32],1usize),
                (vec![1f32,1f32],0usize)
            ];

            // Execution
            // ------------------------------------------------
            neural_network.train(&data,2)
                .learning_rate(2f32)
                .evaluation_data(EvaluationData::Actual(&data)) // Use testing data as evaluation data.
                .early_stopping_condition(MeasuredCondition::Iteration(3000))
                .lambda(0f32)
            .go();

            // Evaluation
            // ------------------------------------------------
            let evaluation = neural_network.evaluate(&data,2);
            assert!(evaluation.1 as usize == data.len());
        }
    }
    // Tests network to recognize handwritten digits of 28x28 pixels (MNIST dataset).
    // Sigmoid->Softmax
    #[test]
    fn train_digits_0() {
        let runs = TEST_RERUN_MULTIPLIER;
        for _ in 0..runs {
            // Setup
            // ------------------------------------------------
            // Sets network
            let mut neural_network = NeuralNetwork::new(784,&[
                Layer::new(100,Activation::Sigmoid),
                Layer::new(10,Activation::Softmax)
            ],None);
            // Sets training and testing data
            let training_data = get_mnist_dataset(false);
            let testing_data = get_mnist_dataset(true);

            // Execution
            // ------------------------------------------------
            neural_network.train(&training_data,10)
                .evaluation_data(EvaluationData::Actual(&testing_data)) // Use testing data as evaluation data.
                .halt_condition(HaltCondition::Accuracy(0.95f32))
            .go();

            // Evaluation
            // ------------------------------------------------

            let evaluation = neural_network.evaluate(&testing_data,10);
            assert!(evaluation.1 >= required_accuracy(&testing_data));
        }
    }
    // Tests network to recognize handwritten digits of 28x28 pixels (MNIST dataset).
    // ReLU->Softmax
    #[test]
    fn train_digits_1() {
        let runs = TEST_RERUN_MULTIPLIER;
        for _ in 0..runs {
            // Setup
            // ------------------------------------------------
            let mut neural_network = NeuralNetwork::new(784,&[
                Layer::new(100,Activation::ReLU),
                Layer::new(10,Activation::Softmax) // You can't have a ReLU output layer
            ],None);
            // Sets training and testing data
            let training_data = get_mnist_dataset(false);
            let testing_data = get_mnist_dataset(true);

            // Execution
            // ------------------------------------------------
            neural_network.train(&training_data,10)
                .evaluation_data(EvaluationData::Actual(&testing_data)) // Use testing data as evaluation data.
                .halt_condition(HaltCondition::Accuracy(0.95f32))
            .go();

            // Evaluation
            // ------------------------------------------------
            let evaluation = neural_network.evaluate(&testing_data,10);
            assert!(evaluation.1 >= required_accuracy(&testing_data));
        }
    }
    // Tests network to recognize handwritten digits of 28x28 pixels (MNIST dataset).
    // Mixed (ReLU -> Sigmoid -> Softmax)
    #[test]
    fn train_digits_2() {
        let runs = TEST_RERUN_MULTIPLIER;
        for _ in 0..runs {
            // Setup
            // ------------------------------------------------
            let mut neural_network = NeuralNetwork::new(784,&[
                Layer::new(300,Activation::ReLU),
                Layer::new(100,Activation::Sigmoid),
                Layer::new(10,Activation::Softmax)
            ],None);
            // Sets training and testing data
            let training_data = get_mnist_dataset(false);
            let testing_data = get_mnist_dataset(true);

            // Execution
            // ------------------------------------------------
            neural_network.train(&training_data,10)
                .evaluation_data(EvaluationData::Actual(&testing_data)) // Use testing data as evaluation data.
                .halt_condition(HaltCondition::Accuracy(0.95f32))
            .go();

            // Evaluation
            // ------------------------------------------------
            let evaluation = neural_network.evaluate(&testing_data,10);
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