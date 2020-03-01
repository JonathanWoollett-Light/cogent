#[cfg(test)]
mod tests {
    use std::time::Instant;
    use rust_neural_network::core::{HaltCondition,EvaluationData,MeasuredCondition,Activation,Layer,NeuralNetwork};
    use std::io::Read;
    use std::io::prelude::*;
    use std::fs::{File,OpenOptions};

    // TODO Figure out better name for this
    const TEST_RERUN_MULTIPLIER:u32 = 1; // Multiplies how many times we rerun tests (we rerun certain tests, due to random variation) (must be > 0)
    // TODO Figure out better name for this
    const TESTING_MIN_ACCURACY:f32 = 0.90f32; // approx 10% min inaccuracy
    // Returns `TESTING_MIN_ACCURACY` percentage as scaler as number of example in dataset.
    fn required_accuracy(test_data:&[(Vec<f32>,usize)]) -> u32 {
        ((test_data.len() as f32) * TESTING_MIN_ACCURACY).ceil() as u32
    }
    // Exports test result to `test_report.txt`.
    fn export_result(test:&str,runs:u32,dataset_length:u32,total_time:u64,total_accuracy:u32,) -> () {
        let avg_time = (total_time / runs as u64) as f32 / 60f32;
        let avg_accuracy = total_accuracy / runs;
        let avg_accuracy_percent = 100f32 * avg_accuracy as f32 / dataset_length as f32;
        let file = OpenOptions::new().append(true).open("test_report.txt");
        let result = format!("{} : {} * {:.2} mins, {}%, {}/{}\n",test,runs,avg_time,avg_accuracy_percent,avg_accuracy,dataset_length);
        
        file.unwrap().write_all(result.as_bytes());
    }
    // Tests network to learn an XOR gate.
    // Softmax output.
    #[test]
    fn train_xor_0() {
        let mut total_accuracy = 0u32;
        let mut total_time = 0u64;
        let runs = 10 * TEST_RERUN_MULTIPLIER;
        for _ in 0..runs {
            let start = Instant::now();

            // Setup
            // ------------------------------------------------
            // Sets network
            let mut neural_network = NeuralNetwork::new(2,&[
                Layer::new(3,Activation::Sigmoid),
                Layer::new(2,Activation::Softmax)
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
            neural_network.train(&data,2)
                .learning_rate(2f32)
                .evaluation_data(EvaluationData::Actual(&data)) // Use testing data as evaluation data.
                .lambda(0f32)
            .go();

            // Evaluation
            // ------------------------------------------------
            total_time += start.elapsed().as_secs();

            let evaluation = neural_network.evaluate(&data,2);
            assert!(evaluation.1 as usize == data.len());

            total_accuracy += evaluation.1;
        }
        export_result("train_xor_0",runs,4u32,total_time,total_accuracy);
    }
    // Tests network to learn an XOR gate.
    // Sigmoid output.
    #[test]
    fn train_xor_1() {
        let mut total_accuracy = 0u32;
        let mut total_time = 0u64;
        let runs = 10 * TEST_RERUN_MULTIPLIER;
        for _ in 0..runs {
            let start = Instant::now();

            // Setup
            // ------------------------------------------------
            // Sets network
            let mut neural_network = NeuralNetwork::new(2,&[
                Layer::new(3,Activation::Sigmoid),
                Layer::new(2,Activation::Sigmoid)
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
            neural_network.train(&data,2)
                .learning_rate(2f32)
                .evaluation_data(EvaluationData::Actual(&data)) // Use testing data as evaluation data.
                .lambda(0f32)
            .go();

            // Evaluation
            // ------------------------------------------------
            total_time += start.elapsed().as_secs();

            let evaluation = neural_network.evaluate(&data,2);
            assert!(evaluation.1 as usize == data.len());

            total_accuracy += evaluation.1;
        }
        export_result("train_xor_1",runs,4u32,total_time,total_accuracy);
    }
    // Tests network to recognize handwritten digits of 28x28 pixels (MNIST dataset).
    // Sigmoid output.
    #[test]
    fn train_digits_0() {
        let mut total_accuracy = 0u32;
        let mut total_time = 0u64;
        let runs = TEST_RERUN_MULTIPLIER;
        for _ in 0..runs {
            let start = Instant::now();

            // Setup
            // ------------------------------------------------
            // Sets network
            let mut neural_network = NeuralNetwork::new(784,&[
                Layer::new(100,Activation::Sigmoid),
                Layer::new(10,Activation::Sigmoid)
            ]);
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
            total_time += start.elapsed().as_secs();

            let evaluation = neural_network.evaluate(&testing_data,10);
            assert!(evaluation.1 >= required_accuracy(&testing_data));
            total_accuracy += evaluation.1;
        }
        export_result("train_digits_0",runs,10000u32,total_time,total_accuracy);
    }

    // This test fails, the cost becomes NAN, which means it is outside the floating point range, my guess is large weights and biases leading to too small activations.
    // Tests network to recognize handwritten digits of 28x28 pixels (MNIST dataset).
    // Softmax output.
    #[test]
    fn train_digits_1() {
        let mut total_accuracy = 0u32;
        let mut total_time = 0u64;
        let runs = TEST_RERUN_MULTIPLIER;
        for _ in 0..runs {
            let start = Instant::now();

            // Setup
            // ------------------------------------------------
            let mut neural_network = NeuralNetwork::new(784,&[
                Layer::new(100,Activation::Sigmoid),
                Layer::new(10,Activation::Softmax)
            ]);
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
            total_time += start.elapsed().as_secs();

            let evaluation = neural_network.evaluate(&testing_data,10);
            assert!(evaluation.1 >= required_accuracy(&testing_data));
            total_accuracy += evaluation.1;
        }
        export_result("train_digits_1",runs,10000u32,total_time,total_accuracy);
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