#[cfg(bench)]
mod benches {
    extern crate rust_neural_network;
    #[bench]
    fn train_full() {
        println!("test start");
        let runs = 1 * TEST_RERUN_MULTIPLIER;
        let mut total_accuracy = 0u32;
        let mut total_time = 0u64;
        for _ in 0..runs {
            let start = Instant::now();
            //Setup
            let training_data = get_combined("/home/jonathan/Projects/data/combined_dataset");
            let mut neural_network = NeuralNetwork::new(training_data[0].0.len(),&[
                Layer::new(training_data[0].0.len()+training_data[0].1.len(),Activation::Sigmoid),
                Layer::new(training_data[0].1.len(),Activation::Sigmoid)
            ]);
            //Execution
            neural_network.train(&training_data)
                .learning_rate(0.05f32)
                .log_interval(MeasuredCondition::Iteration(1u32))
                .checkpoint_interval(MeasuredCondition::Duration(Duration::new(1800,0)))
                .tracking()
            .go();
            //Evaluation
            total_time += start.elapsed().as_secs();
            let evaluated_outputs = neural_network.evaluate_outputs(&training_data);

            println!("{:.?}",evaluated_outputs.0)
            // assert!(evaluation.1 >= required_accuracy(&testing_data));
            // println!("train_full: accuracy: {}",evaluation.1);
            // println!();
            // total_accuracy += evaluation.1;
        }
        export_result("train_full",runs,450000,total_time,total_accuracy);
        assert!(false);
    }
    fn get_combined(path:&str) -> Vec<(Vec<f32>,Vec<f32>)> {
        let mut file = File::open(path).unwrap();
        let mut combined_buffer = Vec::new();
        file.read_to_end(&mut combined_buffer).expect("Couldn't read combined");

        let mut combined_vec = Vec::new();
        let multiplier:usize = (35*35)+1;
        let length = combined_buffer.len() / multiplier;
        println!("length: {}",length);
        let mut last_logged = Instant::now();
        for i in (0..length).rev() {
            let image_index = i * multiplier;
            let label_index = image_index + multiplier -1;

            let label:u8 = combined_buffer.split_off(label_index)[0]; // Array is only 1 element regardless
            let mut label_vec:Vec<f32> = vec!(0f32;95usize);
            label_vec[label as usize] = 1f32;

            let image:Vec<u8> = combined_buffer.split_off(image_index);
            let image_f32 = image.iter().map(|&x| x as f32).collect();

            combined_vec.push((image_f32,label_vec));

            
            if last_logged.elapsed().as_secs() > 5 {
                println!("{:.2}%",((length-i) as f32 / length as f32 *100f32));
                last_logged = Instant::now();
            }
        }
        return combined_vec;
    }

}