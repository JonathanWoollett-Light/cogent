#![feature(test)]
extern crate test;

#[cfg(test)]
mod benches {
    use test::Bencher;

    extern crate rust_neural_network;
    use rust_neural_network::core::{HaltCondition,EvaluationChange,MeasuredCondition,Activation,Layer,NeuralNetwork};
    
    use std::time::{Instant,Duration};
    
    use std::fs::File;
    use std::io::Read;

    #[bench]
    fn combined_dataset_training(b: &mut Bencher) {
        let combined_dataset = get_combined("data/combined_dataset");
        b.iter(|| train_on_combined(&combined_dataset));
    }
    fn train_on_combined(combined_dataset: &Vec<(Vec<f32>,Vec<f32>)>) {
        //Setup
        let training_data = combined_dataset;
        let mut neural_network = NeuralNetwork::new(training_data[0].0.len(),&[
            Layer::new(training_data[0].0.len()+training_data[0].1.len(),Activation::Sigmoid),
            Layer::new(training_data[0].1.len(),Activation::Sigmoid)
        ]);
        //Execution
        neural_network.train(training_data)
            .learning_rate(0.05f32)
            .halt_condition(HaltCondition::Accuracy(0.95f32))
            .early_stopping_condition(MeasuredCondition::Duration(Duration::new(600,0)))
            .evaluation_min_change(EvaluationChange::Percent(0.1f32))
        .go();
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