#[cfg(test)]
mod tests {
    use cogent::core::{
        HaltCondition,EvaluationData,MeasuredCondition,Proportion,
        Activation,Layer,
        NeuralNetwork
    };
    
    use arrayfire::{Array,Dim4,HasAfEnum};

    use std::{
        io::Read,
        fs::File
    };
    use itertools::izip;

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
    #[test]
    fn relu_run() {
        let data = Array::new(
            &[
                0.,0.,
                1.,0.,
                0.,1.,
                1.,1.
            ],
            Dim4::new(&[2,4,1,1])
        );

        let activation = Activation::ReLU;

        let observed = activation.run(&data);

        let predicted = data;

        assert_eq!(to_vec(observed),to_vec(predicted));
    }
    #[test]
    fn relu_run_randn() {
        let data:Array<f32> = Array::new(
            &[
                -0.92466253,  0.1808258  ,  2.5440972  ,  0.35158235 , -0.34516734,
                 0.2191234 , -0.76866555 ,  0.24125518 , -1.1948396  ,  0.8926924 ,
                -0.5377797 ,  0.22698349 ,  0.9353601  , -0.7612852  ,  0.5787065 ,
                -0.6173673 ,  0.58894515 ,  0.78968173 , -0.064518586,  0.9520342 ,
                -1.1410632 ,  0.8281328  , -0.7363409  , -0.7445752  , -0.84145886,
                 1.2990668 ,  0.088310346,  0.6588016  , -0.27883467 ,  1.4143448 ,
                -1.1649874 , -0.66291463 , -0.58066535 , -0.16993414 , -0.72647095,
                -0.15060124, -0.2785236  , -0.006335576,  0.40205446 ,  1.3925238 ,
                -0.24172749,  0.11392703 , -1.627895   ,  0.1487814  ,  0.25124982,
                 0.6435624 , -2.365118   , -0.7734198  , -0.05110575 ,  1.6692852
            ],
            Dim4::new(&[5,10,1,1])
        );

        let activation = Activation::ReLU;

        let observed = activation.run(&data);

        let predicted:Array<f32> = Array::new(
            &[
                0.       , 0.1808258  , 2.5440972 , 0.35158235, 0.        , 
                0.2191234, 0.         , 0.24125518, 0.        , 0.8926924 ,
                0.       , 0.22698349 , 0.9353601 , 0.        , 0.5787065 ,
                0.       , 0.58894515 , 0.78968173, 0.        , 0.9520342 ,
                0.       , 0.8281328  , 0.        , 0.        , 0.        ,
                1.2990668, 0.088310346, 0.6588016 , 0.        , 1.4143448 ,
                0.       , 0.         , 0.        , 0.        , 0.        ,
                0.       , 0.         , 0.        , 0.40205446, 1.3925238 ,
                0.       , 0.11392703 , 0.        , 0.1487814 , 0.25124982,
                0.6435624, 0.         , 0.        , 0.        , 1.6692852
            ],
            Dim4::new(&[5,10,1,1])
        );

        assert_eq!(to_vec(observed),to_vec(predicted));
    }
    #[test]
    fn relu_run_randn_big() {
        let data = Array::<f32>::new(
            &[
                -0.92466253,  0.1808258  ,  2.5440972  ,  0.35158235 , -0.34516734,
                 0.2191234 , -0.76866555 ,  0.24125518 , -1.1948396  ,  0.8926924,
                -0.5377797 ,  0.22698349 ,  0.9353601  , -0.7612852  ,  0.5787065,
                -0.6173673 ,  0.58894515 ,  0.78968173 , -0.064518586,  0.9520342,
                -1.1410632 ,  0.8281328  , -0.7363409  , -0.7445752  , -0.84145886,
                 1.2990668 ,  0.088310346,  0.6588016  , -0.27883467 ,  1.4143448,
                -1.1649874 , -0.66291463 , -0.58066535 , -0.16993414 , -0.72647095,
                -0.15060124, -0.2785236  , -0.006335576,  0.40205446 ,  1.3925238,
                -0.24172749,  0.11392703 , -1.627895   ,  0.1487814  ,  0.25124982,
                 0.6435624 , -2.365118   , -0.7734198  , -0.05110575 ,  1.6692852
            ],
            Dim4::new(&[5,10,1,1])
        ) * 10e+30f32;
        
        let activation = Activation::ReLU;

        let observed = activation.run(&data) * 10e-32f32;

        let predicted:Array<f32> = Array::new(
            &[
                0.       , 0.1808258  , 2.5440972 , 0.35158235, 0.        , 
                0.2191234, 0.         , 0.24125518, 0.        , 0.8926924 ,
                0.       , 0.22698349 , 0.9353601 , 0.        , 0.5787065 ,
                0.       , 0.58894515 , 0.78968173, 0.        , 0.9520342 ,
                0.       , 0.8281328  , 0.        , 0.        , 0.        ,
                1.2990668, 0.088310346, 0.6588016 , 0.        , 1.4143448 ,
                0.       , 0.         , 0.        , 0.        , 0.        ,
                0.       , 0.         , 0.        , 0.40205446, 1.3925238 ,
                0.       , 0.11392703 , 0.        , 0.1487814 , 0.25124982,
                0.6435624, 0.         , 0.        , 0.        , 1.6692852
            ],
            Dim4::new(&[5,10,1,1])
        );

        close_enough(to_vec(observed),to_vec(predicted),10e-5);
    }
    #[test]
    fn relu_run_randn_small() {
        let data = Array::<f32>::new(
            &[
                -0.92466253,  0.1808258  ,  2.5440972  ,  0.35158235 , -0.34516734,
                 0.2191234 , -0.76866555 ,  0.24125518 , -1.1948396  ,  0.8926924,
                -0.5377797 ,  0.22698349 ,  0.9353601  , -0.7612852  ,  0.5787065,
                -0.6173673 ,  0.58894515 ,  0.78968173 , -0.064518586,  0.9520342,
                -1.1410632 ,  0.8281328  , -0.7363409  , -0.7445752  , -0.84145886,
                 1.2990668 ,  0.088310346,  0.6588016  , -0.27883467 ,  1.4143448,
                -1.1649874 , -0.66291463 , -0.58066535 , -0.16993414 , -0.72647095,
                -0.15060124, -0.2785236  , -0.006335576,  0.40205446 ,  1.3925238,
                -0.24172749,  0.11392703 , -1.627895   ,  0.1487814  ,  0.25124982,
                 0.6435624 , -2.365118   , -0.7734198  , -0.05110575 ,  1.6692852
            ],
            Dim4::new(&[5,10,1,1])
        ) * 10e-30f32;

        let activation = Activation::ReLU;

        let observed = activation.run(&data) * 10e+28f32;

        let predicted:Array<f32> = Array::new(
            &[
                0.       , 0.1808258  , 2.5440972 , 0.35158235, 0.        , 
                0.2191234, 0.         , 0.24125518, 0.        , 0.8926924 ,
                0.       , 0.22698349 , 0.9353601 , 0.        , 0.5787065 ,
                0.       , 0.58894515 , 0.78968173, 0.        , 0.9520342 ,
                0.       , 0.8281328  , 0.        , 0.        , 0.        ,
                1.2990668, 0.088310346, 0.6588016 , 0.        , 1.4143448 ,
                0.       , 0.         , 0.        , 0.        , 0.        ,
                0.       , 0.         , 0.        , 0.40205446, 1.3925238 ,
                0.       , 0.11392703 , 0.        , 0.1487814 , 0.25124982,
                0.6435624, 0.         , 0.        , 0.        , 1.6692852
            ],
            Dim4::new(&[5,10,1,1])
        );

        close_enough(to_vec(observed),to_vec(predicted),10e-5);
    }
    #[test]
    fn sigmoid_run() {
        let data:Array<f32> = Array::new(
            &[
                0.,0.,
                1.,0.,
                0.,1.,
                1.,1.
            ],
            Dim4::new(&[2,4,1,1])
        );
        let activation = Activation::Sigmoid;

        let observed = activation.run(&data);

        let predicted:Array<f32> = Array::new(
            &[
                0.5      ,0.5      ,
                0.7310586,0.5      ,
                0.5      ,0.7310586,
                0.7310586,0.7310586
            ],
            Dim4::new(&[2,4,1,1])
        );

        assert_eq!(to_vec(observed),to_vec(predicted));
    }
    #[test]
    fn sigmoid_run_randn() {
        let data:Array<f32> = Array::new(
            &[
                -0.92466253,  0.1808258  ,  2.5440972  ,  0.35158235 , -0.34516734,
                 0.2191234 , -0.76866555 ,  0.24125518 , -1.1948396  ,  0.8926924 ,
                -0.5377797 ,  0.22698349 ,  0.9353601  , -0.7612852  ,  0.5787065 ,
                -0.6173673 ,  0.58894515 ,  0.78968173 , -0.064518586,  0.9520342 ,
                -1.1410632 ,  0.8281328  , -0.7363409  , -0.7445752  , -0.84145886,
                 1.2990668 ,  0.088310346,  0.6588016  , -0.27883467 ,  1.4143448 ,
                -1.1649874 , -0.66291463 , -0.58066535 , -0.16993414 , -0.72647095,
                -0.15060124, -0.2785236  , -0.006335576,  0.40205446 ,  1.3925238 ,
                -0.24172749,  0.11392703 , -1.627895   ,  0.1487814  ,  0.25124982,

                 0.6435624 , -2.365118   , -0.7734198  , -0.05110575 ,  1.6692852
            ],
            Dim4::new(&[5,10,1,1])
        );

        let activation = Activation::Sigmoid;

        let observed = activation.run(&data);

        let predicted:Array<f32> = Array::new(
            &[
                0.284008825 , 0.5450836715 , 0.927175957 , 0.587001242 , 0.414554817 , 
                0.5545627052, 0.3167678453 , 0.5600229454, 0.2323944948, 0.7094454768,
                0.3687042316, 0.5565034852 , 0.7181614672, 0.3183673004, 0.6407697177,
                0.3503804545, 0.6431230775 , 0.6877629878, 0.4838759463, 0.721524088 ,
                0.2421252097, 0.6959599755 , 0.3238048058, 0.3220044848, 0.3012276202,
                0.7856778851, 0.5220632496 , 0.6589911332, 0.4307394954, 0.8044503283,
                0.2377622301, 0.3400851846 , 0.358779511 , 0.4576184059, 0.3259696304,
                0.4624206904, 0.4308157723 , 0.4984161113, 0.599181166 , 0.8009948478,
                0.439860682 , 0.5284509912 , 0.1641189292, 0.5371268888, 0.5624841002,
                0.6555583014, 0.08587159102, 0.3157397996, 0.4872263426, 0.8414804965
            ],
            Dim4::new(&[5,10,1,1])
        );

        close_enough(to_vec(observed),to_vec(predicted),10e-5);
    }
    #[test]
    fn sigmoid_run_randn_big() {
        let data = Array::<f32>::new(
            &[
                -0.92466253,  0.1808258  ,  2.5440972  ,  0.35158235 , -0.34516734,
                 0.2191234 , -0.76866555 ,  0.24125518 , -1.1948396  ,  0.8926924,
                -0.5377797 ,  0.22698349 ,  0.9353601  , -0.7612852  ,  0.5787065,
                -0.6173673 ,  0.58894515 ,  0.78968173 , -0.064518586,  0.9520342,
                -1.1410632 ,  0.8281328  , -0.7363409  , -0.7445752  , -0.84145886,
                 1.2990668 ,  0.088310346,  0.6588016  , -0.27883467 ,  1.4143448,
                -1.1649874 , -0.66291463 , -0.58066535 , -0.16993414 , -0.72647095,
                -0.15060124, -0.2785236  , -0.006335576,  0.40205446 ,  1.3925238,
                -0.24172749,  0.11392703 , -1.627895   ,  0.1487814  ,  0.25124982,
                 0.6435624 , -2.365118   , -0.7734198  , -0.05110575 ,  1.6692852
            ],
            Dim4::new(&[5,10,1,1])
        ) * 10f32;
        
        let activation = Activation::Sigmoid;

        let observed = activation.run(&data);

        let predicted:Array<f32> = Array::new(
            &[
                0.000096427284, 0.8591512          , 1.0             , 0.9711347     , 0.030718993  , 
                0.89945954    , 0.0004586999       , 0.91777945      , 0.000006469562, 0.9998672    , 
                0.0045967554  , 0.9063478          , 0.99991333      , 0.00049381674 , 0.9969424    , 
                0.0020792366  , 0.9972391          , 0.9996282       , 0.3440752     , 0.9999267    , 
                0.00001107696 , 0.99974686         , 0.0006336316    , 0.0005835759  , 0.00022156148,
                0.99999774    , 0.7074649          , 0.99862516      , 0.057957154   , 0.9999993    ,
                0.000008720069, 0.0013195467       , 0.0029984599    , 0.1545513     , 0.00069931516, 
                0.1815305     , 0.058127232        , 0.48416635      , 0.9823731     , 0.99999905   ,
                0.08186485    , 0.75754565         , 0.00000008514163, 0.81574994    , 0.92501336   , 
                0.9983992     , 0.00000000005350852, 0.00043741177   , 0.37494564    , 1.0
            ],
            Dim4::new(&[5,10,1,1])
        );

        close_enough(to_vec(observed),to_vec(predicted),10e-15);
    }
    #[test]
    fn sigmoid_run_randn_small() {
        let data = Array::<f32>::new(
            &[
                -0.92466253,  0.1808258  ,  2.5440972  ,  0.35158235 , -0.34516734,
                 0.2191234 , -0.76866555 ,  0.24125518 , -1.1948396  ,  0.8926924,
                -0.5377797 ,  0.22698349 ,  0.9353601  , -0.7612852  ,  0.5787065,
                -0.6173673 ,  0.58894515 ,  0.78968173 , -0.064518586,  0.9520342,
                -1.1410632 ,  0.8281328  , -0.7363409  , -0.7445752  , -0.84145886,
                 1.2990668 ,  0.088310346,  0.6588016  , -0.27883467 ,  1.4143448,
                -1.1649874 , -0.66291463 , -0.58066535 , -0.16993414 , -0.72647095,
                -0.15060124, -0.2785236  , -0.006335576,  0.40205446 ,  1.3925238,
                -0.24172749,  0.11392703 , -1.627895   ,  0.1487814  ,  0.25124982,
                 0.6435624 , -2.365118   , -0.7734198  , -0.05110575 ,  1.6692852
            ],
            Dim4::new(&[5,10,1,1])
        ) * 10e-10f32;

        let activation = Activation::Sigmoid;

        let observed = activation.run(&data);

        let predicted:Array<f32> = Array::new(
            &[
                0.49999999976883436, 0.5000000000452065 , 0.5000000006360243 , 0.5000000000878956 , 0.49999999991370814, 
                0.5000000000547808 , 0.4999999998078336 , 0.5000000000603138 , 0.49999999970129005, 0.5000000002231731 , 
                0.4999999998655551 , 0.5000000000567459 , 0.5000000002338401 , 0.4999999998096787 , 0.5000000001446766 , 
                0.4999999998456581 , 0.5000000001472363 , 0.5000000001974204 , 0.49999999998387035, 0.5000000002380086 , 
                0.4999999997147342 , 0.5000000002070332 , 0.4999999998159148 , 0.49999999981385623, 0.4999999997896353 , 
                0.5000000003247667 , 0.5000000000220776 , 0.5000000001647004 , 0.4999999999302913 , 0.5000000003535862 , 
                0.4999999997087532 , 0.49999999983427135, 0.4999999998548337 , 0.49999999995751643, 0.4999999998183823 , 
                0.4999999999623497 , 0.49999999993036903, 0.49999999999841616, 0.5000000001005136 , 0.500000000348131  , 
                0.4999999999395681 , 0.5000000000284818 , 0.4999999995930262 , 0.5000000000371954 , 0.5000000000628124 , 
                0.5000000001608906 , 0.49999999940872053, 0.4999999998066451 , 0.49999999998722355, 0.5000000004173213
            ],
            Dim4::new(&[5,10,1,1])
        );

        close_enough(to_vec(observed),to_vec(predicted),10e-10);
    }
    #[test]
    fn softmax_run() {
        let data:Array<f32> = Array::new(
            &[
                0.,0.,
                1.,0.,
                0.,1.,
                1.,1.
            ],
            Dim4::new(&[2,4,1,1])
        );

        let activation = Activation::Softmax;

        let observed = activation.run(&data);

        let predicted:Array<f32> = Array::new(
            &[
                0.5       ,0.5       ,
                0.7310586 ,0.26894143,
                0.26894143,0.7310586 ,
                0.5       ,0.5
            ],
            Dim4::new(&[2,4,1,1])
        );

        assert_eq!(to_vec(observed),to_vec(predicted));
    }
    #[test]
    fn softmax_run_randu() {
        let data:Array<f32> = Array::new(
            &[
                -0.92466253,  0.1808258  ,  2.5440972  ,  0.35158235 , -0.34516734,
                 0.2191234 , -0.76866555 ,  0.24125518 , -1.1948396  ,  0.8926924 ,
                -0.5377797 ,  0.22698349 ,  0.9353601  , -0.7612852  ,  0.5787065 ,
                -0.6173673 ,  0.58894515 ,  0.78968173 , -0.064518586,  0.9520342 ,
                -1.1410632 ,  0.8281328  , -0.7363409  , -0.7445752  , -0.84145886,
                 1.2990668 ,  0.088310346,  0.6588016  , -0.27883467 ,  1.4143448 ,
                -1.1649874 , -0.66291463 , -0.58066535 , -0.16993414 , -0.72647095,
                -0.15060124, -0.2785236  , -0.006335576,  0.40205446 ,  1.3925238 ,
                -0.24172749,  0.11392703 , -1.627895   ,  0.1487814  ,  0.25124982,
                 0.6435624 , -2.365118   , -0.7734198  , -0.05110575 ,  1.6692852
            ],
            Dim4::new(&[5,10,1,1])
        );

        let activation = Activation::Softmax;

        let observed = activation.run(&data);

        let predicted:Array<f32> = Array::new(
            &[
                0.02410457 , 0.072812654, 0.77368224 , 0.08637051 , 0.043029964, 
                0.21743007 , 0.08097078 , 0.22229582 , 0.052874133, 0.42642915 , 
                0.087987795, 0.18904051 , 0.38388303 , 0.070364766, 0.26872385 , 
                0.06681366 , 0.2232339  , 0.27285942 , 0.116135344, 0.32095763 , 
                0.08000179 , 0.57320595 , 0.119913585, 0.11893023 , 0.10794841 , 
                0.3170861  , 0.09448272 , 0.16715276 , 0.065449044, 0.3558294  , 
                0.1149268  , 0.18987542 , 0.20615277 , 0.3108619  , 0.17818314 , 
                0.10579587 , 0.09309209 , 0.12221444 , 0.18385865 , 0.495039   , 
                0.17265008 , 0.2463914  , 0.043167997, 0.25513065 , 0.28265983 , 
                0.21833338 , 0.010776227, 0.05293374 , 0.10900077 , 0.6089559
            ],
            Dim4::new(&[5,10,1,1])
        );

        close_enough(to_vec(observed),to_vec(predicted),10e-10);
    }
    // TODO Name this better
    // Checks all percentage differences between all value pairs from `vec_1` and `vec_2` are less than `allowed_inequality`.
    fn close_enough(vec_1:Vec<f32>,vec_2:Vec<f32>,allowed_inequality:f32) {
        for (a,b) in izip!(vec_1,vec_2) {
            assert!((a-b).abs() <= allowed_inequality,format!("{} dif {} > {}",a,b,allowed_inequality))
        }
    }
    fn to_vec<T:HasAfEnum+Default+Clone>(array:Array<T>) -> Vec<T> {
        let mut vec = vec!(T::default();array.elements());
        array.host(&mut vec);
        return vec;
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
                //.log_interval(MeasuredCondition::Iteration(100))
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
                .tracking().log_interval(MeasuredCondition::Iteration(1))
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