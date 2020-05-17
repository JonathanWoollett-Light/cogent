/// Core functionality of training a neural network.
pub mod core {
    use rand::prelude::SliceRandom;
    use itertools::izip;

    // TODO Is this really a good way to include these?
    use arrayfire::{
        Array, randu, Dim4, matmul, MatProp, constant, sigmoid, cols, col, exp, maxof, sum, pow,
        transpose, imax, eq, sum_all, log, diag_extract, sum_by_key, mul,div,sub, gt, and, max
        ,mem_info,device_mem_info,print_gen,af_print,add
    };

    use crossterm::{QueueableCommand, cursor};

    use serde::{Serialize, Deserialize};

    use std::{
        time::{Duration, Instant},
        io::{Read, Write, stdout},
        fs::{create_dir, remove_dir_all, File},
        path::Path,
        collections::HashMap
    };

    // Default percentage of training data to set as evaluation data (0.1=5%).
    const DEFAULT_EVALUTATION_DATA:f32 = 0.05f32;
    // Default percentage of size of training data to set batch size (0.01=1%).
    const DEFAULT_BATCH_SIZE:f32 = 0.01f32;
    // Default learning rate.
    const DEFAULT_LEARNING_RATE:f32 = 0.1f32;
    // Default interval in iterations before early stopping.
    // early stopping = default early stopping * (size of examples / number of examples) Iterations
    const DEFAULT_EARLY_STOPPING:f32 = 400f32;
    // Default percentage minimum positive accuracy change required to prevent early stopping or learning rate decay (0.005=0.5%).
    const DEFAULT_EVALUATION_MIN_CHANGE:f32 = 0.001f32;
    // Default amount to decay learning rate after period of un-notable (what word should I use here?) change.
    // `new learning rate = learning rate decay * old learning rate`
    const DEFAULT_LEARNING_RATE_DECAY:f32 = 0.5f32;
    // Default interval in iterations before learning rate decay.
    // interval = default learning rate interval * (size of examples / number of examples) iterations.
    const DEFAULT_LEARNING_RATE_INTERVAL:f32 = 200f32;

    /// For setting `evaluation_data`.
    pub enum EvaluationData<'a> {
        /// Set as a given number of examples from training data.
        Scalar(usize),
        /// Set as a given percentage of examples from training data.
        Percent(f32),
        /// Set as a given dataset.
        Actual(&'a Vec<(Vec<f32>,usize)>)
    }
    /// For setting a hyperparameter with measured intervals.
    #[derive(Clone,Copy)]
    pub enum MeasuredCondition {
        Iteration(u32),
        Duration(Duration)
    }
    /// For setting `halt_condition`.
    ///
    /// The training halt condition.
    #[derive(Clone,Copy)]
    pub enum HaltCondition {
        /// Halt after completing a given number of iterations (epochs)
        Iteration(u32),
        /// Halt after a given duration has elapsed.
        Duration(Duration),
        /// Halt after acheiving a given accuracy.
        Accuracy(f32)
    }
    /// For setting a hyperparameter as a proportion of another.
    #[derive(Clone,Copy)]
    pub enum Proportion {
        Scalar(u32),
        Percent(f32),
    }
    
    /// To practicaly implement optional setting of training hyperparameters.
    pub struct Trainer<'a> {
        training_data: Vec<(Vec<f32>,usize)>,
        evaluation_data: EvaluationData<'a>,
        cost:Cost,
        // Will halt after at a certain iteration, accuracy or duration.
        halt_condition: Option<HaltCondition>,
        // Can log after a certain number of iterations, a certain duration, or not at all.
        log_interval: Option<MeasuredCondition>,
        batch_size: usize,
        learning_rate: f32,
        // Lambda value if using L2
        l2:Option<f32>, 
        // Can stop after a lack of cost improvement over a certain number of iterations/durations, or not at all.
        early_stopping_condition: MeasuredCondition,
        // Minimum change required to log positive evaluation change.
        evaluation_min_change: Proportion, 
        // Amount to decrease learning rate by (less than 1)(`learning_rate` *= learning_rate_decay`).
        learning_rate_decay: f32, 
        // Time without notable improvement to wait until decreasing learning rate.
        learning_rate_interval: MeasuredCondition,
        // Duration/iterations between outputting neural network weights and biases to file.
        checkpoint_interval: Option<MeasuredCondition>,
        // Sets what to pretend to checkpoint files. Used to differentiate between nets when checkpointing multiple.
        name: Option<&'a str>,
        // Whether to print percantage progress in each iteration of backpropagation
        tracking: bool,
        neural_network: &'a mut NeuralNetwork
    }
    impl<'a> Trainer<'a> {
        /// Sets `evaluation_data`.
        /// 
        /// `evaluation_data` determines how to set the evaluation data.
        pub fn evaluation_data(&mut self, evaluation_data:EvaluationData<'a>) -> &mut Trainer<'a> {
            self.evaluation_data = evaluation_data;
            return self;
        }
        /// Sets `cost`.
        /// 
        /// `cost` determines cost function of network.
        pub fn cost(&mut self, cost:Cost) -> &mut Trainer<'a> {
            self.cost = cost;
            return self;
        }
        /// Sets `halt_condition`.
        /// 
        /// `halt_condition` sets after which Iteration/Duration or reached accuracy to stop training.
        pub fn halt_condition(&mut self, halt_condition:HaltCondition) -> &mut Trainer<'a> {
            self.halt_condition = Some(halt_condition);
            return self;
        }
        /// Sets `log_interval`.
        /// 
        /// `log_interval` sets some amount of Iterations/Duration to print the cost and accuracy of the neural net.
        pub fn log_interval(&mut self, log_interval:MeasuredCondition) -> &mut Trainer<'a> {
            self.log_interval = Some(log_interval);
            return self;
        }
        /// Sets `batch_size`.
        pub fn batch_size(&mut self, batch_size:Proportion) -> &mut Trainer<'a> {
            self.batch_size = match batch_size {
                Proportion::Percent(percent) => { (self.training_data.len() as f32 * percent) as usize },
                Proportion::Scalar(scalar) => { scalar as usize } 
            };
            return self;
        }
        /// Sets `learning_rate`.
        pub fn learning_rate(&mut self, learning_rate:f32) -> &mut Trainer<'a> {
            self.learning_rate = learning_rate;
            return self;
        }
        /// Sets lambda ($ \lambda $) for `l2`.
        /// 
        /// If $ \lambda $ set, implements L2 regularization with $ \lambda $ value.
        pub fn l2(&mut self, lambda:f32) -> &mut Trainer<'a> {
            self.l2 = Some(lambda);
            return self;
        }
        /// Sets `early_stopping_condition`.
        /// 
        /// `early_stopping_condition` sets some amount of Iterations/Duration to stop after without notable cost improvement.
        pub fn early_stopping_condition(&mut self, early_stopping_condition:MeasuredCondition) -> &mut Trainer<'a> {
            self.early_stopping_condition = early_stopping_condition;
            return self;
        }
        /// Sets `evaluation_min_change`.
        /// 
        /// Minimum change required to log positive evaluation change.
        pub fn evaluation_min_change(&mut self, evaluation_min_change:Proportion) -> &mut Trainer<'a> {
            self.evaluation_min_change = evaluation_min_change;
            return self;
        }
        /// Sets `learning_rate_decay`.
        /// 
        /// `learning_rate_decay` is the mulipliers by which to decay the learning rate.
        pub fn learning_rate_decay(&mut self, learning_rate_decay:f32) -> &mut Trainer<'a> {
            self.learning_rate_decay = learning_rate_decay;
            return self;
        }
        /// Sets `learning_rate_interval`.
        pub fn learning_rate_interval(&mut self, learning_rate_interval:MeasuredCondition) -> &mut Trainer<'a> {
            self.learning_rate_interval = learning_rate_interval;
            return self;
        }
        /// Sets `checkpoint_interval`.
        /// 
        /// `checkpoint_interval` sets how often (if at all) to serialize and output neural network to .txt file.
        pub fn checkpoint_interval(&mut self, checkpoint_interval:MeasuredCondition) -> &mut Trainer<'a> {
            self.checkpoint_interval = Some(checkpoint_interval);
            return self;
        }
        /// Sets `name`
        /// 
        /// `name` sets what to pretend to checkpoint files. Used to differentiate between nets when checkpointing multiple.
        pub fn name(&mut self, name:&'a str) -> &mut Trainer<'a> {
            self.name=Some(name);
            return self;
        }
        /// Sets `tracking`.
        /// 
        /// `tracking` determines whether to output percentage progress during backpropgation.
        pub fn tracking(&mut self) -> &mut Trainer<'a> {
            self.tracking = true;
            return self;
        }
        /// Begins training.
        pub fn go(&mut self) -> () {
            // TODO How expensive is `rand::thread_rng()` would it be worth passing reference to `train_details`? (so as to avoid calling it again).
            // Shuffles training data
            self.training_data.shuffle(&mut rand::thread_rng());

            // Sets evaluation data
            let evaluation_data = match self.evaluation_data {
                EvaluationData::Scalar(scalar) => { self.training_data.split_off(self.training_data.len() - scalar) }
                EvaluationData::Percent(percent) => { self.training_data.split_off(self.training_data.len() - (self.training_data.len() as f32 * percent) as usize) }
                EvaluationData::Actual(actual) => { actual.clone() }
            };
            // Calls `train_details` starting training.
            self.neural_network.train_details(
                &mut self.training_data,
                &evaluation_data,
                &self.cost,
                self.halt_condition,
                self.log_interval,
                self.batch_size,
                self.learning_rate,
                self.l2,
                self.early_stopping_condition,
                self.evaluation_min_change,
                self.learning_rate_decay,
                self.learning_rate_interval,
                self.checkpoint_interval,
                self.name,
                self.tracking,
            );
        }
    }
    /// Defines cost function of neural network.
    #[derive(Serialize,Deserialize)]
    pub enum Cost {
        /// Quadratic cost function.
        ///
        /// $ C(w,b)=\frac{1}{2n}\sum_{x} ||y(x)-a(x) ||^2 $
        Quadratic,
        /// Crossentropy cost function.
        ///
        /// $ C(w,b) = -\frac{1}{n} \sum_{x} (y(x) \ln{(a(x))}  + (1-y(x)) \ln{(1-a(x))}) $
        Crossentropy
    }
    impl Cost {
        /// Runs cost functions
        /// 
        /// y: Target out, a: Actual out
        fn run(&self,y:&Array<f32>,a:&Array<f32>) -> f32 {
            return match self {
                Self::Quadratic => { quadratic(y,a) },
                Self::Crossentropy => { cross_entropy(y,a) }
            };
            // Quadratic cost
            fn quadratic(y: &Array<f32>, a: &Array<f32>) -> f32 {
                sum_all(&pow(&(y - a),&2,false)).0 as f32 / (2f32*a.dims().get()[0] as f32)
            }
            // Cross entropy cost
            // TODO Need to double check this
            fn cross_entropy(y: &Array<f32>, a: &Array<f32>) -> f32 {
                // if sum_all(&arrayfire::isnan(a)).0 > 0f64 { 
                //     af_print!("a",cols(&a,0,10));
                //     panic!("nan a"); 
                // }

                //af_print!("y",cols(&y,0,10));
                

                // Adds very small value to a, to prevent log(0)=nan
                let part1 = log(&(a+1e-20)) * y;
                // Add very small value to prevent log(1-1)=log(0)=nan
                let part2 = log(&(1f32 - a + 1e-20)) * (1f32 - y);

                //af_print!("part1",cols(&part1,0,10));
                //af_print!("part2",cols(&part2,0,10));

                let mut cost:f32 = sum_all(&(part1+part2)).0 as f32;

                //if cost.is_nan() { panic!("nan cost"); }

                cost /= -(a.dims().get()[0] as f32);

                return cost;
            }
        }
        /// Derivative wrt layer output (∂C/∂a)
        /// 
        /// y: Target out, a: Actual out
        fn derivative(&self,y:&Array<f32>,a:&Array<f32>) -> Array<f32> {
            return match self {
                Self::Quadratic => { a-y },
                Self::Crossentropy => {
                    // TODO Double check we don't need to add a val to prevent 1-a=0 (commented out code below checks count of values where a>=1)
                    //let check = sum_all(&arrayfire::ge(a,&1f32,false)).0;
                    //if check != 0f64 { panic!("check: {}",check); }
                    
                    // if sum_all(&arrayfire::isinf(a)).0 > 0f64 { 
                    //     af_print!("a inf",cols(a,0,10));
                    //     panic!("a inf");
                    // }
                    // if sum_all(&arrayfire::isnan(a)).0 > 0f64 { 
                    //     af_print!("a nan",cols(a,0,10));
                    //     panic!("a nan");
                    // }
                    // if sum_all(&eq(a,&1f32,false)).0 > 0f64 { 
                    //     af_print!("a eq 1",cols(a,0,10));
                    //     panic!("nan a eq 1");
                    // }
                    // if sum_all(&eq(a,&0f32,false)).0 > 0f64 { 
                    //     af_print!("a eq 0",cols(a,0,10));
                    //     panic!("nan a eq 0");
                    // }
                    return (-1*y)/a + (1f32-y)/(1f32-a);
                } // -y/a + (1-y)/(1-a)
            }
        }
    }
    /// Defines activations of layers in neural network.
    #[derive(Clone,Copy,Debug,Serialize,Deserialize)]
    pub enum Activation {
        /// Sigmoid activation functions.
        /// 
        /// $ A(z)=\frac{1}{1+e^-z} $
        Sigmoid,
        /// Softmax activation function.
        /// 
        /// $ A(\begin{bmatrix}z_1,\dots,z_k\end{bmatrix})=\begin{bmatrix}\frac{e^{z_1}}{\Sigma_{i=1}^k e^{z_i}} & \dots &\frac{e^{z_k}}{\Sigma_{i=1}^k e^{z_i}}\end{bmatrix} $
        Softmax,
        /// ReLU activation function.
        /// 
        /// $ A(z)=max(z,0) $
        ReLU // Name it 'ReLU' or 'Relu'?
    }
    impl Activation {
        /// Computes activations given inputs.
        pub fn run(&self,z:&Array<f32>) -> Array<f32> {
            return match self {
                Self::Sigmoid => sigmoid(z),
                Self::Softmax => Activation::softmax(z),
                Self::ReLU => Activation::relu(z),
            };
        }
        // Derivative wrt layer input (∂a/∂z)
        fn derivative(&self,z:&Array<f32>) -> Array<f32> {
            // What should we name the derivative functions?
            return match self {
                Self::Sigmoid => sigmoid_derivative(z),
                Self::Softmax => softmax_derivative(z),
                Self::ReLU => relu_derivative(z),
            };

            // Derivative of sigmoid
            // s' = s(1-s)
            fn sigmoid_derivative(z:&Array<f32>) -> Array<f32> {
                let s = sigmoid(z);
                return s.clone()*(1f32-s); // TODO Can we remove the clone here?
            }
            // Derivative of softmax
            // e^z * (sum of other inputs e^input) / (sum of all inputs e^input)^2 = e^z * (exp_sum-e^z) / (exp_sum)^2
            fn softmax_derivative(z:&Array<f32>) -> Array<f32> {
                let exponents = exp(z);
                //af_print!("exponents",exponents);
                // Gets sum of each example (column)
                let sums = sum(&exponents,0);
                //af_print!("sums",sums);
                // Sets squared sum of each example
                let sqrd_sums = pow(&sums,&2,false); // is this better than `&sums*&sums`?
                //af_print!("sqrd_sums",sqrd_sums);
                let ones = constant(1f32,Dim4::new(&[z.dims().get()[0],1,1,1]));
                //af_print!("ones",ones);
                let sums_matrix = matmul(&ones,&sums,MatProp::NONE,MatProp::NONE);
                //af_print!("sums_matrix",sums_matrix);
                let sums_sub = sums_matrix - &exponents;
                //af_print!("sums_sub",sums_sub);
                // TODO Is it more efficient to do this matrix multiplication before or after squaring?
                let sqrd_sums_matrix = matmul(&ones,&sqrd_sums,MatProp::NONE,MatProp::NONE);
                //af_print!("sqrd_sums_matrix",sqrd_sums_matrix);
                let derivatives = exponents * sums_sub / sqrd_sums_matrix;
                
                return derivatives;
            }
            //Deritvative of ReLU
            // ReLU(z)/1 = if >0 1 else 0
            fn relu_derivative(z:&Array<f32>) -> Array<f32> {
                // return Activation::relu(z) / z;
                // Follow code replaces the above line.
                // Above line replaced becuase it is prone to floating point error leading to f32:NAN.
                // Similar performance.
                let gt = gt(z,&0f32,false);
                return and(z,&gt,false);
            }
        }
        // TODO Make this better
        // Applies softmax activation
        fn softmax(y: &Array<f32>) -> Array<f32> {
            let ones = constant(1f32,Dim4::new(&[y.dims().get()[0],1,1,1]));
            //af_print!("ones",ones);

            // Subtracts example max output from all example outputs.
            //  Allowing softmax to handle large values in y.
            // ------------------------------------------------
            // Gets max values in each example
            let max_axis_vals = max(&y,0);
            // Matrix where each value is example max
            let max_axis_vals_matrix = matmul(&ones,&max_axis_vals,MatProp::NONE,MatProp::NONE);
            // All values minus there example maxes
            let max_reduced = y - max_axis_vals_matrix;

            // Applies softmax
            // ------------------------------------------------
            // Apply e^(x) to every value in matrix
            let exp_matrix = exp(&max_reduced);
            // Calculates sums of examples
            let row_sums = sum(&exp_matrix,0);
            // Matrix where each value is example sum
            let row_sums_matrix = matmul(&ones,&row_sums,MatProp::NONE,MatProp::NONE);
            // Divides each value by example sum
            let softmax = exp_matrix / row_sums_matrix; // TODO Could this div be done using batch operation with `arrayfire::div(...)` using `row_sums`?
            
            return softmax;
        }
        // Applies ReLU activation
        fn relu(y: &Array<f32>) -> Array<f32> {
            let zeros = constant(0f32,y.dims());
            return maxof(y,&zeros,false);
        }
    }
    // Defines a dense layer
    struct DenseLayer {
        activation:Activation,
        biases:Array<f32>,
        weights:Array<f32>
    }
    impl DenseLayer {
        // Constructs new `DenseLayer`
        fn new(from:u64,size:u64,activation:Activation) -> DenseLayer {
            if size == 0 { panic!("All dense layer sizes must be >0."); }
            return DenseLayer {
                activation,
                biases: (randu::<f32>(Dim4::new(&[size,1,1,1])) * 2f32) - 1f32,
                weights:
                    ((randu::<f32>(Dim4::new(&[size,from,1,1])) * 2f32) - 1f32)
                    / (from as f32).sqrt()
            };
        }
        // Constructs new `DenseLayer` using a given value for all weights and biases.
        fn new_constant(from:u64,size:u64,activation:Activation,val:f32) -> DenseLayer {
            if size == 0 { panic!("All dense layer sizes must be >0."); }
            return DenseLayer {
                activation,
                biases: constant(val,Dim4::new(&[size,1,1,1])),
                weights: constant(val,Dim4::new(&[size,from,1,1]))
            };
        }
        // Forward propagates.
        fn forepropagate(&self,a:&Array<f32>) -> (Array<f32>,Array<f32>) {
            //println!("activation: {:.?}",self.activation);
            //af_print!("self.weights",self.weights);
            //af_print!("a",a);
            
            let weighted_inputs:Array<f32> = matmul(&self.weights,&a,MatProp::NONE,MatProp::NONE);
            //af_print!("weighted_inputs",weighted_inputs);
            //af_print!("cols(&weighted_inputs,0,10)",cols(&weighted_inputs,0,10));

            //NeuralNetwork::mem_info("weighted inputs computed",false);

            let input = add(&weighted_inputs,&self.biases,true);

            //af_print!("input",input);
            //af_print!("cols(&input,0,10)",cols(&input,0,10));

            //NeuralNetwork::mem_info("z computed",false);

            let activation = self.activation.run(&input);
            //af_print!("activation",activation);
            //af_print!("cols(&activation,0,10)",cols(&activation,0,10));

            //NeuralNetwork::mem_info("a computed",false);

            return (activation,input);
        }
        // TODO name `from_error` better
        // TODO We only need `training_set_length` if `l2 = Some()..`, how can we best pass `training_set_length`?
        // Backpropagates.
        // (Updates weights and biases during this process).
        fn backpropagate(&mut self, partial_error:&Array<f32>,z:&Array<f32>,a:&Array<f32>,learning_rate:f32,l2:Option<f32>,training_set_length:usize) -> Array<f32> {

            // δ
            let error = self.activation.derivative(z) * partial_error;

            NeuralNetwork::mem_info("error computed",false);

            // Number of examples in batch
            let batch_len = z.dims().get()[1] as f32;

            // Sets errors/gradients and sums through examples
            // ∂C/∂b
            let bias_error = sum(&error,1);
            NeuralNetwork::mem_info("bias error computed",false);

            println!("{} | {} -> {}",error.dims(),a.dims(),calc_weight_errors(&error,a).dims());

            // ∂C/∂w
            let weight_error = sum(&calc_weight_errors(&error,a),2);
            NeuralNetwork::mem_info("weight error computed",false);

            // w^T dot δ
            let nxt_partial_error = matmul(&self.weights,&error,MatProp::TRANS,MatProp::NONE);

            NeuralNetwork::mem_info("partial error computed",false);

            // TODO Figure out best way to do weight and bias updates
            // = old weights - avg weight errors
            if let Some(lambda) = l2 {
                self.weights = ((1f32 - (learning_rate * lambda / training_set_length as f32)) * &self.weights) - (learning_rate * weight_error / batch_len)
            } 
            else {
                self.weights = &self.weights - (learning_rate * weight_error / batch_len);
            }

            NeuralNetwork::mem_info("weights updated",false);
            
            // = old biases - avg bias errors
            self.biases = &self.biases - (learning_rate * bias_error / batch_len);

            NeuralNetwork::mem_info("biases updated",false);

            // w^T dot δ
            return nxt_partial_error;

            // TODO Better document this
            fn calc_weight_errors(errors:&Array<f32>,activations:&Array<f32>) -> arrayfire::Array<f32> {
                let examples:u64 = activations.dims().get()[1];
                
                let er_size:u64 = errors.dims().get()[0];
                let act_size:u64 = activations.dims().get()[0];
                let dims = arrayfire::Dim4::new(&[er_size,act_size,examples,1]);
            
                let temp:arrayfire::Array<f32> = arrayfire::Array::<f32>::new_empty(dims);
                
                //NeuralNetwork::mem_info("set weight error slicing 0",false);
                for i in 0..examples {
                    let holder = arrayfire::matmul(
                        &col(errors,i),
                        &col(activations,i),
                        MatProp::NONE,
                        MatProp::TRANS
                    );
                    //NeuralNetwork::mem_info("set weight error slicing 1",false);
                    arrayfire::set_slice(&temp,&holder,i); // TODO Why does this work? I don't think this should work.
                    //NeuralNetwork::mem_info("set weight error slicing 2",false);
                    //if i > 2 { panic!("stop spam"); }
                }
                
                return temp;
            }
        }
    }
    // Defines a dropout layer (mask)
    struct DropoutLayer {
        p:f32,
        mask:Array<f32>
    }
    impl DropoutLayer {
        // Constructs new `DropoutLayer`
        fn new(p:f32) -> DropoutLayer {
            DropoutLayer {p,mask:Array::<f32>::new_empty(Dim4::new(&[1,1,1,1]))}
        }
        // Forward propgates.
        // Creates a mask to fit given data.
        fn forepropagate(&mut self,z:&Array<f32>,ones:&Array<f32>) -> Array<f32> {
            // Sets mask dimensions
            let z_dims = z.dims();
            let z_dim_arr = z_dims.get();
            let mask_dims = Dim4::new(&[z_dim_arr[0],1,1,1]);
            // TODO Look into using `tile`
            // Updates mask
            self.mask = matmul(&gt(&randu::<f32>(mask_dims),&self.p,false).cast::<f32>(),ones,MatProp::NONE,MatProp::NONE);
            // Applies mask
            return mul(z,&self.mask,false);
        }
        // Backpropgates
        // Using mask used for last forepropgate (cannot backpropgate dropout layer without first forepropagating).
        fn backpropagate(&self,partial_error:&Array<f32>) -> Array<f32> {
            return mul(partial_error,&self.mask,false);
        }
    }
    // Specifies layers within neural net.
    enum InnerLayer {
        Dropout(DropoutLayer),Dense(DenseLayer)
    }
    /// Specifies layers to cosntruct neural net.
    pub enum Layer {
        Dropout(f32),Dense(u64,Activation)
    }

    /// Strcut used to import/export neural net.
    #[derive(Serialize,Deserialize)]
    struct ImportExportNet {
        inputs: u64,
        layers: Vec<InnerLayerEnum>,
    }
    // Defines layers for import/export struct.
    #[derive(Serialize,Deserialize)]
    enum InnerLayerEnum {
        Dropout(f32),Dense(Activation,[u64;4],Vec<f32>,[u64;4],Vec<f32>)
    }
    /// Neural network.
    pub struct NeuralNetwork {
        // Inputs to network.
        inputs: u64,
        // Activations of layers.
        layers: Vec<InnerLayer>,
    }
    impl NeuralNetwork {
        /// Constructs network of given layers.
        /// 
        /// Returns constructed network.
        /// ```
        /// use cogent::core::{NeuralNetwork,Layer,Activation};
        /// 
        /// let mut net = NeuralNetwork::new(2,&[
        ///     Layer::Dense(3,Activation::Sigmoid),
        ///     Layer::Dense(2,Activation::Softmax)
        /// ]);
        /// ```
        pub fn new(mut inputs:u64,layers: &[Layer]) -> NeuralNetwork {
            NeuralNetwork::new_checks(inputs,layers);

            // Necessary variable to use mutable `inputs` to nicely specify right layer sizes.
            let net_inputs = inputs;

            // Sets holder for neural net layers
            let mut inner_layers:Vec<InnerLayer> = Vec::with_capacity(layers.len());

            // Sets iterator across given layers data
            let mut layers_iter = layers.iter();
            let layer_1 = layers_iter.next().unwrap();
            
            

            // Constructs first non-input layer
            if let &Layer::Dense(size,activation) = layer_1 {
                inner_layers.push(InnerLayer::Dense(DenseLayer::new(inputs,size,activation)));
                inputs = size;
            } 
            else if let &Layer::Dropout(p) = layer_1 {
                inner_layers.push(InnerLayer::Dropout(DropoutLayer::new(p)));
            }

            // Constructs other layers
            for layer in layers_iter {
                if let &Layer::Dense(size,activation) = layer {
                    inner_layers.push(InnerLayer::Dense(DenseLayer::new(inputs,size,activation)));
                    inputs = size;
                } 
                else if let &Layer::Dropout(p) = layer {
                    inner_layers.push(InnerLayer::Dropout(DropoutLayer::new(p)));
                }
            }

            // Constructs and returns neural network
            return NeuralNetwork{ inputs:net_inputs, layers:inner_layers };
        }
        /// Constructs network of given layers with all weights and biases set to given value.
        /// IMPORTANT: This function seems to cause issues in training and HAS NOT been properly tested, I DO NOT recommend you use this.
        pub fn new_constant(mut inputs:u64,layers: &[Layer],val:f32) -> NeuralNetwork {
            NeuralNetwork::new_checks(inputs,layers);

            // Neccessary variable to use mutable `inputs` to nicely specify right layer sizes.
            let net_inputs = inputs;

            // Sets holder for neural net layers
            let mut inner_layers:Vec<InnerLayer> = Vec::with_capacity(layers.len());

            // Sets iterator across given layers data
            let mut layers_iter = layers.iter();
            let layer_1 = layers_iter.next().unwrap();

            // Constructs first non-input layer
            if let &Layer::Dense(size,activation) = layer_1 {
                inner_layers.push(InnerLayer::Dense(DenseLayer::new_constant(inputs,size,activation,val)));
                inputs = size;
            } 
            else if let &Layer::Dropout(p) = layer_1 {
                inner_layers.push(InnerLayer::Dropout(DropoutLayer::new(p)));
            }

            // Constructs other layers
            for layer in layers_iter {
                if let &Layer::Dense(size,activation) = layer {
                    inner_layers.push(InnerLayer::Dense(DenseLayer::new_constant(inputs,size,activation,val)));
                    inputs = size;
                } 
                else if let &Layer::Dropout(p) = layer {
                    inner_layers.push(InnerLayer::Dropout(DropoutLayer::new(p)));
                }
            }

            // Constructs and returns neural network
            return NeuralNetwork{ inputs:net_inputs, layers:inner_layers };
        }
        // Checks that given data to construct neural network from valid.
        fn new_checks(inputs:u64,layers: &[Layer]) {
            // Checks network contains output layer
            if layers.len() == 0 { panic!("Requires output layer (layers.len() must be >0)."); }
            // Checks inputs != 0
            if inputs == 0 { panic!("Input size must be >0."); }
            // Chekcs last layer is not a dropout layer
            if let Layer::Dropout(_) = layers[layers.len()-1] { panic!("Last layer cannot be a dropout layer."); }
        }
        /// Sets activation of layer specified by index (excluding input layer).
        /// ```
        /// use cogent::core::{NeuralNetwork,Layer,Activation};
        /// 
        /// // Net (2 -Sigmoid-> 3 -Sigmoid-> 2)
        /// let mut net = NeuralNetwork::new(2,&[
        ///     Layer::Dense(3,Activation::Sigmoid),
        ///     Layer::Dense(2,Activation::Sigmoid)
        /// ]);
        /// 
        /// net.activation(1,Activation::Softmax); // Changes activation of output layer.
        /// // Net will now be (2 -Sigmoid-> 3 -Softmax-> 2)
        /// ```
        pub fn activation(&mut self, index:usize, activation:Activation) {
            // Checks lyaer exists
            if index >= self.layers.len() {
                panic!("Layer {} does not exist. 0 <= given index < {}",index,self.layers.len()); 
            }
            // Checks layer has activation function
            if let InnerLayer::Dense(dense_layer) = &mut self.layers[index] {
                dense_layer.activation = activation;
            }
            else {
                panic!("Layer {} does not have an activation function.",index); 
            }
        }
        /// Runs a batch of examples through the network.
        /// 
        /// Returns classes.
        pub fn run(&mut self, inputs:&Vec<Vec<f32>>) -> Vec<usize> {
            let in_len = inputs[0].len();
            let example_len = inputs.len();

            // Converts 2d vec to array for input
            let in_vec:Vec<f32> = inputs.iter().flat_map(|x| x.clone()).collect();
            let input:Array<f32> = Array::<f32>::new(&in_vec,Dim4::new(&[example_len as u64,in_len as u64,1,1]));

            // Forepropagates
            let output = self.inner_run(&input);
            // Computes classes of each example
            let classes = arrayfire::imax(&output,0).1;
  
            // Converts classes array to classes vec
            let classes_vec:Vec<u32> = NeuralNetwork::to_vec(&classes);

            // Returns classes vec casted from `Vec<u32>` to `Vec<usize>`
            return classes_vec.into_iter().map(|x| x as usize).collect(); 
        }
        fn to_vec<T:arrayfire::HasAfEnum+Default+Clone>(array:&arrayfire::Array<T>) -> Vec<T> {
            let mut vec = vec!(T::default();array.elements());
            array.host(&mut vec);
            return vec;
        }
        /// Runs a batch of examples through the network.
        /// 
        /// Returns output.
        pub fn inner_run(&mut self, inputs:&Array<f32>) -> Array<f32> {
            // Number of examples in input.
            let examples = inputs.dims().get()[1];
            let ones = &constant(1f32,Dim4::new(&[1,examples,1,1]));

            // Forepropagates.
            let mut activation = inputs.clone(); // Sets input layer
            for layer in self.layers.iter_mut() {
                activation = match layer {
                    InnerLayer::Dropout(dropout_layer) => dropout_layer.forepropagate(&activation,ones),
                    InnerLayer::Dense(dense_layer) => dense_layer.forepropagate(&activation).0,
                };
            }

            // Returns activation of last layer.
            return activation;
        }
        /// Begins setting hyperparameters for training.
        /// 
        /// Returns `Trainer` struct used to specify hyperparameters
        /// 
        /// Training a network to learn an XOR gate:
        /// ```
        /// use cogent::core::{NeuralNetwork,Layer,Activation,EvaluationData};
        /// 
        /// // Sets network
        /// let mut neural_network = NeuralNetwork::new(2,&[
        ///     Layer::Dense(3,Activation::Sigmoid),
        ///     Layer::Dense(2,Activation::Softmax)
        /// ]);
        /// // Sets data
        /// // 0=false,  1=true.
        /// let data = vec![
        ///     (vec![0f32,0f32],0),
        ///     (vec![1f32,0f32],1),
        ///     (vec![0f32,1f32],1),
        ///     (vec![1f32,1f32],0)
        /// ];
        /// // Trains network
        /// neural_network.train(&data)
        ///     .learning_rate(2f32)
        ///     .evaluation_data(EvaluationData::Actual(&data)) // Use training data as evaluation data.
        /// .go();
        /// ```
        pub fn train(&mut self,training_data:&[(Vec<f32>,usize)]) -> Trainer {
            // Sets number of network outputs
            let class_outs = match &self.layers[self.layers.len()-1] {
                InnerLayer::Dense(dense_layer) => dense_layer.biases.dims().get()[0] as usize,
                _ => panic!("Last layer is somehow a dropout layer, this should not be possible")
            };

            // Checks all examples fit the neural network.
            for i in 0..training_data.len() {
                // Checks number of input values of each example matchs number of network inputs.
                if training_data[i].0.len() != self.inputs as usize {
                    panic!("Input size of example {} ({}) != size of input layer ({}).",i,training_data[i].0.len(),self.inputs);
                }
                // Checks number of classes in dataset does not exceed number of network outputs (presumes classes in data are labelled 0..n).
                if training_data[i].1 >= class_outs {
                    panic!("Output class of example {} ({}) >= number of output classes of network ({}).",i,training_data[i].1,class_outs);
                }
            }

            let multiplier:f32 = training_data[0].0.len() as f32 / training_data.len() as f32;
            let early_stopping_condition:u32 = (DEFAULT_EARLY_STOPPING * multiplier).ceil() as u32;
            let learning_rate_interval:u32 = (DEFAULT_LEARNING_RATE_INTERVAL * multiplier).ceil() as u32;
            
            let batch_holder:f32 = DEFAULT_BATCH_SIZE * training_data.len() as f32;
            // TODO What should we use as min batch size here instead of `100`?
            let batch_size:usize = 
                if training_data.len() < 100usize { training_data.len() }
                else if batch_holder < 100f32 { 100usize }
                else { batch_holder.ceil() as usize };

            return Trainer {
                training_data: training_data.to_vec(),
                evaluation_data: EvaluationData::Percent(DEFAULT_EVALUTATION_DATA),
                cost:Cost::Crossentropy,
                halt_condition: None,
                log_interval: None,
                batch_size: batch_size,
                learning_rate: DEFAULT_LEARNING_RATE,
                l2:None,
                early_stopping_condition: MeasuredCondition::Iteration(early_stopping_condition),
                evaluation_min_change: Proportion::Percent(DEFAULT_EVALUATION_MIN_CHANGE),
                learning_rate_decay: DEFAULT_LEARNING_RATE_DECAY,
                learning_rate_interval: MeasuredCondition::Iteration(learning_rate_interval),
                checkpoint_interval: None,
                name: None,
                tracking: false,
                neural_network:self
            };
            
        }

        // TODO Name this better
        // Runs training.
        fn train_details(&mut self,
            training_data: &mut [(Vec<f32>,usize)], // TODO Look into `&mut [(Vec<f32>,Vec<f32>)]` vs `&mut Vec<(Vec<f32>,Vec<f32>)>`
            evaluation_data: &[(Vec<f32>,usize)],
            cost:&Cost,
            halt_condition: Option<HaltCondition>,
            log_interval: Option<MeasuredCondition>,
            batch_size: usize,
            intial_learning_rate: f32,
            l2:Option<f32>,
            early_stopping_n: MeasuredCondition,
            evaluation_min_change: Proportion,
            learning_rate_decay: f32,
            learning_rate_interval: MeasuredCondition,
            checkpoint_interval: Option<MeasuredCondition>,
            name: Option<&str>,
            tracking:bool
        ) -> (){
            if let Some(_) = checkpoint_interval {
                if !Path::new("checkpoints").exists() {
                    // Create folder
                    create_dir("checkpoints").unwrap();
                }
                if let Some(folder) = name {
                    let path = format!("checkpoints/{}",folder);
                    // If folder exists, empty it.
                    if Path::new(&path).exists() {
                        remove_dir_all(&path).unwrap();// Delete folder
                    }
                    create_dir(&path).unwrap(); // Create folder
                }
            }

            let mut learning_rate:f32 = intial_learning_rate;

            let mut stdout = stdout(); // Handle for standard output for this process.
            let mut rng = rand::thread_rng(); // Random number generator.

            let start_instant = Instant::now(); // Beginning instant to compute duration of training.
            let mut iterations_elapsed = 0u32; // Iteration counter of training.
            
            let mut best_accuracy_iteration = 0u32;// Iteration of best accuracy.
            let mut best_accuracy_instant = Instant::now();// Instant of best accuracy.
            let mut best_accuracy = 0u32; // Value of best accuracy.

            NeuralNetwork::mem_info("Before any alloocation",false);
            // Sets array of evaluation data.
            let matrix_evaluation_data = self.matrixify(evaluation_data);
            NeuralNetwork::mem_info("Evaluation data allocated",false);

            // Computes intial evaluation.
            let starting_evaluation = self.inner_evaluate(&matrix_evaluation_data,evaluation_data,cost);
            NeuralNetwork::mem_info("Evaluation ran",false);
            
            
            // If `log_interval` has been defined, print intial evaluation.
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

            // Backpropgation loop
            // ------------------------------------------------
            loop {
                // Sets array of training data.
                let training_data_matrix = self.matrixify(training_data);
                NeuralNetwork::mem_info("Training data allocated",false);

                // Split training data into batchs.
                let batches = batch_chunks(&training_data_matrix,batch_size);
                NeuralNetwork::mem_info("Training data batchs set",false);

                

                // Runs backpropagation on all batches:
                //  If `tracking` output backpropagation percentage progress.
                if tracking {
                    let mut percentage:f32 = 0f32;
                    stdout.queue(cursor::SavePosition).unwrap();
                    let backprop_start_instant = Instant::now();
                    let percent_change:f32 = 100f32 * batch_size as f32 / training_data_matrix.0.dims().get()[1] as f32;

                    for batch in batches {
                        stdout.write(format!("Backpropagating: {:.2}%",percentage).as_bytes()).unwrap();
                        percentage += percent_change;
                        stdout.queue(cursor::RestorePosition).unwrap();
                        stdout.flush().unwrap();

                        self.backpropagate(&batch,learning_rate,cost,l2,training_data.len());

                        
                    }
                    stdout.write(format!("Backpropagated: {}\n",NeuralNetwork::time(backprop_start_instant)).as_bytes()).unwrap();
                }
                else {
                    for batch in batches {
                        self.backpropagate(&batch,learning_rate,cost,l2,training_data.len());

                        NeuralNetwork::mem_info("Backpropagated",false);
                        panic!("stop here");
                    }
                    
                }
                iterations_elapsed += 1;

                // Computes iteration evaluation.
                let evaluation = self.inner_evaluate(&matrix_evaluation_data,evaluation_data,cost);

                // If `checkpoint_interval` number of iterations or length of duration passed, export weights  (`connections`) and biases (`biases`) to file.
                match checkpoint_interval {
                    Some(MeasuredCondition::Iteration(iteration_interval)) => if iterations_elapsed % iteration_interval == 0 {
                        checkpoint(self,iterations_elapsed.to_string(),name);
                    },
                    Some(MeasuredCondition::Duration(duration_interval)) => if last_checkpointed_instant.elapsed() >= duration_interval {
                        checkpoint(self,NeuralNetwork::time(start_instant),name);
                        last_checkpointed_instant = Instant::now();
                    },
                    _ => {},
                }
                // If `log_interval` number of iterations or length of duration passed, print evaluation of network.
                match log_interval {// TODO Reduce code duplication here
                    Some(MeasuredCondition::Iteration(iteration_interval)) => if iterations_elapsed % iteration_interval == 0 {
                        log_fn(&mut stdout,iterations_elapsed,start_instant,learning_rate,evaluation,evaluation_data.len()
                        );
                    },
                    Some(MeasuredCondition::Duration(duration_interval)) => if last_logged_instant.elapsed() >= duration_interval {
                        log_fn(&mut stdout,iterations_elapsed,start_instant,learning_rate,evaluation,evaluation_data.len()
                        );
                        last_logged_instant = Instant::now();
                    },
                    _ => {},
                }

                // If 100% accuracy, halt.
                if evaluation.1 as usize == evaluation_data.len() { break; }

                // If `halt_condition` number of iterations occured, duration passed or accuracy acheived, halt training.
                match halt_condition {
                    Some(HaltCondition::Iteration(iteration)) => if iterations_elapsed == iteration { break; },
                    Some(HaltCondition::Duration(duration)) => if start_instant.elapsed() > duration { break; },
                    Some(HaltCondition::Accuracy(accuracy)) => if evaluation.1 >= (evaluation_data.len() as f32 * accuracy) as u32  { break; },
                    _ => {},
                }

                // TODO Reduce code duplication here
                // If change in evaluation more than `evaluation_min_change` update `best_accuracy`,`best_accuracy_iteration` and `best_accuracy_instant`.
                match evaluation_min_change {
                    Proportion::Percent(percent) => if (evaluation.1 as f32 / evaluation_data.len() as f32) > (best_accuracy as f32 / evaluation_data.len() as f32) + percent {
                        best_accuracy = evaluation.1;
                        best_accuracy_iteration = iterations_elapsed;
                        best_accuracy_instant = Instant::now();
                    },
                    Proportion::Scalar(scalar) => if evaluation.1 > best_accuracy + scalar {
                        best_accuracy = evaluation.1;
                        best_accuracy_iteration = iterations_elapsed;
                        best_accuracy_instant = Instant::now();
                    }
                }

                // If `early_stopping_n` number of iterations or length of duration passed, without improvement in accuracy (`evaluation.1`), halt training. (early_stopping_n<=halt_condition)
                match early_stopping_n {
                    MeasuredCondition::Iteration(stopping_iteration) =>  if iterations_elapsed - best_accuracy_iteration == stopping_iteration { println!("---------------\nEarly stoppage!\n---------------"); break; },
                    MeasuredCondition::Duration(stopping_duration) => if best_accuracy_instant.elapsed() >= stopping_duration { println!("---------------\nEarly stoppage!\n---------------"); break; }
                }

                // If `learning_rate_interval` number of iterations or length of duration passed, without improvement in accuracy (`evaluation.1`), reduce learning rate. (learning_rate_interval<early_stopping_n<=halt_condition)
                match learning_rate_interval {
                    MeasuredCondition::Iteration(interval_iteration) =>  if iterations_elapsed - best_accuracy_iteration == interval_iteration { learning_rate *= learning_rate_decay },
                    MeasuredCondition::Duration(interval_duration) => if best_accuracy_instant.elapsed() >= interval_duration { learning_rate *= learning_rate_decay }
                }

                // Shuffles training data
                // Given trainin data is already shuffled, so don't need to shuffle on 1st itereation.
                training_data.shuffle(&mut rng);
            }

            // Compute and print final evaluation.
            // ------------------------------------------------
            let evaluation = self.inner_evaluate(&matrix_evaluation_data,evaluation_data,cost); 
            let new_percent = (evaluation.1 as f32)/(evaluation_data.len() as f32) * 100f32;
            let starting_percent = (starting_evaluation.1 as f32)/(evaluation_data.len() as f32) * 100f32;
            println!();
            println!("Cost: {:.4} -> {:.4}",starting_evaluation.0,evaluation.0);
            println!("Classified: {} ({:.2}%) -> {} ({:.2}%)",starting_evaluation.1,starting_percent,evaluation.1,new_percent);
            println!("Cost: {:.4}",evaluation.0-starting_evaluation.0);
            println!("Classified: +{} (+{:.3}%)",evaluation.1-starting_evaluation.1,new_percent - starting_percent);
            println!("Time: {}, Iterations: {}",NeuralNetwork::time(start_instant),iterations_elapsed);
            println!();

            // Prints evaluation of network
            fn log_fn(
                stdout:&mut std::io::Stdout,
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
            // TODO This doesn't seem to require any more memory, look into that.
            // Splits data into chunks of examples.
            fn batch_chunks(data:&(Array<f32>,Array<f32>),batch_size:usize) -> Vec<(Array<f32>,Array<f32>)>{
                // Number of examples in dataset
                let examples = data.0.dims().get()[1];

                // Number of batches
                let batches = (examples as f32 / batch_size as f32).ceil() as usize;

                // vec containg array input and out for each batch
                let mut chunks:Vec<(Array<f32>,Array<f32>)> = Vec::with_capacity(batches);

                // Iterate over batches setting inputs and outputs
                for i in 0..batches-1 {
                    let batch_indx:usize = i * batch_size;
                    let in_batch:Array<f32> = cols(&data.0,batch_indx as u64,(batch_indx+batch_size-1) as u64);
                    let out_batch:Array<f32> = cols(&data.1,batch_indx as u64,(batch_indx+batch_size-1) as u64);

                    chunks.push((in_batch,out_batch));
                }
                // Since length of final batch may be less than `batch_size`, set final batch out of loop.
                let batch_indx:usize = (batches-1) * batch_size;
                let in_batch:Array<f32> = cols(&data.0,batch_indx as u64,examples-1);
                let out_batch:Array<f32> = cols(&data.1,batch_indx as u64,examples-1);
                chunks.push((in_batch,out_batch));

                return chunks;
            }
            // Outputs a checkpoint file
            fn checkpoint(net:&NeuralNetwork,marker:String,name:Option<&str>) {
                if let Some(folder) = name {
                    net.export(&format!("checkpoints/{}/{}",folder,marker));
                }
                else {
                    net.export(&format!("checkpoints/{}",marker));
                }
            }
            
        }
        // Runs batch backpropgation.
        fn backpropagate(&mut self, (net_input,target):&(Array<f32>,Array<f32>),learning_rate: f32,cost:&Cost,l2:Option<f32>,training_set_length:usize) {
            // Feeds forward
            // --------------

            let examples = net_input.dims().get()[1];
            let ones = &constant(1f32,Dim4::new(&[1,examples,1,1]));

            // Represents activations and weighted outputs of layers.
            //  For element i we have the activation of layer i and the weighted inputs of layer i+1.
            //  All layers have activations, but not all layers have useful weighted inputs (.e.g dropout), this is why we use `Option<..>`
            let mut layer_outs:Vec<(Array<f32>,Option<Array<f32>>)> = Vec::with_capacity(self.layers.len());

            // TODO Name this better
            // Sets input layer activation
            let mut input = net_input.clone();
            

            for layer in self.layers.iter_mut() {
                let (a,z) = match layer {
                    InnerLayer::Dropout(dropout_layer) => (dropout_layer.forepropagate(&input,ones),None),
                    InnerLayer::Dense(dense_layer) => { let (a,z) = dense_layer.forepropagate(&input); (a,Some(z)) },
                };
                layer_outs.push((input,z));
                input = a;
                NeuralNetwork::mem_info("Forepropagated layer",false);
            }
            layer_outs.push((input,None));

            NeuralNetwork::mem_info("Forepropagated",false);

            println!("step size: {:.4}mb",arrayfire::get_mem_step_size() as f32 / (1024f32*1024f32));

            //panic!("panic after foreprop");

            // Backpropagates
            // --------------

            let mut out_iter = layer_outs.into_iter().rev();
            let l_iter = self.layers.iter_mut().rev();

            let last_activation = &out_iter.next().unwrap().0;

            // ∇(a)C
            let mut partial_error = cost.derivative(target,last_activation);

            for (layer,(a,z)) in  izip!(l_iter,out_iter) {
                // w(i)^T dot δ(i) 
                // Error of layer i matrix multiplied by transposition of weights connections layer i-1 to layer i.
                partial_error = match layer {
                    InnerLayer::Dropout(dropout_layer) => dropout_layer.backpropagate(&partial_error),
                    InnerLayer::Dense(dense_layer) => dense_layer.backpropagate(&partial_error,&z.unwrap(),&a,learning_rate,l2,training_set_length),
                };
                NeuralNetwork::mem_info("Backpropagated layer",false);
            }
        }
        

        fn mem_info(msg:&str,bytes:bool) {
            let mem_info = device_mem_info();
            println!("{} : {:.4}mb | {:.4}mb",msg,mem_info.0 as f32/(1024f32*1024f32),mem_info.2 as f32/(1024f32*1024f32),);
            println!("buffers: {} | {}",mem_info.1,mem_info.3);
            if bytes { println!("bytes: {} | {}",mem_info.0,mem_info.2); }
        }

        /// Returns tuple: (Average cost across batch, Number of examples correctly classified).
        /// ```
        /// # use cogent::core::{EvaluationData,MeasuredCondition,Activation,Layer,NeuralNetwork};
        /// # 
        /// # let mut net = NeuralNetwork::new(2,&[
        /// #     Layer::Dense(3,Activation::Sigmoid),
        /// #     Layer::Dense(2,Activation::Softmax)
        /// # ]);
        /// # 
        /// let mut data = vec![
        ///     (vec![0f32,0f32],0usize),
        ///     (vec![1f32,0f32],1usize),
        ///     (vec![0f32,1f32],1usize),
        ///     (vec![1f32,1f32],0usize)
        /// ];
        /// 
        /// # net.train(&data)
        /// #     .learning_rate(2f32)
        /// #     .evaluation_data(EvaluationData::Actual(&data)) // Use testing data as evaluation data.
        /// #     .early_stopping_condition(MeasuredCondition::Iteration(2000))
        /// # .go();
        /// # 
        /// // `net` is neural network trained to 100% accuracy to mimic an XOR gate.
        /// // Passing `None` for the cost uses the default cost function (crossentropy).
        /// let (cost,accuracy) = net.evaluate(&mut data,None); 
        /// 
        /// assert_eq!(accuracy,4u32);
        pub fn evaluate(&mut self, test_data:&[(Vec<f32>,usize)],cost:Option<&Cost>) -> (f32,u32) {
            if let Some(cost_function) = cost {
                return self.inner_evaluate(&self.matrixify(test_data),test_data,cost_function);
            } else {
                return self.inner_evaluate(&self.matrixify(test_data),test_data,&Cost::Crossentropy);
            }
        }
        /// Returns tuple: (Average cost across batch, Number of examples correctly classified).
        fn inner_evaluate(&mut self,(input,target):&(Array<f32>,Array<f32>),test_data:&[(Vec<f32>,usize)],cost:&Cost) -> (f32,u32) {
            // Forepropgatates input
            let output = self.inner_run(input);
            // Computes cost
            let cost:f32 = cost.run(target,&output);
            // Computes example output classes
            let output_classes = imax(&output,0).1;
            
            // Sets array of target classes
            let target_classes = Array::<u32>::new(
                &test_data.iter().map(|&(_,class)|class as u32).collect::<Vec<u32>>(),
                Dim4::new(&[1,test_data.len() as u64,1,1])
            );
            
            // Gets number of correct classifications.
            let correct_classifications = eq(&output_classes,&target_classes,false); // TODO Can this be a bitwise AND?
            let correct_classifications_numb:u32 = sum_all(&correct_classifications).0 as u32;

            // Returns average cost and number of examples correctly classified.
            return (cost / test_data.len() as f32, correct_classifications_numb);
        }
        /// Returns tuple of: (Vector of class percentage accuracies, Percentage confusion matrix).
        /// ```ignore
        /// # use cogent::core::{EvaluationData,MeasuredCondition,Activation,Layer,NeuralNetwork};
        /// # 
        /// # let mut net = NeuralNetwork::new(2,&[
        /// #     Layer::new(3,Activation::Sigmoid),
        /// #     Layer::new(2,Activation::Softmax)
        /// # ]);
        /// # 
        /// let mut data = vec![
        ///     (vec![0f32,0f32],0usize),
        ///     (vec![1f32,0f32],1usize),
        ///     (vec![0f32,1f32],1usize),
        ///     (vec![1f32,1f32],0usize)
        /// ];
        /// 
        /// # net.train(&data)
        /// #     .learning_rate(2f32)
        /// #     .evaluation_data(EvaluationData::Actual(&data)) // Use testing data as evaluation data.
        /// #     .early_stopping_condition(MeasuredCondition::Iteration(2000))
        /// # .go();
        /// # 
        /// // `net` is neural network trained to 100% accuracy to mimic an XOR gate.
        /// let (correct_vector,confusion_matrix) = net.analyze(&mut data);
        /// 
        /// assert_eq!(correct_vector,vec![1f32,1f32]);
        /// assert_eq!(confusion_matrix,vec![[1f32,0f32],[0f32,1f32]]);
        /// ```
        #[deprecated(note = "Not deprecated, just broken until ArrayFire update installer to match git (where issue has been reported and fixed).")]
        pub fn analyze(&mut self, data:&mut [(Vec<f32>,usize)]) -> (Vec<f32>,Vec<Vec<f32>>) {
            // Sorts by class
            data.sort_by(|(_,a),(_,b)| a.cmp(b));

            let (input,classes) = matrixify_inputs(data);
            let outputs = self.inner_run(&input);

            let maxs:Array<f32> = arrayfire::max(&outputs,1i32);
            let class_vectors:Array<bool> = eq(&outputs,&maxs,true);
            let confusion_matrix:Array<f32> = sum_by_key(&classes,&class_vectors,0i32).1.cast::<f32>();

            let class_lengths:Array<f32> = sum(&confusion_matrix,1i32); // Number of examples of each class

            let percent_confusion_matrix:Array<f32> = div(&confusion_matrix,&class_lengths,true); // Divides each row (example) by number of examples of that class.

            let dims = percent_confusion_matrix.dims();
            let mut flat_vec = vec!(f32::default();(dims.get()[0]*dims.get()[1]) as usize); // dims.get()[0] == dims.get()[1]
            // `x.host(...)` outputs in column-major order, calling `tranpose(x).host(...)` effectively outputs in row-major order.
            transpose(&percent_confusion_matrix,false).host(&mut flat_vec);
            let matrix_vec:Vec<Vec<f32>> = flat_vec.chunks(dims.get()[0] as usize).map(|x| x.to_vec()).collect();

            // Gets diagonal from matrix, representing what percentage of examples where correctly identified as each class.
            let diag = diag_extract(&percent_confusion_matrix,0i32);
            let mut diag_vec:Vec<f32> = vec!(f32::default();diag.dims().get()[0] as usize);
            diag.host(&mut diag_vec);

            return (diag_vec,matrix_vec);
            
            fn matrixify_inputs(examples:&[(Vec<f32>,usize)]) -> (Array<f32>,Array<u32>) { // Array(in,examples,1,1), Array(examples,1,1,1)
                let in_len = examples[0].0.len();
                let example_len = examples.len();

                // Flattens examples into `in_vec` and `out_vec`
                let in_vec:Vec<f32> = examples.iter().flat_map(|(input,_)| input.clone() ).collect();
                let out_vec:Vec<u32> = examples.iter().map(|(_,class)| class.clone() as u32 ).collect();

                let input:Array<f32> = Array::<f32>::new(&in_vec,Dim4::new(&[in_len as u64,example_len as u64,1,1]));
                let output:Array<u32> = Array::<u32>::new(&out_vec,Dim4::new(&[example_len as u64,1,1,1]));

                return (input,output);
            }
        }
        /// Returns tuple of pretty strings of: (Vector of class percentage accuracies, Percentage confusion matrix).
        /// 
        /// Example without dictionairy:
        /// ```ignore
        /// # use cogent::core::{EvaluationData,MeasuredCondition,Activation,Layer,NeuralNetwork};
        /// # 
        /// # let mut net = NeuralNetwork::new(2,&[
        /// #     Layer::new(3,Activation::Sigmoid),
        /// #     Layer::new(2,Activation::Softmax)
        /// # ]);
        /// # 
        /// let mut data = vec![
        ///     (vec![0f32,0f32],0usize),
        ///     (vec![1f32,0f32],1usize),
        ///     (vec![0f32,1f32],1usize),
        ///     (vec![1f32,1f32],0usize)
        /// ];
        /// 
        /// # net.train(&data)
        /// #     .learning_rate(2f32)
        /// #     .evaluation_data(EvaluationData::Actual(&data)) // Use testing data as evaluation data.
        /// #     .early_stopping_condition(MeasuredCondition::Iteration(2000))
        /// # .go();
        /// # 
        /// // `net` is neural network trained to 100% accuracy to mimic an XOR gate.
        /// let (correct_vector,confusion_matrix) = net.analyze_string(&mut data,2,None);
        /// 
        /// let expected_vector:&str = 
        /// "    0    1   
        ///   ┌           ┐
        /// % │ 1.00 1.00 │
        ///   └           ┘\n";
        /// assert_eq!(&correct_vector,expected_vector);
        /// 
        /// let expected_matrix:&str = 
        /// "%   0    1   
        ///   ┌           ┐
        /// 0 │ 1.00 0.00 │
        /// 1 │ 0.00 1.00 │
        ///   └           ┘\n";
        /// assert_eq!(&confusion_matrix,expected_matrix);
        /// ```
        /// Example with dictionairy:
        /// ```ignore
        /// # use cogent::core::{EvaluationData,MeasuredCondition,Activation,Layer,NeuralNetwork};
        /// # use std::collections::HashMap;
        /// # 
        /// # let mut net = NeuralNetwork::new(2,&[
        /// #     Layer::new(3,Activation::Sigmoid),
        /// #     Layer::new(2,Activation::Softmax)
        /// # ]);
        /// # 
        /// let mut data = vec![
        ///     (vec![0f32,0f32],0usize),
        ///     (vec![1f32,0f32],1usize),
        ///     (vec![0f32,1f32],1usize),
        ///     (vec![1f32,1f32],0usize)
        /// ];
        /// 
        /// # net.train(&data)
        /// #     .learning_rate(2f32)
        /// #     .evaluation_data(EvaluationData::Actual(&data)) // Use testing data as evaluation data.
        /// #     .early_stopping_condition(MeasuredCondition::Iteration(2000))
        /// # .go();
        /// # 
        /// let mut dictionairy:HashMap<usize,&str> = HashMap::new();
        /// dictionairy.insert(0,"False");
        /// dictionairy.insert(1,"True");
        /// 
        /// // `net` is neural network trained to 100% accuracy to mimic an XOR gate.
        /// let (correct_vector,confusion_matrix) = net.analyze_string(&mut data,2,Some(dictionairy));
        /// 
        /// let expected_vector:&str = 
        /// "     False True 
        ///   ┌              ┐
        /// % │  1.00  1.00  │
        ///   └              ┘\n";
        /// assert_eq!(&correct_vector,expected_vector);
        /// 
        /// let expected_matrix:&str = 
        /// "    %    False True 
        ///       ┌              ┐
        /// False │  1.00  0.00  │
        ///  True │  0.00  1.00  │
        ///       └              ┘\n";
        /// assert_eq!(&confusion_matrix,expected_matrix);
        /// ```
        #[deprecated(note = "Not deprecated, just broken until ArrayFire update installer to match git (where issue has been reported and fixed).")]
        pub fn analyze_string(&mut self, data:&mut [(Vec<f32>,usize)],precision:usize,dict_opt:Option<HashMap<usize,&str>>) -> (String,String) {
            let (vector,matrix) = self.analyze(data);

            let class_outs = match &self.layers[self.layers.len()-1] {
                InnerLayer::Dense(dense_layer) => dense_layer.biases.dims().get()[0] as usize,
                _ => panic!("Last layer is somehow a dropout layer, this should not be possible")
            };

            let classes:Vec<String> = if let Some(dictionary) = dict_opt {
                (0..class_outs).map(|class|
                    if let Some(label) = dictionary.get(&class) { 
                        String::from(*label) 
                    } else { format!("{}",class) } // TODO Do this conversion better
                ).collect()
            } else {
                (0..class_outs).map(|class| format!("{}",class) ).collect() // TODO See above todo
            };

            let widest_class:usize = classes.iter().fold(1usize,|max,x| std::cmp::max(max,x.chars().count()));
            let class_spacing:usize = std::cmp::max(precision+2,widest_class);

            let vector_string = vector_string(&vector,&classes,precision,class_spacing);
            let matrix_string = matrix_string(&matrix,&classes,precision,widest_class,class_spacing);

            return (vector_string,matrix_string);

            fn vector_string(vector:&Vec<f32>,classes:&Vec<String>,precision:usize,spacing:usize) -> String {
                let mut string = String::new(); // TODO Change this to `::with_capacity();`
                
                let precision_width = precision+2;
                let space_between_vals = spacing-precision_width+1;
                let row_width = ((spacing+1) * vector.len()) + space_between_vals;

                string.push_str(&format!("  {:1$}","",space_between_vals));
                for class in classes {
                    string.push_str(&format!(" {:1$}",class,spacing));
                }
                string.push_str("\n");
                string.push_str(&format!("{:1$}","",2));
                string.push_str(&format!("┌{:1$}┐\n","",row_width));
                string.push_str(&format!("% │{:1$}","",space_between_vals));
                for val in vector {
                    string.push_str(&format!("{:.1$}",val,precision));
                    string.push_str(&format!("{:1$}","",space_between_vals))
                }
                string.push_str("│\n");
                string.push_str(&format!("{:1$}","",2));
                string.push_str(&format!("└{:1$}┘\n","",row_width));

                return string;
            }
            fn matrix_string(matrix:&Vec<Vec<f32>>,classes:&Vec<String>,precision:usize,class_width:usize,spacing:usize) -> String {
                let mut string = String::new(); // TODO Change this to `::with_capacity();`
                let precision_width = precision+2;
                let space_between_vals = spacing-precision_width+1;
                let row_width = ((spacing+1) * matrix[0].len()) + space_between_vals;

                string.push_str(&format!("{:2$}% {:3$}","","",class_width-1,space_between_vals));
                
                for class in classes {
                    string.push_str(&format!(" {:1$}",class,spacing));
                }
                string.push_str("\n");
                
                string.push_str(&format!("{:2$} ┌{:3$}┐\n","","",class_width,row_width));

                for i in 0..matrix.len() {
                    string.push_str(&format!("{: >2$} │{:3$}",classes[i],"",class_width,space_between_vals));
                    for val in matrix[i].iter() {
                        string.push_str(&format!("{:.1$}",val,precision));
                        string.push_str(&format!("{:1$}","",space_between_vals))
                    }
                    string.push_str("│\n");
                }
                string.push_str(&format!("{:2$} └{:3$}┘\n","","",class_width,row_width));

                return string;
            }
        }
        // TODO Document this better
        // Converts [Vec<f32>,usize] to (Array<f32>,Array<f32>).
        fn matrixify(&self,examples:&[(Vec<f32>,usize)]) -> (Array<f32>,Array<f32>) { 
            // Gets number of inputs.
            let in_len = self.inputs;
            // Gets number of outputs.
            let out_len = match &self.layers[self.layers.len()-1] {
                InnerLayer::Dense(dense_layer) => dense_layer.biases.dims().get()[0] as usize,
                _ => panic!("Last layer is somehow a dropout layer, this should not be possible")
            };
            // Gets number of exampels.
            let example_len = examples.len();

            // TODO Is there a better way to do either of these?
            // Flattens examples into `in_vec` and `out_vec`
            let in_vec:Vec<f32> = examples.iter().flat_map(|(input,_)| (*input).clone()).collect();
            let out_vec:Vec<f32> = examples.iter().flat_map(|(_,class)| { 
                let mut vec = vec!(0f32;out_len);
                vec[*class]=1f32;
                return vec;
            }).collect();

            // Constructs input and output array
            let input:Array<f32> = Array::<f32>::new(&in_vec,Dim4::new(&[in_len as u64,example_len as u64,1,1]));
            let output:Array<f32> = Array::<f32>::new(&out_vec,Dim4::new(&[out_len as u64,example_len as u64,1,1]));
            
            // Returns input and output array
            // Array(in,examples,1,1), Array(out,examples,1,1)
            return (input,output);
        }
        // Returns Instant::elapsed() as hh:mm:ss string.
        fn time(instant:Instant) -> String {
            let mut seconds = instant.elapsed().as_secs();
            let hours = (seconds as f32 / 3600f32).floor();
            seconds = seconds % 3600;
            let minutes = (seconds as f32 / 60f32).floor();
            seconds = seconds % 60;
            let time = format!("{:#02}:{:#02}:{:#02}",hours,minutes,seconds);
            return time;
        }
        /// Exports neural network to `path.json`.
        /// ```ignore
        /// use cogent::core::{Activation,Layer,NeuralNetwork};
        /// 
        /// let net = NeuralNetwork::new(2,&[
        ///     Layer::new(3,Activation::Sigmoid),
        ///     Layer::new(2,Activation::Softmax)
        /// ]);
        /// 
        /// net.export("my_neural_network");
        /// ```
        pub fn export(&self,path:&str) {
            let mut layers:Vec<InnerLayerEnum> = Vec::with_capacity(self.layers.len()-1);

            for layer in self.layers.iter() {
                layers.push(match layer {
                    InnerLayer::Dropout(dropout_layer) => InnerLayerEnum::Dropout(dropout_layer.p),
                    InnerLayer::Dense(dense_layer) => {
                        let mut bias_holder = vec!(f32::default();dense_layer.biases.elements());
                        let mut weight_holder = vec!(f32::default();dense_layer.weights.elements());
                        dense_layer.biases.host(&mut bias_holder);
                        dense_layer.weights.host(&mut weight_holder);
                        InnerLayerEnum::Dense(
                            dense_layer.activation,
                            *dense_layer.biases.dims().get(),
                            bias_holder,
                            *dense_layer.weights.dims().get(),
                            weight_holder
                        )
                    },
                });
            }

            let export_struct = ImportExportNet { inputs:self.inputs, layers };

            let file = File::create(format!("{}.json",path));
            let serialized:String = serde_json::to_string(&export_struct).unwrap();
            file.unwrap().write_all(serialized.as_bytes()).unwrap();
        }
        /// Imports neural network from `path.json`.
        /// ```ignore
        /// use cogent::core::NeuralNetwork;
        /// let net = NeuralNetwork::import("my_neural_network");
        /// ```
        pub fn import(path:&str) -> NeuralNetwork {
            let file = File::open(format!("{}.json",path));
            let mut string_contents:String = String::new();
            file.unwrap().read_to_string(&mut string_contents).unwrap();
            let import_struct:ImportExportNet = serde_json::from_str(&string_contents).unwrap();

            let mut layers:Vec<InnerLayer> = Vec::with_capacity(import_struct.layers.len());

            for layer in import_struct.layers {
                layers.push(match layer {
                    InnerLayerEnum::Dropout(p) => InnerLayer::Dropout(DropoutLayer::new(p)),
                    InnerLayerEnum::Dense(activation,b_dims,biases,w_dims,weights) => {
                        InnerLayer::Dense(DenseLayer{
                            activation,
                            biases:Array::new(&biases,Dim4::new(&b_dims)),
                            weights:Array::new(&weights,Dim4::new(&w_dims))
                        })
                    }
                });
            }

            return NeuralNetwork { inputs:import_struct.inputs, layers };
        }
    }
}