use crate::activations::Activation;

use arrayfire::{constant, gt, matmul, mul, randu, sum, Array, Dim4, MatProp};

/// A dense layer.
pub struct DenseLayer {
    pub activation: Activation,
    pub biases: Array<f32>,
    pub weights: Array<f32>,
}
impl DenseLayer {
    // Constructs new `DenseLayer`
    pub fn new(from: u64, size: u64, activation: Activation) -> DenseLayer {
        if size == 0 {
            panic!("All dense layer sizes must be >0.");
        }
        return DenseLayer {
            activation,
            biases: (randu::<f32>(Dim4::new(&[size, 1, 1, 1])) * 2f32) - 1f32,
            weights: ((randu::<f32>(Dim4::new(&[size, from, 1, 1])) * 2f32) - 1f32)
                / (from as f32).sqrt(),
        };
    }
    // Constructs new `DenseLayer` using a given value for all weights and biases.
    pub fn new_constant(from: u64, size: u64, activation: Activation, val: f32) -> DenseLayer {
        if size == 0 {
            panic!("All dense layer sizes must be >0.");
        }
        return DenseLayer {
            activation,
            biases: constant(val, Dim4::new(&[size, 1, 1, 1])),
            weights: constant(val, Dim4::new(&[size, from, 1, 1])),
        };
    }
    // Forward propagates.
    pub fn forepropagate(&self, a: &Array<f32>, ones: &Array<f32>) -> (Array<f32>, Array<f32>) {
        let weighted_inputs: Array<f32> = matmul(&self.weights, &a, MatProp::NONE, MatProp::NONE);

        // Using batch `arrayfire::add` is sooo slow, this is why we do it like this
        let bias_matrix: Array<f32> = matmul(&self.biases, &ones, MatProp::NONE, MatProp::NONE);

        // z
        let input = weighted_inputs + bias_matrix;

        // a
        let activation = self.activation.run(&input);

        return (activation, input);
    }
    // TODO name `from_error` better
    // TODO We only need `training_set_length` if `l2 = Some()..`, how can we best pass `training_set_length`?
    // Backpropagates.
    // (Updates weights and biases during this process).
    pub fn backpropagate(
        &mut self,
        partial_error: &Array<f32>, // ∂C/∂a as formed by ∇(a)C or (w^{l+1})^T * δ^{l+1}
        z: &Array<f32>,             // l (input of this layer)
        a: &Array<f32>,             // l-1 (activation from previous layer)
        learning_rate: f32,
        l2: Option<f32>,
        training_set_length: usize,
    ) -> Array<f32> {
        // ∂C/∂z = ∂a/∂z * ∂C/∂a
        // (∂C/∂z = δ)
        let error = self.activation.derivative(z) * partial_error;

        // ∂C/∂b = ∂C/∂z
        let bias_error = sum(&error, 1);

        // ∂C/∂w = δ matmul a^T
        let weight_error = matmul(&error, a, MatProp::NONE, MatProp::TRANS);

        // ∂C/∂a^{l-1} = w^T matmul ∂C/∂z
        let nxt_partial_error = matmul(&self.weights, &error, MatProp::TRANS, MatProp::NONE);

        // Number of examples in batch
        let batch_len = z.dims().get()[1] as f32;

        // TODO Figure out best way to do weight and bias updates
        // = old weights - avg weight errors
        if let Some(lambda) = l2 {
            self.weights = ((1f32 - (learning_rate * lambda / training_set_length as f32))
                * &self.weights)
                - (learning_rate * weight_error / batch_len)
        } else {
            self.weights = &self.weights - (learning_rate * weight_error / batch_len);
        }

        // = old biases - avg bias errors
        self.biases = &self.biases - (learning_rate * bias_error / batch_len);

        // ∂C/∂a^{l-1}
        return nxt_partial_error;
    }
}
/// A dropout layer.
pub struct DropoutLayer {
    pub p: f32,
    mask: Array<f32>,
}
impl DropoutLayer {
    // Constructs new `DropoutLayer`
    pub fn new(p: f32) -> DropoutLayer {
        DropoutLayer {
            p,
            mask: Array::<f32>::new_empty(Dim4::new(&[1, 1, 1, 1])),
        }
    }
    // Forward propgates.
    // Creates a mask to fit given data.
    pub fn forepropagate(&mut self, z: &Array<f32>, ones: &Array<f32>) -> Array<f32> {
        // Sets mask dimensions
        let z_dims = z.dims();
        let z_dim_arr = z_dims.get();
        let mask_dims = Dim4::new(&[z_dim_arr[0], 1, 1, 1]);
        // TODO Look into using `tile`
        // Updates mask
        self.mask = matmul(
            &gt(&randu::<f32>(mask_dims), &self.p, false).cast::<f32>(),
            ones,
            MatProp::NONE,
            MatProp::NONE,
        );
        // Applies mask
        return mul(z, &self.mask, false);
    }
    // Backpropgates
    // Using mask used for last forepropgate (cannot backpropgate dropout layer without first forepropagating).
    pub fn backpropagate(&self, partial_error: &Array<f32>) -> Array<f32> {
        return mul(partial_error, &self.mask, false);
    }
}
