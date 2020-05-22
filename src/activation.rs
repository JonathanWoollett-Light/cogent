use serde::{Deserialize, Serialize};

use arrayfire::{
    and, constant, exp, gt, matmul, max, maxof, pow, sigmoid, sum, tanh, Array, Dim4, MatProp,
};

/// Defines activations of layers in neural network.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum Activation {
    /// Sigmoid activation functions.
    ///
    /// $ A(z)=\frac{1}{1+e^-z} $
    Sigmoid,
    /// Tanh activation functions.
    ///
    /// $ A(z)=\frac{2}{1+e^{-2z}}-1 $
    Tanh,
    /// Softmax activation function.
    ///
    /// $ A(\begin{bmatrix}z_1,\dots,z_k\end{bmatrix})=\begin{bmatrix}\frac{e^{z_1}}{\Sigma_{i=1}^k e^{z_i}} & \dots &\frac{e^{z_k}}{\Sigma_{i=1}^k e^{z_i}}\end{bmatrix} $
    Softmax,
    /// ReLU activation function.
    ///
    /// $ A(z)=max(z,0) $
    ReLU, // Name it 'ReLU' or 'Relu'?
}
impl Activation {
    /// Computes activations given inputs.
    pub fn run(&self, z: &Array<f32>) -> Array<f32> {
        return match self {
            Self::Sigmoid => sigmoid(z),
            Self::Tanh => tanh(z),
            Self::Softmax => Activation::softmax(z),
            Self::ReLU => Activation::relu(z),
        };
    }
    // Derivative wrt layer input (∂a/∂z)
    pub fn derivative(&self, z: &Array<f32>) -> Array<f32> {
        // What should we name the derivative functions?
        return match self {
            Self::Sigmoid => sigmoid_derivative(z),
            Self::Tanh => tanh_derivative(z),
            Self::Softmax => softmax_derivative(z),
            Self::ReLU => relu_derivative(z),
        };

        // Derivative of sigmoid
        // s' = s(1-s)
        fn sigmoid_derivative(z: &Array<f32>) -> Array<f32> {
            let s = sigmoid(z);
            return s.clone() * (1f32 - s); // TODO Can we remove the clone here?
        }
        // Derivative of sigmoid
        // t' = 1-t^2
        fn tanh_derivative(z: &Array<f32>) -> Array<f32> {
            1 - pow(&tanh(z), &2, false)
        }
        // Derivative of softmax
        // e^z * (sum of other inputs e^input) / (sum of all inputs e^input)^2 = e^z * (exp_sum-e^z) / (exp_sum)^2
        fn softmax_derivative(z: &Array<f32>) -> Array<f32> {
            let exponents = exp(z);
            //af_print!("exponents",exponents);
            // Gets sum of each example (column)
            let sums = sum(&exponents, 0);
            //af_print!("sums",sums);
            // Sets squared sum of each example
            let sqrd_sums = pow(&sums, &2, false); // is this better than `&sums*&sums`?
                                                   //af_print!("sqrd_sums",sqrd_sums);
            let ones = constant(1f32, Dim4::new(&[z.dims().get()[0], 1, 1, 1]));
            //af_print!("ones",ones);
            let sums_matrix = matmul(&ones, &sums, MatProp::NONE, MatProp::NONE);
            //af_print!("sums_matrix",sums_matrix);
            let sums_sub = sums_matrix - &exponents;
            //af_print!("sums_sub",sums_sub);
            // TODO Is it more efficient to do this matrix multiplication before or after squaring?
            let sqrd_sums_matrix = matmul(&ones, &sqrd_sums, MatProp::NONE, MatProp::NONE);
            //af_print!("sqrd_sums_matrix",sqrd_sums_matrix);
            let derivatives = exponents * sums_sub / sqrd_sums_matrix;

            return derivatives;
        }
        //Deritvative of ReLU
        // ReLU(z)/1 = if >0 1 else 0
        fn relu_derivative(z: &Array<f32>) -> Array<f32> {
            // return Activation::relu(z) / z;
            // Follow code replaces the above line.
            // Above line replaced becuase it is prone to floating point error leading to f32:NAN.
            // Similar performance.
            let gt = gt(z, &0f32, false);
            return and(z, &gt, false);
        }
    }
    // TODO Make this better
    // Applies softmax activation
    fn softmax(y: &Array<f32>) -> Array<f32> {
        let ones = constant(1f32, Dim4::new(&[y.dims().get()[0], 1, 1, 1]));
        //af_print!("ones",ones);

        // Subtracts example max output from all example outputs.
        //  Allowing softmax to handle large values in y.
        // ------------------------------------------------
        // Gets max values in each example
        let max_axis_vals = max(&y, 0);
        // Matrix where each value is example max
        let max_axis_vals_matrix = matmul(&ones, &max_axis_vals, MatProp::NONE, MatProp::NONE);
        // All values minus there example maxes
        let max_reduced = y - max_axis_vals_matrix;

        // Applies softmax
        // ------------------------------------------------
        // Apply e^(x) to every value in matrix
        let exp_matrix = exp(&max_reduced);
        // Calculates sums of examples
        let row_sums = sum(&exp_matrix, 0);
        // Matrix where each value is example sum
        let row_sums_matrix = matmul(&ones, &row_sums, MatProp::NONE, MatProp::NONE);
        // Divides each value by example sum
        let softmax = exp_matrix / row_sums_matrix; // TODO Could this div be done using batch operation with `arrayfire::div(...)` using `row_sums`?

        return softmax;
    }
    // Applies ReLU activation
    fn relu(y: &Array<f32>) -> Array<f32> {
        let zeros = constant(0f32, y.dims());
        return maxof(y, &zeros, false);
    }
}
