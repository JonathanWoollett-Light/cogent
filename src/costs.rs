use serde::{Deserialize, Serialize};

use arrayfire::{log, pow, sum_all, Array};

/// Defines cost function of a neural network.
#[derive(Serialize, Deserialize)]
pub enum Cost {
    /// Quadratic cost function.
    ///
    /// $ C(w,b)=\frac{1}{2n}\sum_{x} ||y(x)-a(x) ||^2 $
    Quadratic,
    /// Crossentropy cost function.
    ///
    /// $ C(w,b) = -\frac{1}{n} \sum_{x} (y(x) \ln{(a(x))}  + (1-y(x)) \ln{(1-a(x))}) $
    Crossentropy,
}
impl Cost {
    /// Runs cost functions.
    ///
    /// y: Target out, a: Actual out.
    pub fn run(&self, y: &Array<f32>, a: &Array<f32>) -> f32 {
        return match self {
            Self::Quadratic => quadratic(y, a),
            Self::Crossentropy => cross_entropy(y, a),
        };
        // Quadratic cost
        fn quadratic(y: &Array<f32>, a: &Array<f32>) -> f32 {
            sum_all(&pow(&(y - a), &2, false)).0 as f32 / (2f32 * a.dims().get()[0] as f32)
        }
        // Cross entropy cost
        // TODO Need to double check this
        fn cross_entropy(y: &Array<f32>, a: &Array<f32>) -> f32 {
            // Adds very small value to a, to prevent log(0)=nan
            let part1 = log(&(a + 1e-20)) * y;
            // Add very small value to prevent log(1-1)=log(0)=nan
            let part2 = log(&(1f32 - a + 1e-20)) * (1f32 - y);

            let mut cost: f32 = sum_all(&(part1 + part2)).0 as f32;

            //if cost.is_nan() { panic!("nan cost"); }

            cost /= -(a.dims().get()[0] as f32);

            return cost;
        }
    }
    /// Derivative w.r.t. layer output (∂C/∂a).
    ///
    /// y: Target out, a: Actual out.
    pub fn derivative(&self, y: &Array<f32>, a: &Array<f32>) -> Array<f32> {
        return match self {
            Self::Quadratic => a - y,
            Self::Crossentropy => {
                // TODO Double check we don't need to add a val to prevent 1-a=0 (commented out code below checks count of values where a>=1)
                //let check = sum_all(&arrayfire::ge(a,&1f32,false)).0;
                //if check != 0f64 { panic!("check: {}",check); }

                return (-1 * y) / a + (1f32 - y) / (1f32 - a);
            } // -y/a + (1-y)/(1-a)
        };
    }
}
impl Default for Cost {
    fn default() -> Self { Cost::Crossentropy }
}