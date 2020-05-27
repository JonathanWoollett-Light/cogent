use std::time::Duration;

/// For setting `evaluation_data`.
pub enum EvaluationData<'a> {
    /// Set as a given number of examples from training data.
    Scalar(usize),
    /// Set as a given percentage of examples from training data.
    Percent(f32),
    /// Set as a given dataset.
    Actual(&'a ndarray::Array2<f32>, &'a ndarray::Array2<usize>),
}
/// For setting a hyperparameter with measured intervals.
#[derive(Clone, Copy)]
pub enum MeasuredCondition {
    Iteration(u32),
    Duration(Duration),
}
/// For setting `halt_condition`.
///
/// The training halt condition.
#[derive(Clone, Copy)]
pub enum HaltCondition {
    /// Halt after completing a given number of iterations (epochs)
    Iteration(u32),
    /// Halt after a given duration has elapsed.
    Duration(Duration),
    /// Halt after acheiving a given accuracy.
    Accuracy(f32),
}
/// For setting a hyperparameter as a proportion of another.
#[derive(Clone, Copy)]
pub enum Proportion {
    Scalar(u32),
    Percent(f32),
}
