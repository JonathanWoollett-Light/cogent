use std::time::Duration;

// Default percentage of training data to set as evaluation data (0.1=5%).
const DEFAULT_EVALUTATION_DATA: EvaluationData = EvaluationData::Percent(0.05);
// Default percentage of size of training data to set batch size (0.01=1%).
const DEFAULT_BATCH_SIZE: Proportion = Proportion::Percent(0.01);
// Default learning rate.
const DEFAULT_LEARNING_RATE: f32 = 0.1f32;
// Default interval in iterations before early stopping.
// early stopping = default early stopping * (size of examples / number of examples) Iterations
const DEFAULT_EARLY_STOPPING: MeasuredCondition = MeasuredCondition::Iteration(10);
// Default percentage minimum positive accuracy change required to prevent early stopping or learning rate decay (0.005=0.5%).
const DEFAULT_EVALUATION_MIN_CHANGE: Proportion = Proportion::Percent(0.001);
// Default amount to decay learning rate after period of negligible change.
// `new learning rate = learning rate decay * old learning rate`
const DEFAULT_LEARNING_RATE_DECAY: f32 = 0.5f32;
// Default interval in iterations before learning rate decay.
// interval = default learning rate interval * (size of examples / number of examples) iterations.
const DEFAULT_LEARNING_RATE_INTERVAL: MeasuredCondition = MeasuredCondition::Iteration(10);

/// For setting `evaluation_data`.
pub enum EvaluationData<'a> {
    /// Set as a given number of examples from training data.
    Scalar(usize),
    /// Set as a given percentage of examples from training data.
    Percent(f32),
    /// Set as a given dataset.
    Actual(&'a ndarray::Array2<f32>, &'a ndarray::Array2<usize>),
}
impl<'a> Default for EvaluationData<'a> {
    fn default() -> Self { DEFAULT_EVALUTATION_DATA }
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


pub struct BatchSize(pub Proportion);
impl Default for BatchSize  {
    fn default() -> Self { BatchSize(DEFAULT_BATCH_SIZE) }
}
pub struct EvaluationMinChange(pub Proportion);
impl Default for EvaluationMinChange  {
    fn default() -> Self { EvaluationMinChange(DEFAULT_EVALUATION_MIN_CHANGE) }
}


pub struct EarlyStoppingCondition(pub MeasuredCondition);
impl Default for EarlyStoppingCondition {
    fn default() -> Self { EarlyStoppingCondition(DEFAULT_EARLY_STOPPING) }
}
pub struct LearningRateInterval(pub MeasuredCondition);
impl Default for LearningRateInterval {
    fn default() -> Self { LearningRateInterval(DEFAULT_LEARNING_RATE_INTERVAL) }
}


pub struct LearningRate(pub f32);
impl Default for LearningRate {
    fn default() -> Self { LearningRate(DEFAULT_LEARNING_RATE) }
}
pub struct LearningRateDecay(pub f32);
impl Default for LearningRateDecay {
    fn default() -> Self { LearningRateDecay(DEFAULT_LEARNING_RATE_DECAY) }
}