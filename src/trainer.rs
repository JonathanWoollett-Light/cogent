use crate::costs::Cost;
use crate::neural_network::NeuralNetwork;
use crate::setter_enums::*;

use ndarray::{Array2, ArrayViewMut2, Axis};

use rand::prelude::*; // TODO Make this more specific to required functionality

/// To practicaly implement optional setting of training hyperparameters.
pub struct Trainer<'a> {
    pub training_data: &'a mut ndarray::Array2<f32>,
    pub training_labels: &'a mut ndarray::Array2<usize>,
    pub evaluation_dataset: EvaluationData<'a>,
    pub cost: Cost,
    // Will halt after at a certain iteration, accuracy or duration.
    pub halt_condition: Option<HaltCondition>,
    // Can log after a certain number of iterations, a certain duration, or not at all.
    pub log_interval: Option<MeasuredCondition>,
    pub batch_size: usize,
    pub learning_rate: f32,
    // Lambda value if using L2
    pub l2: Option<f32>,
    // Can stop after a lack of cost improvement over a certain number of iterations/durations, or not at all.
    pub early_stopping_condition: MeasuredCondition,
    // Minimum change required to log positive evaluation change.
    pub evaluation_min_change: Proportion,
    // Amount to decrease learning rate by (less than 1)(`learning_rate` *= learning_rate_decay`).
    pub learning_rate_decay: f32,
    // Time without notable improvement to wait until decreasing learning rate.
    pub learning_rate_interval: MeasuredCondition,
    // Duration/iterations between outputting neural network weights and biases to file.
    pub checkpoint_interval: Option<MeasuredCondition>,
    // Sets what to pretend to checkpoint files. Used to differentiate between nets when checkpointing multiple.
    pub name: Option<&'a str>,
    // Whether to print percantage progress in each iteration of backpropagation
    pub tracking: bool,
    pub neural_network: &'a mut NeuralNetwork,
}
impl<'a> Trainer<'a> {
    /// Sets `evaluation_data`.
    ///
    /// `evaluation_data` determines how to set the evaluation data.
    pub fn evaluation_data(&mut self, evaluation_data: EvaluationData<'a>) -> &mut Trainer<'a> {
        // Checks data fits net
        if let EvaluationData::Actual(data, labels) = evaluation_data {
            self.neural_network.check_dataset(data, labels);
        }

        self.evaluation_dataset = evaluation_data;
        return self;
    }
    /// Sets `cost`.
    ///
    /// `cost` determines cost function of network.
    pub fn cost(&mut self, cost: Cost) -> &mut Trainer<'a> {
        self.cost = cost;
        return self;
    }
    /// Sets `halt_condition`.
    ///
    /// `halt_condition` sets after which Iteration/Duration or reached accuracy to stop training.
    pub fn halt_condition(&mut self, halt_condition: HaltCondition) -> &mut Trainer<'a> {
        self.halt_condition = Some(halt_condition);
        return self;
    }
    /// Sets `log_interval`.
    ///
    /// `log_interval` sets some amount of Iterations/Duration to print the cost and accuracy of the neural net.
    pub fn log_interval(&mut self, log_interval: MeasuredCondition) -> &mut Trainer<'a> {
        self.log_interval = Some(log_interval);
        return self;
    }
    /// Sets `batch_size`.
    pub fn batch_size(&mut self, batch_size: Proportion) -> &mut Trainer<'a> {
        self.batch_size = match batch_size {
            Proportion::Percent(percent) => {
                (self.training_data.len_of(Axis(0)) as f32 * percent) as usize
            }
            Proportion::Scalar(scalar) => scalar as usize,
        };
        return self;
    }
    /// Sets `learning_rate`.
    pub fn learning_rate(&mut self, learning_rate: f32) -> &mut Trainer<'a> {
        self.learning_rate = learning_rate;
        return self;
    }
    /// Sets lambda ($ \lambda $) for `l2`.
    ///
    /// If $ \lambda $ set, implements L2 regularization with $ \lambda $ value.
    pub fn l2(&mut self, lambda: f32) -> &mut Trainer<'a> {
        self.l2 = Some(lambda);
        return self;
    }
    /// Sets `early_stopping_condition`.
    ///
    /// `early_stopping_condition` sets some amount of Iterations/Duration to stop after without notable cost improvement.
    pub fn early_stopping_condition(
        &mut self,
        early_stopping_condition: MeasuredCondition,
    ) -> &mut Trainer<'a> {
        self.early_stopping_condition = early_stopping_condition;
        return self;
    }
    /// Sets `evaluation_min_change`.
    ///
    /// Minimum change required to log positive evaluation change.
    pub fn evaluation_min_change(&mut self, evaluation_min_change: Proportion) -> &mut Trainer<'a> {
        self.evaluation_min_change = evaluation_min_change;
        return self;
    }
    /// Sets `learning_rate_decay`.
    ///
    /// `learning_rate_decay` is the mulipliers by which to decay the learning rate.
    pub fn learning_rate_decay(&mut self, learning_rate_decay: f32) -> &mut Trainer<'a> {
        self.learning_rate_decay = learning_rate_decay;
        return self;
    }
    /// Sets `learning_rate_interval`.
    pub fn learning_rate_interval(
        &mut self,
        learning_rate_interval: MeasuredCondition,
    ) -> &mut Trainer<'a> {
        self.learning_rate_interval = learning_rate_interval;
        return self;
    }
    /// Sets `checkpoint_interval`.
    ///
    /// `checkpoint_interval` sets how often (if at all) to serialize and output neural network to .txt file.
    pub fn checkpoint_interval(
        &mut self,
        checkpoint_interval: MeasuredCondition,
    ) -> &mut Trainer<'a> {
        self.checkpoint_interval = Some(checkpoint_interval);
        return self;
    }
    /// Sets `name`
    ///
    /// `name` sets what to pretend to checkpoint files. Used to differentiate between nets when checkpointing multiple.
    pub fn name(&mut self, name: &'a str) -> &mut Trainer<'a> {
        self.name = Some(name);
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
        // Shuffles training dataset
        shuffle_dataset(self.training_data, self.training_labels);

        // Sets evaluation data
        let number_of_examples = self.training_data.len_of(Axis(0));

        // TODO Make this better (remove the `.to_owned()`s and `.clone()`s).
        //  If `.split_at()` could return an `ArrayView` and an `ArrayViewMut` this would make this easier, maybe put feature request on ndarray github?
        let ((eval_data, train_data), (eval_labels, train_labels)): (
            (Array2<f32>, ArrayViewMut2<f32>),
            (Array2<usize>, ArrayViewMut2<usize>),
        ) = match self.evaluation_dataset {
            EvaluationData::Scalar(scalar) => {
                let (e_data, t_data) = self.training_data.view_mut().split_at(Axis(0), scalar);
                let (e_labels, t_labels) =
                    self.training_labels.view_mut().split_at(Axis(0), scalar);
                ((e_data.to_owned(), t_data), (e_labels.to_owned(), t_labels))
            }
            EvaluationData::Percent(percent) => {
                let (e_data, t_data) = self
                    .training_data
                    .view_mut()
                    .split_at(Axis(0), (number_of_examples as f32 * percent) as usize);
                let (e_labels, t_labels) = self
                    .training_labels
                    .view_mut()
                    .split_at(Axis(0), (number_of_examples as f32 * percent) as usize);
                ((e_data.to_owned(), t_data), (e_labels.to_owned(), t_labels))
            }
            EvaluationData::Actual(evaluation_data, evaluation_labels) => (
                (evaluation_data.clone(), self.training_data.view_mut()),
                (evaluation_labels.clone(), self.training_labels.view_mut()),
            ),
        };

        // Calls `inner_train` starting training.
        self.neural_network.inner_train(
            train_data,
            train_labels,
            eval_data.view(),
            eval_labels.view(),
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
// TODO Can this be consended with `neural_network::shuffle_dataset(..)`?
fn shuffle_dataset(data: &mut ndarray::Array2<f32>, labels: &mut ndarray::Array2<usize>) {
    let examples = data.len_of(Axis(0));
    let input_size = data.len_of(Axis(1));

    let mut data_slice = data.as_slice_mut().unwrap();
    let mut label_slice = labels.as_slice_mut().unwrap();

    for i in 0..examples - 1 {
        let new_index: usize = thread_rng().gen_range(i, examples);

        let (data_indx_1, data_indx_2) = (i * input_size, new_index * input_size);
        // TODO Can we swap slices better?
        for t in 0..input_size {
            swap(&mut data_slice, data_indx_1 + t, data_indx_2 + t);
        }
        swap(&mut label_slice, i, new_index);
    }

    fn swap<T: Copy>(list: &mut [T], a: usize, b: usize) {
        let temp = list[a];
        list[a] = list[b];
        list[b] = temp;
    }
}
