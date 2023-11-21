use std::ops::IndexMut;

use crate::{matrix::Matrix, activation_function::ActivationFuncion};
#[derive(Default)]
pub struct Network<'a> {
    pub layers: Vec<usize>,
    pub weights: Vec<Matrix>,
    pub biases: Vec<Matrix>,
    pub data: Vec<Matrix>,
    pub learning_rate: f64,
    pub activation_strategy: Vec<ActivationFuncion<'a>> 
}

impl Network<'_> {
    pub fn new(layers: Vec<usize>, activations: Vec<ActivationFuncion>) -> Network {
        let mut weights = vec![];
        let mut biases = vec![];
        for index in 0..layers.len()-1{
            weights.push(Matrix::rand(layers[index+1], layers[index]));
            biases.push(Matrix::rand(layers[index+1], 1));
        }
        Network { layers, weights, biases, data: vec![], learning_rate: 0.1, activation_strategy: activations}
    }
}

impl Network<'_> {
    pub fn feed_foward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        self.data = vec![Matrix::from_vec(&inputs).transpose()];
        let mut inputs = Matrix::from_vec(&inputs).transpose();
        for index in 0..self.layers.len()-1 {
            let weigthed_inputs = self.weights[index].multiply(&inputs);
            let biased_inputs = weigthed_inputs.add(&self.biases[index]);
            
            inputs = biased_inputs.map_mut(self.activation_strategy[index].function);
            
            self.data.push(inputs.transpose().clone())
        }
        inputs.transpose().to_vec()
    }

    fn cost_node(value: &f64, target: &f64) -> f64 {
        (target - value).powi(2)
    }

    pub fn cost(&mut self, inputs: Vec<f64>, expected: Vec<f64>) -> f64{
        let outputs = self.feed_foward(inputs);

        outputs
            .iter()
            .enumerate()
            .fold(0.0, |acc, (index, x)| acc + Network::cost_node(x, &expected[index]))
            / (expected.len() as f64)
    }
}