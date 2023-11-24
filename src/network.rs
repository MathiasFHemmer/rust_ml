use std::ops::IndexMut;

use crate::{matrix::Matrix, activation_function::ActivationFuncion};
#[derive(Default)]
pub struct Network<'a> {
    pub layers: Vec<usize>,
    pub weights: Vec<Matrix>,
    pub biases: Vec<Matrix>,
    
    pub weights_gradient: Vec<Matrix>,
    pub weighted_inputs:  Vec<Matrix>,
    pub activation_weighted_inputs: Vec<Matrix>,
    
    pub learning_rate: f64,
    pub activation_strategy: Vec<ActivationFuncion<'a>> 
}

pub struct Input{
    pub data: Vec<f64>,
    pub expected: Vec<f64>
}

impl Input {
    pub fn new(data: Vec<f64>, expected: Vec<f64>) -> Input{
        Input { data, expected }
    }
}

impl Network<'_> {
    pub fn new(layers: Vec<usize>, activations: Vec<ActivationFuncion>) -> Network {
        let mut weights = vec![];
        let mut biases = vec![];
        for index in 0..layers.len()-1{
            weights.push(Matrix::rand(layers[index+1], layers[index]));
            biases.push(Matrix::rand(layers[index+1], 1));
        }
        Network { layers, weights, biases, learning_rate: 0.05, activation_strategy: activations, ..Default::default()}
    }
}

impl Network<'_> {
    pub fn feed_foward(&mut self, inputs: &Vec<f64>) -> Vec<f64> {
        self.weighted_inputs = vec![Matrix::from_vec(&inputs).transpose()];
        let mut outputs = Matrix::from_vec(&inputs).transpose();
        for index in 0..self.layers.len()-1 {
            let weigthed_inputs = self.weights[index]
                .multiply(&outputs);
            
            let biased_inputs = weigthed_inputs
                .add(&self.biases[index]);
            
            self.weighted_inputs.push(biased_inputs.clone());
            
            outputs = biased_inputs.map_mut(self.activation_strategy[index].function);
            self.activation_weighted_inputs.push(outputs.clone());   
        }
        outputs.transpose().to_vec()
    }

    fn cost_node(value: &f64, target: &f64) -> f64 {
        (target - value).powi(2)
    }

    pub fn cost(&mut self, inputs: Vec<f64>, expected: Vec<f64>) -> f64{
        let outputs = self.feed_foward(&inputs);

        outputs
            .iter()
            .enumerate()
            .fold(0.0, |acc, (index, x)| acc + Network::cost_node(x, &expected[index]))
            / (expected.len() as f64)
    }

    pub fn train(&mut self, inputs: &Vec<Input>) -> Vec<f64> {
        
        for sample in 0..inputs.len(){
            let outputs = self.feed_foward(&inputs[sample].data);
            let mut node_values = self.output_node_values(inputs, sample, outputs);
                 
            for index in (0..self.layers.len() - 1).rev() {
                
            }
    
            for index in 0..self.weights.len(){
                self.weights[index] = self.weights[index].add(&self.weights_gradient[index])
            }
        }

        self.feed_foward(&inputs[0].data)
    }

    fn output_node_values(&mut self, inputs: &Vec<Input>, sample: usize, outputs: Vec<f64>) -> Matrix {
        let mut node_values = vec![0.0; inputs[sample].expected.len()];
        let curr_expected = &inputs[sample].expected;
    
        for index in 0..curr_expected.len(){
            let cost_derivative = outputs[index]-curr_expected[index];
            let activation_derivative = (self.activation_strategy.last().unwrap().derivative)(self.weighted_inputs.last().unwrap().data[index]);
            node_values[index] = cost_derivative*activation_derivative;
        }
        Matrix::from_vec(&node_values).transpose()
    }
}