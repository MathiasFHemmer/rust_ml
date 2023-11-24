use network::{Network, Input};
use rand::{thread_rng, seq::SliceRandom};
use std::io;

use crate::activation_function::SIGMOID;

pub mod matrix;
pub mod network;
pub mod activation_function;

fn main() {
    let mut network = Network::new(vec![2,2,2], vec![SIGMOID, SIGMOID]);
    let mut input = String::new();

    let mut inputs = Vec::new();
    for _ in 0..100 {
        inputs.push(Input::new(vec![0.0, 0.0], vec![0.0, 1.0]));
        inputs.push(Input::new(vec![0.0, 1.0], vec![0.0, 1.0]));
        inputs.push(Input::new(vec![1.0, 0.0], vec![0.0, 1.0]));
        inputs.push(Input::new(vec![1.0, 1.0], vec![1.0, 0.0]));
    }

    let mut rng = thread_rng();
    inputs.shuffle(&mut rng);

    // Read a line of input from the console
    for _ in 0..5000 {
        let output = network.feed_foward(&inputs[0].data);
        //print!("{:#?}", output);    
    }

    print!("{:#?}", network.feed_foward(&vec![0.0, 0.0]));
    print!("{:#?}", network.feed_foward(&vec![1.0, 0.0]));
    print!("{:#?}", network.feed_foward(&vec![0.0, 1.0]));
    print!("{:#?}", network.feed_foward(&vec![1.0, 1.0]));
}
