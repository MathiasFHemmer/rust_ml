use network::Network;

use crate::activation_function::SIGMOID;

pub mod matrix;
pub mod network;
pub mod activation_function;

fn main() {
    let mut network = Network::new(vec![2,2,2], vec![SIGMOID, SIGMOID]);

    let output = network.cost(vec![1.0, 1.0], vec![1.0, 0.0]);
    print!("{:#?}", output);
}
