#![allow(dead_code, unused)]

use model::{
    network::{ActivationFunction, CostFunction, Network, Ready},
    training::Datapoint,
};

mod model;

const CONFIG_PATH: &str = "./config/config.json";

#[derive(serde::Serialize, serde::Deserialize)]
struct Config {
    settings_path: String,
    test_path: String,
}

fn main() {
    let contents = std::fs::read_to_string(CONFIG_PATH).unwrap();
    let Config {
        settings_path,
        test_path,
    } = serde_json::from_str(contents.as_str()).unwrap();

    let datapoint = Datapoint::from(image::open(test_path).unwrap());
    let mut network = Network::deserialize(&settings_path);

    let (label, outputs) = network.run(&datapoint);

    println!("Label: {:?}, Outputs: {:?}", label, outputs);
}

fn create_network() -> Network<Ready> {
    Network::new(64 * 64 * 4)
        .add_layer(64, ActivationFunction::Sigmoid)
        .add_layer(64, ActivationFunction::Sigmoid)
        .add_layer(64, ActivationFunction::Sigmoid)
        .add_layer(2, ActivationFunction::Sigmoid)
        .build(CostFunction::MSE)
}
