use serde::{Deserialize, Serialize};

use super::training::{Datapoint, Dataset, Genetic, Label};

#[derive(Debug, Serialize, Deserialize)]
pub enum ActivationFunction {
    Linear,
    ReLU,
    Sigmoid,
    Softmax,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum CostFunction {
    MSE,
    CCE,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Neuron {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub output: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Layer {
    pub neurons: Vec<Neuron>,
    pub function: ActivationFunction,
}

pub struct Construction;
pub struct Ready;

#[derive(Debug, Serialize, Deserialize)]
pub struct Network<Status = Construction> {
    input_size: usize,
    layers: Vec<Layer>,
    cost_function: CostFunction,
    marker: std::marker::PhantomData<Status>,
}

impl ActivationFunction {
    pub fn apply(&self, outputs: &Vec<f64>) -> Vec<f64> {
        match self {
            ActivationFunction::Linear => outputs.clone(),
            ActivationFunction::ReLU => outputs
                .iter()
                .map(|&output| f64::max(0.0, output))
                .collect(),
            ActivationFunction::Sigmoid => outputs
                .iter()
                .map(|&output| 1.0 / (1.0 + f64::exp(-output)))
                .collect(),
            ActivationFunction::Softmax => {
                let partition = outputs
                    .iter()
                    .fold(0.0, |accumulator, &output| accumulator + f64::exp(output));

                outputs
                    .iter()
                    .map(|&output| f64::exp(output) / partition)
                    .collect()
            }
        }
    }
}
impl Default for ActivationFunction {
    fn default() -> Self {
        Self::Linear
    }
}

impl CostFunction {
    pub fn apply(&self, outputs: &Vec<f64>, targets: Vec<f64>) -> f64 {
        match self {
            CostFunction::CCE => -ActivationFunction::Softmax
                .apply(outputs)
                .iter()
                .zip(targets)
                .map(|(&output, target)| target * f64::ln(output))
                .sum::<f64>(),
            CostFunction::MSE => {
                outputs
                    .iter()
                    .zip(targets)
                    .map(|(&output, target)| (target - output) * (target - output))
                    .sum::<f64>()
                    / outputs.len() as f64
            }
        }
    }
}
impl Default for CostFunction {
    fn default() -> Self {
        Self::MSE
    }
}

impl Neuron {
    pub fn random() -> f64 {
        rand::random::<f64>() * 2.0 - 1.0
    }

    pub fn new(size: usize) -> Self {
        Self {
            weights: (0..size).into_iter().map(|_| Self::random()).collect(),
            bias: Self::random(),
            output: Default::default(),
        }
    }

    pub fn weighted_sum(&mut self, inputs: &Vec<f64>) -> f64 {
        if inputs.len() != self.weights.len() {
            panic!(
                "Invalid dot product! {} != {}",
                inputs.len(),
                self.weights.len()
            );
        }

        self.output = self
            .weights
            .iter()
            .zip(inputs)
            .fold(0.0, |accumulator, (weight, input)| {
                accumulator + weight * input
            })
            + self.bias;

        self.output
    }
}
impl Genetic for Neuron {
    fn mutate(&mut self, alpha: f64) {
        self.weights.iter_mut().for_each(|weight| {
            *weight += Self::random() * alpha;
        });
        self.bias += Self::random() * alpha;
    }
}

impl Layer {
    pub fn new(input_size: usize, size: usize, function: ActivationFunction) -> Self {
        Self {
            neurons: (0..size)
                .into_iter()
                .map(|_| Neuron::new(input_size))
                .collect(),
            function,
        }
    }

    pub fn get_size(&self) -> usize {
        self.neurons.len()
    }

    pub fn outputs(&self) -> Vec<f64> {
        self.neurons.iter().map(|neuron| neuron.output).collect()
    }

    pub fn forward(&mut self, inputs: &Vec<f64>) -> Vec<f64> {
        let weighted_sums = self
            .neurons
            .iter_mut()
            .map(|neuron| neuron.weighted_sum(inputs))
            .collect();

        let activations = self.function.apply(&weighted_sums);
        activations
    }
}
impl Genetic for Layer {
    fn mutate(&mut self, alpha: f64) {
        self.neurons
            .iter_mut()
            .for_each(|neuron| neuron.mutate(alpha));
    }
}

impl Network {
    pub fn output_size(&self) -> usize {
        self.layers
            .last()
            .map_or(self.input_size, |layer| layer.get_size())
    }
}
impl Network<Construction> {
    pub fn new(input_size: usize) -> Self {
        Self {
            input_size,
            cost_function: CostFunction::default(),
            layers: Vec::new(),
            marker: std::marker::PhantomData::<Construction>,
        }
    }

    pub fn add_layer(mut self, size: usize, function: ActivationFunction) -> Self {
        self.layers
            .push(Layer::new(self.output_size(), size, function));
        self
    }

    pub fn build(self, cost_function: CostFunction) -> Network<Ready> {
        Network {
            input_size: self.input_size,
            cost_function,
            layers: self.layers,
            marker: std::marker::PhantomData::<Ready>,
        }
    }
}
impl Network<Ready> {
    pub fn run(&mut self, datapoint: &Datapoint) -> (Label, Vec<f64>) {
        let mut inputs = datapoint.inputs().clone();

        for layer in self.layers.iter_mut() {
            inputs = layer.forward(&inputs);
        }

        (Label::from(&inputs), inputs)
    }
    pub fn train(&mut self) {
        todo!()
    }
    pub fn cost(&mut self, dataset: Dataset) -> f64 {
        let mut cost = 0.0;

        for datapoint in dataset.datapoints().iter() {
            let (label, outputs) = self.run(datapoint);

            let loss = self.cost_function.apply(&outputs, datapoint.targets());
            cost += loss;
        }

        cost / dataset.size() as f64
    }

    pub fn serialize(&self, path: &str) {
        let contents = serde_json::to_string(&self).unwrap();
        std::fs::write(path, contents.as_str());
    }

    pub fn deserialize(path: &str) -> Self {
        let contents = std::fs::read_to_string(path).unwrap();
        serde_json::from_str(contents.as_str()).unwrap()
    }
}
impl Genetic for Network<Ready> {
    fn mutate(&mut self, alpha: f64) {
        self.layers.iter_mut().for_each(|layer| layer.mutate(alpha));
    }
}
