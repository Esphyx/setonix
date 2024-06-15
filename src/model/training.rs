use image::{DynamicImage, GenericImage, GenericImageView};
use strum::EnumCount;
use strum_macros::EnumCount;

pub trait Genetic {
    fn mutate(&mut self, alpha: f64);
}

#[derive(Debug)]
pub struct Dataset {
    datapoints: Vec<Datapoint>,
}

#[derive(Debug)]
pub struct Datapoint {
    inputs: Vec<f64>,
    label: Label,
}

#[derive(Debug, Clone, Copy, EnumCount)]
pub enum Label {
    Real,
    Fake,
}

impl Dataset {
    pub fn datapoints(&self) -> &Vec<Datapoint> {
        &self.datapoints
    }
    pub fn size(&self) -> usize {
        self.datapoints.len()
    }
}

impl Datapoint {
    pub fn inputs(&self) -> &Vec<f64> {
        &self.inputs
    }
    pub fn targets(&self) -> Vec<f64> {
        self.label.one_hot()
    }
    pub fn add_noise(&mut self, alpha: f64) -> (Self, Vec<f64>) {
        let noise = self
            .inputs
            .iter()
            .map(|&input| {
                let (min, max) = (-input, 1.0 - input);
                (rand::random::<f64>() * (max - min) + min) * alpha
            })
            .collect::<Vec<f64>>();
        (
            Self {
                inputs: self
                    .inputs
                    .iter()
                    .zip(&noise)
                    .map(|(input, noise)| input + noise)
                    .collect(),
                label: self.label,
            },
            noise,
        )
    }
}
impl Into<DynamicImage> for &Datapoint {
    fn into(self) -> DynamicImage {
        const CHANNEL_COUNT: usize = 4;
        let length = self.inputs.len() as usize / CHANNEL_COUNT;

        let root = f64::sqrt(length as f64);
        let (mut width, mut height) = (root.floor() as usize, root.ceil() as usize);

        while width * height != length {
            if width * height > length {
                height -= 1;
            } else {
                width += 1;
            }
        }

        let mut image = image::DynamicImage::new_rgba8(width as u32, height as u32);

        (0..height).into_iter().for_each(|y| {
            (0..width).into_iter().for_each(|x| {
                let index = (x + y * width) * CHANNEL_COUNT;

                let values: Vec<u8> = (&self.inputs[index..index + 4])
                    .iter()
                    .map(|channel| (channel * 255.0) as u8)
                    .collect();

                if let &[r, g, b, a] = values.as_slice() {
                    image.put_pixel(x as u32, y as u32, image::Rgba([r, g, b, a]));
                }
            });
        });

        image
    }
}
impl From<DynamicImage> for Datapoint {
    fn from(value: DynamicImage) -> Self {
        let mut inputs = Vec::new();

        let (width, height) = value.dimensions();

        (0..height).for_each(|y| {
            (0..width).for_each(|x| {
                let channels = &mut value
                    .get_pixel(x, y)
                    .0
                    .iter()
                    .map(|&channel| channel as f64 / 256.0)
                    .collect();

                inputs.append(channels);
            });
        });

        Self {
            label: Label::Real,
            inputs,
        }
    }
}

impl Label {
    pub fn one_hot(&self) -> Vec<f64> {
        match self {
            Label::Real => [1.0, 0.0],
            Label::Fake => [0.0, 1.0],
        }
        .to_vec()
    }
}
impl From<&Vec<f64>> for Label {
    fn from(outputs: &Vec<f64>) -> Self {
        if outputs.len() != Label::COUNT {
            panic!("Outputs are invalid!");
        }

        let (index, _) = outputs
            .iter()
            .enumerate()
            .max_by(|&(_, a), &(_, b)| a.total_cmp(b))
            .unwrap();

        if index == 0 {
            Self::Real
        } else {
            Self::Fake
        }
    }
}
