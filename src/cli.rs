use clap::Parser;
use na::{Complex, ComplexField};
use crate::{IndexList};
use crate::maths::isqrt;
use std::str::FromStr;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct Configuration {
    /// An output file.
    #[arg(short, long, default_value = "./negie.png")]
    pub output: std::path::PathBuf,
    /// Dimension of the random square matrix to generate.
    #[arg(short = 'N', long, default_value = "5")]
    pub dim: usize,
    /// Number of variables to use.
    #[arg(short = 'k', long, default_value = "2")]
    pub nvar: usize,
    /// Number of samples to take. Default is 300000.
    #[arg(short = 'n', long, default_value = "300000")]
    pub samples: usize,
    /// Output image's side length. Default is 12000.
    #[arg(short, long, default_value = "12000")]
    pub size: u32,
    /// Integer radius of each eigenvalue's marker on the output picture. Default is heuristically
    /// determined.
    #[arg(short, long, default_value = "default")]
    pub radius: RadiusDescriptor,
    /// Statistical distribution to use for sampling. Ex: uniformunits, uniformrects:20.0,
    /// normal:20.0
    #[arg(short = 'd', long, default_value = "uniformunits")]
    pub distrib: SamplingDistribution,
    /// Matrix to use. Overrides -N. Must be either "random" or a comma-separated list of complex
    /// numbers of length n^2 for some n. Coefficients are expected in column-first order.
    #[arg(short, long, default_value = "random")]
    pub matrix: ComplexMatrixDescriptor,
    /// Variable indices. Overrides -v. Semicolon-separated list of comma-separated triplets i,j,k to use t_k in
    /// the spot (i,j). 0,0,0;1,1,1 replaces the first two coefficients of the main diagonal with
    /// t_0, t_1 respectively. Use "random" for the default behavior.
    #[arg(short, long, default_value = "random")]
    pub variables: VariableIndicesDescriptor,
    /// Population to choose coefficients from when using a random matrix. Comma-separated list of
    /// complex numbers. Value "default" is [i,-i,0,1,.5].
    #[arg(short, long, default_value = "default")]
    pub population: ComplexPopulation
}

#[derive(Clone, Debug)]
pub enum RadiusDescriptor {
    Default,
    Radius(i32)
}

impl From<&str> for RadiusDescriptor {
    fn from(value: &str) -> Self {
        match <i32 as FromStr>::from_str(value) {
            Ok(i) => Self::Radius(i),
            Err(_) => Self::Default
        }
    }
}

#[derive(Clone, Debug)]
pub enum VariableIndicesDescriptor {
    Random,
    List { list : IndexList }
}

impl From<&str> for VariableIndicesDescriptor {
    fn from(value: &str) -> Self {
        if value == "random" {
            Self::Random
        } else {
            let parts : IndexList = value.split(";").map(|s| {
                let ijk : Vec<usize> = s.split(",").map(|x| { usize::from_str(x).expect("Could not parse index in variable index list") }).collect();
                assert_eq!(ijk.len(), 3, "Variable index element was not a triplet.");
                (ijk[0], ijk[1], ijk[2])
            }).collect();
            Self::List { list: parts }
        }
    }
}

/// Types of distribution to use for sampling
#[derive(Clone, Debug)]
pub enum SamplingDistribution {
    Normal { s: f64 },
    UniformRects { r: f64 },
    UniformUnits
}

impl From<&str> for SamplingDistribution {
    fn from(value: &str) -> Self {
        if value == "taurus" || value == "units" || value == "uniformunits" {
            return Self::UniformUnits
        }
        let parts: Vec<&str> = value.split(':').collect();
        assert_eq!(parts.len(), 2);

        let word = parts[0].to_string();
        let num = parts[1].parse::<f64>().map_err(|_| "Unable to parse number").expect("Could not parse f64 in distribution");

        if word == "uniform" || word == "uniformrects" || word == "rects" {
            return Self::UniformRects { r: num }
        };
        if word == "normal" || word == "normalrects" {
            return Self::Normal { s: num }
        };
        panic!("Could not parse distribution")
    }
}
#[derive(Clone, Debug)]
pub enum ComplexMatrixDescriptor {
    Random,
    Matrix { coefs: Vec<Complex<f64>>, size: usize }
}

impl From<&str> for ComplexMatrixDescriptor {
    fn from(value: &str) -> Self {
        if value == "random" {
            Self::Random
        } else {
            let coefs: Vec<&str> = value.split(",").collect();
            assert_ne!(1, coefs.len(), "Can't operate on 1x1 matrix;");
            let size = isqrt(coefs.len());
            assert_eq!(size*size, coefs.len(), "Provided matrix was not a square matrix");
            Self::Matrix {
                coefs: coefs.iter()
                    .map(|s| {
                        Complex::from_str(s).expect("Could nor parse coefficient in the provided matrix!")
                    })
                    .collect(),
                size
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct ComplexPopulation {
    pub(crate) pop: Vec<Complex<f64>>
}

impl From<&str> for ComplexPopulation {
    fn from(value: &str) -> Self {
        if value == "default" {
            Self { pop: vec![Complex::i(), -Complex::i(), Complex::from_real(0.), Complex::from_real(1.), Complex::from_real(0.5)] }
        } else {
            let coefs: Vec<&str> = value.split(",").collect();
            Self {
                pop: coefs.iter()
                    .map(|s| {
                        Complex::from_str(s).expect("Could nor parse coefficient in the provided matrix!")
                    })
                    .collect()
            }
        }
    }
}
