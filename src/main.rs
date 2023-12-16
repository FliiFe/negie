mod cli;
pub mod maths;
mod exif;

extern crate rand_distr;
extern crate image;
extern crate imageproc;

extern crate nalgebra as na;

use na::{DMatrix, Complex, RealField, DVector};
use image::{RgbImage, Rgb};

use rand::{seq::SliceRandom, Rng};
use rand_distr::{Normal, Distribution};

use std::time::Instant;

use clap::Parser;
use crate::cli::{Configuration, ComplexMatrixDescriptor, SamplingDistribution, VariableIndicesDescriptor, RadiusDescriptor};
use crate::exif::write_exif_data;
use crate::maths::{clamp, round};

fn main() {
    let mut config = Configuration::parse();
    let matrix = match config.matrix {
        ComplexMatrixDescriptor::Random => random_matrix(config.dim, config.population.pop.to_owned()),
        ComplexMatrixDescriptor::Matrix { coefs, size } => {
            config.dim = size;
            DMatrix::from_vec(size, size, coefs)
        }
    };
    config.matrix = ComplexMatrixDescriptor::Matrix {
        coefs: matrix.iter().cloned().collect(),
        size: matrix.nrows()
    };
    println!("{}", matrix);
    let variable_indices = match config.variables {
        VariableIndicesDescriptor::Random => random_indices_distinct_variables(config.dim, config
            .nvar),
        VariableIndicesDescriptor::List { list } => {
            config.nvar = list.iter().cloned().fold(0, |a, (i, j, k)| {
                assert!(i < config.dim && j < config.dim, "Index {},{} in variable indices list was out of boundss", i, j);
                a.max(k)
            }) + 1;
            list
        }
    };
    config.variables = VariableIndicesDescriptor::List { list: variable_indices.clone() };
    println!("Indices for variables: {:?}", variable_indices);
    let start = Instant::now();
    println!("Using distribution: {:?}", config.distrib);
    let samples = sample_ts(config.samples, config.nvar, config.distrib.to_owned());
    let (egvs, bounds) = collect_eigenvalues(&matrix, &variable_indices, config.nvar, &samples);
    println!("Eigenvalue collection took {:?}.", start.elapsed());
    let radius = match config.radius {
        RadiusDescriptor::Radius(i) => i,
        RadiusDescriptor::Default => i32::min(round(50. * (config.size as f64 / (config.samples as f64)))
                                                       .try_into().unwrap(), <u32 as TryInto<i32>>::try_into(config.size).unwrap() / 1000)
    };
    config.radius = RadiusDescriptor::Radius(radius);
    save_eigen_image(&egvs, bounds, &config);
    write_exif_data(&config);
}

type Bounds = (f64, f64, f64, f64);

fn save_eigen_image(eigenvalues: &Vec<Complex<f64>>, bounds: Bounds, config: &Configuration)
{
    let width = config.size;
    let height = width;
    let radius= match config.radius {
        RadiusDescriptor::Radius(i) => i,
        _ => 0
    };
    let path = config.output.as_path();
    let mut img = RgbImage::new(width, height);
    for x in 0..width {
        for y in 0..height {
            img.put_pixel(x, y, Rgb([244, 240, 232]));
        }
    }
    let (xmin, xmax, ymin, ymax) = bounds;
    let (min, max) = (xmin.min(ymin).max(-8.), xmax.max(ymax).min(8.));
    let margin = width.max(height) / 20;
    println!("Drawing with radius: {}", radius);
    for egv in eigenvalues.iter() {
        if egv.re < max && egv.re > min && egv.im > min && egv.im < max {
            let center: (i32, i32) =
                (round(clamp(egv.re, min, max, margin as f64, (width - margin) as f64)).try_into().unwrap(),
                 round(clamp(egv.im, min, max, (height - margin) as f64, margin as f64)).try_into
                 ().unwrap());
            imageproc::drawing::draw_filled_circle_mut(&mut img, center, radius, Rgb([56, 59, 62]));
        }
    }
    print!("Saving picture...");
    img.save(path).expect("Couldn't save image");
    println!("DONE");
}

/// Collect a vector of all eigenvalues of the matrix instances with sampled variables
fn collect_eigenvalues(mat: &DMatrix<Complex<f64>>, indices: &IndexList, n_ts: usize, samples: &Vec<Vec<Complex<f64>>>)
                       -> (Vec<Complex<f64>>, Bounds)
{
    let mut egvs = Vec::with_capacity(n_ts * samples.len());
    let (mut xmin, mut xmax, mut ymin, mut ymax): Bounds = (0., 0., 0., 0.);
    for sample in samples.iter() {
        assert_eq!(sample.len(), n_ts);
        let temp_mat = replace_indices(&mat, &indices, sample).expect("Could not replace indices!!");
        for ev in eigenvalues(temp_mat).iter().cloned() {
            egvs.push(ev);
            (xmin, xmax, ymin, ymax) = (f64::min(xmin, ev.re), f64::max(xmax, ev.re), f64::min
                (ymin, ev.im), f64::max(ymax, ev.im));
        }
    }
    (egvs, (xmin, xmax, ymin, ymax))
}

/// Sample ns different values of all n_ts t values according to the given distribution
/// ns: how many samples to take
/// n_ts: how many variable in each sample
/// distrib: distribution to follow for sampling
fn sample_ts(ns: usize, n_ts: usize, distrib: SamplingDistribution) -> Vec<Vec<Complex<f64>>> {
    let mut rng = rand::thread_rng();
    let mut samples = vec![];
    match distrib {
        SamplingDistribution::UniformUnits => {
            for _ in 0..ns {
                samples.push(
                    (0..n_ts).map(|_| {
                        let phase = rng.gen_range((0.0_f64)..(f64::two_pi()));
                        Complex::new(phase.cos(), phase.sin())
                    }).collect()
                )
            }
        }
        SamplingDistribution::UniformRects { r } => {
            for _ in 0..ns {
                samples.push(
                    (0..n_ts).map(|_| {
                        Complex::new(rng.gen_range(-r..r), 0.)
                    }).collect()
                )
            }
        }
        SamplingDistribution::Normal { s } => {
            let dist = Normal::new(0.0_f64, s).unwrap();
            for _ in 0..ns {
                samples.push(
                    (0..n_ts).map(|_| {
                        let val: f64 = dist.sample(&mut rng);
                        Complex::new(val, 0.0_f64)
                    }).collect()
                )
            }
        }
    }
    samples
}

type IndexList = Vec<(usize, usize, usize)>;

/// Get random indices in an n*n matrix, each associated to a different variable t
fn random_indices_distinct_variables(n: usize, n_ts: usize) -> IndexList {
    let indices: Vec<(usize, usize)> = (0..n * n).map(|i| { (i % n, i / n) }).clone().collect();
    indices.choose_multiple(&mut rand::thread_rng(), n_ts).cloned().zip(0..n_ts).map(|((i, j), k)| { (i, j, k) }).collect()
}

/// Random n*n matrix with coefficients in the given population
fn random_matrix(n: usize, population: Vec<Complex<f64>>)
                 -> DMatrix<Complex<f64>>
{
    let iter = std::iter::from_fn(|| {
        population.choose(&mut rand::thread_rng()).map(|v| v.to_owned())
    }).take(n * n);

    let matrix = DMatrix::from_iterator(n, n, iter);
    matrix
}

/// Replace the given indices ind in matrix mat with values in ts
fn replace_indices(mat: &DMatrix<Complex<f64>>, ind: &IndexList, ts: &Vec<Complex<f64>>)
                   -> Result<DMatrix<Complex<f64>>, String>
{
    let len_ts = ts.len();
    let mut new_mat = mat.clone();
    for index in ind.iter() {
        let (i, j, k) = index.to_owned();
        if k >= len_ts {
            return Err(String::from(
                format!("Tried to insert t{} but only {} were provided", k + 1, len_ts)
            ))
        }
        new_mat[(i, j)] = ts[k]
    }
    Ok(new_mat)
}

// Get a square complex matrix' eigenvalues
fn eigenvalues(mat: DMatrix<Complex<f64>>) -> DVector<Complex<f64>> {
    mat.eigenvalues().expect("No eigenvalues!")
}
