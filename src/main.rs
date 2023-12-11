extern crate rand_distr;
extern crate image;
extern crate imageproc;

extern crate nalgebra as na;

use na::{DMatrix, Complex, RealField, ComplexField, DVector};
use image::{RgbImage, Rgb};

use rand::{seq::SliceRandom, Rng};
use rand_distr::{Normal, Distribution};

use std::{time::Instant, path::Path, str::FromStr};

use clap::Parser;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// An output file. Extension must be png. Default is ./negie.png
    #[arg(short, long)]
    output: Option<std::path::PathBuf>,
    /// Dimension of the random square matrix to generate
    #[arg(short = 'N', long)]
    dim: Option<usize>,
    /// Number of variables to use. Defaults is 2.
    #[arg(short = 'k', long)]
    nvar: Option<usize>,
    /// Number of samples to take. Default is 300000.
    #[arg(short = 'n', long)]
    samples: Option<usize>,
    /// Output image's side length. Default is 12000.
    #[arg(short, long)]
    size: Option<u32>,
    /// Radius of each eigenvalue's circle. Default is heuristically determined.
    #[arg(short, long)]
    radius: Option<i32>,
    /// Statistical distribution to use for sampling. Ex: uniformunits, uniformrects:20.0,
    /// normal:20.0
    #[arg(short = 'd', long)]
    distrib: Option<SamplingDistribution>,
    /// Matrix to use. Overrides -d. Must be either "random" or a comma-separated list of complex
    /// numbers of length n^2 for some n.
    #[arg(short, long, default_value = "random")]
    matrix: ComplexMatrixDescriptor,
    /// Variable indices. Semicolon-separated list of comma-separated triplets i,j,k to use t_k in
    /// the spot (i,j). 0,0,0;1,1,1 replaces the first two coefficients of the main diagonal with
    /// t_0, t_1 respectively. Use "random" for the default behavior.
    #[arg(short, long, default_value = "random")]
    variable: VariableIndicesDescriptor
}

#[derive(Clone, Debug)]
enum VariableIndicesDescriptor {
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
enum SamplingDistribution {
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

fn isqrt(num: usize) -> usize {
    let mut sqrt = num;
    let mut next_sqrt = (sqrt + num / sqrt) / 2;
    while next_sqrt < sqrt {
        sqrt = next_sqrt;
        next_sqrt = (sqrt + num / sqrt) / 2;
    }
    sqrt
}

#[derive(Clone, Debug)]
enum ComplexMatrixDescriptor {
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

fn main() {
    let args = Cli::parse();
    let mut dim = args.dim.unwrap_or(5);
    let mut n_ts = args.nvar.unwrap_or(2);
    let nsamples = args.samples.unwrap_or(300000);
    let i= Complex::i();
    let pop = vec![i, -i, Complex::from_real(0.), Complex::from_real(1.), Complex::from_real(0.5)];
    let matrix = match args.matrix {
        ComplexMatrixDescriptor::Random => random_matrix(dim, pop),
        ComplexMatrixDescriptor::Matrix { coefs, size } => {
            dim = size;
            DMatrix::from_vec(size, size, coefs)
        }
    };
    println!("{}", matrix);
    let variable_indices = match args.variable {
        VariableIndicesDescriptor::Random => random_indices_distinct_variables(dim, n_ts),
        VariableIndicesDescriptor::List { list } => {
            n_ts = list.iter().cloned().fold(0, |a, (i,j,k)| { 
                assert!(i < dim && j < dim, "Index {},{} in variable indices list was out of bounds", i, j);
                a.max(k)
            }) + 1;
            list
        }
    };
    println!("Indices for varibales: {:?}", variable_indices);
    let start = Instant::now();
    let distrib = args.distrib.unwrap_or(SamplingDistribution::UniformUnits);
    println!("Using distribution: {:?}", distrib);
    let samples = sample_ts(nsamples, n_ts, distrib);
    let egvs = collect_eigenvalues(&matrix, &variable_indices, n_ts, &samples);
    println!("Eigenvalue collection took {:?}.", start.elapsed());
    let width = args.size.unwrap_or(12000);
    save_eigen_image(args.output.as_deref(), egvs, nsamples, width, width, args.radius);
}


fn save_eigen_image(path: Option<&Path>, eigenvalues: Vec<Complex<f64>>, nsamples: usize, width: u32, height: u32, radius: Option<i32>) {
    let mut img = RgbImage::new(width, height);
    for x in 0..width {
        for y in 0..height {
            img.put_pixel(x, y, Rgb([244, 240, 232]));
        }
    }
    let (xmin, xmax, ymin, ymax) = eigenvalues.iter().fold((0.,0.,0.,0.), |(xmin, xmax, ymin, ymax), z| {
        (f64::min(xmin, z.re), f64::max(xmax, z.re), f64::min(ymin, z.im), f64::max(ymax, z.im))
    });
    let (min, max) = (xmin.min(ymin).max(-8.), xmax.max(ymax).min(8.));
    let margin = width.max(height) / 20;
    let radius : i32 = radius.unwrap_or(i32::min(round(50. * (width as f64 / (nsamples as f64))).try_into().unwrap(), <u32 as TryInto<i32>>::try_into(width).unwrap() / 1000));
    println!("Drawing with radius: {}", radius);
    for egv in eigenvalues.iter() {
        if egv.re < max && egv.re > min && egv.im > min && egv.im < max {
            imageproc::drawing::draw_filled_circle_mut(&mut img,
                (round(clamp(egv.re, min, max, margin as f64, (width - margin) as f64)).try_into().unwrap(),
                round(clamp(egv.im, min, max, (height - margin) as f64, margin as f64)).try_into().unwrap()),
                radius,
                Rgb([56, 59, 62]));
        }
    }
    print!("Saving picture...");
    let actual_path = path.unwrap_or(Path::new("negie.png"));
    assert_eq!(actual_path.extension().expect("No file extension!"), "png");
    img.save(actual_path).expect("Couldn't save image");
    println!("DONE");
}

fn round(x: f64) -> u32 {
    x.round() as u32
}

fn clamp(x: f64, mina: f64, maxa: f64, minb: f64, maxb: f64) -> f64 {
    assert!(x >= mina);
    assert!(x <= maxa);
    (x - mina) * (maxb - minb) / (maxa - mina) + minb
}

/// Collect a vector of all eigenvalues of the matrix instances with sampled variables
fn collect_eigenvalues(mat: &DMatrix<Complex<f64>>, indices: &IndexList, n_ts: usize, samples: &Vec<Vec<Complex<f64>>>)
    -> Vec<Complex<f64>>
{
    let mut egvs = Vec::with_capacity(n_ts * samples.len());
    for sample in samples.iter() {
        assert_eq!(sample.len(), n_ts);
        let temp_mat = replace_indices(&mat, &indices, sample).expect("Could not replace indices!!");
        for ev in eigenvalues(temp_mat).iter().cloned() {
            egvs.push(ev)
        }
    }
    egvs
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
                        Complex::new(rng.gen_range(-r..r),0.)
                    }).collect()
                )
            }
        }
        SamplingDistribution::Normal { s } => {
            let dist = Normal::new(0.0_f64, s).unwrap();
            for _ in 0..ns {
                samples.push(
                    (0..n_ts).map(|_| { 
                        let val : f64 = dist.sample(&mut rng);
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
    let indices: Vec<(usize, usize)> = (0..n*n).map(|i| { (i % n, i / n) }).clone().collect();
    indices.choose_multiple(&mut rand::thread_rng(), n_ts).cloned().zip(0..n_ts).map(|((i,j),k)| { (i,j,k) }).collect()
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
                    format!("Tried to insert t{} but only {} were provided", k+1, len_ts)
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
