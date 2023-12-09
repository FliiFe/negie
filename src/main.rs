extern crate nalgebra as na;
extern crate nalgebra_lapack as nal;

use na::{DMatrix, Complex, ComplexField, OVector, Dyn};
use nal::Eigen;
use rand::{seq::SliceRandom, Rng};

fn main() {
    let dim = 5;
    let n_ts = 2;
    let nsamples = 100000;
    let pop = vec![Complex::i(),
                    -Complex::i(),
                    Complex::from_real(0.),
                    Complex::from_real(1.),
                    Complex::from_real(0.5),
                    ];
    let matrix = random_matrix(dim, pop);
    let variable_indices = random_indices_distinct_variables(dim, n_ts);
    let samples = sample_ts(nsamples, n_ts, Distribution::UniformUnits);
    let egvs = collect_eigenvalues(&matrix, &variable_indices, n_ts, &samples);
    println!("{}", matrix);
    println!("{:?}", egvs.len());
}

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

/// Types of distribution to use for sampling
enum Distribution {
    Normal { s: f64 },
    UniformRects { r: f64 },
    UniformUnits
}

/// Sample ns different values of all n_ts t values according to the given distribution
/// ns: how many samples to take
/// n_ts: how many variable in each sample
/// distrib: distribution to follow for sampling
fn sample_ts(ns: usize, n_ts: usize, distrib: Distribution) -> Vec<Vec<Complex<f64>>> {
    let mut rng = rand::thread_rng();
    let mut samples = vec![];
    match distrib {
        Distribution::UniformUnits => {
            for _ in 0..ns {
                samples.push(
                    (0..n_ts).map(|_| { 
                        let phase = rng.gen_range(0.0..1.0);
                        Complex::new(phase.cos(), phase.sin())
                    }).collect()
                )
            }
        }
        _ => {}
    }
    samples
}

type IndexList = Vec<(usize, usize, usize)>;

/// Get random indices in an n*n matrix, each associated to a different variable t
fn random_indices_distinct_variables(n: usize, n_ts: usize) -> IndexList {
    let indices: Vec<(usize, usize)> = (0..n).zip(0..n).clone().collect();
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
fn eigenvalues(mat: DMatrix<Complex<f64>>) -> OVector<Complex<f64>, Dyn> {
    let owned_matrix = mat.clone_owned();
    let eigen = Eigen::new_complex(owned_matrix, false, false).expect("No eigen!!!!");
    eigen.eigenvalues
}
