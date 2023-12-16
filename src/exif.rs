extern crate itertools;

use std::process::Command;
use std::string::String;
use itertools::Itertools;
use na::{Complex};
use crate::cli::{ComplexMatrixDescriptor, Configuration, SamplingDistribution, VariableIndicesDescriptor};

fn serialize_matrix(mat_desc: &ComplexMatrixDescriptor) -> String {
    let mat = match mat_desc {
        ComplexMatrixDescriptor::Random => { panic!("Matrix unknown at exif write time")}
        ComplexMatrixDescriptor::Matrix { coefs, ..} => { coefs }
    };
    serialize_complexvec(&mat.to_vec())
}

fn serialize_indexlist(list: &VariableIndicesDescriptor) -> String {
    match list {
        VariableIndicesDescriptor::Random => { panic!("IndexList unknown at exif write time") }
        VariableIndicesDescriptor::List { list } => {
            list.iter().map(|(i,j,k)| {
                format!("{},{},{}", i, j, k)
            }).join(":")
        }
    }
}

fn serialize_complexvec(vec: &Vec<Complex<f64>>) -> String {
    vec.iter().map(|z| { z.to_string() }).join(",")
}

fn serialize_distrib(distrib: &SamplingDistribution) -> String {
    match distrib {
        SamplingDistribution::Normal { s } => { format!("normal:{}", s)}
        SamplingDistribution::UniformRects { r } => { format!("rects:{}", r)}
        SamplingDistribution::UniformUnits => { String::from("units") }
    }
}

pub fn write_exif_data(cli: &Configuration) {
    // exiftool -overwrite_original -UserComment="Your descriptive string here" /path/to/yourimage.jpg
    print!("Writing exif data to output picture ... ");
    let cli_run = Command::new("exiftool.exe")
        .arg("-overwrite_original")
        .arg(format!("-UserComment=\"--matrix={} --variables={} --population={} --distrib={}\"",
                     serialize_matrix(&cli.matrix),
                     serialize_indexlist(&cli.variables),
                     serialize_complexvec(&cli.population.pop),
                     serialize_distrib(&cli.distrib)))
        .arg(cli.output.as_path())
        .output();
    match cli_run {
        Ok(output) => { println!("{}.", if output.status.success() { "DONE" } else { "FAILED" })}
        Err(err) => { println!("FAILED: {} (you can ignore this).", err) }
    }
}