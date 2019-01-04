extern crate itertools;

use std::fmt;
use std::fs::read_to_string;
use std::io::Result as IoResult;

use itertools::Itertools;

fn main() {
    // Load the matrices
    let matrix_a = Matrix::load("matrix_a.txt")
        .expect("failed to load matrix A from file");
    let matrix_b = Matrix::load("matrix_b.txt")
        .expect("failed to load matrix B from file");

    println!("Matrix A:\n{}", matrix_a);
    println!("Matrix B:\n{}", matrix_b);
}

#[derive(Debug, Clone)]
pub struct Matrix {
    m: Vec<Vec<char>>,
}

impl Matrix {
    /// Load a matrix from a file at the given path.
    pub fn load(path: &str) -> IoResult<Self> {
        Ok(Self {
            m: read_to_string(path)
                .expect("failed to load matrix from file")
                .lines()
                .filter(|line| !line.chars().all(char::is_whitespace))
                .map(|line| line
                    .chars()
                    .filter(|c| !c.is_whitespace())
                    .collect::<Vec<char>>()
                )
                .collect::<Vec<Vec<char>>>(),
        })
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for row in &self.m {
            write!(f, "{}\n", row.iter().join(" "))?;
        }
        Ok(())
    }
}
