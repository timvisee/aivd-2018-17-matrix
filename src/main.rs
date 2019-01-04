extern crate itertools;
extern crate nalgebra;
extern crate permutator;

use std::fmt;
use std::fs::read_to_string;
use std::io::Result as IoResult;

use itertools::Itertools;
use nalgebra::base::{
    Matrix as NMatrix,
    ArrayStorage,
    U1,
    U12,
    U13,
};
use permutator::*;

const EMPTY: char = '.';

const ROWS: usize = 13;
const COLS: usize = 12;

/// The generic matrix types we're using
type N = u8;
type R = U13;
type C = U12;
type S = ArrayStorage<N, R, C>;
type M = NMatrix<N, R, C, S>;

fn main() {
    // Load the matrices
    let matrix_a = Matx::load("matrix_a.txt")
        .expect("failed to load matrix A from file");
    let matrix_b = Matx::load("matrix_b.txt")
        .expect("failed to load matrix B from file");

    println!("Matx A:\n{}", matrix_a);
    println!("Matx B:\n{}", matrix_b);

    let mut field = Field::empty(matrix_a, matrix_b);
    println!("Field:\n{}", field);

    println!("Started solving...");
    field.solve();
}

#[derive(Debug, Clone, Copy)]
pub struct Matx {
    pub m: M,
}

impl Matx {
    /// Construct a new matrix.
    pub fn zero() -> Self {
        Self {
            m: M::zeros(),
        }
    }

    /// Construct a new matrix from the data in the given vector.
    /// The given data should be row major.
    fn from_vec(vec: Vec<u8>) -> Self {
        type RowMajorMatrix = NMatrix<N, C, R, ArrayStorage<N, C, R>>;
        Self {
            m: RowMajorMatrix::from_vec(vec).transpose(),
        }
        // Self {
        //     m: M::from_vec(vec),
        // }
    }

    /// Load a matrix from a file at the given path.
    pub fn load(path: &str) -> IoResult<Self> {
        Ok(Self::from_vec(
            read_to_string(path)
                .expect("failed to load matrix from file")
                .lines()
                .filter(|line| !line.chars().all(char::is_whitespace))
                .map(|line| line
                    .chars()
                    .filter(|c| !c.is_whitespace())
                    .map(to_number)
                    .collect::<Vec<u8>>()
                )
                .flatten()
                .collect::<Vec<u8>>(),
        ))
    }

    /// Build an iterator over matrix rows.
    ///
    /// Note: this is expensive.
    pub fn iter_rows<'a>(&'a self) -> impl Iterator<Item = Vec<u8>> + 'a {
        (0..ROWS)
            .map(move |r| self
                 .m
                 .row(r)
                 .iter()
                 .map(|c| *c)
                 .collect::<Vec<u8>>()
            )
    }

    /// Convert the matrix into a humanly readable string.
    /// Characters in a row are separated by a space.
    pub fn to_string(&self) -> String {
        self.iter_rows()
            .map(|row| row
                 .iter()
                 .map(|c| to_char(*c))
                 .join(" ")
            )
            .join("\n")
    }
}

impl fmt::Display for Matx {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}\n", self.to_string())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Field {
    left: Matx,
    top: Matx,
    field: Matx,
}

impl Field {
    /// Build a new empty field with the given `left` and `top` matrix.
    pub fn empty(left: Matx, top: Matx) -> Self {
        Self {
            field: Matx::zero(),
            left,
            top,
        }
    }

    /// Attempt to solve the empty field based on the left and top matrices.
    pub fn solve(&mut self) {
        let rows_permutated: Vec<_> = self
            .left
            .iter_rows()
            // .map(|row| row.permutation())
            .collect();

        println!("count: {}", rows_permutated.len());
    }
}

impl fmt::Display for Field {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Determine the width of the left matrix, with separating spaces
        let left_width = COLS * 2 - 1;

        // Print the top matrix first
        write!(f, "{}\n\n",
            self.top.to_string()
                .lines()
                .map(|row| format!("{}{}", vec![' '; left_width + 3].iter().join(""), row))
                .join("\n")
        )?;

        // Print the left and field matrix
        write!(f, "{}\n",
            self.left.to_string()
                .lines()
                .zip(self.field.to_string().lines())
                .map(|(left, field)| format!("{}   {}", left, field))
                .join("\n")
        )
    }
}

/// Convert the given character to a number.
fn to_number(c: char) -> u8 {
    c as u8 - 'A' as u8 + 1
}

/// Convert a given character number into a displayable character.
fn to_char(x: u8) -> char {
    if x == 0 {
        EMPTY
    } else {
        (x + 'A' as u8 - 1) as char
    }
}
