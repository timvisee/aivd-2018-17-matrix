extern crate itertools;
extern crate permutator;

use std::cmp::max;
use std::fmt;
use std::fs::read_to_string;
use std::io::Result as IoResult;

use itertools::Itertools;
use permutator::*;

const EMPTY: char = '.';

fn main() {
    // Load the matrices
    let matrix_a = Matrix::load("matrix_a.txt")
        .expect("failed to load matrix A from file");
    let matrix_b = Matrix::load("matrix_b.txt")
        .expect("failed to load matrix B from file");

    println!("Matrix A:\n{}", matrix_a);
    println!("Matrix B:\n{}", matrix_b);

    let mut field = Field::empty(matrix_a, matrix_b);
    println!("Field:\n{}", field);

    println!("Started solving...");
    field.solve();
}

#[derive(Debug, Clone)]
pub struct Matrix {
    m: Vec<Vec<u8>>,
}

impl Matrix {
    /// Construct a new matrix.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            m: vec![vec![0; width]; height],
        }
    }

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
                    .map(to_number)
                    .collect::<Vec<u8>>()
                )
                .collect::<Vec<Vec<u8>>>(),
        })
    }

    /// Get the width of the matrix.
    pub fn width(&self) -> usize {
        self.m
            .get(0)
            .map(|row| row.len())
            .unwrap_or(0)
    }

    /// Get the height of the matrix.
    pub fn height(&self) -> usize {
        self.m.len()
    }

    /// Get a row of the matrix.
    pub fn row(&self, row: usize) -> &Vec<u8> {
        &self.m[row]
    }

    /// Get an iterator over the rows.
    pub fn iter_rows<'a>(&'a self) -> impl Iterator<Item = &'a Vec<u8>> {
        self.m.iter()
    }

    /// Get an iterator over a row.
    pub fn iter_row<'a>(&'a self, row: usize) -> impl Iterator<Item = u8> + 'a{
        self.m[row]
            .iter()
            .map(|c| *c)
    }

    /// Get an iterator over a column.
    pub fn iter_col<'a>(&'a self, col: usize) -> impl Iterator<Item = u8> + 'a{
        self.m
            .iter()
            .map(move |row| row[col])
    }

    /// Convert the matrix into a humanly readable string.
    /// Characters in a row are separated by a space.
    pub fn to_string(&self) -> String {
        self.m
            .iter()
            .map(|row| row
                 .iter()
                 .map(|c| to_char(*c))
                 .join(" ")
            )
            .join("\n")
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}\n", self.to_string())
    }
}

#[derive(Debug, Clone)]
pub struct Field {
    left: Matrix,
    top: Matrix,
    field: Matrix,
}

impl Field {
    /// Build a new empty field with the given `left` and `top` matrix.
    pub fn empty(left: Matrix, top: Matrix) -> Self {
        Self {
            field: Matrix::new(top.width(), left.height()),
            left,
            top,
        }
    }

    /// Attempt to solve the empty field based on the left and top matrices.
    pub fn solve(&mut self) {
        let mut row = self.left.row(0).clone();
        let a = row.permutation();

        // a.for_each(|a|
        //     println!("Some: {:?}", a)
        // );
    }
}

impl fmt::Display for Field {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Determine the width of the left matrix, with separating spaces
        let left_width = max(self.left.width() as i64 * 2 - 1, 0) as usize;

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
