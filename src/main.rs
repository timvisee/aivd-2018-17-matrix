extern crate itertools;
extern crate nalgebra;
extern crate permutator;

use std::collections::HashMap;
use std::fmt;
use std::fs::read_to_string;
use std::io::Result as IoResult;

use itertools::Itertools;
use nalgebra::base::{ArrayStorage, Matrix as NMatrix, U12, U13};
// use permutator::*;

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
    let matrix_a = Matx::load("matrix_a.txt").expect("failed to load matrix A from file");
    let matrix_b = Matx::load("matrix_b.txt").expect("failed to load matrix B from file");

    println!("Matx A:\n{}", matrix_a);
    println!("Matx B:\n{}", matrix_b);

    let mut field = Field::empty(matrix_a, matrix_b);
    println!("Field:\n{}", field);

    println!("Started solving...");
    field.solve();
    field.solve();
    println!("First solve attempt:\n{}", field);
}

#[derive(Debug, Clone, Copy)]
pub struct Matx {
    pub m: M,
}

impl Matx {
    /// Construct a new matrix.
    pub fn zero() -> Self {
        Self { m: M::zeros() }
    }

    /// Construct a new matrix from the data in the given vector.
    /// The given data should be row major.
    fn from_vec(vec: Vec<u8>) -> Self {
        type RowMajorMatrix = NMatrix<N, C, R, ArrayStorage<N, C, R>>;
        Self {
            m: RowMajorMatrix::from_vec(vec).transpose(),
        }
    }

    /// Load a matrix from a file at the given path.
    pub fn load(path: &str) -> IoResult<Self> {
        Ok(Self::from_vec(
            read_to_string(path)
                .expect("failed to load matrix from file")
                .lines()
                .filter(|line| !line.chars().all(char::is_whitespace))
                .map(|line| {
                    line.chars()
                        .filter(|c| !c.is_whitespace())
                        .map(to_number)
                        .collect::<Vec<_>>()
                })
                .flatten()
                .collect(),
        ))
    }

    /// Get the value at the given coordinate.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> u8 {
        self.m[(row, col)]
    }

    /// Check whether the given coordinate has a value set that is not `0`.
    #[inline]
    pub fn has(&self, row: usize, col: usize) -> bool {
        self.get(row, col) != 0
    }

    /// Set the `value` at the given coordinate.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: N) {
        self.m[(row, col)] = value;
    }

    /// Remove the given `value` from a `row` by it's index.
    /// The cell that contains this value is set to `0`.
    /// This panics if the given value is not found in the row.
    pub fn remove_from_row(&mut self, row: usize, value: N) {
        self.set(
            row,
            self.m
                .row(row)
                .iter()
                .position(|x| x == &value)
                .expect("failed to remove item from row, does not exist"),
            0,
        )
    }

    /// Remove the given `value` from a `col` by it's index.
    /// The cell that contains this value is set to `0`.
    /// This panics if the given value is not found in the row.
    pub fn remove_from_col(&mut self, col: usize, value: N) {
        self.set(
            self.m
                .column(col)
                .iter()
                .position(|x| x == &value)
                .expect("failed to remove item from column, does not exist"),
            col,
            0,
        )
    }

    /// Build an iterator over matrix rows.
    ///
    /// Note: this is expensive.
    pub fn iter_rows<'a>(&'a self) -> impl Iterator<Item = Vec<u8>> + 'a {
        (0..ROWS).map(move |r| self.m.row(r).iter().map(|c| *c).collect::<Vec<u8>>())
    }

    /// Build an iterator over matrix columns.
    ///
    /// Note: this is expensive.
    pub fn iter_cols<'a>(&'a self) -> impl Iterator<Item = Vec<u8>> + 'a {
        (0..COLS).map(move |r| self.m.column(r).iter().map(|c| *c).collect::<Vec<u8>>())
    }

    /// Convert the matrix into a humanly readable string.
    /// Characters in a row are separated by a space.
    pub fn to_string(&self) -> String {
        self.iter_rows()
            .map(|row| row.iter().map(|c| to_char(*c)).join(" "))
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
        self._solve_naked_intersections();
        self._cell_posibilities();
    }

    // TODO: do not clone in here
    fn _solve_naked_intersections(&mut self) {
        // Obtain the values left in the rows and columns
        let rows: Vec<Vec<u8>> = self.left.iter_rows().collect();
        let cols: Vec<Vec<u8>> = self.top.iter_cols().collect();

        // Find naked intersections for each row
        for r in 0..ROWS {
            // Count items in current row and all empty columns
            let row_count = count_map(&rows[r]);
            let col_count: HashMap<u8, u8> = count_map(
                &cols
                    .iter()
                    .enumerate()
                    .filter(|(c, _)| !self.field.has(r, *c))
                    .map(|(_, col)| col.clone())
                    .flatten()
                    .collect(),
            );

            // For each item with the same row/column count, fill it in
            row_count
                .into_iter()
                .filter(|(item, count)| col_count.get(item).unwrap_or(&0) == count)
                .for_each(|(item, _)| {
                    cols.iter()
                        .enumerate()
                        .filter(|(_, col)| col.iter().any(|entry| *entry == item))
                        .for_each(|(c, _)| {
                            self.field.set(r, c, item);
                            self.left.remove_from_row(r, item);
                            self.top.remove_from_col(c, item);
                        })
                });
        }

        // Find naked intersections for each column
        for c in 0..COLS {
            // Count items in current column and all empty rows
            let row_count: HashMap<u8, u8> = count_map(
                &rows
                    .iter()
                    .enumerate()
                    .filter(|(r, _)| !self.field.has(*r, c))
                    .map(|(_, row)| row.clone())
                    .flatten()
                    .collect(),
            );
            let col_count = count_map(&cols[c]);

            // For each item with the same row/column count, fill it in
            col_count
                .into_iter()
                .filter(|(item, count)| row_count.get(item).unwrap_or(&0) == count)
                .for_each(|(item, _)| {
                    rows.iter()
                        .enumerate()
                        .filter(|(_, row)| row.iter().any(|entry| *entry == item))
                        .for_each(|(r, _)| {
                            self.field.set(r, c, item);
                            self.left.remove_from_row(r, item);
                            self.top.remove_from_col(c, item);
                        })
                });
        }
    }

    fn _cell_posibilities(&mut self) {
        // Obtain the values left in the rows and columns
        let rows: Vec<Vec<u8>> = self
            .left
            .iter_rows()
            .map(|r| r.iter().map(|x| *x).filter(|x| x != &0).collect())
            .collect();
        let cols: Vec<Vec<u8>> = self
            .left
            .iter_cols()
            .map(|c| c.iter().map(|x| *x).filter(|x| x != &0).collect())
            .collect();

        let mut map: Vec<Vec<u8>> = Vec::new();
        for r in 0..ROWS {
            let row = &rows[r];
            for c in 0..COLS {
                let possibilities = row
                    .iter()
                    .filter(|x| cols[c].contains(x))
                    .map(|x| *x)
                    .collect::<Vec<u8>>();
                map.push(possibilities);
            }
        }
    }
}

impl fmt::Display for Field {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Determine the width of the left matrix, with separating spaces
        let left_width = COLS * 2 - 1;

        // Print the top matrix first
        write!(
            f,
            "{}\n\n",
            self.top
                .to_string()
                .lines()
                .map(|row| format!("{}{}", vec![' '; left_width + 3].iter().join(""), row))
                .join("\n")
        )?;

        // Print the left and field matrix
        write!(
            f,
            "{}\n",
            self.left
                .to_string()
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

/// Make an item count map for the given list of `items`.
/// The `0` item is not counted.
fn count_map(items: &Vec<u8>) -> HashMap<u8, u8> {
    items
        .into_iter()
        .filter(|i| **i != 0)
        .fold(HashMap::new(), |mut map, item| {
            *map.entry(*item).or_insert(0) += 1;
            map
        })
}
