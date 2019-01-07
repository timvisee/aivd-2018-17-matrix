#![feature(vec_remove_item)]

extern crate itertools;

use std::cmp::{max, min};
use std::collections::HashMap;
use std::fmt;
use std::fs::read_to_string;
use std::io::Result as IoResult;
use std::ops::{Index, IndexMut};

use itertools::Itertools;

const EMPTY: char = '.';

const ROWS: usize = 13;
const COLS: usize = 12;

type NumMatx = Matx<u8>;

fn main() {
    // Load the matrices
    let matrix_a = Matx::load("matrix_a.txt").expect("failed to load matrix A from file");
    let matrix_b = Matx::load("matrix_b.txt").expect("failed to load matrix B from file");

    let mut field = Field::empty(matrix_a, matrix_b);

    println!("Start solving...");
    if field.solve() {
        println!("Solving stalled, could not progress any further");
    } else {
        println!("Failed to solve any cell");
    }
}

#[derive(Debug, Clone)]
pub struct Matx<T> {
    pub cells: Vec<T>,
}

impl<T> Matx<T> {
    /// Construct a new matrix from the data in the given vector.
    /// The given data should be row major.
    fn new(cells: Vec<T>) -> Self {
        Self { cells }
    }

    /// Get a slice of a row based on the given `row` index.
    pub fn row(&self, row: usize) -> &[T] {
        &self.cells[rc_to_i(row, 0)..rc_to_i(row + 1, 0)]
    }

    /// Get a mutable slice of a row based on the given `row` index.
    pub fn row_mut(&mut self, row: usize) -> &mut [T] {
        &mut self.cells[rc_to_i(row, 0)..rc_to_i(row + 1, 0)]
    }

    /// Iterate over a row based by the given `row` index.
    pub fn iter_row(&self, row: usize) -> impl Iterator<Item = &T> {
        self.row(row).iter()
    }

    /// Iterate mutably over a row based by the given `row` index.
    pub fn iter_row_mut(&mut self, row: usize) -> impl Iterator<Item = &mut T> {
        self.row_mut(row).iter_mut()
    }

    /// Iterate over a column based by the given `col` index.
    pub fn iter_col(&self, col: usize) -> impl Iterator<Item = &T> {
        self.cells[col..].iter().step_by(COLS)
    }

    /// Iterate mutably over a column based by the given `col` index.
    pub fn iter_col_mut(&mut self, col: usize) -> impl Iterator<Item = &mut T> {
        self.cells[col..].iter_mut().step_by(COLS)
    }

    /// Build an iterator over matrix rows, returning a slice for each row.
    pub fn iter_rows(&self) -> impl Iterator<Item = &[T]> {
        (0..ROWS).map(move |r| self.row(r))
    }

    /// Build an iterator over matrix rows, returning a slice for each row.
    pub fn iter_rows_iter(&self) -> impl Iterator<Item = impl Iterator<Item = &T>> {
        self.cells.chunks(COLS).map(|r| r.iter())
    }

    /// Build an iterator over matrix columns.
    /// This iterator returns a vector with the column items, because this allocates a vector this
    /// is considered expensive.
    pub fn iter_cols(&self) -> impl Iterator<Item = Vec<&T>> {
        self.iter_cols_iter().map(|c| c.collect())
    }

    /// Build an iterator over matrix columns.
    /// This iterator returns a new iterator for each column.
    pub fn iter_cols_iter(&self) -> impl Iterator<Item = impl Iterator<Item = &T>> {
        (0..COLS).map(move |c| self.iter_col(c))
    }
}

impl Matx<u8> {
    /// Construct a new matrix with just zeros.
    pub fn zero() -> Self {
        Self::new(vec![0u8; ROWS * COLS])
    }

    /// Load a matrix from a file at the given path.
    pub fn load(path: &str) -> IoResult<Self> {
        Ok(Self::new(
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

    /// Check whether the given coordinate has a value set that is not `0`.
    #[inline]
    pub fn has(&self, row: usize, col: usize) -> bool {
        self[(row, col)] != 0
    }

    /// Remove the given `value` from a `row` by it's index.
    /// The cell that contains this value is set to `0`.
    /// This panics if the given value is not found in the row.
    pub fn remove_from_row(&mut self, row: usize, value: u8) {
        let col = self
            .iter_row(row)
            .position(|x| x == &value)
            .expect("failed to remove item from row, does not exist");
        self[(row, col)] = 0;
    }

    /// Remove the given `value` from a `col` by it's index.
    /// The cell that contains this value is set to `0`.
    /// This panics if the given value is not found in the row.
    pub fn remove_from_col(&mut self, col: usize, value: u8) {
        let row = self
            .iter_col(col)
            .position(|x| x == &value)
            .expect("failed to remove item from column, does not exist");
        self[(row, col)] = 0;
    }

    /// Convert the matrix into a humanly readable string.
    /// Characters in a row are separated by a space.
    pub fn to_string(&self) -> String {
        self.iter_rows()
            .map(|row| row.iter().map(|c| to_char(*c)).join(" "))
            .join("\n")
    }
}

impl<T> Index<usize> for Matx<T> {
    type Output = T;

    fn index(&self, i: usize) -> &T {
        &self.cells[i]
    }
}

impl<T> Index<(usize, usize)> for Matx<T> {
    type Output = T;

    fn index(&self, (r, c): (usize, usize)) -> &T {
        &self.cells[rc_to_i(r, c)]
    }
}

impl<T> IndexMut<usize> for Matx<T> {
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.cells[i]
    }
}

impl<T> IndexMut<(usize, usize)> for Matx<T> {
    fn index_mut(&mut self, (r, c): (usize, usize)) -> &mut T {
        &mut self.cells[rc_to_i(r, c)]
    }
}

impl fmt::Display for Matx<u8> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}\n", self.to_string())
    }
}

#[derive(Debug, Clone)]
pub struct Band {
    /// A key-value map with `item > count`.
    map: HashMap<u8, u8>,
}

impl Band {
    /// Build a band from the given iterator holding the items.
    pub fn from<'a>(iter: impl Iterator<Item = &'a u8>) -> Self {
        Self {
            map: iter.fold(HashMap::new(), |mut map, item| {
                *map.entry(*item).or_insert(0) += 1;
                map
            }),
        }
    }

    /// Subtract a single item, decreasing it' s count in the map, or removing it if the count has
    /// reached `0`.
    pub fn subtract(&mut self, item: u8) {
        // Obtain the item, subtract from it
        let entry = self.map.get_mut(&item).unwrap();
        *entry -= 1;

        // If the item is zero, remove it from the map
        if entry == &0 {
            self.map.remove(&item);
        }
    }

    // /// Get all items in this band.
    // /// If `unique` is `true`, the same items are only once in the list.
    // pub fn items(&self, unique: bool) -> Vec<u8> {
    //     if unique {
    //         self.map.keys().map(|c| *c).collect()
    //     } else {
    //         self.map.iter().map(|(k, v)| vec![*k; *v as usize]).flatten().collect()
    //     }
    // }

    /// Do an `and` operation on this band and the given `other`, returning a list of result items.
    /// If `unique` is `true`, the same items are only once in the list, this is less expensive.
    pub fn and(&self, other: &Band, unique: bool) -> Vec<u8> {
        if unique {
            self.map
                .keys()
                .filter(|k| other.map.contains_key(k))
                .map(|c| *c)
                .collect()
        } else {
            self.map
                .iter()
                .filter_map(|(k, v)| other.map.get(k).map(|c| vec![*k; min(*v, *c) as usize]))
                .flatten()
                .collect()
        }
    }

    /// Do an `or` operation on this band and the given `other`, returning a list of result items.
    /// If `unique` is `true`, the same items are only once in the list, this is less expensive.
    pub fn or(&self, other: &Band, unique: bool) -> Vec<u8> {
        if unique {
            self.map
                .keys()
                .chain(other.map.keys())
                .unique()
                .map(|c| *c)
                .collect()
        } else {
            // Clone the current count map, merge the map from other
            let mut map = self.map.clone();
            other.map
                .iter()
                .for_each(|(item, count)| {
                    map.entry(*item).and_modify(|x| *x = max(*x, *count)).or_insert(*count);
                });

            // Transform count map into items vector
            map.into_iter()
                .flat_map(|(item, count)| vec![item; count as usize])
                .collect()
        }
    }
}

/// A set of bands.
#[derive(Debug, Clone)]
pub struct BandSet {
    pub bands: Vec<Band>,
}

impl BandSet {
    /// Construct a new band set from the given list of bands.
    pub fn new(bands: Vec<Band>) -> Self {
        Self { bands }
    }

    /// Construct a new set of bands from the given iterator, producing iterators for band items.
    pub fn from<'a>(band_iter: impl Iterator<Item = impl Iterator<Item = &'a u8>>) -> Self {
        Self::new(band_iter.map(|band| Band::from(band)).collect())
    }
}

impl Index<usize> for BandSet {
    type Output = Band;

    fn index(&self, i: usize) -> &Band {
        &self.bands[i]
    }
}

impl IndexMut<usize> for BandSet {
    fn index_mut(&mut self, i: usize) -> &mut Band {
        &mut self.bands[i]
    }
}

#[derive(Debug, Clone)]
pub struct Field {
    /// The original left matrix, still holding all values
    orig_left: NumMatx,

    /// The original top matrix, still holding all values
    orig_top: NumMatx,

    /// The current left matrix, holding all values left
    left: NumMatx,

    /// The current top matrix, holding all values left
    top: NumMatx,

    /// The current field state in which we solve
    field: NumMatx,

    /// A band set with rows for the left matrix, holding unused items for each row.
    left_bands: BandSet,

    /// A band set with columns for the top matrix, holding unused items for each column.
    top_bands: BandSet,

    /// A map holding all possible values for each cell.
    possibilities: Matx<Option<Vec<u8>>>,
}

impl Field {
    /// Build a new empty field with the given `left` and `top` matrix.
    pub fn empty(left: NumMatx, top: NumMatx) -> Self {
        // Calculate the left and top bands
        let left_bands = BandSet::from(left.iter_rows_iter());
        let top_bands = BandSet::from(top.iter_cols_iter());

        // For each matrix cell, find possibilities
        let possibilities = Matx::new(
            (0..ROWS)
                .cartesian_product(0..COLS)
                .map(|(r, c)| Some(left_bands[r].and(&top_bands[c], false)))
                .collect(),
        );

        Self {
            orig_left: left.clone(),
            orig_top: top.clone(),
            left,
            top,
            field: Matx::zero(),
            left_bands,
            top_bands,
            possibilities,
        }
    }

    /// A cell is solved, set it's value and update the possibility registries.
    pub fn solved_cell(&mut self, r: usize, c: usize, value: u8) {
        // Make sure the cell isn't solved already
        if self.field.has(r, c) {
            panic!("Attempting to solve cell that has already been solved");
        }

        self.left_bands[r].subtract(value);
        self.top_bands[c].subtract(value);
        self.left.remove_from_row(r, value);
        self.top.remove_from_col(c, value);
        self.field[(r, c)] = value;
        self.possibilities[(r, c)] = None;

        // TODO: update `possibilities`, row and column cells may not have possibility more than in
        // band list
    }

    /// Attempt to solve the empty field based on the left and top matrices.
    ///
    /// If nothing could be solved `false` is returned.
    pub fn solve(&mut self) -> bool {
        let count: usize = self.possibilities.cells.iter().filter_map(|c| c.as_ref().map(|c| c.len())).sum();
        println!("Input (remaining cell candidates: {}):\n{}", count, self);

        // Keep solving steps until no step finds anything anymore
        let mut step = 0;
        while self.solve_step() {
            let count: usize = self.possibilities.cells.iter().filter_map(|c| c.as_ref().map(|c| c.len())).sum();
            println!("State after pass #{} (remaining cell candidates: {}):\n{}", step, count, self);
            step += 1;
        }

        step > 0
    }

    /// Attempt to solve the empty field based on the left and top matrices.
    /// This method returns after one step has been solved.
    /// Use `solve()` to attempt to solve the whole field.
    ///
    /// `true` is returned if any step solved anything, `false` if nothing was found.
    pub fn solve_step(&mut self) -> bool {
        self.solve_naked_singles()
            || self.solve_naked_intersections()
            || self.solve_naked_pairs()
            || self.solve_naked_combis()
    }

    // TODO: do not clone in here
    // TODO: improve cell collection logic into `solved_list`
    fn solve_naked_intersections(&mut self) -> bool {
        // Obtain the values left in the rows and columns
        let rows: Vec<Vec<u8>> = self.left.iter_rows().map(|r| r.to_vec()).collect();
        let cols: Vec<Vec<u8>> = self
            .top
            .iter_cols_iter()
            .map(|c| c.map(|c| *c).collect())
            .collect();

        // List of solved fields, to apply in the end
        let mut solved_list: Vec<(usize, usize, u8)> = Vec::new();

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
                        .filter(|(c, _)| !self.field.has(r, *c))
                        .filter(|(_, col)| col.iter().any(|entry| *entry == item))
                        .for_each(|(c, _)| solved_list.push((r, c, item)))
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
                        .filter(|(r, _)| !self.field.has(*r, c))
                        .filter(|(_, row)| row.iter().any(|entry| *entry == item))
                        .for_each(|(r, _)| solved_list.push((r, c, item)))
                });
        }

        // Apply the solved cells
        solved_list.iter()
            .for_each(|(r, c, item)| {
                println!("# solved naked intersection");
                self.solved_cell(*r, *c, *item);
            });

        // Return true if any cell was solved
        solved_list.len() > 0
    }

    // TODO: remove this?
    // /// Build a matrix of all cell possibilities.
    // /// `field` cells that already have a value are `None`.
    // /// If `unique` is `true`, all cells have each possible item once, this is less expensive.
    // fn _cell_posibilities(&self, unique: bool) -> Matx<Option<Vec<u8>>> {
    //     // For each matrix cell, find possibilities
    //     Matx::new(
    //         (0..ROWS)
    //             .cartesian_product(0..COLS)
    //             .map(|(r, c)| {
    //                 if self.field.has(r, c) {
    //                     None
    //                 } else {
    //                     Some(self.left_bands[r].and(&self.top_bands[c], unique))
    //                 }
    //             })
    //             .collect(),
    //     )
    // }

    /// Solve cells that have one possible value.
    ///
    /// Inspired by: http://www.sudokuwiki.org/Getting_Started
    fn solve_naked_singles(&mut self) -> bool {
        // Collect cells we can solve
        let solved: Vec<(usize, u8)> = self
            .possibilities
            .cells
            .iter()
            .enumerate()
            .filter_map(|(i, c)| c.as_ref().map(|c| (i, c)))
            .filter(|(_, c)| c.iter().unique().count() == 1)
            .map(|(i, possibilities)| (i, possibilities[0]))
            .collect();

        // Fill in the solved cells
        solved.iter().for_each(|(i, x)| {
            let (r, c) = i_to_rc(*i);
            self.solved_cell(r, c, *x);
            println!("# solved naked single");
        });

        solved.len() > 0
    }

    /// Solve naked/conjugate pairs. Two cells in a row or column that have the same two
    /// possibilities, eliminating these options from other cells.
    ///
    /// Inspired by: http://www.sudokuwiki.org/Naked_Candidates#NP
    fn solve_naked_pairs(&mut self) -> bool {
        // Build iterators through row and column cell possibilities
        let rows = self
            .possibilities
            .iter_rows_iter()
            .enumerate()
            .map(|(r, row_iter)| {
                row_iter
                    .enumerate()
                    .filter_map(|(c, cell)| cell.as_ref().map(|cell| (r, c, cell)))
                    .collect::<Vec<_>>()
            });
        let cols = self
            .possibilities
            .iter_cols_iter()
            .enumerate()
            .map(|(c, col_iter)| {
                col_iter
                    .enumerate()
                    .filter_map(|(r, cell)| cell.as_ref().map(|cell| (r, c, cell)))
                    .collect::<Vec<_>>()
            });

        // Find possibility pairs on the rows and columns
        let pairs: Vec<((usize, usize), (usize, usize), Vec<u8>)> = rows
            .chain(cols)
            .filter_map(|cells| {
                // Find eligible cells, having two possible values, skip if not enough
                let eligible: Vec<(usize, usize, Vec<u8>)> = cells
                    .into_iter()
                    .filter(|(_, _, possib)| possib.len() == 2)
                    .map(|(r, c, possib)| (r, c, possib.clone()))
                    .collect();
                if eligible.len() < 2 {
                    return None;
                }

                // Find cells with matching possibilities
                let pairs: Vec<((usize, usize), (usize, usize), Vec<u8>)> = eligible
                    .into_iter()
                    .fold(HashMap::new(), |mut map, (r, c, possibilities)| {
                        map.entry(possibilities).or_insert_with(|| Vec::new()).push((r, c));
                        map
                    })
                    .into_iter()
                    .filter(|(_, i)| i.len() >= 2)
                    .inspect(|(_, i)| if i.len() > 2 {
                        panic!("# naked pairs: more than two cells for possibilities, not yet implemented");
                        // TODO: find all pairs in a row/column for the collected cell coordinates
                        // with matching cell possibilities
                    })
                    .map(|(possib, coords)| (coords[0], coords[1], possib))
                    .collect();

                // Return the list with possibilities if there are any
                if pairs.is_empty() {
                    None
                } else {
                    Some(pairs)
                }
            })
            .flatten()
            .collect();

        // Keep track whether we solved anything
        let mut solved = false;

        // For each possibility pair, update the surrounding cell possibilities on the same line
        pairs
            .into_iter()
            .for_each(|(coords_a, coords_b, pair_possib)| {
                println!("# found naked pair");

                // Clone the left and top bands
                let left_bands = self.left_bands.clone();
                let top_bands = self.top_bands.clone();

                // Find the row/column cells iterator, if on a row or column depending on how the
                // pairs are aligned
                let cells_iter: Box<dyn Iterator<Item = (usize, usize, &mut Option<_>)>> =
                    if coords_a.0 == coords_b.0 {
                        Box::new(
                            self.possibilities
                                .iter_row_mut(coords_a.0)
                                .enumerate()
                                .map(|(c, cell)| (coords_a.0, c, cell)),
                        )
                    } else {
                        Box::new(
                            self.possibilities
                                .iter_col_mut(coords_a.1)
                                .enumerate()
                                .map(|(r, cell)| (r, coords_a.1, cell)),
                        )
                    };

                // Update cell possibilities for other cells on the same line
                cells_iter
                    .filter_map(|(r, c, p)| p.as_mut().map(|p| (r, c, p)))
                    .filter(|(r, c, _)| coords_a != (*r, *c) && coords_b != (*r, *c))
                    .for_each(|(r, c, cell_possib)| {
                        // Find all cell possibilities based on bands, remove pair items
                        let mut band_possib = left_bands[r].and(&top_bands[c], false);
                        pair_possib.iter().for_each(|p| {
                            band_possib.remove_item(p);
                        });

                        // Make sure the cell doesn't contain more possibilities than the
                        // band list
                        cell_possib.retain(|p| {
                            if band_possib.remove_item(p).is_some() {
                                true
                            } else {
                                println!("# removed cell possibility due to naked pair");
                                solved = true;
                                false
                            }
                        });
                    });
            });

        solved
    }

    /// Solve naked/conjugate combinations (being pairs, triples, quads or a larger combination).
    /// A combination of cells on the same row or column that together only have the same number of
    /// possibilities as the number of cells in the combination, have these possibilities in either
    /// cell for sure, which allows eliminating these options from other cells in the same line.
    ///
    /// Inspired by: http://www.sudokuwiki.org/Naked_Candidates#NT
    fn solve_naked_combis(&mut self) -> bool {
        // Build iterators through row and column cell possibilities
        let rows = self
            .possibilities
            .iter_rows_iter()
            .enumerate()
            .map(|(r, row_iter)| (
                row_iter
                    .enumerate()
                    .filter_map(|(c, cell)| cell.as_ref().map(|cell| (r, c, cell)))
                    .collect::<Vec<_>>(),
                COLS,
            ));
        let cols = self
            .possibilities
            .iter_cols_iter()
            .enumerate()
            .map(|(c, col_iter)| (
                col_iter
                    .enumerate()
                    .filter_map(|(r, cell)| cell.as_ref().map(|cell| (r, c, cell)))
                    .collect::<Vec<_>>(),
                ROWS,
            ));

        // Find combination possibilities on rows and columns
        let combis: Vec<(Vec<(usize, usize, Vec<u8>)>, HashMap<u8, u8>)> = rows
            .chain(cols)
            .map(|(cells, length)| (2..length - 1)
                // TODO: skip cells that have been solved
                // TODO: determine appropriate combination size, `max-1`?
                // TODO: filter sizes larger than `max - 1` depending on row/column
                .map(|size| cells
                    .iter()
                    .map(|(r, c, p)| (*r, *c, p.iter().cloned().collect::<Vec<u8>>()))
                    // TODO: skip cells that have more possibilities than combination size here,
                    // possible with combinations?
                    .combinations(size)
                    .filter_map(|combis| {
                        // Skip if any cell has more possibilities than the combination size
                        if combis.iter().any(|(_, _, possib)| possib.len() > size) {
                            return None;
                        }

                        // Count possibilities in each cell
                        let combi_possibs: HashMap<u8, u8> = combis
                            .iter()
                            .map(|(_, _, possib)| count_map(possib))
                            .fold(HashMap::new(), |mut map, possib| {
                                possib.into_iter()
                                    .for_each(|(item, count)| {
                                        map.entry(item)
                                            .and_modify(|cur| *cur = max(*cur, count))
                                            .or_insert(count);
                                    });
                                map
                            });

                        // Total must not be larger than combination size
                        if combi_possibs.values().sum::<u8>() > size as u8 {
                            return None;
                        }

                        Some((combis, combi_possibs))
                    })
                    .collect::<Vec<(Vec<(usize, usize, Vec<u8>)>, HashMap<u8, u8>)>>()
                )
                .flatten()
                .collect::<Vec<(Vec<(usize, usize, Vec<u8>)>, HashMap<u8, u8>)>>()
            )
            .flatten()
            .collect();

        // Keep track whether we solved anything
        let mut solved = false;

        // For each possibility combination, update the surrounding cell possibilities on the same line
        combis
            .into_iter()
            .for_each(|(combis, combi_posibs)| {
                // TODO: remove after debugging
                // println!("# found naked combination (size: {})", combis.len());

                // Clone the left and top bands
                let left_bands = self.left_bands.clone();
                let top_bands = self.top_bands.clone();

                // Find the row/column cells iterator, if on a row or column depending on how the
                // pairs are aligned
                let cells_iter: Box<dyn Iterator<Item = (usize, usize, &mut Option<_>)>> =
                    if combis[0].0 == combis[1].0 {
                        Box::new(
                            self.possibilities
                                .iter_row_mut(combis[0].0)
                                .enumerate()
                                .map(|(c, cell)| (combis[0].0, c, cell)),
                        )
                    } else {
                        Box::new(
                            self.possibilities
                                .iter_col_mut(combis[0].1)
                                .enumerate()
                                .map(|(r, cell)| (r, combis[0].1, cell)),
                        )
                    };

                // Update cell possibilities for other cells on the same line
                cells_iter
                    .filter_map(|(r, c, p)| p.as_mut().map(|p| (r, c, p)))
                    .filter(|(r, c, _)| !combis.iter().any(|(rr, cc, _)| r == rr && c == cc))
                    .for_each(|(r, c, cell_possib)| {
                        // Find all cell possibilities based on bands, remove combination items
                        // TODO: should we OR here, or use a single band?
                        let mut band_possib = left_bands[r].or(&top_bands[c], false);
                        combi_posibs.iter().for_each(|(item, count)| for _ in 0..*count {
                            band_possib.remove_item(item);
                        });

                        // Make sure the cell doesn't contain more possibilities than the
                        // band list
                        cell_possib.retain(|p| {
                            if band_possib.remove_item(p).is_some() {
                                true
                            } else {
                                println!("# removed cell possibility due to naked combination");
                                solved = true;
                                false
                            }
                        });
                    });
            });

        solved
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

/// Transform an unified index into a `(row, column)` index.
fn i_to_rc(i: usize) -> (usize, usize) {
    (i / COLS, i % COLS)
}

/// Transform a `(row, column)` index into a unified index.
fn rc_to_i(r: usize, c: usize) -> usize {
    r * COLS + c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_i_to_rc() {
        assert_eq!(i_to_rc(0), (0, 0));
        assert_eq!(i_to_rc(1), (0, 1));
        assert_eq!(i_to_rc(COLS), (1, 0));
        assert_eq!(i_to_rc(COLS * 2 + 3), (2, 3));
    }

    #[test]
    fn test_rc_to_i() {
        assert_eq!(rc_to_i(0, 0), 0);
        assert_eq!(rc_to_i(0, 1), 1);
        assert_eq!(rc_to_i(1, 0), COLS);
        assert_eq!(rc_to_i(2, 3), COLS * 2 + 3);
    }
}
