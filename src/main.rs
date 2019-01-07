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
            other.map.iter().for_each(|(item, count)| {
                map.entry(*item)
                    .and_modify(|x| *x = max(*x, *count))
                    .or_insert(*count);
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
    // TODO: switch to just use vectors, remove option
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

    /// A cell is solved, set it's value, update intersecting bands and elimimnate candidates from
    /// cells in the same row or column.
    pub fn solved_cell(&mut self, r: usize, c: usize, value: u8) {
        // Make sure the cell isn't solved already
        if self.field.has(r, c) {
            panic!("Attempting to solve cell that has already been solved");
        }

        // Remove item from possibilities and bands
        self.left_bands[r].subtract(value);
        self.top_bands[c].subtract(value);
        self.left.remove_from_row(r, value);
        self.top.remove_from_col(c, value);
        self.possibilities[(r, c)] = None;

        // Set the cell
        self.field[(r, c)] = value;

        // List the cell coordinates in the current row and column, except this cell
        let row_cells = (0..ROWS).map(|r| (r, c));
        let col_cells = (0..COLS).map(|c| (r, c));
        let line_cells = row_cells
            .chain(col_cells)
            .filter(|coord| (r, c) != *coord)
            .collect();

        // Eliminate used items from all other cells in the row or column
        self.eliminate_candidates(line_cells, &vec![value]);
    }

    /// Eliminate the `used` items from available candidates in all `cells`.
    ///
    /// For each unsolved cell in `cells`, the candidates are redetermined based on their current
    /// row and column band state. The `used` items are subtracted from this set. Any cell
    /// candidates that exceed this list are removed from the cell.
    fn eliminate_candidates(&mut self, mut cells: Vec<(usize, usize)>, used: &Vec<u8>) -> bool {
        // Keep track whehter any cell candidates have changed
        let mut changed = false;

        // Drop cells that have no possibilities
        cells.retain(|coord| self.possibilities[*coord].is_some());

        // For each given cell, eliminate exceeding candidates
        for (r, c) in cells {
            // Find all cell possibilities based on bands, remove used items
            let mut new_possibs = self.left_bands[r].or(&self.top_bands[c], false);
            used.iter().for_each(|item| {
                new_possibs.remove_item(item);
            });

            // Retain excess items not in new_possibs from current cell
            let cell_possibs = self.possibilities[(r, c)].as_mut().unwrap();
            cell_possibs.retain(|item| {
                if new_possibs.remove_item(item).is_some() {
                    true
                } else {
                    changed = true;

                    // TODO: remove after debugging
                    println!("ELIMINATED {} FROM ({}, {})", to_char(*item), r, c);

                    false
                }
            });

            // Panic if no candidates are left
            if cell_possibs.is_empty() {
                panic!(
                    "Left 0 possibilities in cell ({}, {}) after eliminating candidates",
                    r, c
                );
            }
        }

        changed
    }

    /// Attempt to solve the empty field based on the left and top matrices.
    ///
    /// If nothing could be solved `false` is returned.
    pub fn solve(&mut self) -> bool {
        let count: usize = self
            .possibilities
            .cells
            .iter()
            .filter_map(|c| c.as_ref().map(|c| c.len()))
            .sum();
        println!("Input (remaining cell candidates: {}):\n{}", count, self);

        // Keep solving steps until no step finds anything anymore
        let mut step = 0;
        while self.solve_step() {
            let count: usize = self
                .possibilities
                .cells
                .iter()
                .filter_map(|c| c.as_ref().map(|c| c.len()))
                .sum();
            println!(
                "State after pass #{} (remaining cell candidates: {}):\n{}",
                step, count, self
            );
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
        let mut solved_list: HashMap<(usize, usize), u8> = HashMap::new();

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
                        .for_each(|(c, _)| {
                            solved_list.insert((r, c), item);
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
                        .filter(|(r, _)| !self.field.has(*r, c))
                        .filter(|(_, row)| row.iter().any(|entry| *entry == item))
                        .for_each(|(r, _)| {
                            solved_list.insert((r, c), item);
                        })
                });
        }

        // Apply the solved cells
        solved_list.iter().for_each(|((r, c), item)| {
            println!("# solved naked intersection");
            self.solved_cell(*r, *c, *item);
        });

        // Return true if any cell was solved
        solved_list.len() > 0
    }

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

                // Build list of coordinates for all other cells in the same line
                let line_cells: Box<dyn Iterator<Item = (usize, usize)>> =
                    if coords_a.0 == coords_b.0 {
                        Box::new((0..COLS).map(|c| (coords_a.0, c)))
                    } else {
                        Box::new((0..ROWS).map(|r| (r, coords_a.1)))
                    };
                let line_cells = line_cells
                    .filter(|coord| coord != &coords_a && coord != &coords_b)
                    .collect();

                // Retain used items in combination from other cells in same line
                if self.eliminate_candidates(line_cells, &pair_possib) {
                    solved = true;
                }
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
        // Find all cell combinations on a line that would use all their candidates
        let combis: Vec<(Vec<(usize, usize)>, HashMap<u8, u8>)> = (2..max(ROWS, COLS))
            .flat_map(|combi_size| {
                // Build iterators through row/column cell possibilities if theres more cells than
                // the current combination size
                let rows = (0..ROWS).map(|r| {
                    (0..COLS)
                        .filter_map(|c| self.possibilities[(r, c)].as_ref().map(|_| (r, c)))
                        .collect::<Vec<_>>()
                });
                let cols = (0..COLS).map(|c| {
                    (0..ROWS)
                        .filter_map(|r| self.possibilities[(r, c)].as_ref().map(|_| (r, c)))
                        .collect::<Vec<_>>()
                });
                let lines = rows.chain(cols).filter(|cells| cells.len() > combi_size);

                // Find combination possibilities on rows and columns
                lines
                    .flat_map(|cells| {
                        cells
                            .iter()
                            .map(|coord| coord.to_owned())
                            .combinations(combi_size)
                            .filter_map(|combi_cells| {
                                // Skip if any cell has more possibilities than the combination size
                                // TODO: do this check when building line iterators
                                if combi_cells.iter().any(|coord| {
                                    self.possibilities[*coord].as_ref().unwrap().len() > combi_size
                                }) {
                                    return None;
                                }

                                // Count possibilities in each cell
                                let combi_used: HashMap<u8, u8> = combi_cells
                                    .iter()
                                    .map(|coord| {
                                        count_map(self.possibilities[*coord].as_ref().unwrap())
                                    })
                                    .fold(HashMap::new(), |mut map, possib| {
                                        possib.into_iter().for_each(|(item, count)| {
                                            map.entry(item)
                                                .and_modify(|cur| *cur = max(*cur, count))
                                                .or_insert(count);
                                        });
                                        map
                                    });

                                // Total must not be larger than combination size
                                if combi_used.values().sum::<u8>() > combi_size as u8 {
                                    return None;
                                }

                                Some((combi_cells, combi_used))
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        // Keep track whether we solved anything
        let mut solved = false;

        // For each possibility combination, update the surrounding cell possibilities on the same line
        combis.into_iter().for_each(|(combi_cells, combi_posibs)| {
            println!("# found naked combination (size: {})", combi_cells.len());

            // Build list of coordinates for all other cells in the same line
            let line_cells: Box<dyn Iterator<Item = (usize, usize)>> =
                if combi_cells[0].0 == combi_cells[1].0 {
                    Box::new((0..COLS).map(|c| (combi_cells[0].0, c)))
                } else {
                    Box::new((0..ROWS).map(|r| (r, combi_cells[0].1)))
                };
            let line_cells = line_cells
                .filter(|a| !combi_cells.iter().any(|b| a == b))
                .collect();

            // Build a list of used items
            let used = combi_posibs
                .into_iter()
                .flat_map(|(item, count)| vec![item; count as usize])
                .collect();

            // Retain used items in combination from other cells in same line
            if self.eliminate_candidates(line_cells, &used) {
                solved = true;
            }
        });

        // TODO: remove after debugging
        // println!("\n1. Candidates before ({}, {}): {:?}", r, c, cell_possib.iter().map(|c| to_char(*c)).sorted().collect::<Vec<_>>());
        // println!("2. Combination cells: {:?}", combi_cells.iter().map(|(r, c, _)| (r, c)).sorted().collect::<Vec<_>>());
        // combi_cells.iter()
        //     .for_each(|(r, c, possib)|
        //         println!("3. Combination cell candidates ({}, {}): {:?}", r, c, possib.iter().map(|c| to_char(*c)).sorted().collect::<Vec<_>>())
        //     );
        // println!("4. Recalculated candidates ({}, {}): {:?}", r, c, was.iter().map(|c| to_char(*c)).sorted().collect::<Vec<_>>());
        // println!("5. Combination uses: ({}, {}): {:?}", r, c, combi_posibs.iter().map(|(k, v)| (to_char(*k), v)).sorted().collect::<HashMap<_, _>>());
        // println!("6. Candidates left ({}, {}): {:?}", r, c, band_possib.iter().map(|c| to_char(*c)).sorted().collect::<Vec<_>>());
        // println!("7. After ({}, {}): {:?}", r, c, cell_possib.iter().map(|c| to_char(*c)).sorted().collect::<Vec<_>>());

        solved
    }
}

impl fmt::Display for Field {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Determine the width of the left matrix, with separating spaces
        let left_width = COLS * 2;

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

        // Print column numbers
        write!(
            f,
            "{}{}\n",
            vec![' '; left_width + 3].iter().join(""),
            (0..COLS).map(|n| n % 10).join(" "),
        )?;

        // Print the left and field matrix
        write!(
            f,
            "{}\n",
            self.left
                .to_string()
                .lines()
                .zip(self.field.to_string().lines())
                .enumerate()
                .map(|(n, (left, field))| format!("{}  {} {}", left, n % 10, field))
                .join("\n"),
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
