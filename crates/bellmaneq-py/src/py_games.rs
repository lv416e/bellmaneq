use pyo3::prelude::*;

use bellmaneq_games::minimax::{self, ZeroSumGame};
use bellmaneq_games::tictactoe::{TicTacToe, TicTacToeState};
use bellmaneq_games::connect_four::{ConnectFour, ConnectFourState, ROWS, COLS};

#[pyclass]
pub struct PyTicTacToe {
    game: TicTacToe,
}

#[pymethods]
impl PyTicTacToe {
    #[new]
    fn new() -> Self {
        Self { game: TicTacToe }
    }

    /// Compute the minimax game value via alpha-beta search.
    fn minimax(&self, board: Vec<u8>, next_player: u8, depth: usize) -> f64 {
        let mut board_arr = [0u8; 9];
        for (i, &v) in board.iter().enumerate().take(9) {
            board_arr[i] = v;
        }
        let state = TicTacToeState {
            board: board_arr,
            next_player,
        };
        minimax::alpha_beta(
            &self.game,
            &state,
            depth,
            f64::NEG_INFINITY,
            f64::INFINITY,
        )
    }

    /// Return the optimal move for the current position.
    fn best_move(&self, board: Vec<u8>, next_player: u8, depth: usize) -> Option<usize> {
        let mut board_arr = [0u8; 9];
        for (i, &v) in board.iter().enumerate().take(9) {
            board_arr[i] = v;
        }
        let state = TicTacToeState {
            board: board_arr,
            next_player,
        };
        minimax::best_action(&self.game, &state, depth)
    }

    /// Return the list of legal moves.
    fn legal_actions(&self, board: Vec<u8>, next_player: u8) -> Vec<usize> {
        let mut board_arr = [0u8; 9];
        for (i, &v) in board.iter().enumerate().take(9) {
            board_arr[i] = v;
        }
        let state = TicTacToeState {
            board: board_arr,
            next_player,
        };
        self.game.legal_actions(&state)
    }
}

#[pyclass]
pub struct PyConnectFour {
    game: ConnectFour,
}

// Use u16 instead of u8 for the Python-facing board representation.
// PyO3 converts Vec<u8> to Python `bytes`, which is immutable and not
// subscriptable as `list[int]`. Using u16 avoids this and produces
// a proper `list[list[int]]` on the Python side.

#[pymethods]
impl PyConnectFour {
    #[new]
    fn new() -> Self {
        Self {
            game: ConnectFour,
        }
    }

    /// Compute the minimax game value via alpha-beta search.
    fn minimax(&self, board: Vec<Vec<u16>>, next_player: u8, depth: usize) -> f64 {
        let state = Self::to_state(&board, next_player);
        minimax::alpha_beta(
            &self.game,
            &state,
            depth,
            f64::NEG_INFINITY,
            f64::INFINITY,
        )
    }

    /// Return the optimal move (column index) for the current position.
    fn best_move(&self, board: Vec<Vec<u16>>, next_player: u8, depth: usize) -> Option<usize> {
        let state = Self::to_state(&board, next_player);
        minimax::best_action(&self.game, &state, depth)
    }

    /// Return the list of legal moves (column indices).
    fn legal_actions(&self, board: Vec<Vec<u16>>, next_player: u8) -> Vec<usize> {
        let state = Self::to_state(&board, next_player);
        self.game.legal_actions(&state)
    }

    /// Apply a move and return the resulting board.
    fn apply_move(&self, board: Vec<Vec<u16>>, next_player: u8, col: usize) -> Vec<Vec<u16>> {
        let state = Self::to_state(&board, next_player);
        let new_state = self.game.apply(&state, &col);
        new_state
            .board
            .iter()
            .map(|row| row.iter().map(|&v| v as u16).collect())
            .collect()
    }

    /// Check for a winner. Returns 0=ongoing, 1=player 1 wins, 2=player 2 wins, 3=draw.
    fn check_winner(&self, board: Vec<Vec<u16>>) -> u8 {
        let state = Self::to_state(&board, 1);
        ConnectFour::check_winner(&state.board)
    }

    /// Return an empty 6x7 board.
    #[staticmethod]
    fn empty_board() -> Vec<Vec<u16>> {
        vec![vec![0u16; COLS]; ROWS]
    }
}

impl PyConnectFour {
    fn to_state(board: &[Vec<u16>], next_player: u8) -> ConnectFourState {
        let mut board_arr = [[0u8; COLS]; ROWS];
        for (r, row) in board.iter().enumerate().take(ROWS) {
            for (c, &val) in row.iter().enumerate().take(COLS) {
                board_arr[r][c] = val as u8;
            }
        }
        ConnectFourState {
            board: board_arr,
            next_player,
        }
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTicTacToe>()?;
    m.add_class::<PyConnectFour>()?;
    Ok(())
}
