use crate::minimax::ZeroSumGame;

/// Connect Four board dimensions.
pub const ROWS: usize = 6;
pub const COLS: usize = 7;

/// Game state for Connect Four.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ConnectFourState {
    /// Board cells: 0 = empty, 1 = first player (red), 2 = second player (yellow).
    pub board: [[u8; COLS]; ROWS],
    /// Player to move next (1 or 2).
    pub next_player: u8,
}

impl ConnectFourState {
    pub fn new() -> Self {
        Self {
            board: [[0; COLS]; ROWS],
            next_player: 1,
        }
    }

    /// Returns the stack height for a column (the row where the next piece will land).
    fn column_height(&self, col: usize) -> usize {
        for row in (0..ROWS).rev() {
            if self.board[row][col] == 0 {
                return row;
            }
        }
        ROWS // Column is full
    }

    /// Returns a human-readable string representation of the board.
    pub fn display(&self) -> String {
        let mut s = String::new();
        for row in &self.board {
            for &cell in row {
                s.push(match cell {
                    1 => 'X',
                    2 => 'O',
                    _ => '.',
                });
                s.push(' ');
            }
            s.push('\n');
        }
        for i in 0..COLS {
            s.push_str(&format!("{} ", i));
        }
        s
    }
}

impl Default for ConnectFourState {
    fn default() -> Self {
        Self::new()
    }
}

/// Connect Four game engine.
pub struct ConnectFour;

impl ConnectFour {
    /// Determines the winner. Returns 0 = ongoing, 1 = first player wins, 2 = second player wins, 3 = draw.
    pub fn check_winner(board: &[[u8; COLS]; ROWS]) -> u8 {
        // Horizontal
        for row in 0..ROWS {
            for col in 0..COLS - 3 {
                let a = board[row][col];
                if a != 0
                    && a == board[row][col + 1]
                    && a == board[row][col + 2]
                    && a == board[row][col + 3]
                {
                    return a;
                }
            }
        }

        // Vertical
        for row in 0..ROWS - 3 {
            for col in 0..COLS {
                let a = board[row][col];
                if a != 0
                    && a == board[row + 1][col]
                    && a == board[row + 2][col]
                    && a == board[row + 3][col]
                {
                    return a;
                }
            }
        }

        // Diagonal (top-left to bottom-right)
        for row in 0..ROWS - 3 {
            for col in 0..COLS - 3 {
                let a = board[row][col];
                if a != 0
                    && a == board[row + 1][col + 1]
                    && a == board[row + 2][col + 2]
                    && a == board[row + 3][col + 3]
                {
                    return a;
                }
            }
        }

        // Diagonal (bottom-left to top-right)
        for row in 3..ROWS {
            for col in 0..COLS - 3 {
                let a = board[row][col];
                if a != 0
                    && a == board[row - 1][col + 1]
                    && a == board[row - 2][col + 2]
                    && a == board[row - 3][col + 3]
                {
                    return a;
                }
            }
        }

        // Draw (all cells occupied)
        if board[0].iter().all(|&c| c != 0) {
            return 3;
        }

        0
    }

    /// Heuristic evaluation function for non-terminal states when search
    /// depth is insufficient.
    ///
    /// Scores sliding windows of four cells across all directions.
    pub fn heuristic_evaluate(board: &[[u8; COLS]; ROWS]) -> f64 {
        let mut score = 0.0;

        // Center column bonus
        for row in 0..ROWS {
            if board[row][3] == 1 {
                score += 3.0;
            } else if board[row][3] == 2 {
                score -= 3.0;
            }
        }

        // Score all four-cell sliding windows in every direction
        let score_window = |window: &[u8]| -> f64 {
            let p1 = window.iter().filter(|&&c| c == 1).count();
            let p2 = window.iter().filter(|&&c| c == 2).count();
            let empty = window.iter().filter(|&&c| c == 0).count();

            if p1 == 4 {
                return 100.0;
            }
            if p2 == 4 {
                return -100.0;
            }
            if p1 == 3 && empty == 1 {
                return 5.0;
            }
            if p2 == 3 && empty == 1 {
                return -5.0;
            }
            if p1 == 2 && empty == 2 {
                return 2.0;
            }
            if p2 == 2 && empty == 2 {
                return -2.0;
            }
            0.0
        };

        // Horizontal
        for row in 0..ROWS {
            for col in 0..COLS - 3 {
                let w = [
                    board[row][col],
                    board[row][col + 1],
                    board[row][col + 2],
                    board[row][col + 3],
                ];
                score += score_window(&w);
            }
        }

        // Vertical
        for row in 0..ROWS - 3 {
            for col in 0..COLS {
                let w = [
                    board[row][col],
                    board[row + 1][col],
                    board[row + 2][col],
                    board[row + 3][col],
                ];
                score += score_window(&w);
            }
        }

        // Diagonals
        for row in 0..ROWS - 3 {
            for col in 0..COLS - 3 {
                let w = [
                    board[row][col],
                    board[row + 1][col + 1],
                    board[row + 2][col + 2],
                    board[row + 3][col + 3],
                ];
                score += score_window(&w);
            }
        }
        for row in 3..ROWS {
            for col in 0..COLS - 3 {
                let w = [
                    board[row][col],
                    board[row - 1][col + 1],
                    board[row - 2][col + 2],
                    board[row - 3][col + 3],
                ];
                score += score_window(&w);
            }
        }

        score
    }
}

impl ZeroSumGame for ConnectFour {
    type State = ConnectFourState;
    type Action = usize; // Column index (0-6)

    fn is_max_player(&self, state: &Self::State) -> bool {
        state.next_player == 1
    }

    fn legal_actions(&self, state: &Self::State) -> Vec<Self::Action> {
        // Search from the center column outward for better pruning efficiency
        let order = [3, 2, 4, 1, 5, 0, 6];
        order
            .iter()
            .filter(|&&col| state.board[0][col] == 0)
            .copied()
            .collect()
    }

    fn apply(&self, state: &Self::State, action: &Self::Action) -> Self::State {
        let mut new_state = state.clone();
        let row = state.column_height(*action);
        new_state.board[row][*action] = state.next_player;
        new_state.next_player = if state.next_player == 1 { 2 } else { 1 };
        new_state
    }

    fn is_terminal(&self, state: &Self::State) -> bool {
        Self::check_winner(&state.board) != 0
    }

    fn evaluate(&self, state: &Self::State) -> f64 {
        match Self::check_winner(&state.board) {
            1 => 1000.0,
            2 => -1000.0,
            3 => 0.0,
            _ => Self::heuristic_evaluate(&state.board),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::minimax::{alpha_beta, best_action};

    #[test]
    fn test_initial_state() {
        let state = ConnectFourState::new();
        assert_eq!(state.next_player, 1);
        assert!(state.board.iter().all(|row| row.iter().all(|&c| c == 0)));
    }

    #[test]
    fn test_legal_actions() {
        let game = ConnectFour;
        let state = ConnectFourState::new();
        let actions = game.legal_actions(&state);
        assert_eq!(actions.len(), 7);
    }

    #[test]
    fn test_vertical_win() {
        let game = ConnectFour;
        let mut state = ConnectFourState::new();
        // Player 1 stacks four pieces in column 0
        for _ in 0..4 {
            state = game.apply(&state, &0);
            if state.next_player == 1 {
                // Player 2 places in column 1
            } else {
                state = game.apply(&state, &1);
            }
        }
        // Construct the board directly for verification
        let mut board = [[0u8; COLS]; ROWS];
        for row in (ROWS - 4)..ROWS {
            board[row][0] = 1;
        }
        assert_eq!(ConnectFour::check_winner(&board), 1);
    }

    #[test]
    fn test_horizontal_win() {
        let mut board = [[0u8; COLS]; ROWS];
        board[ROWS - 1][0] = 1;
        board[ROWS - 1][1] = 1;
        board[ROWS - 1][2] = 1;
        board[ROWS - 1][3] = 1;
        assert_eq!(ConnectFour::check_winner(&board), 1);
    }

    #[test]
    fn test_diagonal_win() {
        let mut board = [[0u8; COLS]; ROWS];
        // Descending diagonal (top-left to bottom-right)
        board[ROWS - 4][0] = 1;
        board[ROWS - 3][1] = 1;
        board[ROWS - 2][2] = 1;
        board[ROWS - 1][3] = 1;
        assert_eq!(ConnectFour::check_winner(&board), 1);
    }

    #[test]
    fn test_best_move_blocks_threat() {
        let game = ConnectFour;
        // Player 2 has three pieces stacked in column 0 -- player 1 must block
        let mut board = [[0u8; COLS]; ROWS];
        board[ROWS - 1][0] = 2;
        board[ROWS - 2][0] = 2;
        board[ROWS - 3][0] = 2;
        let state = ConnectFourState {
            board,
            next_player: 1,
        };
        let best = best_action(&game, &state, 4);
        assert_eq!(best, Some(0), "Should block the vertical threat at column 0");
    }

    #[test]
    fn test_shallow_search() {
        let game = ConnectFour;
        let state = ConnectFourState::new();
        // Even a shallow search should return a legal move
        let action = best_action(&game, &state, 4);
        assert!(action.is_some());
        let val = alpha_beta(&game, &state, 4, f64::NEG_INFINITY, f64::INFINITY);
        assert!(val.is_finite());
    }
}
