use crate::minimax::ZeroSumGame;

/// Game state for Tic-Tac-Toe.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TicTacToeState {
    /// Board cells: 0 = empty, 1 = X (first player), 2 = O (second player).
    pub board: [u8; 9],
    /// Player to move next (1 = X, 2 = O).
    pub next_player: u8,
}

impl TicTacToeState {
    pub fn new() -> Self {
        Self {
            board: [0; 9],
            next_player: 1,
        }
    }
}

impl Default for TicTacToeState {
    fn default() -> Self {
        Self::new()
    }
}

/// Tic-Tac-Toe game engine.
pub struct TicTacToe;

impl TicTacToe {
    /// Determines the winner. Returns 0 = ongoing, 1 = X wins, 2 = O wins, 3 = draw.
    pub fn check_winner(board: &[u8; 9]) -> u8 {
        const LINES: [[usize; 3]; 8] = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8], // rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8], // columns
            [0, 4, 8], [2, 4, 6],             // diagonals
        ];

        for line in &LINES {
            let a = board[line[0]];
            if a != 0 && a == board[line[1]] && a == board[line[2]] {
                return a;
            }
        }

        // Check for draw
        if board.iter().all(|&c| c != 0) {
            return 3;
        }

        0 // Game still in progress
    }
}

impl ZeroSumGame for TicTacToe {
    type State = TicTacToeState;
    type Action = usize; // Cell index 0-8

    fn is_max_player(&self, state: &Self::State) -> bool {
        state.next_player == 1
    }

    fn legal_actions(&self, state: &Self::State) -> Vec<Self::Action> {
        state
            .board
            .iter()
            .enumerate()
            .filter(|(_, &cell)| cell == 0)
            .map(|(i, _)| i)
            .collect()
    }

    fn apply(&self, state: &Self::State, action: &Self::Action) -> Self::State {
        let mut new_state = state.clone();
        new_state.board[*action] = state.next_player;
        new_state.next_player = if state.next_player == 1 { 2 } else { 1 };
        new_state
    }

    fn is_terminal(&self, state: &Self::State) -> bool {
        Self::check_winner(&state.board) != 0
    }

    fn evaluate(&self, state: &Self::State) -> f64 {
        match Self::check_winner(&state.board) {
            1 => 1.0,   // X wins
            2 => -1.0,  // O wins
            _ => 0.0,   // Draw or still in progress
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::minimax::{alpha_beta, best_action};

    #[test]
    fn test_initial_state() {
        let state = TicTacToeState::new();
        assert_eq!(state.board, [0; 9]);
        assert_eq!(state.next_player, 1);
    }

    #[test]
    fn test_tic_tac_toe_is_draw() {
        // Full-depth search: the theoretical game value of Tic-Tac-Toe is a draw (0.0)
        let game = TicTacToe;
        let state = TicTacToeState::new();
        let value = alpha_beta(&game, &state, 9, f64::NEG_INFINITY, f64::INFINITY);
        assert!(
            (value - 0.0).abs() < 1e-10,
            "Tic-Tac-Toe game value should be 0 (draw), got {}",
            value
        );
    }

    #[test]
    fn test_best_action_exists() {
        let game = TicTacToe;
        let state = TicTacToeState::new();
        let action = best_action(&game, &state, 9);
        assert!(action.is_some());
    }

    #[test]
    fn test_winning_detection() {
        let board = [1, 1, 1, 0, 0, 0, 0, 0, 0];
        assert_eq!(TicTacToe::check_winner(&board), 1);

        let board = [2, 0, 0, 2, 0, 0, 2, 0, 0];
        assert_eq!(TicTacToe::check_winner(&board), 2);
    }
}
