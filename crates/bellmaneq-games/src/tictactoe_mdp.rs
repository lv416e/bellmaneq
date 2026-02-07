/// Adapter that wraps Tic-Tac-Toe as a Markov Decision Process.
///
/// The opponent (player 2) is modelled as selecting moves uniformly at
/// random. This makes the transitions genuinely stochastic and allows
/// the standard MDP solver (Value Iteration / Policy Iteration) to
/// compute an optimal policy for player 1 against a random opponent.
///
/// Comparing the MDP value (random opponent) with the minimax value
/// (optimal opponent) demonstrates the conceptual link and the practical
/// difference between the two formulations.
use std::collections::{HashSet, VecDeque};

use bellmaneq_core::mdp::MDP;

use crate::minimax::ZeroSumGame;
use crate::tictactoe::{TicTacToe, TicTacToeState};

/// TicTacToe viewed as a single-agent MDP with a random opponent.
pub struct TicTacToeAsMDP {
    game: TicTacToe,
    all_states: Vec<TicTacToeState>,
}

impl TicTacToeAsMDP {
    pub fn new() -> Self {
        let game = TicTacToe;
        let all_states = Self::enumerate_reachable(&game);
        Self { game, all_states }
    }

    /// BFS from the empty board to enumerate every reachable state.
    fn enumerate_reachable(game: &TicTacToe) -> Vec<TicTacToeState> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let start = TicTacToeState::new();
        visited.insert(start.clone());
        queue.push_back(start);

        while let Some(state) = queue.pop_front() {
            if game.is_terminal(&state) {
                continue;
            }
            for action in game.legal_actions(&state) {
                let next = game.apply(&state, &action);
                if visited.insert(next.clone()) {
                    queue.push_back(next);
                }
            }
        }

        visited.into_iter().collect()
    }
}

impl Default for TicTacToeAsMDP {
    fn default() -> Self {
        Self::new()
    }
}

impl MDP for TicTacToeAsMDP {
    type State = TicTacToeState;
    type Action = usize;

    fn states(&self) -> Vec<TicTacToeState> {
        self.all_states.clone()
    }

    fn actions(&self, state: &TicTacToeState) -> Vec<usize> {
        // Only player 1 makes decisions. Player-2 turns and terminal
        // states are handled by returning an empty action set (the solver
        // assigns them value 0.0).
        if self.game.is_terminal(state) || state.next_player != 1 {
            return vec![];
        }
        self.game.legal_actions(state)
    }

    fn transitions(
        &self,
        state: &TicTacToeState,
        action: &usize,
    ) -> Vec<(f64, TicTacToeState, f64)> {
        // Player 1 takes the action.
        let after_p1 = self.game.apply(state, action);

        // If the game ended (player 1 won or board full), return a single
        // deterministic transition with the terminal reward.
        if self.game.is_terminal(&after_p1) {
            let reward = self.game.evaluate(&after_p1);
            return vec![(1.0, after_p1, reward)];
        }

        // The opponent (player 2) plays each legal move with equal probability.
        let opp_actions = self.game.legal_actions(&after_p1);
        let prob = 1.0 / opp_actions.len() as f64;

        opp_actions
            .iter()
            .map(|opp_a| {
                let after_p2 = self.game.apply(&after_p1, opp_a);
                let reward = if self.game.is_terminal(&after_p2) {
                    self.game.evaluate(&after_p2)
                } else {
                    0.0
                };
                (prob, after_p2, reward)
            })
            .collect()
    }

    fn is_terminal(&self, state: &TicTacToeState) -> bool {
        // Both game-over states and player-2 states are terminal from the
        // MDP's perspective (the opponent's responses are folded into the
        // transition probabilities of the preceding player-1 state).
        self.game.is_terminal(state) || state.next_player != 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bellmaneq_core::solver::value_iteration::ValueIteration;
    use crate::minimax::alpha_beta;

    #[test]
    fn test_mdp_value_vs_minimax() {
        let mdp = TicTacToeAsMDP::new();
        let solver = ValueIteration::new(1.0).with_tolerance(1e-10);
        let result = solver.solve(&mdp);

        let start = TicTacToeState::new();
        let mdp_value = *result.values.get(&start).unwrap_or(&0.0);

        // Minimax value with optimal opponent is exactly 0 (draw).
        let game = TicTacToe;
        let minimax_value = alpha_beta(&game, &start, 9, f64::NEG_INFINITY, f64::INFINITY);
        assert!(
            (minimax_value - 0.0).abs() < 1e-10,
            "Minimax value should be 0 (draw), got {}",
            minimax_value
        );

        // Against a random opponent, player 1 should win more often than
        // not, so the MDP value must be strictly greater than 0.
        assert!(
            mdp_value > 0.01,
            "MDP value ({}) should exceed minimax value ({}) — a random opponent is weaker",
            mdp_value,
            minimax_value
        );
    }

    #[test]
    fn test_mdp_converges() {
        let mdp = TicTacToeAsMDP::new();
        let solver = ValueIteration::new(1.0).with_tolerance(1e-10);
        let result = solver.solve(&mdp);
        assert!(
            result.converged,
            "Value Iteration should converge on the episodic TicTacToe MDP"
        );
    }

    #[test]
    fn test_mdp_policy_nonempty() {
        let mdp = TicTacToeAsMDP::new();
        let solver = ValueIteration::new(1.0).with_tolerance(1e-10);
        let result = solver.solve(&mdp);
        assert!(
            !result.policy.is_empty(),
            "Policy should contain entries for player-1 states"
        );
    }
}
