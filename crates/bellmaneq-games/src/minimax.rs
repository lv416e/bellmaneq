/// Trait defining a two-player zero-sum game.
pub trait ZeroSumGame {
    type State: Clone;
    type Action: Clone;

    /// Returns whether the current player is the maximizing player.
    fn is_max_player(&self, state: &Self::State) -> bool;

    /// Generates all legal actions from the given state.
    fn legal_actions(&self, state: &Self::State) -> Vec<Self::Action>;

    /// Applies an action and returns the resulting state.
    fn apply(&self, state: &Self::State, action: &Self::Action) -> Self::State;

    /// Returns whether the given state is terminal.
    fn is_terminal(&self, state: &Self::State) -> bool;

    /// Evaluates a terminal state from the maximizing player's perspective.
    fn evaluate(&self, state: &Self::State) -> f64;
}

/// Minimax search with alpha-beta pruning.
///
/// Solves the minimax Bellman equation:
///   V(s) = max_a V(apply(s,a))  (maximizer's turn)
///   V(s) = min_a V(apply(s,a))  (minimizer's turn)
pub fn alpha_beta<G: ZeroSumGame>(
    game: &G,
    state: &G::State,
    depth: usize,
    mut alpha: f64,
    mut beta: f64,
) -> f64 {
    if depth == 0 || game.is_terminal(state) {
        return game.evaluate(state);
    }

    let actions = game.legal_actions(state);
    if actions.is_empty() {
        return game.evaluate(state);
    }

    if game.is_max_player(state) {
        let mut value = f64::NEG_INFINITY;
        for action in &actions {
            let child = game.apply(state, action);
            value = value.max(alpha_beta(game, &child, depth - 1, alpha, beta));
            alpha = alpha.max(value);
            if alpha >= beta {
                break;
            }
        }
        value
    } else {
        let mut value = f64::INFINITY;
        for action in &actions {
            let child = game.apply(state, action);
            value = value.min(alpha_beta(game, &child, depth - 1, alpha, beta));
            beta = beta.min(value);
            if alpha >= beta {
                break;
            }
        }
        value
    }
}

/// Returns the best action from the given state.
pub fn best_action<G: ZeroSumGame>(
    game: &G,
    state: &G::State,
    depth: usize,
) -> Option<G::Action> {
    let actions = game.legal_actions(state);
    if actions.is_empty() {
        return None;
    }

    let is_max = game.is_max_player(state);

    actions
        .into_iter()
        .map(|a| {
            let child = game.apply(state, &a);
            let val = alpha_beta(game, &child, depth - 1, f64::NEG_INFINITY, f64::INFINITY);
            (a, val)
        })
        .max_by(|(_, v1), (_, v2)| {
            if is_max {
                v1.partial_cmp(v2).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                v2.partial_cmp(v1).unwrap_or(std::cmp::Ordering::Equal)
            }
        })
        .map(|(a, _)| a)
}
