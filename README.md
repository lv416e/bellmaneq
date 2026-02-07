# bellmaneq

Bellman equation solver in Rust with Python bindings.

Covers three domains through a unified dynamic programming lens:

- **MDP solvers** -- Value Iteration, Policy Iteration, Modified Policy Iteration
- **Game theory** -- Minimax with alpha-beta pruning (Tic-Tac-Toe, Connect Four)
- **Finance** -- American option pricing (CRR binomial, finite difference, optimal stopping)

## Structure

```
crates/
  bellmaneq-core/      # MDP trait, Bellman operator, solvers
  bellmaneq-games/     # Minimax games, TicTacToe-as-MDP adapter
  bellmaneq-finance/   # American options, optimal stopping, FD method
  bellmaneq-py/        # PyO3 bindings
python/bellmaneq/      # Python package with visualization helpers
notebooks/             # Jupyter tutorials
tests/                 # Python test suite
```

## Requirements

- Rust (stable)
- Python 3.12+
- [maturin](https://github.com/PyO3/maturin)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install maturin numpy matplotlib pytest
maturin develop --release
```

## Usage

```python
import bellmaneq

# Solve an MDP with Value Iteration
result = bellmaneq.solve_value_iteration(rewards, transitions, gamma=0.95)
print(result.get_values())

# Price an American put option
option = bellmaneq.price_american_option(
    spot=100, strike=100, rate=0.05,
    volatility=0.2, maturity=1.0, steps=500,
)
print(option.price)

# Play Tic-Tac-Toe with minimax
game = bellmaneq.TicTacToe()
move = game.best_move([0]*9, next_player=1, depth=9)
```

## Tests

```bash
# Rust
cargo test -p bellmaneq-core -p bellmaneq-games -p bellmaneq-finance

# Python
pytest tests/ -v
```

## License

MIT
