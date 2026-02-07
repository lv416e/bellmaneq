"""Python tests for the bellmaneq-core solvers."""

import numpy as np
import pytest
import bellmaneq


class TestValueIteration:
    """Tests for Value Iteration."""

    def _make_two_state_mdp(self):
        """Two-state MDP with a known analytical solution."""
        # State 0: action 0 -> state 0 (reward 1), action 1 -> state 1 (reward 0)
        # State 1: action 0 -> state 0 (reward 0), action 1 -> state 1 (reward 2)
        n_s, n_a = 2, 2
        rewards = np.array([[1.0, 0.0], [0.0, 2.0]])
        transitions = np.zeros(n_s * n_a * n_s)
        # P(s'|s,a) -- deterministic transitions
        transitions[0 * n_a * n_s + 0 * n_s + 0] = 1.0  # s=0, a=0 → s'=0
        transitions[0 * n_a * n_s + 1 * n_s + 1] = 1.0  # s=0, a=1 → s'=1
        transitions[1 * n_a * n_s + 0 * n_s + 0] = 1.0  # s=1, a=0 → s'=0
        transitions[1 * n_a * n_s + 1 * n_s + 1] = 1.0  # s=1, a=1 → s'=1
        return rewards, transitions

    def test_basic_solve(self):
        rewards, transitions = self._make_two_state_mdp()
        result = bellmaneq.solve_value_iteration(rewards, transitions, gamma=0.9)

        assert result.converged
        assert result.iterations > 0

        values = result.get_values()
        assert values.shape == (2,)
        # Optimal: s0→a1 (R=0,→s1), s1→a1 (R=2,→s1)
        # V(1) = 2/(1-0.9) = 20, V(0) = 0 + 0.9*20 = 18
        assert abs(values[0] - 18.0) < 0.1
        assert abs(values[1] - 20.0) < 0.1

    def test_policy_is_correct(self):
        rewards, transitions = self._make_two_state_mdp()
        result = bellmaneq.solve_value_iteration(rewards, transitions, gamma=0.9)

        policy = result.get_policy()
        assert policy.shape == (2,)
        # State 0: action 1 (transitions to state 1; reward 0 but state 1 has higher value)
        # State 1: action 1 (reward 2)
        # At γ=0.9, state 0 prefers action 0 (stay): 1+0.9*10=10 vs 0+0.9*20=18
        # => action 1 is optimal
        assert policy[0] == 1
        assert policy[1] == 1

    def test_convergence_history(self):
        rewards, transitions = self._make_two_state_mdp()
        result = bellmaneq.solve_value_iteration(rewards, transitions, gamma=0.9)

        history = result.get_convergence_history()
        assert len(history) > 0
        # Convergence history should be approximately monotonically decreasing
        assert history[-1] < history[0]


class TestPolicyIteration:
    """Tests for Policy Iteration."""

    def test_matches_value_iteration(self):
        """VI and PI should produce identical results."""
        n_s, n_a = 3, 2
        rng = np.random.default_rng(42)
        rewards = rng.standard_normal((n_s, n_a))

        # Build stochastic transitions
        transitions = np.zeros(n_s * n_a * n_s)
        for s in range(n_s):
            for a in range(n_a):
                probs = rng.dirichlet(np.ones(n_s))
                for sp in range(n_s):
                    transitions[s * n_a * n_s + a * n_s + sp] = probs[sp]

        vi_result = bellmaneq.solve_value_iteration(rewards, transitions, gamma=0.95)
        pi_result = bellmaneq.solve_policy_iteration(rewards, transitions, gamma=0.95)

        vi_values = vi_result.get_values()
        pi_values = pi_result.get_values()
        np.testing.assert_allclose(vi_values, pi_values, atol=1e-6)

        vi_policy = vi_result.get_policy()
        pi_policy = pi_result.get_policy()
        np.testing.assert_array_equal(vi_policy, pi_policy)


class TestInputValidation:
    """Tests for input validation."""

    def test_transitions_shape_mismatch(self):
        rewards = np.array([[1.0, 0.0]])
        transitions = np.zeros(10)  # Should be 1*2*1=2
        with pytest.raises(ValueError):
            bellmaneq.solve_value_iteration(rewards, transitions, gamma=0.9)
