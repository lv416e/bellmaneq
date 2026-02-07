use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::prelude::*;
use std::collections::HashMap;

use bellmaneq_core::mdp::MDP;
use bellmaneq_core::operator::{bellman_operator, bellman_operator_par};
use bellmaneq_core::solver::value_iteration::ValueIteration;
use bellmaneq_core::solver::policy_iteration::PolicyIteration;

/// A randomly generated tabular MDP for benchmarking.
struct RandomMDP {
    n_states: usize,
    n_actions: usize,
    rewards: Vec<Vec<f64>>,
    transitions: Vec<Vec<Vec<f64>>>,
}

impl RandomMDP {
    fn generate(n_states: usize, n_actions: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut rewards = vec![vec![0.0; n_actions]; n_states];
        let mut transitions = vec![vec![vec![0.0; n_states]; n_actions]; n_states];

        for s in 0..n_states {
            for a in 0..n_actions {
                rewards[s][a] = rng.random_range(-1.0..1.0);
                // Generate Dirichlet-distributed transition probabilities
                let alphas: Vec<f64> = (0..n_states).map(|_| rng.random_range(0.01..1.0)).collect();
                let sum: f64 = alphas.iter().sum();
                for sp in 0..n_states {
                    transitions[s][a][sp] = alphas[sp] / sum;
                }
            }
        }

        Self {
            n_states,
            n_actions,
            rewards,
            transitions,
        }
    }
}

impl MDP for RandomMDP {
    type State = usize;
    type Action = usize;

    fn states(&self) -> Vec<usize> {
        (0..self.n_states).collect()
    }

    fn actions(&self, _state: &usize) -> Vec<usize> {
        (0..self.n_actions).collect()
    }

    fn transitions(&self, state: &usize, action: &usize) -> Vec<(f64, usize, f64)> {
        let reward = self.rewards[*state][*action];
        self.transitions[*state][*action]
            .iter()
            .enumerate()
            .filter(|(_, &p)| p > 1e-10)
            .map(|(sp, &p)| (p, sp, reward))
            .collect()
    }

    fn is_terminal(&self, _state: &usize) -> bool {
        false
    }
}

fn bench_value_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("value_iteration");

    for n_states in [10, 50, 100, 200] {
        let mdp = RandomMDP::generate(n_states, 4, 42);
        let solver = ValueIteration::new(0.95).with_tolerance(1e-8);

        group.bench_with_input(
            BenchmarkId::new("states", n_states),
            &mdp,
            |b, mdp| b.iter(|| solver.solve(black_box(mdp))),
        );
    }

    group.finish();
}

fn bench_policy_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_iteration");

    for n_states in [10, 50, 100, 200] {
        let mdp = RandomMDP::generate(n_states, 4, 42);
        let solver = PolicyIteration::new(0.95);

        group.bench_with_input(
            BenchmarkId::new("states", n_states),
            &mdp,
            |b, mdp| b.iter(|| solver.solve(black_box(mdp))),
        );
    }

    group.finish();
}

fn bench_bellman_operator(c: &mut Criterion) {
    let mut group = c.benchmark_group("bellman_operator");

    for n_states in [10, 50, 100, 200, 500] {
        let mdp = RandomMDP::generate(n_states, 4, 42);
        let values: HashMap<usize, f64> = (0..n_states).map(|s| (s, 0.0)).collect();

        group.bench_with_input(
            BenchmarkId::new("states", n_states),
            &(&mdp, &values),
            |b, &(mdp, values)| b.iter(|| bellman_operator(black_box(mdp), black_box(values), 0.95)),
        );
    }

    group.finish();
}

fn bench_vi_vs_pi(c: &mut Criterion) {
    let mut group = c.benchmark_group("vi_vs_pi");

    let mdp = RandomMDP::generate(100, 4, 42);

    group.bench_function("value_iteration", |b| {
        let solver = ValueIteration::new(0.95).with_tolerance(1e-8);
        b.iter(|| solver.solve(black_box(&mdp)))
    });

    group.bench_function("policy_iteration", |b| {
        let solver = PolicyIteration::new(0.95);
        b.iter(|| solver.solve(black_box(&mdp)))
    });

    group.finish();
}

fn bench_sequential_vs_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("seq_vs_par");

    for n_states in [100, 500, 1000] {
        let mdp = RandomMDP::generate(n_states, 4, 42);
        let values: HashMap<usize, f64> = (0..n_states).map(|s| (s, 0.0)).collect();

        group.bench_with_input(
            BenchmarkId::new("sequential", n_states),
            &(&mdp, &values),
            |b, &(mdp, values)| {
                b.iter(|| bellman_operator(black_box(mdp), black_box(values), 0.95))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("parallel", n_states),
            &(&mdp, &values),
            |b, &(mdp, values)| {
                b.iter(|| bellman_operator_par(black_box(mdp), black_box(values), 0.95))
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_value_iteration,
    bench_policy_iteration,
    bench_bellman_operator,
    bench_vi_vs_pi,
    bench_sequential_vs_parallel,
);
criterion_main!(benches);
