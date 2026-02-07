use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use bellmaneq_finance::american_option::price_american_option;

fn bench_american_option_steps(c: &mut Criterion) {
    let mut group = c.benchmark_group("american_option_pricing");

    for steps in [50, 100, 200, 500, 1000] {
        group.bench_with_input(
            BenchmarkId::new("steps", steps),
            &steps,
            |b, &steps| {
                b.iter(|| {
                    price_american_option(
                        black_box(100.0),
                        black_box(100.0),
                        black_box(0.05),
                        black_box(0.2),
                        black_box(1.0),
                        black_box(steps),
                        black_box(false),
                    )
                })
            },
        );
    }

    group.finish();
}

fn bench_option_call_vs_put(c: &mut Criterion) {
    let mut group = c.benchmark_group("call_vs_put");
    let steps = 500;

    group.bench_function("put", |b| {
        b.iter(|| {
            price_american_option(
                black_box(100.0),
                black_box(100.0),
                black_box(0.05),
                black_box(0.2),
                black_box(1.0),
                black_box(steps),
                black_box(false),
            )
        })
    });

    group.bench_function("call", |b| {
        b.iter(|| {
            price_american_option(
                black_box(100.0),
                black_box(100.0),
                black_box(0.05),
                black_box(0.2),
                black_box(1.0),
                black_box(steps),
                black_box(true),
            )
        })
    });

    group.finish();
}

criterion_group!(benches, bench_american_option_steps, bench_option_call_vs_put);
criterion_main!(benches);
