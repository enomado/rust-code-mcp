//! Time each pass of `LanceDbBackend::maintain` separately on a COPY of a
//! vector store: `cargo run --release -p rmc-engine --features vector-store
//! --example maintain_probe -- <collection dir> [merge N]`.
//!
//! The live log only prints the whole pass (136–425 s on a large workspace), which
//! cannot tell compaction, the scalar/ANN index update and prune apart. Never
//! point it at a store a daemon has open: the passes write.

use std::time::Instant;

use lancedb::table::{OptimizeAction, OptimizeOptions};

async fn report(table: &lancedb::Table, when: &str) {
    let stats = table.stats().await.unwrap();
    println!(
        "[{when}] version {} rows {} fragments {} small {}",
        table.version().await.unwrap(),
        stats.num_rows,
        stats.fragment_stats.num_fragments,
        stats.fragment_stats.num_small_fragments,
    );
    // `list_indices` lists one entry per DELTA of an index, all under the
    // same name (196 for the ANN index of a large workspace on 2026-10-31),
    // and every `index_stats` costs seconds there: one line per name.
    let mut seen = std::collections::HashSet::new();
    let configs = table.list_indices().await.unwrap();
    println!("  list_indices: {} entries", configs.len());
    for config in configs {
        if !seen.insert(config.name.clone()) {
            continue;
        }
        let index = table.index_stats(&config.name).await.unwrap().unwrap();
        println!(
            "  index {} {:?} {:?}: parts {:?}, indexed {}, unindexed {}",
            config.name,
            config.columns,
            index.index_type,
            index.num_indices,
            index.num_indexed_rows,
            index.num_unindexed_rows,
        );
    }
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = &args[1];
    let merge = args.get(2).map(|n| n.parse::<usize>().unwrap());

    let db = lancedb::connect(dir).execute().await.unwrap();
    let table = db.open_table("vectors").execute().await.unwrap();
    report(&table, "before").await;

    let started = Instant::now();
    table
        .optimize(OptimizeAction::Index(OptimizeOptions::default()))
        .await
        .unwrap();
    println!("Index(default) took {:.2}s", started.elapsed().as_secs_f64());
    report(&table, "after Index(default)").await;

    let started = Instant::now();
    table
        .optimize(OptimizeAction::Index(OptimizeOptions::default()))
        .await
        .unwrap();
    println!(
        "Index(default) again, nothing new: {:.2}s",
        started.elapsed().as_secs_f64()
    );

    if let Some(n) = merge {
        let started = Instant::now();
        table
            .optimize(OptimizeAction::Index(OptimizeOptions::merge(n)))
            .await
            .unwrap();
        println!("Index(merge {n}) took {:.2}s", started.elapsed().as_secs_f64());
        report(&table, "after merge").await;

        let started = Instant::now();
        table
            .optimize(OptimizeAction::Index(OptimizeOptions::default()))
            .await
            .unwrap();
        println!(
            "Index(default) after merge: {:.2}s",
            started.elapsed().as_secs_f64()
        );
    }

    let started = Instant::now();
    table
        .optimize(OptimizeAction::Prune {
            older_than: Some(lancedb::table::optimize::Duration::seconds(600)),
            delete_unverified: None,
            error_if_tagged_old_versions: None,
        })
        .await
        .unwrap();
    println!("Prune took {:.2}s", started.elapsed().as_secs_f64());
}
