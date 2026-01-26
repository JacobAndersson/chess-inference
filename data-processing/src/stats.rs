use serde::Serialize;
use std::collections::HashMap;

use crate::buckets::{elo_bucket, ply_bucket};
use crate::parser::GameData;

#[derive(Default)]
pub struct Statistics {
    buckets: HashMap<(u16, u16), u64>,
    pub games_processed: u64,
    pub games_skipped: u64,
}

#[derive(Serialize)]
pub struct BucketEntry {
    pub elo_min: u16,
    pub elo_max: u16,
    pub ply_min: u16,
    pub ply_max: u16,
    pub count: u64,
}

#[derive(Serialize)]
pub struct StatsOutput {
    pub total_games: u64,
    pub skipped_games: u64,
    pub buckets: Vec<BucketEntry>,
}

impl Statistics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_game(&mut self, game: &GameData) {
        let elo = elo_bucket(game.avg_elo);
        let ply = ply_bucket(game.ply_count);
        *self.buckets.entry((elo, ply)).or_insert(0) += 1;
        self.games_processed += 1;
    }

    pub fn record_skipped(&mut self) {
        self.games_skipped += 1;
    }

    pub fn to_output(&self) -> StatsOutput {
        let mut buckets: Vec<BucketEntry> = self
            .buckets
            .iter()
            .map(|(&(elo, ply), &count)| BucketEntry {
                elo_min: elo,
                elo_max: elo + 100,
                ply_min: ply,
                ply_max: ply + 10,
                count,
            })
            .collect();

        buckets.sort_by_key(|b| (b.elo_min, b.ply_min));

        StatsOutput {
            total_games: self.games_processed,
            skipped_games: self.games_skipped,
            buckets,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_single_game() {
        let mut stats = Statistics::new();
        stats.record_game(&GameData {
            avg_elo: 1550,
            ply_count: 45,
        });

        assert_eq!(stats.games_processed, 1);
        assert_eq!(stats.games_skipped, 0);

        let output = stats.to_output();
        assert_eq!(output.total_games, 1);
        assert_eq!(output.buckets.len(), 1);
        assert_eq!(output.buckets[0].elo_min, 1500);
        assert_eq!(output.buckets[0].ply_min, 40);
        assert_eq!(output.buckets[0].count, 1);
    }

    #[test]
    fn test_record_multiple_same_bucket() {
        let mut stats = Statistics::new();
        stats.record_game(&GameData {
            avg_elo: 1500,
            ply_count: 40,
        });
        stats.record_game(&GameData {
            avg_elo: 1550,
            ply_count: 45,
        });

        let output = stats.to_output();
        assert_eq!(output.total_games, 2);
        assert_eq!(output.buckets.len(), 1);
        assert_eq!(output.buckets[0].count, 2);
    }

    #[test]
    fn test_record_different_buckets() {
        let mut stats = Statistics::new();
        stats.record_game(&GameData {
            avg_elo: 1500,
            ply_count: 40,
        });
        stats.record_game(&GameData {
            avg_elo: 2000,
            ply_count: 80,
        });

        let output = stats.to_output();
        assert_eq!(output.total_games, 2);
        assert_eq!(output.buckets.len(), 2);
    }

    #[test]
    fn test_record_skipped() {
        let mut stats = Statistics::new();
        stats.record_skipped();
        stats.record_skipped();

        assert_eq!(stats.games_skipped, 2);
        assert_eq!(stats.games_processed, 0);

        let output = stats.to_output();
        assert_eq!(output.skipped_games, 2);
    }

    #[test]
    fn test_output_sorted() {
        let mut stats = Statistics::new();
        stats.record_game(&GameData {
            avg_elo: 2000,
            ply_count: 80,
        });
        stats.record_game(&GameData {
            avg_elo: 1500,
            ply_count: 40,
        });
        stats.record_game(&GameData {
            avg_elo: 1500,
            ply_count: 80,
        });

        let output = stats.to_output();
        assert_eq!(output.buckets[0].elo_min, 1500);
        assert_eq!(output.buckets[0].ply_min, 40);
        assert_eq!(output.buckets[1].elo_min, 1500);
        assert_eq!(output.buckets[1].ply_min, 80);
        assert_eq!(output.buckets[2].elo_min, 2000);
    }
}
