use serde::Serialize;
use std::collections::HashMap;

use crate::buckets::elo_bucket;
use crate::parser::GameData;
use crate::time_control::TimeControl;

#[derive(Default)]
pub struct Statistics {
    elo_buckets: HashMap<u16, u64>,
    time_control_counts: HashMap<TimeControl, u64>,
    pub games_processed: u64,
    pub games_skipped: u64,
}

#[derive(Serialize)]
pub struct EloBucketEntry {
    pub elo_min: u16,
    pub elo_max: u16,
    pub count: u64,
}

#[derive(Serialize)]
pub struct TimeControlEntry {
    pub category: TimeControl,
    pub count: u64,
}

#[derive(Serialize)]
pub struct StatsOutput {
    pub total_games: u64,
    pub skipped_games: u64,
    pub elo_distribution: Vec<EloBucketEntry>,
    pub time_control_distribution: Vec<TimeControlEntry>,
}

impl Statistics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_game(&mut self, game: &GameData) {
        let elo = elo_bucket(game.max_elo);
        *self.elo_buckets.entry(elo).or_insert(0) += 1;
        *self
            .time_control_counts
            .entry(game.time_control)
            .or_insert(0) += 1;
        self.games_processed += 1;
    }

    pub fn record_skipped(&mut self) {
        self.games_skipped += 1;
    }

    pub fn merge(&mut self, other: Statistics) {
        for (elo, count) in other.elo_buckets {
            *self.elo_buckets.entry(elo).or_insert(0) += count;
        }
        for (tc, count) in other.time_control_counts {
            *self.time_control_counts.entry(tc).or_insert(0) += count;
        }
        self.games_processed += other.games_processed;
        self.games_skipped += other.games_skipped;
    }

    pub fn to_output(&self) -> StatsOutput {
        let mut elo_distribution: Vec<EloBucketEntry> = self
            .elo_buckets
            .iter()
            .map(|(&elo, &count)| EloBucketEntry {
                elo_min: elo,
                elo_max: elo + 100,
                count,
            })
            .collect();
        elo_distribution.sort_by_key(|b| b.elo_min);

        let mut time_control_distribution: Vec<TimeControlEntry> = self
            .time_control_counts
            .iter()
            .map(|(&category, &count)| TimeControlEntry { category, count })
            .collect();
        time_control_distribution.sort_by_key(|t| match t.category {
            TimeControl::Bullet => 0,
            TimeControl::Blitz => 1,
            TimeControl::Rapid => 2,
            TimeControl::Classical => 3,
            TimeControl::Unknown => 4,
        });

        StatsOutput {
            total_games: self.games_processed,
            skipped_games: self.games_skipped,
            elo_distribution,
            time_control_distribution,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn game(max_elo: u16, time_control: TimeControl) -> GameData {
        GameData {
            max_elo,
            ply_count: 40,
            time_control,
        }
    }

    #[test]
    fn test_record_single_game() {
        let mut stats = Statistics::new();
        stats.record_game(&game(1550, TimeControl::Blitz));

        assert_eq!(stats.games_processed, 1);
        assert_eq!(stats.games_skipped, 0);

        let output = stats.to_output();
        assert_eq!(output.total_games, 1);
        assert_eq!(output.elo_distribution.len(), 1);
        assert_eq!(output.elo_distribution[0].elo_min, 1500);
        assert_eq!(output.elo_distribution[0].count, 1);
        assert_eq!(output.time_control_distribution.len(), 1);
        assert_eq!(
            output.time_control_distribution[0].category,
            TimeControl::Blitz
        );
        assert_eq!(output.time_control_distribution[0].count, 1);
    }

    #[test]
    fn test_record_multiple_same_bucket() {
        let mut stats = Statistics::new();
        stats.record_game(&game(1500, TimeControl::Blitz));
        stats.record_game(&game(1550, TimeControl::Blitz));

        let output = stats.to_output();
        assert_eq!(output.total_games, 2);
        assert_eq!(output.elo_distribution.len(), 1);
        assert_eq!(output.elo_distribution[0].count, 2);
    }

    #[test]
    fn test_record_different_elo_buckets() {
        let mut stats = Statistics::new();
        stats.record_game(&game(1500, TimeControl::Blitz));
        stats.record_game(&game(2000, TimeControl::Blitz));

        let output = stats.to_output();
        assert_eq!(output.total_games, 2);
        assert_eq!(output.elo_distribution.len(), 2);
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
    fn test_elo_output_sorted() {
        let mut stats = Statistics::new();
        stats.record_game(&game(2000, TimeControl::Blitz));
        stats.record_game(&game(1500, TimeControl::Blitz));
        stats.record_game(&game(1800, TimeControl::Blitz));

        let output = stats.to_output();
        assert_eq!(output.elo_distribution[0].elo_min, 1500);
        assert_eq!(output.elo_distribution[1].elo_min, 1800);
        assert_eq!(output.elo_distribution[2].elo_min, 2000);
    }

    #[test]
    fn test_time_control_distribution() {
        let mut stats = Statistics::new();
        stats.record_game(&game(1500, TimeControl::Bullet));
        stats.record_game(&game(1500, TimeControl::Bullet));
        stats.record_game(&game(1500, TimeControl::Blitz));
        stats.record_game(&game(1500, TimeControl::Rapid));

        let output = stats.to_output();
        assert_eq!(output.time_control_distribution.len(), 3);

        assert_eq!(
            output.time_control_distribution[0].category,
            TimeControl::Bullet
        );
        assert_eq!(output.time_control_distribution[0].count, 2);
        assert_eq!(
            output.time_control_distribution[1].category,
            TimeControl::Blitz
        );
        assert_eq!(output.time_control_distribution[1].count, 1);
        assert_eq!(
            output.time_control_distribution[2].category,
            TimeControl::Rapid
        );
        assert_eq!(output.time_control_distribution[2].count, 1);
    }

    #[test]
    fn test_time_control_sorted_order() {
        let mut stats = Statistics::new();
        stats.record_game(&game(1500, TimeControl::Classical));
        stats.record_game(&game(1500, TimeControl::Bullet));
        stats.record_game(&game(1500, TimeControl::Unknown));
        stats.record_game(&game(1500, TimeControl::Rapid));
        stats.record_game(&game(1500, TimeControl::Blitz));

        let output = stats.to_output();
        assert_eq!(
            output.time_control_distribution[0].category,
            TimeControl::Bullet
        );
        assert_eq!(
            output.time_control_distribution[1].category,
            TimeControl::Blitz
        );
        assert_eq!(
            output.time_control_distribution[2].category,
            TimeControl::Rapid
        );
        assert_eq!(
            output.time_control_distribution[3].category,
            TimeControl::Classical
        );
        assert_eq!(
            output.time_control_distribution[4].category,
            TimeControl::Unknown
        );
    }

    #[test]
    fn test_merge_empty_stats() {
        let mut stats1 = Statistics::new();
        stats1.record_game(&game(1500, TimeControl::Blitz));

        let stats2 = Statistics::new();
        stats1.merge(stats2);

        assert_eq!(stats1.games_processed, 1);
        assert_eq!(stats1.games_skipped, 0);
    }

    #[test]
    fn test_merge_combines_elo_buckets() {
        let mut stats1 = Statistics::new();
        stats1.record_game(&game(1500, TimeControl::Blitz));
        stats1.record_game(&game(1550, TimeControl::Blitz));

        let mut stats2 = Statistics::new();
        stats2.record_game(&game(1520, TimeControl::Rapid));
        stats2.record_game(&game(2000, TimeControl::Bullet));

        stats1.merge(stats2);

        let output = stats1.to_output();
        assert_eq!(output.total_games, 4);
        assert_eq!(output.elo_distribution.len(), 2);

        let elo_1500 = output
            .elo_distribution
            .iter()
            .find(|e| e.elo_min == 1500)
            .expect("should have 1500 bucket");
        assert_eq!(elo_1500.count, 3);

        let elo_2000 = output
            .elo_distribution
            .iter()
            .find(|e| e.elo_min == 2000)
            .expect("should have 2000 bucket");
        assert_eq!(elo_2000.count, 1);
    }

    #[test]
    fn test_merge_combines_time_controls() {
        let mut stats1 = Statistics::new();
        stats1.record_game(&game(1500, TimeControl::Blitz));
        stats1.record_game(&game(1500, TimeControl::Blitz));

        let mut stats2 = Statistics::new();
        stats2.record_game(&game(1500, TimeControl::Blitz));
        stats2.record_game(&game(1500, TimeControl::Rapid));

        stats1.merge(stats2);

        let output = stats1.to_output();
        let blitz = output
            .time_control_distribution
            .iter()
            .find(|t| t.category == TimeControl::Blitz)
            .expect("should have blitz");
        assert_eq!(blitz.count, 3);

        let rapid = output
            .time_control_distribution
            .iter()
            .find(|t| t.category == TimeControl::Rapid)
            .expect("should have rapid");
        assert_eq!(rapid.count, 1);
    }

    #[test]
    fn test_merge_combines_skipped() {
        let mut stats1 = Statistics::new();
        stats1.record_skipped();
        stats1.record_skipped();

        let mut stats2 = Statistics::new();
        stats2.record_skipped();
        stats2.record_game(&game(1500, TimeControl::Blitz));

        stats1.merge(stats2);

        assert_eq!(stats1.games_processed, 1);
        assert_eq!(stats1.games_skipped, 3);
    }
}
