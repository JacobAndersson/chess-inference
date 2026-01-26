use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::error::PgnError;
use crate::training_visitor::TrainingGameData;

const ELO_BUCKETS: &[(u16, &str)] = &[
    (1200, "elo_1200.txt"),
    (1500, "elo_1500.txt"),
    (1800, "elo_1800.txt"),
    (2000, "elo_2000.txt"),
    (2500, "elo_2500.txt"),
];

const ALL_BUCKET: &str = "elo_all.txt";

pub struct TrainingWriter {
    writers: HashMap<&'static str, BufWriter<File>>,
    counts: HashMap<&'static str, u64>,
}

impl TrainingWriter {
    pub fn new(output_dir: &Path) -> Result<Self, PgnError> {
        fs::create_dir_all(output_dir).map_err(|e| PgnError::io(output_dir, e))?;

        let mut writers = HashMap::new();
        let mut counts = HashMap::new();

        for (_, filename) in ELO_BUCKETS {
            let path = output_dir.join(filename);
            let file = File::create(&path).map_err(|e| PgnError::io(&path, e))?;
            writers.insert(*filename, BufWriter::new(file));
            counts.insert(*filename, 0);
        }

        let all_path = output_dir.join(ALL_BUCKET);
        let all_file = File::create(&all_path).map_err(|e| PgnError::io(&all_path, e))?;
        writers.insert(ALL_BUCKET, BufWriter::new(all_file));
        counts.insert(ALL_BUCKET, 0);

        Ok(Self { writers, counts })
    }

    pub fn write_game(&mut self, game: &TrainingGameData) -> Result<(), PgnError> {
        let line = game.moves.join(" ");

        for (threshold, filename) in ELO_BUCKETS {
            if game.avg_elo < *threshold {
                self.write_to_bucket(filename, &line)?;
            }
        }

        self.write_to_bucket(ALL_BUCKET, &line)?;

        Ok(())
    }

    fn write_to_bucket(&mut self, bucket: &'static str, line: &str) -> Result<(), PgnError> {
        if let Some(writer) = self.writers.get_mut(bucket) {
            writeln!(writer, "{}", line).map_err(|e| PgnError::Parse(e.to_string()))?;
            *self.counts.entry(bucket).or_insert(0) += 1;
        }
        Ok(())
    }

    pub fn flush(&mut self) -> Result<(), PgnError> {
        for writer in self.writers.values_mut() {
            writer.flush().map_err(|e| PgnError::Parse(e.to_string()))?;
        }
        Ok(())
    }

    pub fn counts(&self) -> &HashMap<&'static str, u64> {
        &self.counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training_visitor::GameResult;
    use std::fs;
    use tempfile::tempdir;

    fn make_game(avg_elo: u16) -> TrainingGameData {
        TrainingGameData {
            avg_elo,
            moves: vec!["e4".to_string(), "e5".to_string(), "Nf3".to_string()],
            result: GameResult::White,
        }
    }

    #[test]
    fn test_creates_all_bucket_files() {
        let dir = tempdir().expect("create tempdir");
        let _writer = TrainingWriter::new(dir.path()).expect("create writer");

        assert!(dir.path().join("elo_1200.txt").exists());
        assert!(dir.path().join("elo_1500.txt").exists());
        assert!(dir.path().join("elo_1800.txt").exists());
        assert!(dir.path().join("elo_2000.txt").exists());
        assert!(dir.path().join("elo_2500.txt").exists());
        assert!(dir.path().join("elo_all.txt").exists());
    }

    #[test]
    fn test_low_elo_goes_to_all_buckets() {
        let dir = tempdir().expect("create tempdir");
        let mut writer = TrainingWriter::new(dir.path()).expect("create writer");

        let game = make_game(1100);
        writer.write_game(&game).expect("write game");
        writer.flush().expect("flush");

        let all = fs::read_to_string(dir.path().join("elo_all.txt")).expect("read all");
        let b1200 = fs::read_to_string(dir.path().join("elo_1200.txt")).expect("read 1200");
        let b1500 = fs::read_to_string(dir.path().join("elo_1500.txt")).expect("read 1500");

        assert!(all.contains("e4 e5 Nf3"));
        assert!(b1200.contains("e4 e5 Nf3"));
        assert!(b1500.contains("e4 e5 Nf3"));
    }

    #[test]
    fn test_mid_elo_skips_lower_buckets() {
        let dir = tempdir().expect("create tempdir");
        let mut writer = TrainingWriter::new(dir.path()).expect("create writer");

        let game = make_game(1600);
        writer.write_game(&game).expect("write game");
        writer.flush().expect("flush");

        let b1200 = fs::read_to_string(dir.path().join("elo_1200.txt")).expect("read 1200");
        let b1500 = fs::read_to_string(dir.path().join("elo_1500.txt")).expect("read 1500");
        let b1800 = fs::read_to_string(dir.path().join("elo_1800.txt")).expect("read 1800");
        let all = fs::read_to_string(dir.path().join("elo_all.txt")).expect("read all");

        assert!(b1200.is_empty());
        assert!(b1500.is_empty());
        assert!(b1800.contains("e4 e5 Nf3"));
        assert!(all.contains("e4 e5 Nf3"));
    }

    #[test]
    fn test_high_elo_only_in_all() {
        let dir = tempdir().expect("create tempdir");
        let mut writer = TrainingWriter::new(dir.path()).expect("create writer");

        let game = make_game(2600);
        writer.write_game(&game).expect("write game");
        writer.flush().expect("flush");

        let b2500 = fs::read_to_string(dir.path().join("elo_2500.txt")).expect("read 2500");
        let all = fs::read_to_string(dir.path().join("elo_all.txt")).expect("read all");

        assert!(b2500.is_empty());
        assert!(all.contains("e4 e5 Nf3"));
    }

    #[test]
    fn test_counts_tracked() {
        let dir = tempdir().expect("create tempdir");
        let mut writer = TrainingWriter::new(dir.path()).expect("create writer");

        writer.write_game(&make_game(1100)).expect("write");
        writer.write_game(&make_game(1400)).expect("write");
        writer.write_game(&make_game(2600)).expect("write");

        let counts = writer.counts();
        assert_eq!(*counts.get("elo_all.txt").unwrap_or(&0), 3);
        assert_eq!(*counts.get("elo_1200.txt").unwrap_or(&0), 1);
        assert_eq!(*counts.get("elo_1500.txt").unwrap_or(&0), 2);
    }
}
