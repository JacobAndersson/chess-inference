use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::error::PgnError;
use crate::tokenizer::ChessTokenizer;
use crate::training_visitor::TrainingGameData;

const ELO_BUCKETS: &[u16] = &[1200, 1500, 1800, 2000, 2500];
const MAX_GAME_LENGTH: usize = 8192;
const TEST_SPLIT_MODULO: u64 = 20; // Every 20th game goes to test (5%)

pub struct TrainingWriter {
    writers: HashMap<String, BufWriter<File>>,
    counts: HashMap<String, u64>,
    tokenize: bool,
    token_buffer: Vec<u8>,
    game_counter: u64,
}

impl TrainingWriter {
    pub fn new(output_dir: &Path, tokenize: bool) -> Result<Self, PgnError> {
        fs::create_dir_all(output_dir).map_err(|e| PgnError::io(output_dir, e))?;

        let mut writers = HashMap::new();
        let mut counts = HashMap::new();

        for elo in ELO_BUCKETS {
            for split in ["train", "test"] {
                let filename = Self::filename(*elo, split, tokenize);
                let path = output_dir.join(&filename);
                let file = File::create(&path).map_err(|e| PgnError::io(&path, e))?;
                writers.insert(filename.clone(), BufWriter::new(file));
                counts.insert(filename, 0);
            }
        }

        for split in ["train", "test"] {
            let filename = Self::all_filename(split, tokenize);
            let path = output_dir.join(&filename);
            let file = File::create(&path).map_err(|e| PgnError::io(&path, e))?;
            writers.insert(filename.clone(), BufWriter::new(file));
            counts.insert(filename, 0);
        }

        Ok(Self {
            writers,
            counts,
            tokenize,
            token_buffer: vec![0u8; MAX_GAME_LENGTH],
            game_counter: 0,
        })
    }

    fn filename(elo: u16, split: &str, tokenize: bool) -> String {
        if tokenize {
            format!("tokens_elo_{elo}_{split}.txt")
        } else {
            format!("elo_{elo}_{split}.txt")
        }
    }

    fn all_filename(split: &str, tokenize: bool) -> String {
        if tokenize {
            format!("tokens_elo_all_{split}.txt")
        } else {
            format!("elo_all_{split}.txt")
        }
    }

    pub fn output_filenames(tokenize: bool) -> Vec<String> {
        let mut filenames = Vec::new();
        for elo in ELO_BUCKETS {
            for split in ["train", "test"] {
                filenames.push(Self::filename(*elo, split, tokenize));
            }
        }
        for split in ["train", "test"] {
            filenames.push(Self::all_filename(split, tokenize));
        }
        filenames
    }

    /// Write a game to the appropriate bucket files.
    /// Returns Ok(true) if written, Ok(false) if skipped (e.g. too long to tokenize).
    pub fn write_game(&mut self, game: &TrainingGameData) -> Result<bool, PgnError> {
        let line = game.moves.join(" ");

        let formatted = if self.tokenize {
            match self.tokenize_game(&line) {
                Some(tokens) => tokens,
                None => return Ok(false),
            }
        } else {
            line
        };

        let split = if self.game_counter.is_multiple_of(TEST_SPLIT_MODULO) {
            "test"
        } else {
            "train"
        };
        self.game_counter += 1;

        for elo in ELO_BUCKETS {
            if game.avg_elo < *elo {
                let filename = Self::filename(*elo, split, self.tokenize);
                self.write_to_bucket(&filename, &formatted)?;
            }
        }

        let all_filename = Self::all_filename(split, self.tokenize);
        self.write_to_bucket(&all_filename, &formatted)?;

        Ok(true)
    }

    fn tokenize_game(&mut self, moves: &str) -> Option<String> {
        let len = ChessTokenizer::encode_with_eos(moves, &mut self.token_buffer)?;

        let token_strings: Vec<String> = self.token_buffer[..len]
            .iter()
            .map(|t| t.to_string())
            .collect();

        Some(token_strings.join(","))
    }

    fn write_to_bucket(&mut self, bucket: &str, line: &str) -> Result<(), PgnError> {
        if let Some(writer) = self.writers.get_mut(bucket) {
            writeln!(writer, "{line}").map_err(|e| PgnError::Parse(e.to_string()))?;
            *self.counts.entry(bucket.to_string()).or_insert(0) += 1;
        }
        Ok(())
    }

    pub fn flush(&mut self) -> Result<(), PgnError> {
        for writer in self.writers.values_mut() {
            writer.flush().map_err(|e| PgnError::Parse(e.to_string()))?;
        }
        Ok(())
    }

    pub fn counts(&self) -> &HashMap<String, u64> {
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

    fn read_both_splits(dir: &Path, base: &str) -> String {
        let train = fs::read_to_string(dir.join(format!("{base}_train.txt"))).unwrap_or_default();
        let test = fs::read_to_string(dir.join(format!("{base}_test.txt"))).unwrap_or_default();
        format!("{train}{test}")
    }

    #[test]
    fn test_creates_all_bucket_files() {
        let dir = tempdir().expect("create tempdir");
        let _writer = TrainingWriter::new(dir.path(), false).expect("create writer");

        for elo in [1200, 1500, 1800, 2000, 2500] {
            assert!(dir.path().join(format!("elo_{elo}_train.txt")).exists());
            assert!(dir.path().join(format!("elo_{elo}_test.txt")).exists());
        }
        assert!(dir.path().join("elo_all_train.txt").exists());
        assert!(dir.path().join("elo_all_test.txt").exists());
    }

    #[test]
    fn test_creates_token_bucket_files() {
        let dir = tempdir().expect("create tempdir");
        let _writer = TrainingWriter::new(dir.path(), true).expect("create writer");

        for elo in [1200, 1500, 1800, 2000, 2500] {
            assert!(dir
                .path()
                .join(format!("tokens_elo_{elo}_train.txt"))
                .exists());
            assert!(dir
                .path()
                .join(format!("tokens_elo_{elo}_test.txt"))
                .exists());
        }
        assert!(dir.path().join("tokens_elo_all_train.txt").exists());
        assert!(dir.path().join("tokens_elo_all_test.txt").exists());
    }

    #[test]
    fn test_low_elo_goes_to_all_buckets() {
        let dir = tempdir().expect("create tempdir");
        let mut writer = TrainingWriter::new(dir.path(), false).expect("create writer");

        let game = make_game(1100);
        writer.write_game(&game).expect("write game");
        writer.flush().expect("flush");

        let all = read_both_splits(dir.path(), "elo_all");
        let b1200 = read_both_splits(dir.path(), "elo_1200");
        let b1500 = read_both_splits(dir.path(), "elo_1500");

        assert!(all.contains("e4 e5 Nf3"));
        assert!(b1200.contains("e4 e5 Nf3"));
        assert!(b1500.contains("e4 e5 Nf3"));
    }

    #[test]
    fn test_mid_elo_skips_lower_buckets() {
        let dir = tempdir().expect("create tempdir");
        let mut writer = TrainingWriter::new(dir.path(), false).expect("create writer");

        let game = make_game(1600);
        writer.write_game(&game).expect("write game");
        writer.flush().expect("flush");

        let b1200 = read_both_splits(dir.path(), "elo_1200");
        let b1500 = read_both_splits(dir.path(), "elo_1500");
        let b1800 = read_both_splits(dir.path(), "elo_1800");
        let all = read_both_splits(dir.path(), "elo_all");

        assert!(b1200.is_empty());
        assert!(b1500.is_empty());
        assert!(b1800.contains("e4 e5 Nf3"));
        assert!(all.contains("e4 e5 Nf3"));
    }

    #[test]
    fn test_high_elo_only_in_all() {
        let dir = tempdir().expect("create tempdir");
        let mut writer = TrainingWriter::new(dir.path(), false).expect("create writer");

        let game = make_game(2600);
        writer.write_game(&game).expect("write game");
        writer.flush().expect("flush");

        let b2500 = read_both_splits(dir.path(), "elo_2500");
        let all = read_both_splits(dir.path(), "elo_all");

        assert!(b2500.is_empty());
        assert!(all.contains("e4 e5 Nf3"));
    }

    #[test]
    fn test_train_test_split_ratio() {
        let dir = tempdir().expect("create tempdir");
        let mut writer = TrainingWriter::new(dir.path(), false).expect("create writer");

        // Write 100 games
        for _ in 0..100 {
            writer.write_game(&make_game(1100)).expect("write");
        }
        writer.flush().expect("flush");

        let counts = writer.counts();
        let train = *counts.get("elo_all_train.txt").unwrap_or(&0);
        let test = *counts.get("elo_all_test.txt").unwrap_or(&0);

        assert_eq!(train, 95); // 95% train
        assert_eq!(test, 5); // 5% test
    }

    #[test]
    fn test_first_game_goes_to_test() {
        let dir = tempdir().expect("create tempdir");
        let mut writer = TrainingWriter::new(dir.path(), false).expect("create writer");

        writer.write_game(&make_game(1100)).expect("write");
        writer.flush().expect("flush");

        let test = fs::read_to_string(dir.path().join("elo_all_test.txt")).expect("read");
        let train = fs::read_to_string(dir.path().join("elo_all_train.txt")).expect("read");

        assert!(test.contains("e4 e5 Nf3"));
        assert!(train.is_empty());
    }

    #[test]
    fn test_tokenized_output_format() {
        let dir = tempdir().expect("create tempdir");
        let mut writer = TrainingWriter::new(dir.path(), true).expect("create writer");

        let game = make_game(1100);
        writer.write_game(&game).expect("write game");
        writer.flush().expect("flush");

        let content = read_both_splits(dir.path(), "tokens_elo_all");
        let line = content.trim();

        let tokens: Vec<u8> = line
            .split(',')
            .map(|s| s.parse::<u8>().expect("parse token"))
            .collect();

        assert_eq!(*tokens.last().expect("last token"), 30);

        let decoded = ChessTokenizer::decode(&tokens);
        assert_eq!(decoded, "e4 e5 Nf3");
    }

    #[test]
    fn test_tokenized_buckets_work() {
        let dir = tempdir().expect("create tempdir");
        let mut writer = TrainingWriter::new(dir.path(), true).expect("create writer");

        writer.write_game(&make_game(1100)).expect("write");
        writer.flush().expect("flush");

        let b1200 = read_both_splits(dir.path(), "tokens_elo_1200");
        let all = read_both_splits(dir.path(), "tokens_elo_all");

        assert!(!b1200.is_empty());
        assert!(!all.is_empty());
        assert_eq!(b1200.trim(), all.trim());
    }

    #[test]
    fn test_tokenize_with_castling() {
        let dir = tempdir().expect("create tempdir");
        let mut writer = TrainingWriter::new(dir.path(), true).expect("create writer");

        let game = TrainingGameData {
            avg_elo: 1500,
            moves: vec!["e4".to_string(), "e5".to_string(), "O-O".to_string()],
            result: GameResult::White,
        };
        writer.write_game(&game).expect("write game");
        writer.flush().expect("flush");

        let content = read_both_splits(dir.path(), "tokens_elo_all");
        let tokens: Vec<u8> = content
            .trim()
            .split(',')
            .map(|s| s.parse::<u8>().expect("parse"))
            .collect();

        let decoded = ChessTokenizer::decode(&tokens);
        assert_eq!(decoded, "e4 e5 O-O");
    }

    #[test]
    fn test_very_long_game_skipped_in_tokenize_mode() {
        let dir = tempdir().expect("create tempdir");
        let mut writer = TrainingWriter::new(dir.path(), true).expect("create writer");

        // Create a game with enough moves to exceed MAX_GAME_LENGTH
        let moves: Vec<String> = (0..3000).map(|_| "Nf3".to_string()).collect();
        let game = TrainingGameData {
            avg_elo: 1500,
            moves,
            result: GameResult::Draw,
        };

        let written = writer.write_game(&game).expect("should not error");
        assert!(!written, "overly long game should be skipped, not written");
    }

    #[test]
    fn test_write_game_returns_true_for_normal_game() {
        let dir = tempdir().expect("create tempdir");
        let mut writer = TrainingWriter::new(dir.path(), true).expect("create writer");

        let game = make_game(1100);
        let written = writer.write_game(&game).expect("write game");
        assert!(written);
    }
}
