use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::error::PgnError;
use crate::tokenizer::ChessTokenizer;
use crate::training_visitor::TrainingGameData;

const ELO_BUCKETS: &[(u16, &str, &str)] = &[
    (1200, "elo_1200.txt", "tokens_elo_1200.txt"),
    (1500, "elo_1500.txt", "tokens_elo_1500.txt"),
    (1800, "elo_1800.txt", "tokens_elo_1800.txt"),
    (2000, "elo_2000.txt", "tokens_elo_2000.txt"),
    (2500, "elo_2500.txt", "tokens_elo_2500.txt"),
];

const ALL_BUCKET: &str = "elo_all.txt";
const ALL_BUCKET_TOKENS: &str = "tokens_elo_all.txt";
const MAX_GAME_LENGTH: usize = 2048;

pub struct TrainingWriter {
    writers: HashMap<&'static str, BufWriter<File>>,
    counts: HashMap<&'static str, u64>,
    tokenize: bool,
    token_buffer: Vec<u8>,
}

impl TrainingWriter {
    pub fn new(output_dir: &Path, tokenize: bool) -> Result<Self, PgnError> {
        fs::create_dir_all(output_dir).map_err(|e| PgnError::io(output_dir, e))?;

        let mut writers = HashMap::new();
        let mut counts = HashMap::new();

        for (_, text_filename, token_filename) in ELO_BUCKETS {
            let filename = if tokenize {
                *token_filename
            } else {
                *text_filename
            };
            let path = output_dir.join(filename);
            let file = File::create(&path).map_err(|e| PgnError::io(&path, e))?;
            writers.insert(filename, BufWriter::new(file));
            counts.insert(filename, 0);
        }

        let all_filename = if tokenize {
            ALL_BUCKET_TOKENS
        } else {
            ALL_BUCKET
        };
        let all_path = output_dir.join(all_filename);
        let all_file = File::create(&all_path).map_err(|e| PgnError::io(&all_path, e))?;
        writers.insert(all_filename, BufWriter::new(all_file));
        counts.insert(all_filename, 0);

        Ok(Self {
            writers,
            counts,
            tokenize,
            token_buffer: vec![0u8; MAX_GAME_LENGTH],
        })
    }

    pub fn write_game(&mut self, game: &TrainingGameData) -> Result<(), PgnError> {
        let line = game.moves.join(" ");

        let formatted = if self.tokenize {
            self.tokenize_game(&line)?
        } else {
            line
        };

        for (threshold, text_filename, token_filename) in ELO_BUCKETS {
            if game.avg_elo < *threshold {
                let filename = if self.tokenize {
                    *token_filename
                } else {
                    *text_filename
                };
                self.write_to_bucket(filename, &formatted)?;
            }
        }

        let all_filename = if self.tokenize {
            ALL_BUCKET_TOKENS
        } else {
            ALL_BUCKET
        };
        self.write_to_bucket(all_filename, &formatted)?;

        Ok(())
    }

    fn tokenize_game(&mut self, moves: &str) -> Result<String, PgnError> {
        let len = ChessTokenizer::encode_with_eos(moves, &mut self.token_buffer)
            .ok_or_else(|| PgnError::Parse(format!("Failed to tokenize game: {moves}")))?;

        let token_strings: Vec<String> = self.token_buffer[..len]
            .iter()
            .map(|t| t.to_string())
            .collect();

        Ok(token_strings.join(","))
    }

    fn write_to_bucket(&mut self, bucket: &'static str, line: &str) -> Result<(), PgnError> {
        if let Some(writer) = self.writers.get_mut(bucket) {
            writeln!(writer, "{line}").map_err(|e| PgnError::Parse(e.to_string()))?;
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
        let _writer = TrainingWriter::new(dir.path(), false).expect("create writer");

        assert!(dir.path().join("elo_1200.txt").exists());
        assert!(dir.path().join("elo_1500.txt").exists());
        assert!(dir.path().join("elo_1800.txt").exists());
        assert!(dir.path().join("elo_2000.txt").exists());
        assert!(dir.path().join("elo_2500.txt").exists());
        assert!(dir.path().join("elo_all.txt").exists());
    }

    #[test]
    fn test_creates_token_bucket_files() {
        let dir = tempdir().expect("create tempdir");
        let _writer = TrainingWriter::new(dir.path(), true).expect("create writer");

        assert!(dir.path().join("tokens_elo_1200.txt").exists());
        assert!(dir.path().join("tokens_elo_1500.txt").exists());
        assert!(dir.path().join("tokens_elo_1800.txt").exists());
        assert!(dir.path().join("tokens_elo_2000.txt").exists());
        assert!(dir.path().join("tokens_elo_2500.txt").exists());
        assert!(dir.path().join("tokens_elo_all.txt").exists());
    }

    #[test]
    fn test_low_elo_goes_to_all_buckets() {
        let dir = tempdir().expect("create tempdir");
        let mut writer = TrainingWriter::new(dir.path(), false).expect("create writer");

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
        let mut writer = TrainingWriter::new(dir.path(), false).expect("create writer");

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
        let mut writer = TrainingWriter::new(dir.path(), false).expect("create writer");

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
        let mut writer = TrainingWriter::new(dir.path(), false).expect("create writer");

        writer.write_game(&make_game(1100)).expect("write");
        writer.write_game(&make_game(1400)).expect("write");
        writer.write_game(&make_game(2600)).expect("write");

        let counts = writer.counts();
        assert_eq!(*counts.get("elo_all.txt").unwrap_or(&0), 3);
        assert_eq!(*counts.get("elo_1200.txt").unwrap_or(&0), 1);
        assert_eq!(*counts.get("elo_1500.txt").unwrap_or(&0), 2);
    }

    #[test]
    fn test_tokenized_output_format() {
        let dir = tempdir().expect("create tempdir");
        let mut writer = TrainingWriter::new(dir.path(), true).expect("create writer");

        let game = make_game(1100);
        writer.write_game(&game).expect("write game");
        writer.flush().expect("flush");

        let content = fs::read_to_string(dir.path().join("tokens_elo_all.txt")).expect("read");
        let line = content.trim();

        // Verify format is comma-separated integers
        let tokens: Vec<u8> = line
            .split(',')
            .map(|s| s.parse::<u8>().expect("parse token"))
            .collect();

        // Should end with EOS token (30)
        assert_eq!(*tokens.last().expect("last token"), 30);

        // Decode back to original
        let decoded = ChessTokenizer::decode(&tokens);
        assert_eq!(decoded, "e4 e5 Nf3");
    }

    #[test]
    fn test_tokenized_buckets_work() {
        let dir = tempdir().expect("create tempdir");
        let mut writer = TrainingWriter::new(dir.path(), true).expect("create writer");

        writer.write_game(&make_game(1100)).expect("write");
        writer.flush().expect("flush");

        let b1200 = fs::read_to_string(dir.path().join("tokens_elo_1200.txt")).expect("read");
        let all = fs::read_to_string(dir.path().join("tokens_elo_all.txt")).expect("read");

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

        let content = fs::read_to_string(dir.path().join("tokens_elo_all.txt")).expect("read");
        let tokens: Vec<u8> = content
            .trim()
            .split(',')
            .map(|s| s.parse::<u8>().expect("parse"))
            .collect();

        let decoded = ChessTokenizer::decode(&tokens);
        assert_eq!(decoded, "e4 e5 O-O");
    }
}
