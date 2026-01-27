use clap::Parser;
use glob::glob;
use pgn_reader::BufferedReader;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use data_processing::error::PgnError;
use data_processing::filters::is_valid_training_game;
use data_processing::training_visitor::TrainingVisitor;
use data_processing::training_writer::TrainingWriter;

#[derive(Parser)]
#[command(name = "extract_training")]
#[command(about = "Extract training data from PGN files, bucketed by ELO")]
struct Args {
    /// Directory containing PGN files
    input_dir: PathBuf,

    /// Output directory for training data
    #[arg(short, long, default_value = "./processed_games/training/")]
    output: PathBuf,

    /// Output tokenized format (comma-separated token IDs) instead of SAN moves
    #[arg(long, default_value = "false")]
    tokenize: bool,
}

fn main() -> Result<(), PgnError> {
    let args = Args::parse();

    let pgn_files = find_pgn_files(&args.input_dir)?;
    if pgn_files.is_empty() {
        return Err(PgnError::Argument(format!(
            "No .pgn files found in {}",
            args.input_dir.display()
        )));
    }

    println!("Found {} PGN files", pgn_files.len());
    if args.tokenize {
        println!("Tokenization mode enabled");
    }

    let mut writer = TrainingWriter::new(&args.output, args.tokenize)?;
    let mut total_processed = 0u64;
    let mut total_skipped = 0u64;
    let mut total_filtered = 0u64;

    for pgn_path in &pgn_files {
        println!("Processing: {}", pgn_path.display());
        let (processed, skipped, filtered) = process_pgn(pgn_path, &mut writer)?;
        total_processed += processed;
        total_skipped += skipped;
        total_filtered += filtered;
    }

    writer.flush()?;

    println!("\n=== Summary ===");
    println!("Games written:  {}", total_processed);
    println!("Games filtered: {} (short/incomplete)", total_filtered);
    println!("Games skipped:  {} (missing ELO)", total_skipped);

    println!("\nOutput files:");
    for (bucket, count) in writer.counts() {
        println!("  {}: {} games", bucket, count);
    }

    Ok(())
}

fn find_pgn_files(dir: &Path) -> Result<Vec<PathBuf>, PgnError> {
    let pattern = dir.join("*.pgn");
    let pattern_str = pattern
        .to_str()
        .ok_or_else(|| PgnError::Argument("Invalid path encoding".to_string()))?;

    let files: Vec<PathBuf> = glob(pattern_str)
        .map_err(|e| PgnError::Argument(format!("Invalid glob pattern: {}", e)))?
        .filter_map(Result::ok)
        .collect();

    Ok(files)
}

fn process_pgn(path: &Path, writer: &mut TrainingWriter) -> Result<(u64, u64, u64), PgnError> {
    let file = File::open(path).map_err(|e| PgnError::io(path, e))?;
    let reader = BufReader::new(file);
    let mut pgn_reader = BufferedReader::new(reader);

    let mut visitor = TrainingVisitor::new();
    let mut processed = 0u64;
    let mut skipped = 0u64;
    let mut filtered = 0u64;

    loop {
        match pgn_reader.read_game(&mut visitor) {
            Ok(Some(Some(game))) => {
                if is_valid_training_game(&game) {
                    writer.write_game(&game)?;
                    processed += 1;
                } else {
                    filtered += 1;
                }
            }
            Ok(Some(None)) => {
                skipped += 1;
            }
            Ok(None) => break,
            Err(e) => return Err(PgnError::Parse(e.to_string())),
        }
    }

    Ok((processed, skipped, filtered))
}
