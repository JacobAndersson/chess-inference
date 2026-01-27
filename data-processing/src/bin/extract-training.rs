use clap::Parser;
use glob::glob;
use pgn_reader::BufferedReader;
use rayon::prelude::*;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use data_processing::error::PgnError;
use data_processing::filters::is_valid_training_game;
use data_processing::training_visitor::{TrainingGameData, TrainingVisitor};
use data_processing::training_writer::TrainingWriter;

#[derive(Parser)]
#[command(name = "extract-training")]
#[command(about = "Extract training data from PGN files, bucketed by ELO")]
struct Args {
    /// Path to PGN file or directory
    input: PathBuf,

    /// Output directory (defaults to training-data/ alongside input)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Number of worker threads (defaults to CPU cores)
    #[arg(short = 'j', long)]
    num_workers: Option<usize>,

    /// Output tokenized format (comma-separated token IDs) instead of SAN moves
    #[arg(long, default_value = "false")]
    tokenize: bool,
}

struct ProcessedFile {
    games: Vec<TrainingGameData>,
    processed: u64,
    skipped: u64,
    filtered: u64,
}

fn main() -> Result<(), PgnError> {
    let args = Args::parse();

    if let Some(num_workers) = args.num_workers {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_workers)
            .build_global()
            .map_err(|e| PgnError::Argument(format!("Failed to set thread count: {e}")))?;
    }

    let pgn_files = collect_pgn_files(&args.input)?;
    if pgn_files.is_empty() {
        return Err(PgnError::Argument(format!(
            "No .pgn files found in {}",
            args.input.display()
        )));
    }

    let output_dir = args.output.unwrap_or_else(|| {
        if args.input.is_file() {
            args.input.parent().map_or_else(
                || PathBuf::from("training-data"),
                |p| p.join("training-data"),
            )
        } else {
            args.input.join("training-data")
        }
    });

    println!("Found {} PGN files", pgn_files.len());
    println!(
        "Using {} worker threads",
        args.num_workers.unwrap_or_else(rayon::current_num_threads)
    );
    if args.tokenize {
        println!("Tokenization mode enabled");
    }
    println!("Output directory: {}", output_dir.display());

    let results: Vec<Result<ProcessedFile, PgnError>> = pgn_files
        .par_iter()
        .map(|path| {
            println!("Processing: {}", path.display());
            process_pgn(path)
        })
        .collect();

    let mut all_games = Vec::new();
    let mut total_processed = 0u64;
    let mut total_skipped = 0u64;
    let mut total_filtered = 0u64;

    for result in results {
        let processed_file = result?;
        total_processed += processed_file.processed;
        total_skipped += processed_file.skipped;
        total_filtered += processed_file.filtered;
        all_games.extend(processed_file.games);
    }

    let mut writer = TrainingWriter::new(&output_dir, args.tokenize)?;
    for game in &all_games {
        writer.write_game(game)?;
    }
    writer.flush()?;

    println!("\n=== Summary ===");
    println!("Games written:  {total_processed}");
    println!("Games filtered: {total_filtered} (short/incomplete)");
    println!("Games skipped:  {total_skipped} (missing ELO)");

    println!("\nOutput files:");
    for (bucket, count) in writer.counts() {
        println!("  {bucket}: {count} games");
    }

    Ok(())
}

fn collect_pgn_files(input: &Path) -> Result<Vec<PathBuf>, PgnError> {
    if input.is_file() {
        return Ok(vec![input.to_path_buf()]);
    }

    let pattern = input.join("*.pgn");
    let pattern_str = pattern
        .to_str()
        .ok_or_else(|| PgnError::Argument("Invalid path encoding".to_string()))?;

    let files: Vec<PathBuf> = glob(pattern_str)
        .map_err(|e| PgnError::Argument(format!("Invalid glob pattern: {e}")))?
        .filter_map(Result::ok)
        .collect();

    Ok(files)
}

fn process_pgn(path: &Path) -> Result<ProcessedFile, PgnError> {
    let file = File::open(path).map_err(|e| PgnError::io(path, e))?;
    let reader = BufReader::new(file);
    let mut pgn_reader = BufferedReader::new(reader);

    let mut visitor = TrainingVisitor::new();
    let mut games = Vec::new();
    let mut processed = 0u64;
    let mut skipped = 0u64;
    let mut filtered = 0u64;

    loop {
        match pgn_reader.read_game(&mut visitor) {
            Ok(Some(Some(game))) => {
                if is_valid_training_game(&game) {
                    games.push(game);
                    processed += 1;
                } else {
                    filtered += 1;
                }
            },
            Ok(Some(None)) => {
                skipped += 1;
            },
            Ok(None) => break,
            Err(e) => return Err(PgnError::Parse(e.to_string())),
        }
    }

    Ok(ProcessedFile {
        games,
        processed,
        skipped,
        filtered,
    })
}
