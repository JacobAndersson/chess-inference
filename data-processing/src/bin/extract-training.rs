use clap::Parser;
use glob::glob;
use pgn_reader::BufferedReader;
use rayon::prelude::*;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use data_processing::error::PgnError;
use data_processing::filters::is_valid_training_game;
use data_processing::training_visitor::TrainingVisitor;
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

struct ProcessingStats {
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

    let mut pgn_files = collect_pgn_files(&args.input)?;
    if pgn_files.is_empty() {
        return Err(PgnError::Argument(format!(
            "No .pgn files found in {}",
            args.input.display()
        )));
    }
    pgn_files.sort();

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

    let temp_dir = output_dir.join(".temp");
    fs::create_dir_all(&temp_dir).map_err(|e| PgnError::io(&temp_dir, e))?;

    let indexed_files: Vec<(usize, &PathBuf)> = pgn_files.iter().enumerate().collect();

    let results: Vec<Result<(usize, ProcessingStats), PgnError>> = indexed_files
        .par_iter()
        .map(|(idx, path)| {
            println!("Processing: {}", path.display());
            let worker_dir = temp_dir.join(idx.to_string());
            let stats = process_pgn_streaming(path, &worker_dir, args.tokenize)?;
            Ok((*idx, stats))
        })
        .collect();

    let mut total_processed = 0u64;
    let mut total_skipped = 0u64;
    let mut total_filtered = 0u64;

    for result in &results {
        let (_, stats) = result
            .as_ref()
            .map_err(|e| PgnError::Parse(e.to_string()))?;
        total_processed += stats.processed;
        total_skipped += stats.skipped;
        total_filtered += stats.filtered;
    }

    println!("\nMerging output files...");
    let output_filenames = TrainingWriter::output_filenames(args.tokenize);
    merge_temp_files(&temp_dir, &output_dir, &output_filenames, pgn_files.len())?;

    fs::remove_dir_all(&temp_dir).map_err(|e| PgnError::io(&temp_dir, e))?;

    println!("\n=== Summary ===");
    println!("Games written:  {total_processed}");
    println!("Games filtered: {total_filtered} (short/incomplete)");
    println!("Games skipped:  {total_skipped} (missing ELO)");

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

fn process_pgn_streaming(
    path: &Path,
    output_dir: &Path,
    tokenize: bool,
) -> Result<ProcessingStats, PgnError> {
    let file = File::open(path).map_err(|e| PgnError::io(path, e))?;
    let reader = BufReader::new(file);
    let mut pgn_reader = BufferedReader::new(reader);

    let mut writer = TrainingWriter::new(output_dir, tokenize)?;
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
            },
            Ok(Some(None)) => {
                skipped += 1;
            },
            Ok(None) => break,
            Err(e) => return Err(PgnError::Parse(e.to_string())),
        }
    }

    writer.flush()?;

    Ok(ProcessingStats {
        processed,
        skipped,
        filtered,
    })
}

fn merge_temp_files(
    temp_dir: &Path,
    output_dir: &Path,
    filenames: &[String],
    num_files: usize,
) -> Result<(), PgnError> {
    fs::create_dir_all(output_dir).map_err(|e| PgnError::io(output_dir, e))?;

    for filename in filenames {
        let output_path = output_dir.join(filename);
        let output_file = File::create(&output_path).map_err(|e| PgnError::io(&output_path, e))?;
        let mut writer = BufWriter::new(output_file);

        for idx in 0..num_files {
            let temp_file_path = temp_dir.join(idx.to_string()).join(filename);
            if temp_file_path.exists() {
                let file =
                    File::open(&temp_file_path).map_err(|e| PgnError::io(&temp_file_path, e))?;
                let reader = BufReader::new(file);
                for line in reader.lines() {
                    let line = line.map_err(|e| PgnError::io(&temp_file_path, e))?;
                    writeln!(writer, "{line}").map_err(|e| PgnError::io(&output_path, e))?;
                }
            }
        }

        writer.flush().map_err(|e| PgnError::io(&output_path, e))?;
    }

    Ok(())
}
