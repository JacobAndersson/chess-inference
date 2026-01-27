use chrono::Local;
use clap::Parser;
use pgn_reader::BufferedReader;
use rayon::prelude::*;
use std::fs::{self, File};
use std::io::BufReader;
use std::path::{Path, PathBuf};

use data_processing::error::PgnError;
use data_processing::parser::GameVisitor;
use data_processing::stats::Statistics;

#[derive(Parser)]
#[command(name = "pgn-stats")]
#[command(about = "Process PGN files and output game statistics bucketed by ELO and game length")]
struct Args {
    /// Path to PGN file or folder containing PGN files
    input: PathBuf,

    /// Output directory for statistics JSON (defaults to stats/ alongside input)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Number of worker threads (defaults to number of CPU cores)
    #[arg(short = 'j', long)]
    num_workers: Option<usize>,
}

fn main() -> Result<(), PgnError> {
    let args = Args::parse();

    if let Some(workers) = args.num_workers {
        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build_global()
            .map_err(|e| PgnError::Parse(format!("Failed to configure thread pool: {}", e)))?;
    }

    let is_folder = args.input.is_dir();
    let pgn_files = collect_pgn_files(&args.input)?;

    if pgn_files.is_empty() {
        return Err(PgnError::Parse(format!(
            "No PGN files found in {}",
            args.input.display()
        )));
    }

    println!("Processing {} PGN file(s)...", pgn_files.len());

    let results: Result<Vec<Statistics>, PgnError> =
        pgn_files.par_iter().map(|path| process_pgn(path)).collect();

    let stats_list = results?;

    let mut unified = Statistics::new();
    for stats in stats_list {
        unified.merge(stats);
    }

    let output_dir = args.output.unwrap_or_else(|| {
        if is_folder {
            args.input.join("stats")
        } else {
            args.input
                .parent()
                .map(|p| p.join("stats"))
                .unwrap_or_else(|| PathBuf::from("stats"))
        }
    });

    write_output(&unified, &output_dir, &args.input, is_folder)?;

    println!(
        "Processed {} games, skipped {} games",
        unified.games_processed, unified.games_skipped
    );

    Ok(())
}

fn collect_pgn_files(path: &Path) -> Result<Vec<PathBuf>, PgnError> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }

    if !path.is_dir() {
        return Err(PgnError::Parse(format!(
            "Path is neither a file nor a directory: {}",
            path.display()
        )));
    }

    let pattern = path.join("*.pgn");
    let pattern_str = pattern
        .to_str()
        .ok_or_else(|| PgnError::Parse("Invalid path encoding".to_string()))?;

    let files: Vec<PathBuf> = glob::glob(pattern_str)
        .map_err(|e| PgnError::Parse(format!("Invalid glob pattern: {}", e)))?
        .filter_map(Result::ok)
        .collect();

    Ok(files)
}

fn process_pgn(path: &Path) -> Result<Statistics, PgnError> {
    println!("Processing: {}", path.display());

    let file = File::open(path).map_err(|e| PgnError::io(path, e))?;
    let reader = BufReader::new(file);
    let mut pgn_reader = BufferedReader::new(reader);

    let mut stats = Statistics::new();
    let mut visitor = GameVisitor::new();

    loop {
        match pgn_reader.read_game(&mut visitor) {
            Ok(Some(Some(game_data))) => {
                stats.record_game(&game_data);
            }
            Ok(Some(None)) => {
                stats.record_skipped();
            }
            Ok(None) => break,
            Err(e) => return Err(PgnError::Parse(e.to_string())),
        }
    }

    Ok(stats)
}

fn write_output(
    stats: &Statistics,
    output_dir: &Path,
    input_path: &Path,
    is_folder: bool,
) -> Result<(), PgnError> {
    fs::create_dir_all(output_dir).map_err(|e| PgnError::io(output_dir, e))?;

    let filename = if is_folder {
        let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S");
        format!("{}_stats.json", timestamp)
    } else {
        let stem = input_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("stats");
        format!("{}_stats.json", stem)
    };

    let output_path = output_dir.join(filename);
    let output = stats.to_output();
    let json = serde_json::to_string_pretty(&output)
        .map_err(|e| PgnError::Parse(format!("JSON serialization failed: {}", e)))?;

    fs::write(&output_path, json).map_err(|e| PgnError::io(&output_path, e))?;

    println!("Wrote statistics to {}", output_path.display());
    Ok(())
}
