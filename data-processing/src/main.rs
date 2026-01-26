use clap::Parser;
use pgn_reader::BufferedReader;
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
    /// Path to the PGN file to process
    pgn_file: PathBuf,

    /// Output directory for statistics JSON
    #[arg(short, long, default_value = "./processed_games/")]
    output: PathBuf,
}

fn main() -> Result<(), PgnError> {
    let args = Args::parse();

    let stats = process_pgn(&args.pgn_file)?;
    write_output(&stats, &args.output, &args.pgn_file)?;

    println!(
        "Processed {} games, skipped {} games",
        stats.games_processed, stats.games_skipped
    );

    Ok(())
}

fn process_pgn(path: &Path) -> Result<Statistics, PgnError> {
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

fn write_output(stats: &Statistics, output_dir: &Path, input_path: &Path) -> Result<(), PgnError> {
    fs::create_dir_all(output_dir).map_err(|e| PgnError::io(output_dir, e))?;

    let stem = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("stats");

    let output_path = output_dir.join(format!("{}_stats.json", stem));
    let output = stats.to_output();
    let json = serde_json::to_string_pretty(&output)
        .map_err(|e| PgnError::Parse(format!("JSON serialization failed: {}", e)))?;

    fs::write(&output_path, json).map_err(|e| PgnError::io(&output_path, e))?;

    println!("Wrote statistics to {}", output_path.display());
    Ok(())
}
