# Chess Data Processing

Tools for processing PGN files and extracting statistics from chess games.

## Binaries

### pgn-stats (`data-processing`)

Processes PGN files and outputs game statistics as JSON, bucketed by max ELO and time control.

```bash
cargo run --release --bin data-processing -- <pgn-file> [-o <output-dir>]
```

Output defaults to `stats/` alongside the input file.

Example:
```bash
cargo run --release --bin data-processing -- ../games/lichess_db_standard_rated_2014-10.pgn
# Writes to ../games/stats/lichess_db_standard_rated_2014-10_stats.json
```

Output JSON structure:
```json
{
  "total_games": 1111238,
  "skipped_games": 64,
  "elo_distribution": [
    { "elo_min": 1500, "elo_max": 1600, "count": 168655 }
  ],
  "time_control_distribution": [
    { "category": "bullet", "count": 311234 },
    { "category": "blitz", "count": 470470 }
  ]
}
```

### extract_training

Extracts training data from PGN files, filtering by ELO thresholds.

```bash
cargo run --release --bin extract_training -- <pgn-file> -o <output-dir>
```

## Visualization Scripts

Generate charts from statistics JSON files.

### Setup

```bash
cd scripts
uv sync  # or just run the script, uv will install deps automatically
```

### Usage

```bash
cd scripts
uv run visualize_stats.py <stats-json> [-o <output-dir>]
```

Output defaults to `figs/` as a sibling to the input file's parent directory (e.g., `stats/` -> `figs/`).

Example:
```bash
uv run visualize_stats.py ../games/stats/lichess_db_standard_rated_2014-10_stats.json
# Writes to ../games/figs/
```

Generates:
- `*_elo_distribution.png` - bar chart of games by max ELO bucket
- `*_time_control.png` - bar chart with percentages for each time control category

## Time Control Classification

Games are classified using the Lichess formula:

```
estimated_time = initial_seconds + (40 * increment_seconds)
```

| Category   | Estimated Time |
|------------|----------------|
| Bullet     | < 3 min        |
| Blitz      | 3-8 min        |
| Rapid      | 8-25 min       |
| Classical  | >= 25 min      |

## Development

```bash
cargo fmt          # format code
cargo clippy       # lint
cargo test         # run tests
```
