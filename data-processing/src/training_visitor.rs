use pgn_reader::{RawHeader, SanPlus, Skip, Visitor};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GameResult {
    White,
    Black,
    Draw,
    #[default]
    Incomplete,
}

pub struct TrainingGameData {
    pub avg_elo: u16,
    pub moves: Vec<String>,
    pub result: GameResult,
}

#[derive(Default)]
pub struct TrainingVisitor {
    white_elo: Option<u16>,
    black_elo: Option<u16>,
    moves: Vec<String>,
    result: GameResult,
}

impl TrainingVisitor {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Visitor for TrainingVisitor {
    type Result = Option<TrainingGameData>;

    fn begin_game(&mut self) {
        self.white_elo = None;
        self.black_elo = None;
        self.moves.clear();
        self.result = GameResult::Incomplete;
    }

    fn header(&mut self, key: &[u8], value: RawHeader<'_>) {
        let value_bytes = value.as_bytes();

        match key {
            b"WhiteElo" => self.white_elo = parse_elo(value_bytes),
            b"BlackElo" => self.black_elo = parse_elo(value_bytes),
            b"Result" => self.result = parse_result(value_bytes),
            _ => {}
        }
    }

    fn san(&mut self, san: SanPlus) {
        self.moves.push(san.to_string());
    }

    fn begin_variation(&mut self) -> Skip {
        Skip(true)
    }

    fn end_game(&mut self) -> Self::Result {
        match (self.white_elo, self.black_elo) {
            (Some(w), Some(b)) => Some(TrainingGameData {
                avg_elo: (w + b) / 2,
                moves: std::mem::take(&mut self.moves),
                result: self.result,
            }),
            _ => None,
        }
    }
}

fn parse_elo(value: &[u8]) -> Option<u16> {
    if value == b"?" {
        return None;
    }
    std::str::from_utf8(value).ok().and_then(|s| s.parse().ok())
}

fn parse_result(value: &[u8]) -> GameResult {
    match value {
        b"1-0" => GameResult::White,
        b"0-1" => GameResult::Black,
        b"1/2-1/2" => GameResult::Draw,
        _ => GameResult::Incomplete,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pgn_reader::BufferedReader;

    #[test]
    fn test_parse_elo_valid() {
        assert_eq!(parse_elo(b"1500"), Some(1500));
        assert_eq!(parse_elo(b"2000"), Some(2000));
    }

    #[test]
    fn test_parse_elo_invalid() {
        assert_eq!(parse_elo(b"?"), None);
        assert_eq!(parse_elo(b"abc"), None);
    }

    #[test]
    fn test_parse_result() {
        assert_eq!(parse_result(b"1-0"), GameResult::White);
        assert_eq!(parse_result(b"0-1"), GameResult::Black);
        assert_eq!(parse_result(b"1/2-1/2"), GameResult::Draw);
        assert_eq!(parse_result(b"*"), GameResult::Incomplete);
    }

    #[test]
    fn test_visitor_extracts_moves() {
        let pgn = br#"[WhiteElo "1500"]
[BlackElo "1600"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0
"#;
        let mut reader = BufferedReader::new_cursor(pgn);
        let mut visitor = TrainingVisitor::new();
        let result = reader.read_game(&mut visitor).expect("read game").flatten();

        let game = result.expect("game data");
        assert_eq!(game.avg_elo, 1550);
        assert_eq!(game.moves, vec!["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"]);
        assert_eq!(game.result, GameResult::White);
    }

    #[test]
    fn test_visitor_missing_elo_returns_none() {
        let pgn = br#"[WhiteElo "?"]
[BlackElo "1600"]
[Result "1-0"]

1. e4 e5 1-0
"#;
        let mut reader = BufferedReader::new_cursor(pgn);
        let mut visitor = TrainingVisitor::new();
        let result = reader.read_game(&mut visitor).expect("read game").flatten();

        assert!(result.is_none());
    }

    #[test]
    fn test_visitor_draw_result() {
        let pgn = br#"[WhiteElo "2000"]
[BlackElo "2100"]
[Result "1/2-1/2"]

1. d4 d5 2. c4 c6 1/2-1/2
"#;
        let mut reader = BufferedReader::new_cursor(pgn);
        let mut visitor = TrainingVisitor::new();
        let result = reader.read_game(&mut visitor).expect("read game").flatten();

        let game = result.expect("game data");
        assert_eq!(game.result, GameResult::Draw);
    }

    #[test]
    fn test_visitor_incomplete_result() {
        let pgn = br#"[WhiteElo "1800"]
[BlackElo "1900"]
[Result "*"]

1. e4 c5 *
"#;
        let mut reader = BufferedReader::new_cursor(pgn);
        let mut visitor = TrainingVisitor::new();
        let result = reader.read_game(&mut visitor).expect("read game").flatten();

        let game = result.expect("game data");
        assert_eq!(game.result, GameResult::Incomplete);
    }

    #[test]
    fn test_visitor_resets_between_games() {
        let pgn = br#"[WhiteElo "1500"]
[BlackElo "1500"]
[Result "1-0"]

1. e4 e5 1-0

[WhiteElo "2000"]
[BlackElo "2000"]
[Result "0-1"]

1. d4 d5 2. c4 c6 0-1
"#;
        let mut reader = BufferedReader::new_cursor(pgn);
        let mut visitor = TrainingVisitor::new();

        let game1 = reader
            .read_game(&mut visitor)
            .expect("read game 1")
            .flatten()
            .expect("game 1 data");
        assert_eq!(game1.moves, vec!["e4", "e5"]);
        assert_eq!(game1.avg_elo, 1500);

        let game2 = reader
            .read_game(&mut visitor)
            .expect("read game 2")
            .flatten()
            .expect("game 2 data");
        assert_eq!(game2.moves, vec!["d4", "d5", "c4", "c6"]);
        assert_eq!(game2.avg_elo, 2000);
    }
}
