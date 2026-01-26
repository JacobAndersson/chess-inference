use pgn_reader::{RawHeader, SanPlus, Skip, Visitor};

use crate::time_control::TimeControl;

pub struct GameData {
    pub max_elo: u16,
    pub ply_count: u16,
    pub time_control: TimeControl,
}

#[derive(Default)]
pub struct GameVisitor {
    white_elo: Option<u16>,
    black_elo: Option<u16>,
    ply_count: u16,
    time_control: TimeControl,
}

impl GameVisitor {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Visitor for GameVisitor {
    type Result = Option<GameData>;

    fn begin_game(&mut self) {
        self.white_elo = None;
        self.black_elo = None;
        self.ply_count = 0;
        self.time_control = TimeControl::Unknown;
    }

    fn header(&mut self, key: &[u8], value: RawHeader<'_>) {
        let value_str = value.as_bytes();

        if key == b"WhiteElo" {
            self.white_elo = parse_elo(value_str);
        } else if key == b"BlackElo" {
            self.black_elo = parse_elo(value_str);
        } else if key == b"TimeControl" {
            self.time_control = TimeControl::from_header(value_str);
        }
    }

    fn san(&mut self, _san: SanPlus) {
        self.ply_count += 1;
    }

    fn begin_variation(&mut self) -> Skip {
        Skip(true)
    }

    fn end_game(&mut self) -> Self::Result {
        match (self.white_elo, self.black_elo) {
            (Some(w), Some(b)) => Some(GameData {
                max_elo: w.max(b),
                ply_count: self.ply_count,
                time_control: self.time_control,
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

#[cfg(test)]
mod tests {
    use super::*;
    use pgn_reader::BufferedReader;

    #[test]
    fn test_parse_elo_valid() {
        assert_eq!(parse_elo(b"1500"), Some(1500));
        assert_eq!(parse_elo(b"2000"), Some(2000));
        assert_eq!(parse_elo(b"800"), Some(800));
    }

    #[test]
    fn test_parse_elo_question_mark() {
        assert_eq!(parse_elo(b"?"), None);
    }

    #[test]
    fn test_parse_elo_invalid() {
        assert_eq!(parse_elo(b"abc"), None);
        assert_eq!(parse_elo(b""), None);
    }

    #[test]
    fn test_visitor_complete_game() {
        let pgn = br#"[WhiteElo "1500"]
[BlackElo "1600"]
[TimeControl "300+0"]

1. e4 e5 2. Nf3 Nc6 *
"#;
        let mut reader = BufferedReader::new_cursor(pgn);
        let mut visitor = GameVisitor::new();
        let result = reader.read_game(&mut visitor).expect("read game").flatten();

        let game = result.expect("game data");
        assert_eq!(game.max_elo, 1600);
        assert_eq!(game.ply_count, 4);
        assert_eq!(game.time_control, TimeControl::Blitz);
    }

    #[test]
    fn test_visitor_missing_elo() {
        let pgn = br#"[WhiteElo "?"]
[BlackElo "1600"]

1. e4 e5 *
"#;
        let mut reader = BufferedReader::new_cursor(pgn);
        let mut visitor = GameVisitor::new();
        let result = reader.read_game(&mut visitor).expect("read game").flatten();

        assert!(result.is_none());
    }

    #[test]
    fn test_visitor_longer_game() {
        let pgn = br#"[WhiteElo "2000"]
[BlackElo "2100"]
[TimeControl "600+5"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 1-0
"#;
        let mut reader = BufferedReader::new_cursor(pgn);
        let mut visitor = GameVisitor::new();
        let result = reader.read_game(&mut visitor).expect("read game").flatten();

        let game = result.expect("game data");
        assert_eq!(game.max_elo, 2100);
        assert_eq!(game.ply_count, 10);
        assert_eq!(game.time_control, TimeControl::Rapid);
    }

    #[test]
    fn test_visitor_no_time_control() {
        let pgn = br#"[WhiteElo "1500"]
[BlackElo "1600"]

1. e4 e5 *
"#;
        let mut reader = BufferedReader::new_cursor(pgn);
        let mut visitor = GameVisitor::new();
        let result = reader.read_game(&mut visitor).expect("read game").flatten();

        let game = result.expect("game data");
        assert_eq!(game.time_control, TimeControl::Unknown);
    }

    #[test]
    fn test_visitor_white_higher_elo() {
        let pgn = br#"[WhiteElo "2000"]
[BlackElo "1800"]

1. e4 *
"#;
        let mut reader = BufferedReader::new_cursor(pgn);
        let mut visitor = GameVisitor::new();
        let result = reader.read_game(&mut visitor).expect("read game").flatten();

        let game = result.expect("game data");
        assert_eq!(game.max_elo, 2000);
    }
}
