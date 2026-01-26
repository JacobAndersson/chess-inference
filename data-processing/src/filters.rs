use crate::training_visitor::{GameResult, TrainingGameData};

const MIN_PLIES: usize = 10;

pub fn is_valid_training_game(game: &TrainingGameData) -> bool {
    game.moves.len() >= MIN_PLIES && game.result != GameResult::Incomplete
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_game(moves_count: usize, result: GameResult) -> TrainingGameData {
        TrainingGameData {
            avg_elo: 1500,
            moves: vec!["e4".to_string(); moves_count],
            result,
        }
    }

    #[test]
    fn test_valid_game_passes() {
        let game = make_game(10, GameResult::White);
        assert!(is_valid_training_game(&game));
    }

    #[test]
    fn test_long_game_passes() {
        let game = make_game(100, GameResult::Draw);
        assert!(is_valid_training_game(&game));
    }

    #[test]
    fn test_short_game_rejected() {
        let game = make_game(9, GameResult::White);
        assert!(!is_valid_training_game(&game));
    }

    #[test]
    fn test_incomplete_game_rejected() {
        let game = make_game(20, GameResult::Incomplete);
        assert!(!is_valid_training_game(&game));
    }

    #[test]
    fn test_edge_case_exactly_10_plies() {
        let game = make_game(10, GameResult::Black);
        assert!(is_valid_training_game(&game));
    }

    #[test]
    fn test_short_and_incomplete_rejected() {
        let game = make_game(5, GameResult::Incomplete);
        assert!(!is_valid_training_game(&game));
    }
}
