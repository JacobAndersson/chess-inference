pub fn elo_bucket(max_elo: u16) -> u16 {
    if max_elo < 1000 {
        0
    } else {
        (max_elo / 100) * 100
    }
}

pub fn ply_bucket(plies: u16) -> u16 {
    (plies / 10) * 10
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elo_bucket_below_1000() {
        assert_eq!(elo_bucket(0), 0);
        assert_eq!(elo_bucket(500), 0);
        assert_eq!(elo_bucket(999), 0);
    }

    #[test]
    fn test_elo_bucket_boundary() {
        assert_eq!(elo_bucket(1000), 1000);
        assert_eq!(elo_bucket(1001), 1000);
        assert_eq!(elo_bucket(1099), 1000);
        assert_eq!(elo_bucket(1100), 1100);
    }

    #[test]
    fn test_elo_bucket_high_values() {
        assert_eq!(elo_bucket(2000), 2000);
        assert_eq!(elo_bucket(2850), 2800);
    }

    #[test]
    fn test_ply_bucket_small() {
        assert_eq!(ply_bucket(0), 0);
        assert_eq!(ply_bucket(1), 0);
        assert_eq!(ply_bucket(9), 0);
    }

    #[test]
    fn test_ply_bucket_boundary() {
        assert_eq!(ply_bucket(10), 10);
        assert_eq!(ply_bucket(11), 10);
        assert_eq!(ply_bucket(19), 10);
        assert_eq!(ply_bucket(20), 20);
    }

    #[test]
    fn test_ply_bucket_typical_game() {
        assert_eq!(ply_bucket(40), 40);
        assert_eq!(ply_bucket(85), 80);
        assert_eq!(ply_bucket(200), 200);
    }
}
