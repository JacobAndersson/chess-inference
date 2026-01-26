use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TimeControl {
    Bullet,
    Blitz,
    Rapid,
    Classical,
    #[default]
    Unknown,
}

impl TimeControl {
    pub fn from_header(value: &[u8]) -> Self {
        let s = match std::str::from_utf8(value) {
            Ok(s) => s,
            Err(_) => return TimeControl::Unknown,
        };

        if s == "-" || s == "?" {
            return TimeControl::Unknown;
        }

        let (initial, increment) = parse_time_control_str(s);
        let estimated_seconds = initial + (40 * increment);

        match estimated_seconds {
            0..180 => TimeControl::Bullet,
            180..480 => TimeControl::Blitz,
            480..1500 => TimeControl::Rapid,
            _ => TimeControl::Classical,
        }
    }
}

fn parse_time_control_str(s: &str) -> (u32, u32) {
    let parts: Vec<&str> = s.split('+').collect();

    let initial = parts.first().and_then(|p| p.parse().ok()).unwrap_or(0);
    let increment = parts.get(1).and_then(|p| p.parse().ok()).unwrap_or(0);

    (initial, increment)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bullet_no_increment() {
        assert_eq!(TimeControl::from_header(b"60+0"), TimeControl::Bullet);
        assert_eq!(TimeControl::from_header(b"120+0"), TimeControl::Bullet);
    }

    #[test]
    fn test_bullet_with_increment() {
        assert_eq!(TimeControl::from_header(b"60+1"), TimeControl::Bullet);
        assert_eq!(TimeControl::from_header(b"30+0"), TimeControl::Bullet);
    }

    #[test]
    fn test_blitz() {
        assert_eq!(TimeControl::from_header(b"180+0"), TimeControl::Blitz);
        assert_eq!(TimeControl::from_header(b"180+2"), TimeControl::Blitz);
        assert_eq!(TimeControl::from_header(b"300+0"), TimeControl::Blitz);
        assert_eq!(TimeControl::from_header(b"300+3"), TimeControl::Blitz);
    }

    #[test]
    fn test_rapid() {
        assert_eq!(TimeControl::from_header(b"600+0"), TimeControl::Rapid);
        assert_eq!(TimeControl::from_header(b"900+10"), TimeControl::Rapid);
        assert_eq!(TimeControl::from_header(b"600+5"), TimeControl::Rapid);
    }

    #[test]
    fn test_classical() {
        assert_eq!(TimeControl::from_header(b"1800+0"), TimeControl::Classical);
        assert_eq!(TimeControl::from_header(b"1500+30"), TimeControl::Classical);
    }

    #[test]
    fn test_unknown() {
        assert_eq!(TimeControl::from_header(b"-"), TimeControl::Unknown);
        assert_eq!(TimeControl::from_header(b"?"), TimeControl::Unknown);
    }

    #[test]
    fn test_boundary_bullet_blitz() {
        // 179 seconds = bullet
        assert_eq!(TimeControl::from_header(b"179+0"), TimeControl::Bullet);
        // 180 seconds = blitz
        assert_eq!(TimeControl::from_header(b"180+0"), TimeControl::Blitz);
    }

    #[test]
    fn test_boundary_blitz_rapid() {
        // 479 estimated = blitz
        assert_eq!(TimeControl::from_header(b"479+0"), TimeControl::Blitz);
        // 480 estimated = rapid
        assert_eq!(TimeControl::from_header(b"480+0"), TimeControl::Rapid);
    }

    #[test]
    fn test_boundary_rapid_classical() {
        // 1499 estimated = rapid
        assert_eq!(TimeControl::from_header(b"1499+0"), TimeControl::Rapid);
        // 1500 estimated = classical
        assert_eq!(TimeControl::from_header(b"1500+0"), TimeControl::Classical);
    }

    #[test]
    fn test_increment_affects_classification() {
        // 60 + 40*3 = 180 = blitz
        assert_eq!(TimeControl::from_header(b"60+3"), TimeControl::Blitz);
    }

    #[test]
    fn test_invalid_utf8() {
        assert_eq!(
            TimeControl::from_header(&[0xFF, 0xFE]),
            TimeControl::Unknown
        );
    }

    #[test]
    fn test_parse_time_control_str() {
        assert_eq!(parse_time_control_str("180+2"), (180, 2));
        assert_eq!(parse_time_control_str("300+0"), (300, 0));
        assert_eq!(parse_time_control_str("600"), (600, 0));
        assert_eq!(parse_time_control_str("invalid"), (0, 0));
    }
}
