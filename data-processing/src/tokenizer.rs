pub const VOCAB_SIZE: usize = 32;
pub const EOS_TOKEN: u8 = 30;
pub const PAD_TOKEN: u8 = 31;

const VOCAB_CHARS: &[u8] = b"abcdefgh12345678KQRBNx+#=O- 0/";

const fn build_encode_table() -> [u8; 128] {
    let mut table = [255u8; 128];
    let mut i = 0;
    while i < VOCAB_CHARS.len() {
        table[VOCAB_CHARS[i] as usize] = i as u8;
        i += 1;
    }
    table
}

const ENCODE_TABLE: [u8; 128] = build_encode_table();

pub struct ChessTokenizer;

impl ChessTokenizer {
    pub fn encode_into(text: &str, buffer: &mut [u8]) -> Option<usize> {
        let bytes = text.as_bytes();
        if bytes.len() > buffer.len() {
            return None;
        }

        for (i, &byte) in bytes.iter().enumerate() {
            if byte >= 128 {
                return None;
            }
            let token = ENCODE_TABLE[byte as usize];
            if token == 255 {
                return None;
            }
            buffer[i] = token;
        }
        Some(bytes.len())
    }

    pub fn encode_with_eos(text: &str, buffer: &mut [u8]) -> Option<usize> {
        let bytes = text.as_bytes();
        let total_len = bytes.len() + 1;
        if total_len > buffer.len() {
            return None;
        }

        for (i, &byte) in bytes.iter().enumerate() {
            if byte >= 128 {
                return None;
            }
            let token = ENCODE_TABLE[byte as usize];
            if token == 255 {
                return None;
            }
            buffer[i] = token;
        }
        buffer[bytes.len()] = EOS_TOKEN;
        Some(total_len)
    }

    pub fn decode(tokens: &[u8]) -> String {
        tokens
            .iter()
            .filter_map(|&t| {
                if t == EOS_TOKEN || t == PAD_TOKEN {
                    None
                } else if (t as usize) < VOCAB_CHARS.len() {
                    Some(VOCAB_CHARS[t as usize] as char)
                } else {
                    None
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_simple_move() {
        let mut buffer = [0u8; 16];
        let len = ChessTokenizer::encode_into("e4", &mut buffer);
        assert_eq!(len, Some(2));
        assert_eq!(buffer[0], 4); // 'e' -> 4
        assert_eq!(buffer[1], 11); // '4' -> 11 ('1' is at 8, so '4' is at 11)
    }

    #[test]
    fn test_encode_with_eos() {
        let mut buffer = [0u8; 16];
        let len = ChessTokenizer::encode_with_eos("e4", &mut buffer);
        assert_eq!(len, Some(3));
        assert_eq!(buffer[0], 4);
        assert_eq!(buffer[1], 11);
        assert_eq!(buffer[2], EOS_TOKEN);
    }

    #[test]
    fn test_roundtrip() {
        let original = "e4 e5 Nf3 Nc6";
        let mut buffer = [0u8; 64];
        let len = ChessTokenizer::encode_into(original, &mut buffer).expect("encode");
        let decoded = ChessTokenizer::decode(&buffer[..len]);
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_buffer_overflow_returns_none() {
        let mut buffer = [0u8; 2];
        let result = ChessTokenizer::encode_into("e4 e5", &mut buffer);
        assert_eq!(result, None);
    }

    #[test]
    fn test_buffer_overflow_with_eos_returns_none() {
        let mut buffer = [0u8; 2];
        let result = ChessTokenizer::encode_with_eos("e4", &mut buffer);
        assert_eq!(result, None);
    }

    #[test]
    fn test_castling_notation() {
        let mut buffer = [0u8; 16];

        let len = ChessTokenizer::encode_into("O-O", &mut buffer);
        assert_eq!(len, Some(3));
        assert_eq!(ChessTokenizer::decode(&buffer[..3]), "O-O");

        let len = ChessTokenizer::encode_into("O-O-O", &mut buffer);
        assert_eq!(len, Some(5));
        assert_eq!(ChessTokenizer::decode(&buffer[..5]), "O-O-O");
    }

    #[test]
    fn test_capture_and_check() {
        let mut buffer = [0u8; 16];
        let len = ChessTokenizer::encode_into("Bxf7+", &mut buffer).expect("encode");
        assert_eq!(ChessTokenizer::decode(&buffer[..len]), "Bxf7+");
    }

    #[test]
    fn test_promotion() {
        let mut buffer = [0u8; 16];
        let len = ChessTokenizer::encode_into("e8=Q", &mut buffer).expect("encode");
        assert_eq!(ChessTokenizer::decode(&buffer[..len]), "e8=Q");
    }

    #[test]
    fn test_checkmate() {
        let mut buffer = [0u8; 16];
        let len = ChessTokenizer::encode_into("Qh7#", &mut buffer).expect("encode");
        assert_eq!(ChessTokenizer::decode(&buffer[..len]), "Qh7#");
    }

    #[test]
    fn test_invalid_character_returns_none() {
        let mut buffer = [0u8; 16];
        let result = ChessTokenizer::encode_into("e4!", &mut buffer);
        assert_eq!(result, None);
    }

    #[test]
    fn test_decode_ignores_eos_and_pad() {
        let tokens = [4, 11, EOS_TOKEN, PAD_TOKEN, PAD_TOKEN];
        let decoded = ChessTokenizer::decode(&tokens);
        assert_eq!(decoded, "e4");
    }

    #[test]
    fn test_vocab_constants() {
        assert_eq!(VOCAB_SIZE, 32);
        assert_eq!(EOS_TOKEN, 30);
        assert_eq!(PAD_TOKEN, 31);
    }

    #[test]
    fn test_full_game_encode() {
        let game = "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O";
        let mut buffer = [0u8; 128];
        let len = ChessTokenizer::encode_with_eos(game, &mut buffer).expect("encode");
        let decoded = ChessTokenizer::decode(&buffer[..len]);
        assert_eq!(decoded, game);
        assert_eq!(buffer[len - 1], EOS_TOKEN);
    }
}
