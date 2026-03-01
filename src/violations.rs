use crate::earley::{EarleyParser, MASK_TOKEN};

/// Three-tier violation detection strategy.
///
/// Given a token sequence and the positions that were recently unmasked,
/// identifies which of those positions violate the grammar.
///
/// Strategy (fast to slow):
/// 1. Fast path: parser reports error position directly. If it falls on an
///    unmasked position, that single position is returned.
/// 2. Medium path: try masking each unmasked position individually. If masking
///    a position makes the parse succeed, that position is a violator.
/// 3. Slow fallback: return all unmasked positions as violations.
pub fn find_violations(
    parser: &EarleyParser,
    tokens: &[String],
    unmasked_positions: &[usize],
) -> Vec<usize> {
    if unmasked_positions.is_empty() {
        return Vec::new();
    }

    // If it already parses, no violations.
    let err = match parser.parse(tokens) {
        Ok(()) => return Vec::new(),
        Err(e) => e,
    };

    // Fast path: parser error position falls on an unmasked position.
    if let Some(err_pos) = err.position {
        if unmasked_positions.contains(&err_pos) {
            return vec![err_pos];
        }
    }

    // Medium path: test each unmasked position by masking it back.
    // If masking position p makes the parse succeed (or removes the error),
    // then p is a violator.
    let mut violators = Vec::new();
    let mut test_tokens: Vec<String> = tokens.to_vec();

    for &pos in unmasked_positions {
        if pos >= tokens.len() {
            continue;
        }
        let original = test_tokens[pos].clone();
        test_tokens[pos] = MASK_TOKEN.to_string();

        if parser.parse(&test_tokens).is_ok() {
            violators.push(pos);
        }

        test_tokens[pos] = original;
    }

    if !violators.is_empty() {
        return violators;
    }

    // Slow fallback: return all unmasked positions.
    unmasked_positions
        .iter()
        .filter(|&&p| p < tokens.len())
        .copied()
        .collect()
}

/// A more aggressive variant: mask all violating positions at once and verify.
/// Returns the minimal set of positions that need remasking.
pub fn find_violations_greedy(
    parser: &EarleyParser,
    tokens: &[String],
    unmasked_positions: &[usize],
) -> Vec<usize> {
    if unmasked_positions.is_empty() {
        return Vec::new();
    }

    if parser.parse(tokens).is_ok() {
        return Vec::new();
    }

    // Greedy approach: mask all unmasked positions, then try unmasking each one.
    // Positions that cause failure when unmasked are violators.
    let mut masked_tokens: Vec<String> = tokens.to_vec();
    for &pos in unmasked_positions {
        if pos < masked_tokens.len() {
            masked_tokens[pos] = MASK_TOKEN.to_string();
        }
    }

    // If masking all doesn't help, the violation is elsewhere.
    if parser.parse(&masked_tokens).is_err() {
        return unmasked_positions
            .iter()
            .filter(|&&p| p < tokens.len())
            .copied()
            .collect();
    }

    // Try adding each position back. Those that cause failure are violators.
    let mut violators = Vec::new();
    for &pos in unmasked_positions {
        if pos >= tokens.len() {
            continue;
        }
        masked_tokens[pos] = tokens[pos].clone();
        if parser.parse(&masked_tokens).is_err() {
            violators.push(pos);
            masked_tokens[pos] = MASK_TOKEN.to_string(); // re-mask violator
        }
    }

    violators
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tokens(strs: &[&str]) -> Vec<String> {
        strs.iter().map(|s| s.to_string()).collect()
    }

    fn make_parser() -> EarleyParser {
        let grammar_str = r#"
            start: S
            S -> a b c d
        "#;
        EarleyParser::from_grammar_str(grammar_str).unwrap()
    }

    #[test]
    fn test_no_violations() {
        let parser = make_parser();
        let tokens = make_tokens(&["a", "b", "c", "d"]);
        let violations = find_violations(&parser, &tokens, &[1, 2]);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_single_violation_fast_path() {
        let parser = make_parser();
        // Position 2 has "x" instead of "c"
        let tokens = make_tokens(&["a", "b", "x", "d"]);
        let violations = find_violations(&parser, &tokens, &[2]);
        assert_eq!(violations, vec![2]);
    }

    #[test]
    fn test_single_violation_medium_path() {
        let parser = make_parser();
        // Position 1 has "x", unmasked positions include 1 and 3
        let tokens = make_tokens(&["a", "x", "c", "d"]);
        let violations = find_violations(&parser, &tokens, &[1, 3]);
        assert!(violations.contains(&1));
        assert!(!violations.contains(&3));
    }

    #[test]
    fn test_multiple_violations() {
        let parser = make_parser();
        // Positions 1 and 2 both wrong. Fast path finds position 1 (first error).
        // This is by design: find one, remask, re-decode, repeat.
        let tokens = make_tokens(&["a", "x", "y", "d"]);
        let violations = find_violations(&parser, &tokens, &[1, 2]);
        assert!(violations.contains(&1));
    }

    #[test]
    fn test_greedy_violations() {
        let parser = make_parser();
        let tokens = make_tokens(&["a", "x", "y", "d"]);
        let violations = find_violations_greedy(&parser, &tokens, &[1, 2]);
        assert!(violations.contains(&1));
        assert!(violations.contains(&2));
    }

    #[test]
    fn test_greedy_single_violation() {
        let parser = make_parser();
        let tokens = make_tokens(&["a", "x", "c", "d"]);
        let violations = find_violations_greedy(&parser, &tokens, &[1, 2, 3]);
        assert_eq!(violations, vec![1]);
    }
}
