use std::collections::{HashMap, HashSet};
use std::fmt;

// --- Grammar representation ---

/// A symbol in the grammar: either a terminal (token) or a nonterminal.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Symbol {
    Terminal(String),
    Nonterminal(String),
    /// Matches any single token (used for MASK positions).
    Wildcard,
}

/// A production rule: lhs -> rhs[0] rhs[1] ...
#[derive(Debug, Clone)]
pub struct Rule {
    pub lhs: String,
    pub rhs: Vec<Symbol>,
}

/// A context-free grammar.
#[derive(Debug, Clone)]
pub struct Grammar {
    pub rules: Vec<Rule>,
    pub start: String,
    rules_by_lhs: HashMap<String, Vec<usize>>,
}

impl Grammar {
    pub fn new(rules: Vec<Rule>, start: String) -> Self {
        let mut rules_by_lhs: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, rule) in rules.iter().enumerate() {
            rules_by_lhs
                .entry(rule.lhs.clone())
                .or_default()
                .push(i);
        }
        Self {
            rules,
            start,
            rules_by_lhs,
        }
    }

    pub fn rules_for(&self, nonterminal: &str) -> &[usize] {
        self.rules_by_lhs
            .get(nonterminal)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }
}

// --- Parse error ---

#[derive(Debug, Clone)]
pub struct ParseError {
    pub message: String,
    /// Position (0-indexed) where the parse failed, if determinable.
    pub position: Option<usize>,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.position {
            Some(pos) => write!(f, "ParseError at position {}: {}", pos, self.message),
            None => write!(f, "ParseError: {}", self.message),
        }
    }
}

// --- Earley item ---

/// An Earley item: (rule_index, dot_position, origin).
/// Represents progress through a rule starting from a given input position.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct EarleyItem {
    rule_idx: usize,
    dot: usize,
    origin: usize,
}

impl EarleyItem {
    fn new(rule_idx: usize, dot: usize, origin: usize) -> Self {
        Self {
            rule_idx,
            dot,
            origin,
        }
    }

    fn is_complete(&self, grammar: &Grammar) -> bool {
        self.dot >= grammar.rules[self.rule_idx].rhs.len()
    }

    fn next_symbol<'a>(&self, grammar: &'a Grammar) -> Option<&'a Symbol> {
        let rhs = &grammar.rules[self.rule_idx].rhs;
        if self.dot < rhs.len() {
            Some(&rhs[self.dot])
        } else {
            None
        }
    }
}

// --- Earley parser ---

/// State set for one position in the input.
#[derive(Debug, Clone, Default)]
struct StateSet {
    items: Vec<EarleyItem>,
    seen: HashSet<EarleyItem>,
}

impl StateSet {
    fn add(&mut self, item: EarleyItem) -> bool {
        if self.seen.insert(item.clone()) {
            self.items.push(item);
            true
        } else {
            false
        }
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

/// The MASK token sentinel.
pub const MASK_TOKEN: &str = "[MASK]";

/// Earley parser with support for MASK (wildcard) tokens.
pub struct EarleyParser {
    grammar: Grammar,
}

impl EarleyParser {
    pub fn new(grammar: Grammar) -> Self {
        Self { grammar }
    }

    /// Parse a grammar from a simple text format:
    /// ```text
    /// start: S
    /// S -> a B
    /// B -> b | c
    /// ```
    /// Terminals are lowercase or quoted strings, nonterminals are uppercase.
    pub fn from_grammar_str(grammar_str: &str) -> Result<Self, String> {
        let grammar = parse_grammar_str(grammar_str)?;
        Ok(Self::new(grammar))
    }

    /// Parse a sequence of tokens. MASK tokens are treated as wildcards.
    /// Returns Ok(()) if the input can be parsed, Err with position info otherwise.
    pub fn parse(&self, tokens: &[String]) -> Result<(), ParseError> {
        let n = tokens.len();
        let mut chart: Vec<StateSet> = vec![StateSet::default(); n + 1];

        // Seed: add all rules for the start symbol at position 0.
        for &rule_idx in self.grammar.rules_for(&self.grammar.start) {
            chart[0].add(EarleyItem::new(rule_idx, 0, 0));
        }

        for i in 0..=n {
            let mut j = 0;
            // Process items in chart[i]. We use index-based iteration because
            // new items may be added during processing.
            while j < chart[i].len() {
                let item = chart[i].items[j].clone();
                j += 1;

                if item.is_complete(&self.grammar) {
                    // Completion: advance items in chart[item.origin] that
                    // were waiting for this nonterminal.
                    let completed_lhs = &self.grammar.rules[item.rule_idx].lhs.clone();
                    let origin = item.origin;
                    let mut to_add = Vec::new();
                    for k in 0..chart[origin].len() {
                        let waiting = &chart[origin].items[k];
                        if let Some(Symbol::Nonterminal(ref nt)) =
                            waiting.next_symbol(&self.grammar)
                        {
                            if nt == completed_lhs {
                                to_add.push(EarleyItem::new(
                                    waiting.rule_idx,
                                    waiting.dot + 1,
                                    waiting.origin,
                                ));
                            }
                        }
                    }
                    for new_item in to_add {
                        chart[i].add(new_item);
                    }
                } else if let Some(sym) = item.next_symbol(&self.grammar) {
                    match sym {
                        Symbol::Nonterminal(ref nt) => {
                            // Prediction: add all rules for the nonterminal.
                            let rule_indices: Vec<usize> =
                                self.grammar.rules_for(nt).to_vec();
                            for rule_idx in rule_indices {
                                chart[i].add(EarleyItem::new(rule_idx, 0, i));
                            }
                        }
                        Symbol::Terminal(_) | Symbol::Wildcard => {
                            // Scanning is handled below when processing input tokens.
                        }
                    }
                }
            }

            // Scan: if there is a next token, advance matching items.
            if i < n {
                let token = &tokens[i];
                let is_mask = token == MASK_TOKEN;

                // Collect items to advance (avoids borrow conflict).
                let to_advance: Vec<EarleyItem> = chart[i]
                    .items
                    .iter()
                    .filter_map(|item| {
                        item.next_symbol(&self.grammar).and_then(|sym| {
                            let matches = match sym {
                                Symbol::Terminal(ref t) => is_mask || t == token,
                                Symbol::Wildcard => true,
                                Symbol::Nonterminal(_) => false,
                            };
                            if matches {
                                Some(EarleyItem::new(
                                    item.rule_idx,
                                    item.dot + 1,
                                    item.origin,
                                ))
                            } else {
                                None
                            }
                        })
                    })
                    .collect();

                for new_item in to_advance {
                    chart[i + 1].add(new_item);
                }
            }
        }

        // Check if any completed start-symbol item spans the full input.
        let start = &self.grammar.start;
        for item in &chart[n].items {
            if item.origin == 0
                && item.is_complete(&self.grammar)
                && self.grammar.rules[item.rule_idx].lhs == *start
            {
                return Ok(());
            }
        }

        // Find the rightmost position that had any items (for error reporting).
        let mut last_active = 0;
        for (pos, state_set) in chart.iter().enumerate() {
            if !state_set.items.is_empty() {
                last_active = pos;
            }
        }

        Err(ParseError {
            message: format!(
                "parse failed: no complete parse found (input length {}, last active position {})",
                n, last_active
            ),
            position: if last_active < n {
                Some(last_active)
            } else {
                None
            },
        })
    }

    /// Parse and return the error position, if any.
    pub fn parse_with_error_position(&self, tokens: &[String]) -> Option<usize> {
        match self.parse(tokens) {
            Ok(()) => None,
            Err(e) => e.position,
        }
    }

    pub fn grammar(&self) -> &Grammar {
        &self.grammar
    }
}

// --- Grammar string parser ---

fn parse_grammar_str(input: &str) -> Result<Grammar, String> {
    let mut rules = Vec::new();
    let mut start: Option<String> = None;

    for line in input.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Handle "start: X"
        if let Some(rest) = line.strip_prefix("start:") {
            start = Some(rest.trim().to_string());
            continue;
        }

        // Handle "LHS -> RHS1 | RHS2"
        let parts: Vec<&str> = line.splitn(2, "->").collect();
        if parts.len() != 2 {
            return Err(format!("invalid rule: {}", line));
        }

        let lhs = parts[0].trim().to_string();
        let alternatives: Vec<&str> = parts[1].split('|').collect();

        for alt in alternatives {
            let symbols: Vec<Symbol> = alt
                .trim()
                .split_whitespace()
                .map(|s| {
                    if s == "*" {
                        Symbol::Wildcard
                    } else if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
                        // Quoted terminal
                        Symbol::Terminal(s[1..s.len() - 1].to_string())
                    } else if s.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                        Symbol::Nonterminal(s.to_string())
                    } else {
                        Symbol::Terminal(s.to_string())
                    }
                })
                .collect();

            if symbols.is_empty() {
                // Epsilon production (empty rhs)
                rules.push(Rule {
                    lhs: lhs.clone(),
                    rhs: Vec::new(),
                });
            } else {
                rules.push(Rule {
                    lhs: lhs.clone(),
                    rhs: symbols,
                });
            }
        }
    }

    let start = start.unwrap_or_else(|| {
        rules
            .first()
            .map(|r| r.lhs.clone())
            .unwrap_or_default()
    });

    if rules.is_empty() {
        return Err("no rules defined".to_string());
    }

    Ok(Grammar::new(rules, start))
}

// --- Unit tests ---

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tokens(strs: &[&str]) -> Vec<String> {
        strs.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn test_simple_grammar() {
        let grammar_str = r#"
            start: S
            S -> a B
            B -> b | c
        "#;
        let parser = EarleyParser::from_grammar_str(grammar_str).unwrap();

        assert!(parser.parse(&make_tokens(&["a", "b"])).is_ok());
        assert!(parser.parse(&make_tokens(&["a", "c"])).is_ok());
        assert!(parser.parse(&make_tokens(&["a", "d"])).is_err());
        assert!(parser.parse(&make_tokens(&["b", "b"])).is_err());
    }

    #[test]
    fn test_mask_token_wildcard() {
        let grammar_str = r#"
            start: S
            S -> a B
            B -> b | c
        "#;
        let parser = EarleyParser::from_grammar_str(grammar_str).unwrap();

        // MASK at position 1 should accept (matches b or c)
        assert!(parser.parse(&make_tokens(&["a", "[MASK]"])).is_ok());
        // MASK at position 0 should fail (nothing produces MASK -> a B pattern starting with wildcard)
        // Actually, MASK matches terminal "a", so it should work
        assert!(parser.parse(&make_tokens(&["[MASK]", "b"])).is_ok());
        // All masks
        assert!(parser.parse(&make_tokens(&["[MASK]", "[MASK]"])).is_ok());
    }

    #[test]
    fn test_recursive_grammar() {
        let grammar_str = r#"
            start: S
            S -> a S b | a b
        "#;
        let parser = EarleyParser::from_grammar_str(grammar_str).unwrap();

        assert!(parser.parse(&make_tokens(&["a", "b"])).is_ok());
        assert!(parser.parse(&make_tokens(&["a", "a", "b", "b"])).is_ok());
        assert!(parser
            .parse(&make_tokens(&["a", "a", "a", "b", "b", "b"]))
            .is_ok());
        assert!(parser.parse(&make_tokens(&["a", "a", "b"])).is_err());
    }

    #[test]
    fn test_epsilon_production() {
        let grammar_str = r#"
            start: S
            S -> a B c
            B -> b |
        "#;
        let parser = EarleyParser::from_grammar_str(grammar_str).unwrap();

        assert!(parser.parse(&make_tokens(&["a", "b", "c"])).is_ok());
        assert!(parser.parse(&make_tokens(&["a", "c"])).is_ok()); // B -> epsilon
    }

    #[test]
    fn test_error_position() {
        let grammar_str = r#"
            start: S
            S -> a b c d
        "#;
        let parser = EarleyParser::from_grammar_str(grammar_str).unwrap();

        let err = parser.parse(&make_tokens(&["a", "b", "x", "d"])).unwrap_err();
        assert_eq!(err.position, Some(2));
    }

    #[test]
    fn test_quoted_terminals() {
        let grammar_str = r#"
            start: S
            S -> "{" Pairs "}"
            Pairs -> Pair | Pair "," Pairs
            Pair -> Key ":" Value
            Key -> "key"
            Value -> "val"
        "#;
        let parser = EarleyParser::from_grammar_str(grammar_str).unwrap();

        assert!(parser
            .parse(&make_tokens(&["{", "key", ":", "val", "}"]))
            .is_ok());
        assert!(parser
            .parse(&make_tokens(&[
                "{", "key", ":", "val", ",", "key", ":", "val", "}"
            ]))
            .is_ok());
    }

    #[test]
    fn test_ambiguous_grammar() {
        // E -> E + E | n  (ambiguous)
        let grammar_str = r#"
            start: E
            E -> E "+" E | n
        "#;
        let parser = EarleyParser::from_grammar_str(grammar_str).unwrap();

        assert!(parser.parse(&make_tokens(&["n"])).is_ok());
        assert!(parser.parse(&make_tokens(&["n", "+", "n"])).is_ok());
        assert!(parser
            .parse(&make_tokens(&["n", "+", "n", "+", "n"]))
            .is_ok());
    }

    #[test]
    fn test_empty_input() {
        let grammar_str = r#"
            start: S
            S ->
        "#;
        let parser = EarleyParser::from_grammar_str(grammar_str).unwrap();
        assert!(parser.parse(&make_tokens(&[])).is_ok());
    }
}
