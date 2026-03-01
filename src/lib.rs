mod earley;
mod violations;

use pyo3::prelude::*;

/// Grammar checker exposed to Python via PyO3.
#[pyclass]
pub struct GrammarChecker {
    parser: earley::EarleyParser,
}

#[pymethods]
impl GrammarChecker {
    #[new]
    fn new(grammar_str: &str) -> PyResult<Self> {
        let parser = earley::EarleyParser::from_grammar_str(grammar_str)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        Ok(Self { parser })
    }

    /// Parse a token sequence and return whether it is valid.
    fn check(&self, tokens: Vec<String>) -> bool {
        self.parser.parse(&tokens).is_ok()
    }

    /// Parse and return the error position, or None if valid.
    fn error_position(&self, tokens: Vec<String>) -> Option<usize> {
        self.parser.parse_with_error_position(&tokens)
    }

    /// Find violation positions among recently unmasked tokens.
    /// Uses three-tier strategy: fast (error position), medium (per-token), slow (all).
    fn find_violations(
        &self,
        tokens: Vec<String>,
        unmasked_positions: Vec<usize>,
    ) -> Vec<usize> {
        violations::find_violations(&self.parser, &tokens, &unmasked_positions)
    }

    /// Greedy violation detection: mask all, then add back one by one.
    /// Returns a tighter set of violators at the cost of more parse calls.
    fn find_violations_greedy(
        &self,
        tokens: Vec<String>,
        unmasked_positions: Vec<usize>,
    ) -> Vec<usize> {
        violations::find_violations_greedy(&self.parser, &tokens, &unmasked_positions)
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GrammarChecker>()?;
    m.add("MASK_TOKEN", earley::MASK_TOKEN)?;
    Ok(())
}
