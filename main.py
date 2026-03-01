"""Entry point for dGrammar experiments."""

from dgrammar import GrammarChecker, MASK_TOKEN


def demo():
    """Quick demo of the grammar checker."""
    grammar = """
    start: S
    S -> "{" Pairs "}"
    Pairs -> Pair | Pair "," Pairs
    Pair -> Key ":" Value
    Key -> "key1" | "key2"
    Value -> "val1" | "val2" | "true" | "false"
    """

    checker = GrammarChecker(grammar)

    valid = ["{", "key1", ":", "val1", "}"]
    invalid = ["{", "key1", ":", "bad", "}"]
    masked = ["{", "key1", ":", MASK_TOKEN, "}"]

    print(f"Valid   {valid}: {checker.check(valid)}")
    print(f"Invalid {invalid}: {checker.check(invalid)}")
    print(f"Masked  {masked}: {checker.check(masked)}")

    violations = checker.find_violations(invalid, [3])
    print(f"Violations in {invalid} at unmasked [3]: {violations}")


if __name__ == "__main__":
    demo()
