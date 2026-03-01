"""Grammar definition helpers.

Converts JSON Schema and other formats into the CFG text format
consumed by the Rust Earley parser.
"""

from __future__ import annotations

import json
from typing import Any


def json_schema_to_cfg(schema: dict[str, Any]) -> str:
    """Convert a JSON Schema into a CFG string for the Earley parser.

    Supports: object, array, string, number, boolean, null, enum.
    """
    rules: list[str] = ["start: Value"]
    _seen: set[str] = set()

    def _add_type_rules(type_name: str, schema_part: dict[str, Any]) -> str:
        """Generate rules for a schema type. Returns the nonterminal name."""
        typ = schema_part.get("type", "any")
        enum_vals = schema_part.get("enum")

        if enum_vals is not None:
            nt = f"Enum{len(_seen)}"
            _seen.add(nt)
            for val in enum_vals:
                rules.append(f'{nt} -> "{json.dumps(val)}"')
            return nt

        if typ == "object":
            return _add_object_rules(schema_part)
        elif typ == "array":
            return _add_array_rules(schema_part)
        elif typ == "string":
            return "StringVal"
        elif typ in ("number", "integer"):
            return "NumberVal"
        elif typ == "boolean":
            return "BoolVal"
        elif typ == "null":
            return "NullVal"
        else:
            return "Value"

    def _add_object_rules(schema_part: dict[str, Any]) -> str:
        nt = f"Obj{len(_seen)}"
        _seen.add(nt)
        props = schema_part.get("properties", {})
        if not props:
            rules.append(f'{nt} -> "{{" "}}"')
            return nt

        pair_nts = []
        for key, val_schema in props.items():
            pair_nt = f"Pair{len(_seen)}"
            _seen.add(pair_nt)
            val_nt = _add_type_rules(f"Val{len(_seen)}", val_schema)
            rules.append(f'{pair_nt} -> "\\"{key}\\"" ":" {val_nt}')
            pair_nts.append(pair_nt)

        # Build object body: pairs separated by commas
        if len(pair_nts) == 1:
            rules.append(f'{nt} -> "{{" {pair_nts[0]} "}}"')
        else:
            body_nt = f"ObjBody{len(_seen)}"
            _seen.add(body_nt)
            # First pair, then comma-separated rest
            body = f'{pair_nts[0]}'
            for pnt in pair_nts[1:]:
                body += f' "," {pnt}'
            rules.append(f'{body_nt} -> {body}')
            rules.append(f'{nt} -> "{{" {body_nt} "}}"')

        return nt

    def _add_array_rules(schema_part: dict[str, Any]) -> str:
        nt = f"Arr{len(_seen)}"
        _seen.add(nt)
        items_schema = schema_part.get("items", {})
        item_nt = _add_type_rules(f"Item{len(_seen)}", items_schema)

        elems_nt = f"Elems{len(_seen)}"
        _seen.add(elems_nt)
        rules.append(f'{elems_nt} -> {item_nt} | {item_nt} "," {elems_nt}')
        rules.append(f'{nt} -> "[" {elems_nt} "]" | "[" "]"')

        return nt

    # Base value types
    rules.extend([
        'Value -> StringVal | NumberVal | BoolVal | NullVal | ObjGeneric | ArrGeneric',
        'StringVal -> string',
        'NumberVal -> number',
        'BoolVal -> "true" | "false"',
        'NullVal -> "null"',
        'ObjGeneric -> "{" "}"',
        'ArrGeneric -> "[" "]"',
    ])

    root_nt = _add_type_rules("Root", schema)
    if root_nt != "Value":
        rules.insert(0, f"start: {root_nt}")
        rules.pop(1)  # Remove the "start: Value"

    return "\n".join(rules)


def cpp_grammar() -> str:
    """Return a simplified C++ grammar in CFG format.

    This is a subset covering common constructs. For full C++ parsing,
    use tree-sitter instead.
    """
    return """
start: Program
Program -> Stmts
Stmts -> Stmt | Stmt Stmts
Stmt -> ExprStmt | IfStmt | ReturnStmt | BlockStmt | VarDecl | FuncDecl
ExprStmt -> Expr ";"
IfStmt -> "if" "(" Expr ")" Stmt | "if" "(" Expr ")" Stmt "else" Stmt
ReturnStmt -> "return" Expr ";" | "return" ";"
BlockStmt -> "{" "}" | "{" Stmts "}"
VarDecl -> Type Ident "=" Expr ";" | Type Ident ";"
FuncDecl -> Type Ident "(" ")" BlockStmt | Type Ident "(" Params ")" BlockStmt
Params -> Param | Param "," Params
Param -> Type Ident
Type -> "int" | "float" | "double" | "char" | "bool" | "void" | "string" | "auto"
Expr -> Ident | Literal | Expr Op Expr | "(" Expr ")" | Ident "(" ")" | Ident "(" Args ")"
Args -> Expr | Expr "," Args
Op -> "+" | "-" | "*" | "/" | "==" | "!=" | "<" | ">" | "<=" | ">=" | "&&" | "||"
Ident -> identifier
Literal -> number | string
""".strip()


def smiles_grammar() -> str:
    """Return a simplified SMILES grammar in CFG format."""
    return """
start: Smiles
Smiles -> Chain
Chain -> Atom | Atom Chain | Atom Bond Chain | Atom Branch Chain
Branch -> "(" Chain ")" | "(" Bond Chain ")"
Atom -> OrganicAtom | BracketAtom
OrganicAtom -> "C" | "N" | "O" | "S" | "P" | "F" | "Cl" | "Br" | "I" | "B" | "c" | "n" | "o" | "s"
BracketAtom -> "[" AtomSpec "]"
AtomSpec -> Symbol | Symbol Charge
Symbol -> "C" | "N" | "O" | "S" | "P" | "F" | "Cl" | "Br" | "I" | "B" | "H" | "Si" | "Se"
Charge -> "+" | "-" | "++" | "--"
Bond -> "-" | "=" | "#" | ":"
""".strip()
