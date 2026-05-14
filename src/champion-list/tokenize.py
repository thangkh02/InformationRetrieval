from __future__ import annotations

import argparse
import json
import importlib.util
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEXT_UTILS_PATH = ROOT / "text_utils.py"

_STD_TOKENIZE_PATH = Path(sysconfig.get_path("stdlib")) / "tokenize.py"
_STD_TOKENIZE_SPEC = importlib.util.spec_from_file_location("_stdlib_tokenize", _STD_TOKENIZE_PATH)
if _STD_TOKENIZE_SPEC is not None and _STD_TOKENIZE_SPEC.loader is not None:
    _STD_TOKENIZE_MODULE = importlib.util.module_from_spec(_STD_TOKENIZE_SPEC)
    _STD_TOKENIZE_SPEC.loader.exec_module(_STD_TOKENIZE_MODULE)
    for _name in dir(_STD_TOKENIZE_MODULE):
        if _name.startswith("_"):
            continue
        globals().setdefault(_name, getattr(_STD_TOKENIZE_MODULE, _name))

spec = importlib.util.spec_from_file_location("text_utils_local", TEXT_UTILS_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load {TEXT_UTILS_PATH}")
text_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(text_utils)
tokenize_text = text_utils.tokenize_underthesea_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tokenize",
        description="Tokenize JSONL files and cache tokenized text as JSONL.",
    )
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--id-field", default="", help="Optional explicit id field name")
    parser.add_argument(
        "--text-fields",
        default="title,text",
        help="Comma-separated legal corpus fields to join and tokenize",
    )
    return parser


def _resolve_id(obj: dict, explicit_field: str) -> str:
    if explicit_field:
        value = obj.get(explicit_field)
        if value is None:
            raise KeyError(f"Missing id field: {explicit_field}")
        return str(value)

    for key in ("_id", "doc_id", "query-id"):
        if key in obj and obj[key] is not None:
            return str(obj[key])

    raise KeyError("Could not resolve an id field. Pass --id-field explicitly.")


def _resolve_text(obj: dict, fields: list[str]) -> str:
    parts: list[str] = []
    for field in fields:
        value = obj.get(field)
        if value:
            parts.append(str(value))
    return " ".join(parts).strip()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    text_fields = [field.strip() for field in args.text_fields.split(",") if field.strip()]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with input_path.open("r", encoding="utf-8", errors="ignore") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            item_id = _resolve_id(obj, args.id_field)
            text = _resolve_text(obj, text_fields)
            tokens = list(tokenize_text(text))
            fout.write(
                json.dumps(
                    {
                        "id": item_id,
                        "doc_id": item_id,
                        "tokens": tokens,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1

    print(f"Wrote {count} tokenized rows to {output_path}")


if __name__ == "__main__":
    main()
