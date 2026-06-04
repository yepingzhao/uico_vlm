#!/usr/bin/env python3
"""Check Prompt B format compliance — reports how many outputs contain
both "Violation:" and "Location:" fields.

Usage:
    python scripts/check_format_compliance.py \\
        outputs/llava/predictions_prompt_b.jsonl \\
        outputs/qwen2vl/predictions_prompt_b.jsonl
"""

import json
import sys


def check(filepath: str) -> tuple[int, int]:
    """Count compliant (both fields present) vs total predictions."""
    total = 0
    compliant = 0
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            caption = rec["caption"]
            total += 1
            if "Violation:" in caption and "Location:" in caption:
                compliant += 1
    return compliant, total


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <predictions.jsonl> [...]")
        sys.exit(1)

    for fp in sys.argv[1:]:
        compliant, total = check(fp)
        rate = compliant / total * 100 if total > 0 else 0
        print(f"{fp}: {compliant}/{total} ({rate:.1f}%) compliant")


if __name__ == "__main__":
    main()
