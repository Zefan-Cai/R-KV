"""Shared output-health checker for R-KV smokes.

A healthy generation is fluent text; broken R-KV shows up as token salad
(e.g. 'GraphUnits' style), tight repetition loops, or truncation crashes.
"""
import re
import sys


def check_text(text: str, min_distinct_words: int | None = None) -> dict:
    words = re.findall(r"[A-Za-z]{2,}", text)
    distinct = len(set(w.lower() for w in words))
    if min_distinct_words is None:
        # Scale with output size: a short, correct answer can legitimately
        # use few distinct words; long outputs should show variety.
        min_distinct_words = min(40, max(15, len(text) // 30))
    # longest run of one repeated 20-char chunk
    max_rep = 0
    if len(text) >= 20:
        seen = {}
        step = 10
        for i in range(0, len(text) - 20, step):
            chunk = text[i : i + 20]
            seen[chunk] = seen.get(chunk, 0) + 1
        max_rep = max(seen.values()) if seen else 0
    # ratio of ascii letters+digits+punct+space (garble tends to add junk)
    printable = sum(1 for c in text if c.isprintable())
    ratio = printable / max(1, len(text))
    ok = distinct >= min_distinct_words and max_rep <= 10 and ratio > 0.95
    return {
        "chars": len(text),
        "distinct_words": distinct,
        "max_20char_repeats": max_rep,
        "printable_ratio": round(ratio, 3),
        "healthy": ok,
    }


def report(name: str, text: str) -> bool:
    r = check_text(text)
    status = "HEALTH_PASS" if r["healthy"] else "HEALTH_FAIL"
    print(f"{status} {name} chars={r['chars']} distinct={r['distinct_words']} "
          f"rep20={r['max_20char_repeats']} printable={r['printable_ratio']}")
    return r["healthy"]


if __name__ == "__main__":
    data = open(sys.argv[1]).read()
    ok = report(sys.argv[1], data)
    sys.exit(0 if ok else 1)
