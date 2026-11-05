"""Frozen subtitle-anchor selector used in the 2026-11 development run.

This module only consumes public questions, choices, subtitles, and an existing
event-coverage frame selection. It must never receive a correct-answer label.
"""
import re
from difflib import SequenceMatcher


def seconds(value):
    if isinstance(value, str):
        return sum(float(x) * 60**i for i, x in enumerate(reversed(value.replace(',', '.').split(':'))))
    return float(value)


def tokens(text):
    return re.findall(r'\w+', text.casefold())


def windows(rows):
    cues = []
    for row in rows:
        start, end = row.get('timestamp') or (row.get('start'), row.get('end'))
        if start is not None and end is not None:
            cues.append((seconds(start), seconds(end), tokens(row.get('text') or row.get('line') or '')))
    cues.sort()
    for index, (start, _, _) in enumerate(cues):
        words = []
        for _, end, text in cues[index:index + 4]:
            if end - start > 20:
                break
            words.extend(text)
            if words:
                yield start, end, words.copy()


def candidates(query, spans):
    query = tokens(query)
    if len(query) < 3:
        return []
    ranked = []
    for start, end, words in spans:
        matcher = SequenceMatcher(None, query, words, autojunk=False)
        coverage = sum(block.size for block in matcher.get_matching_blocks()) / len(query)
        if coverage >= .7:
            ranked.append((coverage, matcher.ratio(), start, end))
    keep = []
    for row in sorted(ranked, reverse=True):
        if all(abs(row[2] - old[2]) >= 5 for old in keep):
            keep.append(row)
            if len(keep) == 3:
                break
    return keep


def queries(question, choices, kind):
    """Reproduce the full-run query rules without accessing the answer label."""
    if kind == 'TOS':
        return list(choices)
    if kind.startswith('T'):
        return [a or b for a, b in re.findall(r"'([^']{8,})'|\"([^\"]{8,})\"", question)]
    return []


def select(base, count, fps, record, limit):
    """Insert up to 16 subtitle-guided frames, then fill from the base selection."""
    offset = float(record['offset'] or 0)
    anchors = []
    for matches in record['matches']:
        possible = []
        for _, _, start, end in matches['candidates']:
            t = (start + end) / 2 - offset
            if 0 <= t < count / fps:
                possible.append(t)
        anchors.append(possible[:limit])
    times = list(dict.fromkeys(t for rank in range(limit) for group in anchors for t in group[rank:rank + 1]))
    targeted = list(dict.fromkeys(
        max(0, min(count - 1, round((t + shift) * fps)))
        for shift in (0, 1, -1, 2, -2, 4, -4, 8, -8)
        for t in times
    ))[:16]
    return sorted((targeted + [i for i in base if i not in targeted])[:len(base)])
