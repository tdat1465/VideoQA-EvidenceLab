"""Public-text event queries and bounded evidence coverage; no model/label inputs."""
from __future__ import annotations

import math
import re

VARIANTS = ('sigma2', 'sigma1', 'time16_sigma1', 'eventmean_sigma1', 'eventcover_sigma1')
PROTOTYPE = {
    'schema': 'sequence-event-coverage-v1',
    'gate': 'public-question-sequence-or-order-and-all-options-explicit-first-finally',
    'dedup': 'exact-normalized-text-union-sorted-no-option-order-or-frequency',
    'max_events': 12, 'content_tokens_per_chunk': 33, 'total_tokens_max': 35,
    'support_raw_max': .5, 'support_raw_span': .1,
    'support_radius': 1, 'support_normalized_score': .5, 'support_min_samples': 2,
    'response_sigma': 1., 'response_floor': .5,
    'time_anchors': 16, 'event_frame_budget': 16, 'max_event_centers': 8,
    'center_min_gap': 2, 'context_radius': 1,
    'eventmean_question_weight': .5,
    'fallback': 'sigma1-if-parser-ineligible-or-no-supported-events',
    'pilot_questions': 20, 'pilot_seed': 20261022,
}


def normalized_event(text):
    # Presentation punctuation only. No semantic/paraphrase merges or substring
    # matching, which could conflate "desert" and "outer space" scenes.
    text = re.sub(r'[^\w\s-]', ' ', text.lower(), flags=re.UNICODE)
    return re.sub(r'\s+', ' ', text).strip()


def event_descriptions(question):
    """Discard every option's order after extracting explicit sequence clauses.

    Only Question.question/choices are accessed. Unsupported syntax fails closed
    to sigma1, without silently treating a complete option as a single event.
    """
    text = question.question.lower()
    empty = lambda reason: {'eligible': False, 'reason': reason, 'events': []}
    if not (re.search(r'\b(?:sequences?|orders?)\b', text)
            and re.search(r'\b(?:scenes?|events?|scenarios?)\b', text)):
        return empty('question-is-not-explicit-scene-event-order')
    if not 2 <= len(question.choices) <= 5:
        raise ValueError('Require 2-5 public choices')
    events = set()
    for choice in question.choices:
        first = re.match(r'^\s*first\b\s*[,：:]?\s*', choice, re.I)
        if first is None:
            return empty('option-without-leading-first')
        body = choice[first.end():]
        markers = list(re.finditer(
            r'[.;,]\s*(?:and\s+)?(next|then|followed\s+by|finally|lastly)\b\s*[,：:]?\s*',
            body, re.I))
        if not 1 <= len(markers) <= 4:
            return empty('ambiguous-number-of-sequence-clauses')
        names = [re.sub(r'\s+', ' ', m.group(1).lower()) for m in markers]
        if names[-1] not in ('finally', 'lastly') or any(x in ('finally', 'lastly') for x in names[:-1]):
            return empty('option-without-unambiguous-final-clause')
        starts = [0] + [m.end() for m in markers]
        ends = [m.start() for m in markers] + [len(body)]
        clauses = [normalized_event(body[a:b]) for a, b in zip(starts, ends)]
        if any(len(c.split()) < 3 for c in clauses):
            return empty('event-description-too-short')
        events.update(clauses)
    if not 2 <= len(events) <= PROTOTYPE['max_events']:
        return empty('event-union-outside-2-12-budget')
    return {'eligible': True, 'reason': 'explicit-sequence-clauses', 'events': sorted(events)}


def event_plan(question, tokenizer):
    parsed = event_descriptions(question)
    queries = []
    for text in parsed['events']:
        ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        if not ids:
            raise ValueError('Empty event token sequence')
        chunks = [ids[i:i+33] for i in range(0, len(ids), 33)]
        if [v for chunk in chunks for v in chunk] != ids:
            raise AssertionError('Lost event tokens')
        queries.append({'text': text, 'content_ids': chunks, 'token_count': len(ids)})
    return {**parsed, 'queries': queries}


def validate_vectors(question_scores, per_event, plan):
    n = len(question_scores)
    if n < 64 or len(per_event) != len(plan['queries']):
        raise ValueError('Invalid event cache shape')
    if plan['eligible'] != bool(plan['queries']) or any(len(x) != n for x in per_event):
        raise ValueError('Incomplete event score vectors')
    if any(not math.isfinite(x) or not 0 <= x <= 1 for v in [question_scores, *per_event] for x in v):
        raise ValueError('Invalid BLIP probability')


def event_evidence(question_scores, per_event, plan):
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    validate_vectors(question_scores, per_event, plan)
    curves, support = [], []
    for query, raw in zip(plan['queries'], per_event):
        values = np.asarray(raw, dtype=np.float64)
        span = float(np.ptp(values))
        peak = int(np.argmax(values))
        normalized = (values-values.min())/span if span > 1e-12 else np.zeros_like(values)
        radius = PROTOTYPE['support_radius']
        local = normalized[max(0, peak-radius):min(len(values), peak+radius+1)]
        count = int(np.sum(local >= PROTOTYPE['support_normalized_score']))
        accepted = bool(values.max() >= PROTOTYPE['support_raw_max']
                        and span >= PROTOTYPE['support_raw_span']
                        and count >= PROTOTYPE['support_min_samples'])
        support.append({'text': query['text'], 'accepted': accepted, 'raw_max': float(values.max()),
                        'raw_span': span, 'peak_position': peak, 'local_support': count})
        if accepted:
            smooth = gaussian_filter1d(values, sigma=PROTOTYPE['response_sigma'])
            width = float(np.ptp(smooth))
            response = (smooth-smooth.min())/width if width > 1e-12 else np.zeros_like(smooth)
            floor = PROTOTYPE['response_floor']
            curves.append(np.maximum(0., (response-floor)/(1.-floor)))
    matrix = np.asarray(curves, dtype=np.float64).reshape(len(curves), len(question_scores))
    return matrix, support


def fill_with_base(required, base, ranking, k=64):
    chosen = set(required)
    if len(chosen) > k:
        raise ValueError('Required frames exceed budget')
    for i in sorted(base, key=lambda p: (-float(ranking[p]), p)):
        if len(chosen) == k:
            break
        chosen.add(i)
    if len(chosen) != k:
        raise ValueError('Cannot fill frame budget from baseline')
    return sorted(chosen)


def select_variant(author, score, weights, variant, k=64):
    import numpy as np
    import ridge_component_ablation as frozen
    if variant not in VARIANTS or k != 64:
        raise ValueError('Frozen 64-frame event experiment')
    raw = score['question_scores']
    base, details = frozen.select_variant(author, raw, weights, 'sigma2' if variant == 'sigma2' else 'sigma1', k)
    if variant in ('sigma2', 'sigma1'):
        return base, details
    responses, support = event_evidence(raw, score['per_event_scores'], score['query_plan'])
    metadata = {'prototype': PROTOTYPE, 'parser_eligible': score['query_plan']['eligible'],
                'parser_reason': score['query_plan']['reason'], 'event_support': support,
                'accepted_events': len(responses), 'fallback': not bool(len(responses))}
    if not len(responses):
        return base, {**details, **metadata}
    if variant == 'eventmean_sigma1':
        # Explicit sequential addition is identical on Python 3.10 and 3.12.
        vectors = [v for v, s in zip(score['per_event_scores'], support) if s['accepted']]
        mean = [0.]*len(raw)
        for values in vectors:
            mean = [a+b for a, b in zip(mean, values)]
        alpha = PROTOTYPE['eventmean_question_weight']
        blended = [alpha*q+(1.-alpha)*m/len(vectors) for q, m in zip(raw, mean)]
        chosen, info = frozen.select_variant(author, blended, weights, 'sigma1', k)
        return chosen, {**info, **metadata}
    anchors = [int(i) for i in np.linspace(0, len(raw)-1, PROTOTYPE['time_anchors'], dtype=int)]
    required, centers, contexts = set(anchors), [], []
    covered = responses[:, anchors].max(axis=1)
    before = covered.tolist()
    if variant == 'eventcover_sigma1':
        for _ in range(min(len(responses), PROTOTYPE['max_event_centers'])):
            gains = np.maximum(0., responses-covered[:, None]).mean(axis=0)
            for i in required:
                gains[i] = -1.
            gap = PROTOTYPE['center_min_gap']
            for p in centers:
                gains[max(0, p-gap+1):min(len(raw), p+gap)] = -1.
            peak = int(np.argmax(gains))
            if gains[peak] <= 1e-12:
                break
            required.add(peak)
            centers.append(peak)
            covered = np.maximum(covered, responses[:, peak])
        # At most 16 event/context additions beyond the same 16 time anchors.
        radius = PROTOTYPE['context_radius']
        for offset in [i for i in range(-radius, radius+1) if i]:
            for peak in centers:
                if len(required)-len(anchors) >= PROTOTYPE['event_frame_budget']:
                    break
                p = peak+offset
                if 0 <= p < len(raw) and p not in required:
                    required.add(p)
                    contexts.append(p)
    chosen = fill_with_base(required, base, details['curve']['s_smooth'], k)
    metadata.update(time_anchors=anchors, event_centers=centers, context_positions=contexts,
                    event_additions=len(required)-len(anchors), retained_sigma1_frames=len(set(base)&set(chosen)),
                    event_coverage_at_anchors=before, event_coverage_after=responses[:, chosen].max(axis=1).tolist())
    return chosen, {**details, **metadata}
