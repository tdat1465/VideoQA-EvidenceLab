"""Frozen parser-dedup and temporal-pair coverage ablation."""
from pathlib import Path
import hashlib
import re
import numpy as np
import events_sigma2 as events

VARIANTS = ('sigma2', 'eventcover_sigma2', 'dedup_sigma2', 'relation_sigma2')


def code_contract():
    return {name + '_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for name, path in [('sigma2_adapter', __file__), ('sigma2_selector', events.__file__)]}


def deduplicate(score):
    """Use one existing BLIP curve per syntactic variant, with no label inputs."""
    plan = score['query_plan']
    groups = {}
    for index, query in enumerate(plan['queries']):
        text = query['text']
        canonical = re.sub(r'^is (?=a scene (?:where|with)\b)', '', text)
        groups.setdefault(canonical, []).append(index)
    chosen = [min(indices, key=lambda i: (plan['queries'][i]['text'] != key, i))
              for key, indices in sorted(groups.items())]
    revised = dict(score)
    revised['query_plan'] = {**plan, 'queries': [plan['queries'][i] for i in chosen]}
    revised['per_event_scores'] = [score['per_event_scores'][i] for i in chosen]
    return revised, len(plan['queries']) - len(chosen)


def pair_coverage(responses, selected, i, j):
    ordered = sorted(selected)
    return max((responses[i, a] * responses[j, b] for a in ordered for b in ordered if a < b), default=0.)


def relation_select(author, score, weights):
    import ridge_component_ablation as frozen
    raw = score['question_scores']
    base, details = frozen.select_variant(author, raw, weights, 'sigma2', 64)
    response, support = events.event_evidence(raw, score['per_event_scores'], score['query_plan'])
    if len(response) < 2:
        chosen, info = events.select_variant(author, score, weights, 'eventcover_sigma2')
        return chosen, {**info, 'relation_fallback': True}
    anchors = np.linspace(0, len(raw)-1, 16, dtype=int).tolist()
    required = set(anchors)
    peaks = response.argmax(axis=1)
    pairs = [(i, j) if peaks[i] < peaks[j] else (j, i)
             for i in range(len(response)) for j in range(i+1, len(response)) if peaks[i] != peaks[j]]
    if not pairs:
        chosen, info = events.select_variant(author, score, weights, 'eventcover_sigma2')
        return chosen, {**info, 'relation_fallback': True}
    centers = []
    for _ in range(min(len(response), 8)):
        covered = response[:, sorted(required)].max(axis=1)
        gains = np.maximum(0., response-covered[:, None]).mean(axis=0)
        relation_gain = np.zeros(len(raw))
        selected = np.array(sorted(required))
        for i, j in pairs:
            old = pair_coverage(response, required, i, j)
            before = np.maximum.accumulate(np.where(np.isin(np.arange(len(raw)), selected), response[i], 0.))
            after = np.maximum.accumulate(np.where(np.isin(np.arange(len(raw)), selected), response[j], 0.)[::-1])[::-1]
            before = np.r_[0., before[:-1]]
            after = np.r_[after[1:], 0.]
            relation_gain += np.maximum(0., np.maximum(response[j]*before, response[i]*after)-old)
        gains = .5*gains + .5*relation_gain/len(pairs)
        for pos in required:
            gains[pos] = -1.
        for pos in centers:
            gains[max(0, pos-1):min(len(raw), pos+2)] = -1.
        peak = int(np.argmax(gains))
        if gains[peak] <= 1e-12:
            break
        required.add(peak)
        centers.append(peak)
    for offset in (-1, 1):
        for peak in centers:
            if len(required)-len(anchors) >= 16:
                break
            pos = peak + offset
            if 0 <= pos < len(raw):
                required.add(pos)
    chosen = events.fill_with_base(required, base, details['curve']['s_smooth'], 64)
    return chosen, {**details, 'relation_fallback': False, 'accepted_events': len(response),
                    'pairs': len(pairs), 'centers': centers, 'event_support': support}


def selected_frames(author, score, weights, variant):
    import ridge_component_ablation as frozen
    candidates = score['key']['candidate_indices']
    if variant not in VARIANTS:
        raise ValueError('Unknown variant')
    if len(candidates) <= 64:
        return list(candidates), {'fallback': True, 'selection_mode': 'all-candidates'}
    if variant in ('sigma2', 'eventcover_sigma2'):
        positions, details = events.select_variant(author, score, weights, variant)
    else:
        revised, removed = deduplicate(score)
        if variant == 'dedup_sigma2':
            positions, details = events.select_variant(author, revised, weights, 'eventcover_sigma2')
        else:
            positions, details = relation_select(author, revised, weights)
        details['deduplicated_queries'] = removed
    return [candidates[p] for p in positions], frozen.jsonable(details)


def configure(canonical):
    canonical.VARIANTS = VARIANTS
    canonical.selected_frames = selected_frames
