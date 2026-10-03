import heapq
import json
import numpy as np
import argparse
import os
from dataclasses import asdict

from watershed_selector import WatershedConfig, select_leaves

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description='AKS frame selection (original or watershed)')

    parser.add_argument('--dataset_name', type=str, default='longvideobench', help='support longvideobench and videomme')
    parser.add_argument('--extract_feature_model', type=str, default='blip', help='blip/clip/sevila')
    parser.add_argument('--score_path', type=str, default='./outscores/longvideobench/blip/scores.json')
    parser.add_argument('--frame_path', type=str, default='./outscores/longvideobench/blip/frames.json')
    parser.add_argument('--max_num_frames', type=int, default=64)
    parser.add_argument('--ratio', type=int, default=1)
    parser.add_argument('--t1', type=float, default=0.8)
    parser.add_argument('--t2', type=float, default=-100)
    parser.add_argument('--all_depth', type=int, default=5)
    parser.add_argument('--output_file', type=str, default='./selected_frames')
    parser.add_argument('--selector', choices=('original', 'watershed'), default='original')
    parser.add_argument('--watershed_sigma', type=float, default=1.0)
    parser.add_argument('--watershed_min_prominence', type=float, default=0.05)
    parser.add_argument('--watershed_min_distance', type=int, default=3,
                        help='minimum gap in candidate ticks AFTER --ratio, including across leaves')
    parser.add_argument('--watershed_prominence_weight', type=float, default=0.25)
    parser.add_argument('--diagnostics_file', help='optional JSON sidecar; selected_frames.json stays unchanged')

    return parser.parse_args(argv)

def meanstd(len_scores, dic_scores, n, fns,t1,t2,all_depth):
        split_scores = []
        split_fn = []
        no_split_scores = []
        no_split_fn = []
        i= 0
        for dic_score, fn in zip(dic_scores, fns):
                # normalized_data = (score - np.min(score)) / (np.max(score) - np.min(score))
                score = dic_score['score']
                depth = dic_score['depth']
                # Empty children contribute no selected frames in upstream.
                # Skip their NaN statistics/recursive expansion without changing
                # the stopping depth or quota of any nonempty sibling.
                if len(score) == 0:
                        continue
                mean = np.mean(score)
                std = np.std(score)

                top_n = heapq.nlargest(n, range(len(score)), score.__getitem__)
                top_score = [score[t] for t in top_n]
                # print(f"split {i}: ",len(score))
                i += 1
                mean_diff = np.mean(top_score) - mean
                if mean_diff > t1 and std > t2:
                        no_split_scores.append(dic_score)
                        no_split_fn.append(fn)
                elif depth < all_depth:
                # elif len(score)>(len_scores/n)*2 and len(score) >= 8:
                        score1 = score[:len(score)//2]
                        score2 = score[len(score)//2:]
                        fn1 = fn[:len(score)//2]
                        fn2 = fn[len(score)//2:]                       
                        split_scores.append(dict(score=score1,depth=depth+1))
                        split_scores.append(dict(score=score2,depth=depth+1))
                        split_fn.append(fn1)
                        split_fn.append(fn2)
                else:
                        no_split_scores.append(dic_score)
                        no_split_fn.append(fn)
        if len(split_scores) > 0:
                all_split_score, all_split_fn = meanstd(len_scores, split_scores, n, split_fn,t1,t2,all_depth)
        else:
                all_split_score = []
                all_split_fn = []
        all_split_score = no_split_scores + all_split_score
        all_split_fn = no_split_fn + all_split_fn


        return all_split_score, all_split_fn

def select_frames(scores, frame_ids, max_num_frames=64, ratio=1, t1=0.8,
                  t2=-100, all_depth=5, selector='original', watershed_config=None):
    """Select one query's original video-frame IDs, plus diagnostics.

    The original judge uses global K even inside child bins (as upstream does).
    Only terminal-leaf selection depends on ``selector``. Quota rounding and
    ratio's floor-and-stride behavior deliberately match upstream.
    """
    for name, value, minimum in (('max_num_frames', max_num_frames, 0),
                                  ('ratio', ratio, 1), ('all_depth', all_depth, 0)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if selector not in ('original', 'watershed'):
        raise ValueError(f'unknown selector: {selector}')
    if not np.isfinite(t1) or not np.isfinite(t2):
        raise ValueError('AKS thresholds must be finite')
    values = np.asarray(scores, dtype=float)
    ids = np.asarray(frame_ids)
    if values.ndim != 1 or ids.ndim != 1 or len(values) != len(ids):
        raise ValueError('scores and frame_ids must be aligned 1D sequences')
    if not np.all(np.isfinite(values)):
        raise ValueError('relevance scores must be finite (NaN/Inf are invalid)')
    # Official VideoMME caches store integer-valued floats (e.g. 29.0).
    # Accept those IDs, rejecting fractions and invalid values before decoding.
    if len(ids) and (ids.dtype.kind not in 'iuf' or not np.all(np.isfinite(ids))
                     or np.any(ids < 0) or np.any(ids >= 2**63)
                     or np.any(ids != np.floor(ids)) or np.any(ids[1:] <= ids[:-1])):
        raise ValueError('frame_ids must be unique, increasing, nonnegative integer-valued numbers')
    ids = ids.astype(np.int64)
    config = watershed_config or WatershedConfig()
    nums = len(values) // ratio
    sampled = np.arange(nums, dtype=int) * ratio
    values = values[sampled]
    ids = ids[sampled].tolist()
    info = {'selector': selector, 'budget': max_num_frames, 'candidate_count': nums,
            'leaf_plan': [], 'distance_relaxations': 0}
    if selector == 'watershed':
        info['watershed_config'] = asdict(config)
    if max_num_frames == 0 or nums == 0:
        info.update(selected_count=0, allocated_capacity=0, short_circuit='empty_or_zero_budget')
        return [], info
    if nums < max_num_frames:
        # This is AKS's original short-video path, not a new allocation policy.
        info.update(selected_count=nums, allocated_capacity=nums, short_circuit='short_video')
        return ids, info

    span = np.max(values) - np.min(values)
    normalized = (values - np.min(values)) / span if span > 0 else np.zeros(nums)
    # Use candidate positions to retain exact interval boundaries for NMS/audit.
    # On a flat curve upstream's division-by-zero makes every judge comparison
    # false. Force the same split decisions with finite scores for the selectors.
    judge_t1 = t1 if span > 0 else float('inf')
    nodes, positions = meanstd(nums, [dict(score=normalized, depth=0)], max_num_frames,
                               [list(range(nums))], judge_t1, t2, all_depth)
    leaves = []
    for node, pos in zip(nodes, positions):
        quota = int(max_num_frames / 2**node['depth'])
        leaves.append({'scores': node['score'], 'positions': pos, 'quota': quota})
        info['leaf_plan'].append({'start': pos[0], 'end': pos[-1], 'depth': node['depth'],
                                  'quota': quota, 'capacity': min(quota, len(pos))})
    info['allocated_capacity'] = sum(leaf['capacity'] for leaf in info['leaf_plan'])
    if selector == 'original':
        chosen = []
        for leaf in leaves:
            topk = heapq.nlargest(leaf['quota'], range(len(leaf['scores'])), leaf['scores'].__getitem__)
            chosen.extend(leaf['positions'][i] for i in topk)
        chosen.sort()
    else:
        chosen, details = select_leaves(leaves, config)
        info.update(details)
    result = [ids[i] for i in chosen]
    info['selected_positions'] = chosen
    info['selected_count'] = len(result)
    if len(result) > max_num_frames or len(set(result)) != len(result):
        raise RuntimeError('selector violated the unique frame budget')
    return result, info


def main(args):
    with open(args.score_path, encoding='utf-8') as f:
        itm_outs = json.load(f)
    with open(args.frame_path, encoding='utf-8') as f:
        fn_outs = json.load(f)
    if not isinstance(itm_outs, list) or not isinstance(fn_outs, list) or len(itm_outs) != len(fn_outs):
        raise ValueError('score/frame JSON must be lists with the same number of queries')
    config = WatershedConfig(args.watershed_sigma, args.watershed_min_prominence,
                             args.watershed_min_distance, args.watershed_prominence_weight)
    outs, diagnostics = [], []
    for query_index, (scores, frames) in enumerate(zip(itm_outs, fn_outs)):
        try:
            out, info = select_frames(scores, frames, args.max_num_frames, args.ratio,
                                     args.t1, args.t2, args.all_depth, args.selector, config)
        except ValueError as error:
            raise ValueError(f'query {query_index}: {error}') from error
        outs.append(out)
        diagnostics.append(info)
    out_score_path = os.path.join(args.output_file, args.dataset_name, args.extract_feature_model)
    os.makedirs(out_score_path, exist_ok=True)
    with open(os.path.join(out_score_path, 'selected_frames.json'), 'w', encoding='utf-8') as f:
        json.dump(outs, f)
    if args.diagnostics_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.diagnostics_file)), exist_ok=True)
        with open(args.diagnostics_file, 'w', encoding='utf-8') as f:
            json.dump(diagnostics, f, indent=2, allow_nan=False)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
