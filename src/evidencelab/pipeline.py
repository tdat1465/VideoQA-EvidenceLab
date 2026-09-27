from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .config import Config
from .data import Question
from .selectors import aks, evidence_frames, expand_uniform, topk, uniform


@dataclass
class Decision:
    scores: list[float]
    input_tokens: int

    @property
    def prediction(self):
        return max(range(len(self.scores)), key=self.scores.__getitem__)

    @property
    def margin(self):
        scores = sorted(self.scores, reverse=True)
        return scores[0] - scores[1]


def infer(question: Question, frames, times: list[float], config: Config, backend,
          scorer=None, aks_file: Path | None = None) -> dict:
    """At most two answerer calls; decisions never receive Sample.answer."""
    method, calls = config.method, []
    n = len(times)
    trace = {"method": method, "candidate_frames": n, "selector_queries": 0, "anchors": []}

    def answer(ids):
        decision = backend.answer(question, [frames[i] for i in ids], [times[i] for i in ids])
        import math
        if (len(decision.scores) != len(question.choices) or
            any(not math.isfinite(s) or not 0 <= s <= 1 for s in decision.scores) or
            abs(sum(decision.scores) - 1) > 1e-4):
            raise ValueError("Backend must return normalized finite option probabilities")
        calls.append({**asdict(decision), "frame_indices": ids, "timestamps": [times[i] for i in ids]})
        return decision

    if method in {"uniform", "evidence", "uniform_refine"}:
        ids = uniform(n, config.initial_frames if method != "uniform" else config.frames)
    else:
        if scorer is None:
            raise ValueError("This method requires a scorer")
        scores = scorer.score(frames, [question.question])[0]
        trace["selector_queries"] += 1
        if method == "clip_topk":
            ids = topk(scores, config.frames)
        else:
            if aks_file is None:
                raise ValueError("Set --aks-file to the pinned upstream frame_select.py")
            ids, trace["aks_status"] = aks(scores, config.frames, aks_file,
                                          config.aks_t1, config.aks_t2, config.aks_depth)
    decision = answer(ids)
    trace["initial_margin"] = decision.margin
    trace["refined"] = False
    if (method in {"evidence", "uniform_refine"} and decision.margin < config.margin_threshold
            and len(ids) < min(config.frames, n)):
        if method == "uniform_refine":
            ids = expand_uniform(n, ids, config.frames)
        else:
            if scorer is None:
                raise ValueError("Evidence refinement requires a scorer")
            competitors = tuple(sorted(range(len(decision.scores)),
                                       key=lambda i: (-decision.scores[i], i))[:2])
            # All choices are public; no correct-answer annotation enters scoring.
            scores = scorer.score(frames, [question.question] +
                                  [f"{question.question} {choice}" for choice in question.choices])
            trace["selector_queries"] += 1 + len(question.choices)
            ids, trace["anchors"] = evidence_frames(times, ids, config.frames, scores[0],
                                                    scores[1:], competitors,
                                                    config.neighbor_seconds, config.relevance_weight)
            trace["competitors"] = list(competitors)
        decision = answer(ids)
        trace["refined"] = True
    trace.update(prediction=decision.prediction, scores=decision.scores, calls=calls,
                 input_tokens=sum(call["input_tokens"] for call in calls),
                 frame_presentations=sum(len(call["frame_indices"]) for call in calls),
                 final_frames=len(ids))
    return trace
