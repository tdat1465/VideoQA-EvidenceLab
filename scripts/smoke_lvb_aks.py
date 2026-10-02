"""Exercise LongVideoBench AKS with verified real upstream source and a synthetic scorer."""
import argparse
import math
from pathlib import Path

from evidencelab.aks_adapter import select_aks
from evidencelab.data import Question
from evidencelab.longvideo_config import LongVideoConfig

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("source", type=Path)
args = p.parse_args()

class Video:
    def __len__(self):
        return 30 * 192

    def get_avg_fps(self):
        return 30.

class Scorer:
    def reset(self):
        self.evaluated, self.calls = [], 0

    def __call__(self, video, query, indices):
        self.evaluated.extend(indices)
        self.calls += 1
        return [.5 + .49 * math.sin(i / 200) for i in indices]

indices, details = select_aks(Video(), Question("test", "What happens?", ("a", "b")),
                             LongVideoConfig(method="aks"), Scorer(), args.source)
assert len(indices) == 64 and indices == sorted(set(indices))
assert all(i % 30 == 0 for i in indices)
assert details['scorer_frame_evaluations'] == 192
assert details['scorer_batches'] == 48
assert details['aks_fallback'] == 'upstream'
print("LongVideoBench AKS: real pinned selector + bounded scoring smoke PASS")
