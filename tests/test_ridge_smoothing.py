import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from evidencelab.config import source_fingerprint
from evidencelab.pipeline import Decision
from evidencelab.ridge_adapter import load_ridge
from evidencelab.store import Store, atomic_json

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("smoothing", REPO/"scripts/ridge_smoothing_ablation.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
WEIGHTS = dict(zip(("peak_similarity", "slope_abs", "rising_slope", "falling_slope",
                    "boundary_change", "context_density"), (8, 6, 5, 7, 4, 3)))


class SmoothingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.author, _ = load_ridge(Path(os.environ["RIDGE_UPSTREAM_DIR"]))

    def test_controls_exactly_match_author_and_all_variants_obey_budget(self):
        rng = np.random.default_rng(18)
        for n in (64, 65, 150, 800):
            for scores in (rng.random(n), np.ones(n)*.4,
                           .2+.6*np.exp(-((np.arange(n)-n//2)/8)**2)):
                for variant in module.VARIANTS:
                    chosen, details = module.select_variant(self.author, scores, WEIGHTS, variant)
                    self.assertEqual(len(chosen), 64)
                    self.assertEqual(chosen, sorted(set(chosen)))
                    self.assertEqual(len(details["labels"]), n)
                    json.dumps(details, allow_nan=False)
                    if variant != "multiscale_v1":
                        self.assertEqual(chosen, self.author.gradselect(scores, 64, WEIGHTS, module.args_for(variant)))

    def test_development_sampling_is_whole_video_and_outcome_independent(self):
        rows = [{"id":f"q{video}-{i}", "video":f"v{video}", "ridge_candidate_count":100,
                 "correct":bool(i%2), "prediction":i} for video in range(200) for i in range(1+video%3)]
        manifest = module.choose_development(rows)
        changed = [{**row, "correct":not row["correct"], "prediction":99} for row in reversed(rows)]
        self.assertEqual(manifest, module.choose_development(changed))
        self.assertEqual(len(manifest["ids"]), 200)
        for row in rows:
            self.assertEqual(row["id"] in manifest["ids"], row["video"] in manifest["videos"])
        with self.assertRaises(ValueError):
            module.choose_development([row for row in rows if row["video"]=="v1"])

    def test_cache_rejects_corruption_and_candidate_or_question_change(self):
        key = {"candidate_indices":list(range(80)), "question_sha256":"q", "video_sha256":"v"}
        record = module.score_record(key, [.5]*80, 2., 20)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"cache.json"
            atomic_json(path, record)
            self.assertEqual(module.read_score_cache(path, key), record)
            for changed in ({**key, "question_sha256":"new"}, {**key, "candidate_indices":list(range(81))}):
                with self.assertRaises(ValueError): module.read_score_cache(path, changed)
            record["scores"][0] = .9
            atomic_json(path, record)
            with self.assertRaises(ValueError): module.read_score_cache(path, key)
        with self.assertRaises(ValueError): module.score_record(key, [float("nan")]*80, 2., 20)

    def test_partial_variant_resume_preserves_answers_and_no_rescoring(self):
        sample = SimpleNamespace(id="q", video="v.mp4", choices=("a","b"), answer=1,
                                 question_type="TAA", duration_group="600")
        scores = np.random.default_rng(1).random(150).tolist()
        key = {"candidate_indices":list(range(150))}
        record = module.score_record(key, scores, 3., 38)
        weight_record = {"weights":WEIGHTS, "record_sha256":"w", "generation_seconds":4.}
        calls = []
        def answer(indices, times):
            calls.append(indices)
            return Decision([.1,.9],100), 1., {}
        contract = {"selected_count":5, "manifest":{"ids":["q"]}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path/"scores").mkdir()
            atomic_json(path/"scores/q.json", record)
            store = Store(path, contract, False)
            with self.assertRaises(InterruptedError):
                module.evaluate_question(sample,list(range(150)),1.,weight_record,record,self.author,
                    store,path/"curve.json",answer,stop=lambda:len(calls)>=2)
            before = store.results()
            self.assertEqual(len(before),2)
            store.close()
            store = Store(path, contract, True)
            module.evaluate_question(sample,list(range(150)),1.,weight_record,record,self.author,
                                     store,path/"curve.json",answer)
            self.assertEqual(len(calls),5)
            self.assertEqual(len(store.results()),5)
            self.assertTrue(all(v in store.results() for v in before))
            summary = module.export(path,contract,store)
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["one_time_scoring_seconds"],3.)
            self.assertEqual(summary["one_time_scored_questions"],1)
            self.assertTrue(all(v["n"]==1 for v in summary["variants"].values()))
            store.close()

    def test_baseline_source_unchanged_and_bad_inputs_fail_closed(self):
        self.assertEqual(source_fingerprint(),module.BASE_SOURCE)
        with self.assertRaises(ValueError): module.select_variant(self.author,[.1]*63,WEIGHTS,"sigma2")
        with self.assertRaises(ValueError): module.select_variant(self.author,[.1]*100,WEIGHTS,"unknown")

    def test_supported_fine_peak_rescue_on_synthetic_shoulder_not_flat_noise(self):
        x = .01 + np.exp(-((np.arange(300)-150)/25)**2/2)
        x[130:132] += .4
        selected, details = module.select_variant(self.author,x,WEIGHTS,"multiscale_v1")
        self.assertIn(131,details["rescued_peaks"])
        self.assertTrue(any(abs(i-131)<=1 for i in selected))
        _, flat = module.select_variant(self.author,[.4]*300,WEIGHTS,"multiscale_v1")
        self.assertEqual(flat["rescued_peaks"],[])

    def test_mock_gpu_workflow_pilot_resume_reuses_preferences_and_shared_cache(self):
        from evidencelab.data import digest, file_hash
        from evidencelab.longvideo_config import LongVideoConfig
        from evidencelab.longvideo_data import parse_annotations
        from evidencelab.ridge_weights import question_hash, weight_contract
        _, weight_author = load_ridge(Path(os.environ["RIDGE_UPSTREAM_DIR"]))
        runtime = {"fake_runtime":"test"}
        execution = {"resolved_dtype":"bfloat16", "gpu":"fake", "node":"fake"}
        counters = {"scored":0, "answers":0, "models":0}
        class Video:
            def __len__(self): return 90
            def get_avg_fps(self): return 1.
            def __getitem__(self,index): return SimpleNamespace(asnumpy=lambda:np.zeros((16,16,3),dtype=np.uint8))
        class Scorer:
            def __init__(self,config): self.reset()
            def reset(self): self.calls=0
            def __call__(self,video,query,indices):
                self.calls+=1
                counters["scored"]+=len(indices)
                return [i/90 for i in indices]
        class Backend:
            def __init__(self,config):
                counters["models"]+=1
                self.torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:None))
            def reset_peak(self): pass
            def measurements(self): return {}
            def answer(self,question,frames,times):
                self_outer=counters
                self_outer["answers"]+=1
                return Decision([.2,.8],100)
        class Reader:
            def __init__(self,source): pass
            def clear_cache(self): pass
        def fetch(reader,entry,path,stop):
            path.write_bytes(b"fake")
            return "a"*64
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            baseline,output,ram=root/"baseline",root/"ablation",root/"ram"
            baseline.mkdir(); ram.mkdir()
            annotations=[{"id":str(i),"video_path":f"v{i//2}.mp4","question":f"question {i}",
                          "candidates":["a","b"],"correct_choice":1,"question_category":"TAA",
                          "duration_group":"600","duration":90.} for i in range(1337)]
            atomic_json(root/"annotations.json",annotations)
            samples=parse_annotations(annotations)
            config=LongVideoConfig.load(REPO/"configs/lvb_molmo2_ridge64_full.json")
            contract={"config":config.contract(),"source_sha256":module.BASE_SOURCE,
                      "resolved_dtype":"bfloat16","runtime":runtime,
                      "selected_ids_hash":digest(sorted(s.id for s in samples)),
                      "annotation_sha256":file_hash(root/"annotations.json")}
            atomic_json(baseline/"contract.json",contract)
            atomic_json(baseline/"summary.json",{"complete":True,"completed_count":1337})
            records={}
            positions=self.author.gradselect([i/90 for i in range(90)],64,WEIGHTS,module.args_for("sigma2"))
            rows=[]
            for q in samples:
                rec={"question_id":q.id,"question_sha256":question_hash(q.public_question()),
                     "weights":WEIGHTS,"input_tokens":32,"output_tokens":64,
                     "generation_seconds":1.,"response_sha256":"f"*64}
                rec["record_sha256"]=digest(rec);records[q.id]=rec
                rows.append({"id":q.id,"video":q.video,"ridge_candidate_count":90,
                             "ridge_weight_record_sha256":rec["record_sha256"],"video_sha256":"a"*64,
                             "calls":[{"frame_indices":positions}],"prediction":1})
            (baseline/"predictions.jsonl").write_text("\n".join(json.dumps(row) for row in rows)+"\n")
            atomic_json(baseline/"ridge-weights.json",{"contract":weight_contract(config,contract,weight_author),"records":records})
            before={p.name:file_hash(p) for p in baseline.iterdir()}
            args=SimpleNamespace(config=REPO/"configs/lvb_molmo2_ridge64_full.json",annotations=root/"annotations.json",
                                 baseline=baseline,output=output,ram_root=ram,index=root/"index.json",
                                 max_new_questions=2,max_seconds=60)
            decord=SimpleNamespace(VideoReader=lambda *a,**k:Video(),cpu=lambda x:None)
            patches=[patch.dict("sys.modules",{"decord":decord}),
                     patch("evidencelab.longvideo.validate_ram_root"),
                     patch("evidencelab.hardware.execution_profile",return_value=execution),
                     patch("evidencelab.longvideo.longvideo_runtime",return_value=runtime),
                     patch("evidencelab.backends.HFBackend",Backend),
                     patch("evidencelab.focus_adapter.BlipITMScorer",Scorer),
                     patch("evidencelab.ridge_adapter.fetch_ridge",return_value=Path(os.environ["RIDGE_UPSTREAM_DIR"])),
                     patch("evidencelab.longvideo_data.HTTPRangeSource",return_value=SimpleNamespace()),
                     patch("evidencelab.longvideo_data.MultipartReader",Reader),
                     patch("evidencelab.longvideo_data.archive_index",return_value={"videos/"+q.video:[0,1] for q in samples}),
                     patch("evidencelab.longvideo_data.fetch_video",side_effect=fetch),
                     patch("evidencelab.longvideo_data.check_space")]
            from contextlib import ExitStack
            with ExitStack() as stack:
                for item in patches: stack.enter_context(item)
                self.assertEqual(module.run(args),75)
                first=module.load(output/"summary.json")
                self.assertEqual(first["completed_variant_answers"],10)
                self.assertEqual(first["baseline_reproduction"],{"n":2,"same_frames":2,"same_prediction":2})
                self.assertEqual(module.run(args),75)
            final=module.load(output/"summary.json")
            self.assertEqual(final["completed_variant_answers"],20)
            self.assertEqual(counters,{"scored":360,"answers":20,"models":2})
            self.assertEqual(len(list((output/"scores").glob("*.json"))),4)
            self.assertEqual(before,{p.name:file_hash(p) for p in baseline.iterdir()})


if __name__ == "__main__":
    unittest.main()
