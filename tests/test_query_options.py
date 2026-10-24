import contextlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'scripts'),str(REPO/'src')]
import ridge_query_options_ablation as mod
from evidencelab.store import Store,atomic_json
from evidencelab.pipeline import Decision

class Tokenizer:
    def encode(self,text,**kwargs):
        assert kwargs=={'add_special_tokens':False,'truncation':False}
        return [int(x) for x in text.split()]
    def build_inputs_with_special_tokens(self,ids):return [101,*ids,102]

class Tensor:
    def __init__(self,v):self.v=np.array(v)
    def to(self,*args,**kwargs):return self
    def float(self):return self
    def cpu(self):return self
    def tolist(self):return self.v.tolist()
    def __getitem__(self,key):return Tensor(self.v[key])

class Torch:
    long=np.int64
    tensor=staticmethod(lambda v,**kwargs:Tensor(v))
    ones=staticmethod(lambda shape,**kwargs:Tensor(np.ones(shape)))
    stack=staticmethod(lambda v:Tensor(v))
    inference_mode=staticmethod(contextlib.nullcontext)
    @staticmethod
    def softmax(v,dim):
        exp=np.exp(v.v-v.v.max(axis=dim,keepdims=True))
        return Tensor(exp/exp.sum(axis=dim,keepdims=True))

class QueryTests(unittest.TestCase):
    def test_all_tokens_preserved_options_equal_and_order_independent(self):
        q=SimpleNamespace(choices=(' '.join(str(i) for i in range(1,101)),'201 202'))
        plan=mod.option_plan(q,Tokenizer())
        self.assertEqual([p['token_count'] for p in plan],[100,2])
        self.assertEqual([len(c) for c in plan[0]['content_ids']],[33,33,33,1])
        self.assertEqual([x for c in plan[0]['content_ids'] for x in c],list(range(1,101)))
        q.choices=tuple(reversed(q.choices))
        self.assertEqual(mod.option_plan(q,Tokenizer()),plan)
        self.assertEqual(mod.combine_scores([.2,.6],[[.9,.3],[.1,.7]]),[.35,.55])
        with self.assertRaises(ValueError):mod.combine_scores([.2],[[float('nan')],[.5]])

    def test_bounded_inference_scores_every_chunk_without_materializing_video(self):
        plan=mod.option_plan(SimpleNamespace(choices=(' '.join(str(i) for i in range(1,70)),'90 91')),Tokenizer())
        transforms=[];forward=[]
        def model(**kwargs):
            ids=kwargs['input_ids'].v
            self.assertTrue((ids[:,0]==30523).all());self.assertLessEqual(ids.shape[1],35)
            forward.append(ids.shape)
            score=ids[:,1:-1].mean(axis=1)/100
            return SimpleNamespace(itm_score=Tensor(np.stack([score*0,score],axis=1)))
        scorer=SimpleNamespace(torch=Torch(),processor=SimpleNamespace(tokenizer=Tokenizer()),enc_id=30523,
            model=model,transform=lambda im:(transforms.append(im.size) or np.zeros((3,2,2))))
        def reset():scorer.evaluated=[];scorer.calls=0
        scorer.reset=reset
        class Video:
            def __getitem__(self,index):return SimpleNamespace(asnumpy=lambda:np.zeros((2,2,3),dtype=np.uint8))
        result=mod.score_options(scorer,Video(),[1,2,3,4,5],plan,2)
        self.assertEqual(len(transforms),5)
        self.assertEqual(scorer.calls,12) # Three frame batches times four text chunks.
        self.assertEqual([len(x) for x in result],[5,5])
        with self.assertRaises(InterruptedError):
            mod.score_options(scorer,Video(),[1,2,3],plan,2,stop=lambda:len(scorer.evaluated)>0)

    def test_query_cache_rejects_changed_choices_scores_and_corruption(self):
        key={'candidate_indices':[0,1],'query':'old'};plan=[{'text':'one'},{'text':'two'}]
        score=mod.query_score(key,[.1,.2],[[.3,.4],[.5,.6]],plan,1.,2)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'score.json';atomic_json(path,score)
            self.assertEqual(mod.read_query_score(path,key,[.1,.2],plan),score)
            with self.assertRaises(ValueError):mod.read_query_score(path,{**key,'query':'changed'},[.1,.2],plan)
            with self.assertRaises(ValueError):mod.read_query_score(path,key,[.2,.1],plan)
            with self.assertRaises(ValueError):mod.read_query_score(path,key,[.1,.2],[{'text':'changed'}])
            bad={**score,'scores':[.9,.9]};atomic_json(path,bad)
            with self.assertRaises(ValueError):mod.read_query_score(path,key,[.1,.2],plan)

    def test_transaction_resume_does_not_rerun_committed_control_or_modify_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'curves').mkdir();(root/'scores').mkdir()
            contract={'selected_count':2,'pilot20_manifest':{'ids':['q']}}
            sample=SimpleNamespace(id='q',question='demo',choices=('one','two'),answer=1,video='v',question_type='SSS',duration_group='600')
            base=list(range(64));modified=base[:-1]+[70]
            weight={'weights':{'weight':1},'record_sha256':'unchanged'}
            score={'sha256':'new','question_scores':[.1]*100,'scores':[.9]*100}
            old=json.dumps(score,sort_keys=True);calls=[]
            ref={'calls':[{'frame_indices':base}],'prediction':1}
            def select(author,values,weights,variant,k=64):return (base if values==score['question_scores'] else modified),{}
            def answer(indices,times):calls.append(indices);return Decision([.1,.9],100),.1,{}
            with patch.object(mod.frozen,'select_variant',side_effect=select):
                store=Store(root,contract,False)
                with self.assertRaises(InterruptedError):
                    mod.evaluate_question(sample,list(range(100)),1.,weight,score,None,store,root/'curves/q.json',answer,
                        stop=lambda:len(calls)==1,reference=ref)
                self.assertEqual(len(store.results()),1);store.close()
                store=Store(root,contract,True)
                mod.evaluate_question(sample,list(range(100)),1.,weight,score,None,store,root/'curves/q.json',answer,reference=ref)
                summary=mod.export(root,contract,store);store.close()
            self.assertTrue(summary['complete']);self.assertEqual(len(calls),2)
            self.assertEqual(json.dumps(score,sort_keys=True),old)
            self.assertEqual(summary['new_qwen_calls'],0)

    def test_control_matches_all_200_real_cached_baselines(self):
        root=REPO.parents[1]
        source=root/'outputs/ridge-smoothing-results-20261021-232214/lvb-ridge-smoothing-v1-dev200'
        if not source.exists():self.skipTest('Real local exports are not part of the repository')
        baseline=root/'outputs/ridge-results-20261021-075851/lvb-val-molmo2-ridge64-full-ridge-v1'
        from evidencelab.ridge_adapter import load_ridge
        author,_=load_ridge(root/'research-ridge/upstream-ridge')
        data=mod.validate_source(source,baseline)
        refs={r['id']:r for r in (json.loads(x) for x in (baseline/'predictions.jsonl').read_text().splitlines())}
        weights=mod.load(baseline/'ridge-weights.json')['records']
        for q in data['contract']['manifest']['ids']:
            rec=mod.load(source/'scores'/(mod.digest(q)+'.json'))
            values=rec['scores']
            score={'question_scores':values,'scores':mod.combine_scores(values,[[.1]*len(values),[.9]*len(values)])}
            positions,_=mod.select_variant(author,score,weights[q]['weights'],'sigma2')
            self.assertEqual([rec['key']['candidate_indices'][i] for i in positions],refs[q]['calls'][0]['frame_indices'])

if __name__=='__main__':unittest.main()
