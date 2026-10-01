import json
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from evidencelab.data import Question
from evidencelab.lens_adapter import (LensSelector, bounded_ssim, candidate_indices, hyperframe,
                                      lens_query, load_lens, neighbors, parse_ratio)
from evidencelab.longvideo import make_trace
from evidencelab.longvideo_config import LongVideoConfig
from evidencelab.pipeline import Decision


class LensTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.sampling, cls.prompts = load_lens()

    def test_ssim_chunking_matches_unmodified_source(self):
        frames = torch.randint(0, 256, (7, 3, 16, 24)).float()
        expected = self.sampling._compute_ssim_matrix(frames, torch.device('cpu'))
        actual = bounded_ssim(frames, 'cpu', batch_size=2)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(actual.diag(), torch.ones(7))

    def test_allocation_parsing_has_explicit_fallback(self):
        for text in ('nan', 'inf', 'reasoning 0.8', ''):
            self.assertEqual(parse_ratio(text), (.5, True))
        self.assertEqual(parse_ratio(' 0.25\n'), (.25, False))
        self.assertEqual(parse_ratio('-2'), (0., False))
        self.assertEqual(parse_ratio('2'), (1., False))

    def test_candidate_pool_covers_video_without_materializing_pixels(self):
        self.assertEqual(candidate_indices(300, 29.97), list(range(0, 300, 30)))
        self.assertEqual(candidate_indices(3, .5), [0, 1, 2])
        long = candidate_indices(30*15000, 30)
        self.assertEqual(len(long), 9999)
        self.assertEqual((long[0], long[-1]), (0, 30*15000-1))

    def test_hyperframe_contains_four_chronological_quadrants(self):
        images = [Image.new('RGB', (16, 16), (n, n, n)) for n in (10, 60, 120, 240)]
        result = np.asarray(hyperframe(images))
        self.assertEqual(result.shape, (16, 16, 3))
        self.assertEqual([result[y, x, 0] for y, x in ((2,2), (2,12), (12,2), (12,12))],
                         [10, 60, 120, 240])
        self.assertEqual(neighbors(0, 2), [0, 1, 1, 1])
        self.assertEqual(neighbors(5, 6), [2, 3, 4, 5])

    def test_selector_mixed_views_lazy_scoring_and_accounting(self):
        selector = LensSelector.__new__(LensSelector)
        selector.config = LongVideoConfig(method='lens', frames=4, blip_batch_size=3)
        selector.sampling, selector.prompts = self.sampling, self.prompts
        selector.guard = lambda: None
        selector.mask = lambda image, query: Image.new('RGB', image.size, (255, 0, 0))
        batch_lengths = []
        class Inputs(dict):
            def to(self, device):
                return self
        def processor(images, text, **kwargs):
            batch_lengths.append(len(images))
            return Inputs(values=torch.tensor([np.asarray(i)[0,0,0] for i in images]).float())
        selector.processor = processor
        selector.blip = lambda values: SimpleNamespace(itm_score=torch.stack((torch.zeros_like(values), values/255), -1))
        class Video:
            def __len__(self): return 360
            def get_avg_fps(self): return 30.
            def __getitem__(self, i):
                return SimpleNamespace(asnumpy=lambda: np.full((16, 16, 3), int(i)%255, np.uint8))
        texts = []
        def generate(text, **kwargs):
            texts.append(text)
            return {'text': '0.5', 'input_tokens': 80, 'output_tokens': 3}
        question = SimpleNamespace(id='q1', question='What happens?', choices=('choice one', 'choice two'))
        original = bounded_ssim
        with patch('evidencelab.lens_adapter.bounded_ssim',
                   side_effect=lambda frames, device, **kw: original(frames, 'cpu', batch_size=3)):
            indices, frames, details = selector.select(Video(), question, SimpleNamespace(generate_text=generate))
        self.assertEqual(len(frames), 4)
        self.assertEqual(indices, sorted(indices))
        self.assertTrue(all(n <= 3 for n in batch_lengths))
        self.assertEqual(sum(batch_lengths), 12)
        self.assertNotIn('choice one', texts[0])
        self.assertIn('choice one', lens_query(question))
        self.assertEqual(details['lens_spatial_views'], 2)
        self.assertEqual(details['lens_temporal_views'], 2)
        self.assertEqual(details['source_frame_presentations'], 10)
        for frame, view in zip(frames, details['lens_views']):
            if view['kind'] == 'spatial':
                self.assertEqual(frame.getpixel((0,0)), (255, 0, 0))
        trace = make_trace(Decision([.7, .3], 90), indices, [i/30 for i in indices], selector.config, details)
        self.assertEqual(trace['input_tokens'], 170)
        self.assertEqual(trace['answer_input_tokens'], 90)
        self.assertEqual(trace['auxiliary_vlm_calls'], 1)
        self.assertEqual(trace['frame_index_space'], 'source-video-anchors-of-transformed-views')
        json.dumps(trace, allow_nan=False)

    def test_all_configs_are_valid_and_share_ids_protocol(self):
        root = Path(__file__).resolve().parents[1]
        configs = [LongVideoConfig.load(path) for path in (root/'configs').glob('lvb_*.json')]
        self.assertEqual({c.method for c in configs}, {'uniform', 'focus', 'lens'})
        self.assertEqual({(c.limit, c.seed, c.protocol) for c in configs}, {(200, 18, 'lvb-video-only-letter-v1')})
        with self.assertRaises(ValueError):
            replace(configs[0], lens_implementation='unknown')

    def test_runner_passes_transformed_views_to_answer_and_releases_video(self):
        from evidencelab.longvideo import run
        from evidencelab.longvideo_data import LongVideoSample
        from evidencelab.store import read_run
        sample = LongVideoSample('q1', 'v.mp4', 'What?', ('one', 'two'), 0, 'S2E', '1', 10.)
        received = []
        class Video:
            def __len__(self): return 300
            def get_avg_fps(self): return 30.
            def __getitem__(self, i):
                raise AssertionError('Runner must not replace LENS views with raw frames')
        class Backend:
            torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
            def reset_peak(self): pass
            def measurements(self): return {'peak_vram_allocated_gib': None}
            def answer(self, question, frames, times):
                received.extend(frames)
                self_test.assertFalse(hasattr(question, 'answer'))
                return Decision([.9, .1], 20)
        self_test = self
        view = Image.new('RGB', (8,8), (12,34,56))
        details = {'auxiliary_vlm_calls': 1, 'auxiliary_input_tokens': 10,
                   'source_frame_presentations': 4}
        def download(reader, entry, path, **kwargs):
            path.write_bytes(b'video')
            return 'a'*64
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp)
            annotations = root/'val.json'
            annotations.write_text('[]')
            stack.enter_context(patch.dict('sys.modules', {'decord': SimpleNamespace(
                bridge=SimpleNamespace(set_bridge=lambda _: None), cpu=lambda _: None,
                VideoReader=lambda *a, **k: Video())}))
            patches = {'parse_annotations': lambda _: [sample]*1337, 'choose_subset': lambda *a: [sample],
                       'validate_ram_root': lambda _: None, 'longvideo_runtime': lambda _: {},
                       'execution_profile': lambda _: {'resolved_dtype': 'bfloat16', 'node': 'fake'},
                       'HTTPRangeSource': lambda **k: SimpleNamespace(bytes_read=0),
                       'archive_index': lambda *a: {'videos/v.mp4': [0,5]},
                       'HFBackend': lambda _: Backend(), 'check_space': lambda *a: None,
                       'fetch_video': download}
            for name, value in patches.items():
                stack.enter_context(patch('evidencelab.longvideo.'+name, value))
            stack.enter_context(patch('evidencelab.lens_adapter.LensSelector',
                return_value=SimpleNamespace(select=lambda *a: ([30], [view], details))))
            index = root/'index.json'
            index.write_text('{}')
            config = LongVideoConfig(method='lens', frames=8, limit=1)
            self.assertEqual(run(config, annotations, root/'ram', root/'out', index), 0)
            self.assertIs(received[0], view)
            self.assertFalse((root/'ram/active-video.mp4').exists())
            summary = json.loads((root/'out/summary.json').read_text())
            self.assertEqual(summary['mean_vlm_calls'], 2)
            self.assertEqual(summary['mean_input_tokens'], 30)
            self.assertEqual(summary['mean_source_frame_presentations'], 4)

    def test_upstream_mask_imports_and_finite_output_on_tiny_clip(self):
        from API_CLIP.clip_prs.utils.model import CLIP
        from API_CLIP.hook import hook_prs_logger
        from API_CLIP.main import gen_mask, merge_mask
        from torchvision.transforms import Compose, Resize, ToTensor
        torch.manual_seed(42)
        model = CLIP(embed_dim=16,
                     vision_cfg={'layers': 2, 'width': 32, 'head_width': 16, 'patch_size': 4, 'image_size': 16},
                     text_cfg={'context_length': 8, 'vocab_size': 20, 'width': 32, 'heads': 2, 'layers': 1}).eval()
        # This upstream class leaves some parameters torch.empty until loading
        # a checkpoint. Initialize all of them for the no-download smoke test.
        from API_CLIP.clip_prs.utils.transformer import LayerNorm
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.normal_(std=.1)
            for module in model.modules():
                if isinstance(module, (LayerNorm, torch.nn.LayerNorm)):
                    module.weight.fill_(1)
                    module.bias.zero_()
        prs = hook_prs_logger(model, 'cpu', layer_index=0)
        tokenizer = lambda texts: torch.tensor([[1, 4, 19, 0, 0, 0, 0, 0] for _ in texts])
        image = Image.fromarray(np.random.default_rng(42).integers(0, 256, (16,16,3), dtype=np.uint8))
        with torch.inference_mode():
            _, attention, tokens = gen_mask(model, prs, Compose([Resize((16,16)), ToTensor()]),
                                            'cpu', tokenizer, [image], ['test'])
            result = merge_mask(attention[0], tokens[0])
        self.assertEqual(result.shape, (4,4))
        self.assertTrue(torch.isfinite(result).all())


if __name__ == '__main__':
    unittest.main()
