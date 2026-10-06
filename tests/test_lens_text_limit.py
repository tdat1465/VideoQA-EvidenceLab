"""Reproduce LENS's 590-token failure using offline HF BLIP components."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from evidencelab.lens_adapter import lens_blip_text_metadata


class LensTextLimitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        from transformers import BertTokenizer, BlipImageProcessor, BlipProcessor
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        vocab = Path(self.temp.name) / 'vocab.txt'
        vocab.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\n', encoding='utf-8')
        self.tokenizer = BertTokenizer(vocab_file=str(vocab), model_max_length=512)
        self.processor = BlipProcessor(image_processor=BlipImageProcessor(size={'height': 16, 'width': 16}),
                                       tokenizer=self.tokenizer)
        self.images = [Image.new('RGB', (16, 16)), Image.new('RGB', (16, 16), (100, 20, 30))]
        self.text_config = SimpleNamespace(max_position_embeddings=512)

    def encode(self, query, bounded=True):
        kwargs = {}
        if bounded:
            details = lens_blip_text_metadata(self.tokenizer, self.text_config, query)
            kwargs = {'truncation': True, 'max_length': details['lens_blip_max_length']}
        return self.processor(images=self.images, text=[query]*2, padding=True, return_tensors='pt', **kwargs)

    def test_short_and_boundary_queries_keep_tokens_images_and_attention(self):
        for count in (8, 510):
            query = 'hello ' * count
            details = lens_blip_text_metadata(self.tokenizer, self.text_config, query)
            self.assertEqual(details['lens_blip_query_tokens'], count+2)
            self.assertFalse(details['lens_blip_query_truncated'])
            self.assertEqual(details['lens_blip_dropped_tokens'], 0)
            before, after = self.encode(query, False), self.encode(query)
            self.assertEqual(before.keys(), after.keys())
            for key in before:
                torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)

    def test_long_query_reproduces_failure_then_runs_at_512_with_same_weights(self):
        from transformers import BlipConfig, BlipForImageTextRetrieval
        query = 'hello ' * 588
        details = lens_blip_text_metadata(self.tokenizer, self.text_config, query)
        self.assertEqual(details['lens_blip_query_tokens'], 590)
        self.assertTrue(details['lens_blip_query_truncated'])
        self.assertEqual(details['lens_blip_dropped_tokens'], 78)
        config = BlipConfig(
            text_config={'vocab_size': 8, 'bos_token_id': 2, 'max_position_embeddings': 512,
                         'hidden_size': 16, 'intermediate_size': 32, 'num_hidden_layers': 1,
                         'num_attention_heads': 2},
            vision_config={'hidden_size': 16, 'intermediate_size': 32, 'num_hidden_layers': 1,
                           'num_attention_heads': 2, 'image_size': 16, 'patch_size': 8},
            image_text_hidden_size=8, projection_dim=8)
        model = BlipForImageTextRetrieval(config).eval()
        embedding = model.text_encoder.get_input_embeddings()
        weights = embedding.weight.detach().clone()
        with torch.inference_mode():
            with self.assertRaisesRegex(RuntimeError, '590.*512'):
                model(**self.encode(query, False))
            inputs = self.encode(query)
            self.assertEqual(tuple(inputs['input_ids'].shape), (2, 512))
            self.assertTrue(torch.all(inputs['input_ids'][:, 0] == self.tokenizer.cls_token_id))
            self.assertTrue(torch.all(inputs['input_ids'][:, -1] == self.tokenizer.sep_token_id))
            scores = model(**inputs).itm_score
        self.assertEqual(tuple(scores.shape), (2, 2))
        self.assertTrue(torch.isfinite(scores).all())
        torch.testing.assert_close(weights, embedding.weight, rtol=0, atol=0)

    def test_checkpoint_and_tokenizer_limits_and_policy_are_validated(self):
        self.tokenizer.model_max_length = 10**30  # HF sentinel, not the actual model capacity.
        details = lens_blip_text_metadata(self.tokenizer, SimpleNamespace(max_position_embeddings=32), 'hello '*40)
        self.assertEqual(details['lens_blip_max_length'], 32)
        self.tokenizer.model_max_length = 24
        self.assertEqual(lens_blip_text_metadata(self.tokenizer, self.text_config, 'hello')['lens_blip_max_length'], 24)
        self.tokenizer.truncation_side = 'left'
        with self.assertRaisesRegex(ValueError, 'right text truncation'):
            lens_blip_text_metadata(self.tokenizer, self.text_config, 'hello')
        self.tokenizer.truncation_side = 'right'
        with self.assertRaisesRegex(ValueError, 'context limit'):
            lens_blip_text_metadata(self.tokenizer, SimpleNamespace(max_position_embeddings=0), 'hello')


if __name__ == '__main__':
    unittest.main()
