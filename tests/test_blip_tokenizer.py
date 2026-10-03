"""Exercise the missing-special-token failure with real offline HF tokenizers."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from evidencelab.focus_adapter import prepare_blip_itm_tokenizer


class BlipTokenizerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.vocab = Path(self.temp.name) / "vocab.txt"
        words = [f"unused{i}" for i in range(30522)]
        for i, token in {0: "[PAD]", 100: "[UNK]", 101: "[CLS]", 102: "[SEP]", 103: "[MASK]"}.items():
            words[i] = token
        self.vocab.write_text("\n".join(words) + "\n", encoding="utf-8")
        self.config = SimpleNamespace(bos_token_id=30522, vocab_size=30524)

    def test_missing_tokens_restored_for_slow_and_fast_tokenizers(self):
        from transformers import BertTokenizer, BertTokenizerFast
        for cls in (BertTokenizer, BertTokenizerFast):
            with self.subTest(tokenizer=cls.__name__):
                tokenizer = cls(vocab_file=str(self.vocab))
                self.assertEqual(tokenizer.convert_tokens_to_ids("[ENC]"), tokenizer.unk_token_id)
                before = tokenizer("unused200 unused201")["input_ids"]
                self.assertEqual(prepare_blip_itm_tokenizer(tokenizer, self.config, 30524), 30523)
                self.assertEqual(tokenizer("unused200 unused201")["input_ids"], before)
                self.assertEqual(tokenizer.bos_token_id, 30522)
                self.assertEqual(tokenizer("[ENC]", add_special_tokens=False)["input_ids"], [30523])
                # Already initialized tokenizers must remain unchanged.
                self.assertEqual(prepare_blip_itm_tokenizer(tokenizer, self.config, 30524), 30523)

    def test_rejects_shifted_ids_and_incompatible_embeddings(self):
        from transformers import BertTokenizer
        tokenizer = BertTokenizer(vocab_file=str(self.vocab))
        with self.assertRaisesRegex(ValueError, "checkpoint vocabulary"):
            prepare_blip_itm_tokenizer(tokenizer, self.config, 30522)
        tokenizer.add_tokens(["unexpected_added_token"])
        with self.assertRaisesRegex(ValueError, "special-token IDs"):
            prepare_blip_itm_tokenizer(tokenizer, self.config, 30524)

    def test_itm_forward_uses_existing_enc_embedding_without_resize(self):
        import torch
        from transformers import BertTokenizer, BlipConfig, BlipForImageTextRetrieval
        config = BlipConfig(
            text_config={"vocab_size": 30524, "bos_token_id": 30522, "hidden_size": 16,
                         "intermediate_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2},
            vision_config={"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1,
                           "num_attention_heads": 2, "image_size": 16, "patch_size": 8},
            image_text_hidden_size=8, projection_dim=8)
        model = BlipForImageTextRetrieval(config).eval()
        embedding = model.text_encoder.get_input_embeddings()
        weight_before = embedding.weight.detach().clone()
        tokenizer = BertTokenizer(vocab_file=str(self.vocab))
        enc = prepare_blip_itm_tokenizer(tokenizer, model.config.text_config, embedding.num_embeddings)
        inputs = tokenizer(["unused200", "unused201"], return_tensors="pt", padding=True)
        inputs["input_ids"][:, 0] = enc
        inputs.pop("token_type_ids", None)
        with torch.inference_mode():
            scores = model(pixel_values=torch.zeros(2, 3, 16, 16), **inputs, use_itm_head=True).itm_score
        self.assertEqual(tuple(scores.shape), (2, 2))
        self.assertTrue(torch.isfinite(scores).all())
        self.assertTrue(torch.equal(weight_before, embedding.weight))
