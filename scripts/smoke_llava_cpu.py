"""No checkpoint downloads: exercise the actual upstream video expansion + generator API.

Run in the LLaVA environment with the pinned llava source installed.
This is an API check with random tiny weights, not a VideoQA evaluation.
"""
import torch
from torch import nn
from transformers.generation.utils import GenerationMixin
from llava.model.language_model.llava_qwen import LlavaQwenConfig, LlavaQwenForCausalLM


def main():
    config = LlavaQwenConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                            max_position_embeddings=2048, bos_token_id=1, eos_token_id=2,
                            mm_patch_merge_type="spatial_unpad", mm_newline_position="grid",
                            mm_spatial_pool_mode="bilinear", mm_spatial_pool_stride=2,
                            tokenizer_model_max_length=None, tokenizer_padding_side="right")
    model = LlavaQwenForCausalLM(config).eval()
    class Tower(nn.Module):
        num_patches_per_side = 4
        def forward(self, images):
            return torch.zeros((len(images), 16, 8), device=images.device)
    model.model.vision_tower = Tower()
    model.model.mm_projector = nn.Linear(8, 32)
    model.model.image_newline = nn.Parameter(torch.zeros(32))
    ids = torch.tensor([[1, -200, 4, 5]])
    with torch.inference_mode():
        _, positions, mask, _, embeddings, _ = model.prepare_inputs_labels_for_multimodal(
            ids, None, torch.ones_like(ids), None, None, [torch.zeros(3, 3, 16, 16)], ["video"])
        assert embeddings.shape[1] == 21  # 3 frames * (2x2 + 2 newline tokens) + 3 text tokens.
        result = GenerationMixin.generate(model, inputs_embeds=embeddings, attention_mask=mask,
                                           position_ids=positions, max_new_tokens=1, do_sample=False,
                                           return_dict_in_generate=True, output_scores=True, pad_token_id=2)
        assert len(result.scores) == 1 and result.scores[0].shape == (1, 128)
        assert torch.isfinite(result.scores[0]).all()
        hidden = model.model(inputs_embeds=embeddings, attention_mask=mask, position_ids=positions,
                             use_cache=False, return_dict=True).last_hidden_state
        logits = model.lm_head(hidden[:, -1, :]).float()
        torch.testing.assert_close(logits, result.scores[0])
        # LENS allocator uses the upstream text-only wrapper. Confirm that it
        # returns only generated IDs, so the adapter must not strip prompt IDs.
        text_ids = torch.tensor([[1, 4, 5]])
        text_result = model.generate(text_ids, attention_mask=torch.ones_like(text_ids),
                                     images=None, modalities=["text"], max_new_tokens=2,
                                     min_new_tokens=2, do_sample=False, pad_token_id=2)
        assert text_result.shape == (1, 2)
    print("PASS: LLaVA video expansion; last-position logits match one-token generation (tiny CPU model)")


if __name__ == "__main__":
    main()
