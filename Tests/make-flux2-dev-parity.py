#!/usr/bin/env python3
"""CPU oracle for FLUX.2-dev guidance, Mistral states and official tokenization."""
import json
from pathlib import Path
import torch
import diffusers
import transformers
from diffusers import Flux2Transformer2DModel
from transformers import MistralConfig, MistralModel, PreTrainedTokenizerFast
from safetensors.torch import save_file

root = Path(__file__).resolve().parents[1]
out = root / '.build/Flux2DevParity'
metadata = root / '.build/Flux2DevMetadata'
out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
torch.manual_seed(739)
with torch.no_grad():
    dit = Flux2Transformer2DModel(in_channels=128, num_layers=2, num_single_layers=2,
        attention_head_dim=16, num_attention_heads=2, joint_attention_dim=96,
        timestep_guidance_channels=16, axes_dims_rope=(4, 4, 4, 4), guidance_embeds=True).eval()
    dit.save_pretrained(out / 'transformer')
    x, text, reference = torch.randn(1, 6, 128), torch.randn(1, 5, 96), torch.randn(1, 2, 128)
    tid = torch.tensor([[0, 0, 0, t] for t in range(5)])
    iid = torch.tensor([[0, y, x, 0] for y in range(2) for x in range(3)])
    rid = torch.tensor([[10, 0, x, 0] for x in range(2)])
    fixtures = dict(latent=x, text=text, reference=reference, ids=torch.cat([tid, iid]),
                    edit_ids=torch.cat([tid, iid, rid]))
    for guidance in [0, 4]:
        fixtures[f'dit_{guidance}'] = dit(x, text, timestep=torch.tensor([.73]),
            guidance=torch.tensor([float(guidance)]), img_ids=iid, txt_ids=tid, return_dict=False)[0]
    fixtures['edit'] = dit(torch.cat([x, reference], dim=1), text, timestep=torch.tensor([.73]),
        guidance=torch.tensor([4.]), img_ids=torch.cat([iid, rid]), txt_ids=tid, return_dict=False)[0]
    config = MistralConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=32, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        sliding_window=None, rope_parameters={'rope_type': 'default', 'rope_theta': 1e9})
    model = MistralModel(config).eval()
    ids = torch.tensor([[0, 0, 0, 4, 2, 6, 7]])
    states = model(ids, attention_mask=torch.tensor([[0, 0, 0, 1, 1, 1, 1]]),
        output_hidden_states=True, use_cache=False).hidden_states
    fixtures['text_features'] = torch.cat([states[i] for i in [10, 20, 30]], dim=-1)
    fixtures['text_tokens'] = ids.flatten()
    folder = out / 'text_encoder'
    folder.mkdir(exist_ok=True)
    (folder / 'config.json').write_text(config.to_json_string())
    save_file({'language_model.model.' + k: v.contiguous() for k, v in model.state_dict().items()}, folder / 'model.safetensors')
    save_file({k: v.contiguous() for k, v in fixtures.items()}, out / 'fixtures.safetensors')

# Transformers 5.18's AutoTokenizer selects a SentencePiece Llama path for this
# older LlamaTokenizerFast export, replacing the byte-level tokenizer and losing
# spaces. Test the original serialized Rust tokenizer, without that override.
tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(metadata / 'tokenizer/tokenizer.json'),
    bos_token='<s>', eos_token='</s>', pad_token='<pad>', padding_side='left')
assert tokenizer.padding_side == 'left'
template = (metadata / 'tokenizer/chat_template.jinja').read_text()
system = 'You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object\nattribution and actions without speculation.'
tests = []
for prompt in ['A red ceramic mug.', '一只红色陶瓷杯，柔和自然光。', 'A cat 🐈 under a tree 🌳',
               'Change [IMG] the cup to blue.', '', 'A vivid red mug. ' * 600]:
    messages = [{'role': 'system', 'content': [{'type': 'text', 'text': system}]},
                {'role': 'user', 'content': [{'type': 'text', 'text': prompt.replace('[IMG]', '')}]}]
    rendered = tokenizer.apply_chat_template(messages, chat_template=template, tokenize=False, add_generation_prompt=False)
    data = tokenizer(rendered, add_special_tokens=False, padding='max_length', truncation=True, max_length=512)
    tests.append({'prompt': prompt, 'tokens': data['input_ids'], 'validCount': sum(data['attention_mask'])})
(out / 'token-tests.json').write_text(json.dumps(tests, ensure_ascii=False))
(out / 'provenance.json').write_text(json.dumps({'diffusers': diffusers.__version__,
    'transformers': transformers.__version__, 'torch': torch.__version__}, indent=2))
print(out)
