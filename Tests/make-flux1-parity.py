"""CPU-only Diffusers/Transformers numerical oracle for native FLUX.1."""
import json
import os
from pathlib import Path
import torch
import diffusers
import transformers
from diffusers import FluxTransformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler
from transformers import CLIPTextConfig, CLIPTextModel, T5Config, T5EncoderModel, AutoTokenizer
from safetensors.torch import save_file

torch.set_num_threads(4)
torch.manual_seed(812)
root = Path(__file__).resolve().parents[1] / '.build/Flux1Parity'
root.mkdir(parents=True, exist_ok=True)
hf_home = Path(os.environ.get('HF_HOME', Path.home() / '.cache/huggingface')).expanduser()
hub = hf_home if hf_home.name == 'hub' else hf_home / 'hub'
cache = hub / 'models--black-forest-labs--FLUX.1-dev'
revision = (cache / 'refs/main').read_text().strip()
snapshot = cache / 'snapshots' / revision

with torch.no_grad():
    dit = FluxTransformer2DModel(in_channels=64, num_layers=2, num_single_layers=2,
        attention_head_dim=16, num_attention_heads=2, joint_attention_dim=48,
        pooled_projection_dim=24, axes_dims_rope=(4, 6, 6), guidance_embeds=True).eval()
    dit.save_pretrained(root / 'transformer')
    x, text, pooled = torch.randn(1, 6, 64), torch.randn(1, 5, 48), torch.randn(1, 24)
    tid = torch.zeros(5, 3)
    iid = torch.tensor([[0, y, x] for y in range(2) for x in range(3)])
    fixtures = dict(latent=x, text=text, pooled=pooled, ids=torch.cat([tid, iid]))
    fixtures['dit'] = dit(x, text, pooled_projections=pooled, timestep=torch.tensor([0.73]),
        guidance=torch.tensor([3.5]), img_ids=iid, txt_ids=tid, return_dict=False)[0]
    fixtures['dit_guidance_zero'] = dit(x, text, pooled_projections=pooled, timestep=torch.tensor([0.73]),
        guidance=torch.tensor([0.0]), img_ids=iid, txt_ids=tid, return_dict=False)[0]
    clip = CLIPTextModel(CLIPTextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=8, eos_token_id=2, bos_token_id=0)).eval()
    clip.save_pretrained(root / 'text_encoder')
    clip_tokens = torch.tensor([[4, 6, 2, 63, 0, 0, 0]])
    fixtures['clip_tokens'] = clip_tokens.flatten()
    fixtures['clip'] = clip(clip_tokens).pooler_output
    t5 = T5EncoderModel(T5Config(vocab_size=64, d_model=32, d_ff=64, d_kv=8,
        num_heads=4, num_layers=2, feed_forward_proj='gated-gelu')).eval()
    t5.config.save_pretrained(root / 'text_encoder_2')
    save_file({k: v.contiguous().clone() for k, v in t5.state_dict().items()}, root / 'text_encoder_2/model.safetensors')
    t5_tokens = torch.tensor([[4, 7, 2, 1, 0, 0]])
    fixtures['t5_tokens'] = t5_tokens.flatten()
    fixtures['t5'] = t5(t5_tokens).last_hidden_state
    vae = AutoencoderKL(block_out_channels=(32, 32, 64, 64), layers_per_block=1,
        down_block_types=('DownEncoderBlock2D',) * 4, up_block_types=('UpDecoderBlock2D',) * 4,
        latent_channels=16, use_quant_conv=False, use_post_quant_conv=False,
        scaling_factor=.3611, shift_factor=.1159).eval()
    vae.save_pretrained(root / 'vae')
    packed = torch.randn(1, 6, 64)
    latent = packed.reshape(1, 2, 3, 16, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(1, 16, 4, 6)
    fixtures['vae_latent'] = packed
    fixtures['decoded'] = vae.decode(latent / .3611 + .1159).sample.permute(0, 2, 3, 1)
    scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)
    for count in [256, 1024, 4096]:
        mu = .5 + (count - 256) * (.65 / (4096 - 256))
        scheduler.set_timesteps(sigmas=[1, .75, .5, .25], mu=mu)
        fixtures[f'sigmas_{count}'] = scheduler.sigmas
    (root / 'scheduler').mkdir(exist_ok=True)
    (root / 'scheduler/scheduler_config.json').write_text((snapshot / 'scheduler/scheduler_config.json').read_text())
    save_file({k: v.contiguous() for k, v in fixtures.items()}, root / 'fixtures.safetensors')
    prompts = ["A red ceramic mug on a white table", "猫咪在海边，柔和光线。", "A café ☕ — 123!", "It's &amp; a\n test.", "", "hello " * 100]
    token_tests = []
    for component, length in [('tokenizer', 77), ('tokenizer_2', 512)]:
        tokenizer = AutoTokenizer.from_pretrained(snapshot / component, local_files_only=True)
        for prompt in prompts:
            ids = tokenizer(prompt, padding='max_length', truncation=True, max_length=length)['input_ids']
            token_tests.append(dict(component=component, prompt=prompt, tokens=ids))
    (root / 'token-tests.json').write_text(json.dumps(token_tests, ensure_ascii=False))
    (root / 'provenance.json').write_text(json.dumps(dict(diffusers=diffusers.__version__,
        transformers=transformers.__version__, torch=torch.__version__, flux_revision=revision), indent=2))
print(root)
