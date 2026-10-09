#!/usr/bin/env python3
"""CPU Diffusers oracle for native Klein DiT, VAE, scheduler and reference tokens."""
import json
from pathlib import Path
import torch
import diffusers
from diffusers import Flux2Transformer2DModel, AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu
from safetensors.torch import save_file
from transformers import Qwen3Config, Qwen3Model

torch.set_num_threads(4)
torch.manual_seed(711)
root = Path(__file__).resolve().parents[1] / '.build/Flux2Parity'
root.mkdir(parents=True, exist_ok=True)
with torch.no_grad():
    dit = Flux2Transformer2DModel(in_channels=128, num_layers=2, num_single_layers=2,
        attention_head_dim=16, num_attention_heads=2, joint_attention_dim=48,
        timestep_guidance_channels=16, axes_dims_rope=(4, 4, 4, 4), guidance_embeds=False).eval()
    dit.save_pretrained(root / 'transformer')
    x, text = torch.randn(1, 6, 128), torch.randn(1, 5, 48)
    tid = torch.tensor([[0, 0, 0, t] for t in range(5)])
    iid = torch.tensor([[0, y, x, 0] for y in range(2) for x in range(3)])
    fixtures = dict(latent=x, text=text, ids=torch.cat([tid, iid]))
    fixtures['dit'] = dit(x, text, timestep=torch.tensor([0.73]), img_ids=iid, txt_ids=tid, return_dict=False)[0]
    ref = torch.randn(1, 2, 128)
    rid = torch.tensor([[10, 0, x, 0] for x in range(2)])
    fixtures.update(reference=ref, edit_ids=torch.cat([tid, iid, rid]))
    fixtures['edit'] = dit(torch.cat([x, ref], dim=1), text, timestep=torch.tensor([0.73]), img_ids=torch.cat([iid, rid]), txt_ids=tid, return_dict=False)[0]
    vae = AutoencoderKLFlux2(block_out_channels=(32, 32, 64, 64), layers_per_block=1).eval()
    vae.bn.running_mean.copy_(torch.randn(128) * 0.2)
    vae.bn.running_var.copy_(torch.rand(128) + 0.5)
    vae.save_pretrained(root / 'vae')
    rgb = torch.randn(1, 3, 32, 48)
    z = vae.encode(rgb).latent_dist.mode()
    packed = z.reshape(1, 32, 2, 2, 3, 2).permute(0, 2, 4, 1, 3, 5).reshape(1, 6, 128)
    packed = (packed - vae.bn.running_mean) / (vae.bn.running_var + vae.config.batch_norm_eps).sqrt()
    fixtures.update(rgb=rgb.permute(0, 2, 3, 1), encoded=packed,
                    decoded=vae.decode(z).sample.permute(0, 2, 3, 1))
    qwen = Qwen3Model(Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=28, num_attention_heads=4, num_key_value_heads=2, head_dim=8)).eval()
    token_ids = torch.tensor([[4, 2, 6, 7, 0, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0]])
    states = qwen(token_ids, attention_mask=mask, output_hidden_states=True, use_cache=False).hidden_states
    fixtures['text_features'] = torch.cat([states[i] for i in [9, 18, 27]], dim=-1)
    fixtures['text_tokens'] = token_ids.flatten()
    text_folder = root / 'text_encoder'
    text_folder.mkdir(exist_ok=True)
    (text_folder / 'config.json').write_text(qwen.config.to_json_string())
    save_file({'model.' + k: v.contiguous() for k, v in qwen.state_dict().items()}, text_folder / 'model.safetensors')
    scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)
    for count in [256, 4096, 5000]:
        scheduler.set_timesteps(sigmas=[1, .75, .5, .25], mu=compute_empirical_mu(count, 4))
        fixtures[f'sigmas_{count}'] = scheduler.sigmas
    save_file({k: v.contiguous() for k, v in fixtures.items()}, root / 'fixtures.safetensors')
    (root / 'provenance.json').write_text(json.dumps({'diffusers': diffusers.__version__, 'torch': torch.__version__}, indent=2))
print(root)
