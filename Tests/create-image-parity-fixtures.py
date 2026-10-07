"""Development-only CPU Diffusers reference; the App and smoke inference use Swift."""
import json
from pathlib import Path
import sys

import torch
from safetensors.torch import save_file
from diffusers import AutoencoderKLQwenImage21
from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21Transformer2DModel, QwenImage21AttnProcessor, QwenImage21KVCache,
    QwenImage21TemporalTimesteps,
)
from diffusers import FlowMatchEulerDiscreteScheduler

root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
torch.manual_seed(42)
torch.set_num_threads(4)
transformer = QwenImage21Transformer2DModel(
    in_channels=8, out_channels=8, num_layers=2, attention_head_dim=16,
    num_attention_heads=2, context_in_dim=16, axes_dims_rope=(4, 6, 6),
).eval()
for block in transformer.transformer_blocks:
    block.attn.set_processor(QwenImage21AttnProcessor())
transformer.save_pretrained(root / "transformer")
text = torch.randn(1, 3, 16)
latent = torch.randn(1, 20, 8)
mask = torch.tensor([[False, True, False, True, True, True, True]])
shapes = [[(1, 2, 2), (1, 4, 4)]]
cache = QwenImage21KVCache(2)
with torch.no_grad():
    expected_first = transformer(latent, text, torch.tensor([0.75]),
        img_shapes=shapes, img_mask=mask, kv_cache=cache, kv_cache_mode="extract").sample[:, -16:]
    expected_cached = transformer(latent, text, torch.tensor([0.25]),
        img_shapes=shapes, img_mask=mask, kv_cache=cache, kv_cache_mode="cached").sample[:, -16:]
    expected_uncached = transformer(latent, text, torch.tensor([0.25]),
        img_shapes=shapes, img_mask=mask).sample[:, -16:]
save_file({"text": text, "latents": latent, "first": expected_first.contiguous(),
           "cached": expected_cached.contiguous(), "uncached": expected_uncached.contiguous()}, root / "dit-inputs.safetensors")
vae = AutoencoderKLQwenImage21(
    base_dim=8, decoder_base_dim=8, z_dim=4, dim_mult=[1, 2, 4, 4, 4],
    num_res_blocks=2, temperal_downsample=[False, True, True, True],
    latents_mean=[0.0] * 4, latents_std=[1.0] * 4, is_residual=True,
    in_channels=4, out_channels=4,
).eval()
vae.save_pretrained(root / "vae")
image = torch.randn(1, 4, 1, 32, 32).clamp(-1, 1)
with torch.no_grad():
    encoded = vae.encode(image).latent_dist.mode()
    decoded = vae.decode(encoded).sample.clamp(-1, 1)
save_file({"image": image.contiguous(), "encoded": encoded.contiguous(),
           "decoded": decoded.contiguous()}, root / "vae-inputs.safetensors")
config = {
    "base_image_seq_len": 256, "base_shift": 0.5, "max_image_seq_len": 8192,
    "max_shift": 0.9, "shift_terminal": 0.02, "use_dynamic_shifting": True,
    "time_shift_type": "exponential", "invert_sigmas": False, "stochastic_sampling": False,
}
scheduler = FlowMatchEulerDiscreteScheduler(**config)
mu = 0.5 + (1024 - 256) * (0.9 - 0.5) / (8192 - 256)
scheduler.set_timesteps(40, sigmas=torch.linspace(1.0, 1 / 40, 40).numpy(), mu=mu)
(root / "scheduler.json").write_text(json.dumps(config))
save_file({"sigmas": scheduler.sigmas}, root / "scheduler.safetensors")
precision = {"sigma": torch.tensor([0.999123, 0.83117, 0.6137, 0.21319, 0.02071])}
time_projection = QwenImage21TemporalTimesteps(256)
for name, dtype in [("bf16", torch.bfloat16), ("fp16", torch.float16), ("fp32", torch.float32)]:
    embeddings = []
    for sigma in precision["sigma"]:
        # QwenImage21Pipeline casts the scheduler's 0...1000 timestep before
        # division, then the transformer casts the normalized input again.
        t = ((sigma * 1000).to(dtype) / 1000).to(dtype)
        embeddings.append(time_projection(torch.stack([t, torch.zeros_like(t)])).to(dtype))
    precision[name + "_time"] = torch.stack(embeddings)
    initial = torch.randn(1, 4, 64).to(dtype)
    predictions = torch.randn(40, 1, 4, 64).to(dtype)
    sample = initial
    trajectory = []
    scheduler.set_timesteps(40, sigmas=torch.linspace(1.0, 1 / 40, 40).numpy(), mu=mu)
    for t, prediction in zip(scheduler.timesteps, predictions):
        # Explicit FP32 velocity avoids torch's weak-scalar promotion rule.
        sample = scheduler.step(prediction.float(), t, sample.float()).prev_sample.to(dtype)
        trajectory.append(sample)
    precision[name + "_initial"] = initial
    precision[name + "_predictions"] = predictions
    precision[name + "_trajectory"] = torch.stack(trajectory)
save_file(precision, root / "precision.safetensors")

# Independent CPU VideoX-Fun control branch with nonzero projection weights.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "transformer"))
from qwen_fun.control import QwenImage21ControlTransformer2DModel
control_model = QwenImage21ControlTransformer2DModel(
    in_channels=8, out_channels=8, num_layers=6, attention_head_dim=16,
    num_attention_heads=2, context_in_dim=16, axes_dims_rope=(4, 6, 6),
    control_in_dim=17, control_layers=[0, 2, 4],
).eval()
with torch.no_grad():
    for block in control_model.control_blocks:
        block.after_proj.weight.normal_(0, 0.025)
        block.after_proj.bias.normal_(0, 0.01)
    control_model.control_blocks[0].before_proj.weight.normal_(0, 0.025)
    control_model.control_blocks[0].before_proj.bias.normal_(0, 0.01)
control_model.save_pretrained(root / "control-transformer")
save_file({k: v.contiguous() for k, v in control_model.state_dict().items() if k.startswith("control_")},
          root / "control.safetensors")
control_latents = torch.randn(1, 16, 8)
control_text = torch.randn(1, 3, 16)
context = torch.cat([torch.randn(1, 16, 8), torch.zeros(1, 16, 9)], dim=-1)
control_inputs = {"latents": control_latents, "text": control_text, "context": context}
with torch.no_grad():
    for name, scale, t in [("zero", 0.0, 0.75), ("half", 0.5, 0.75), ("full", 1.0, 0.75), ("later", 1.0, 0.25)]:
        control_inputs[name] = control_model(control_latents, control_text, torch.tensor([t]),
            img_shapes=[[(1, 4, 4)]], img_mask=torch.tensor([[False, False, False, True, True, True, True]]),
            control_context=context, control_context_scale=scale).sample[:, -16:].contiguous()
save_file(control_inputs, root / "control-inputs.safetensors")
print("PASS: CPU reference fixtures: two-layer DiT including reference/cache, residual RGBA VAE and Euler schedule")
