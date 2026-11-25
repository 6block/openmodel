"""is_model_complete must recognise TWO model layouts.

The diffusers case is the one that bites: a diffusion model directory has
`model_index.json` and component subdirectories, NOT a `config.json`. Before this was
handled, a perfectly good image-generation model was judged corrupt — and a corrupt
cache is DELETED and re-downloaded, so the failure mode was a multi-gigabyte download
every start, not a clear error.
"""
from pathlib import Path

from src.model.manager import is_model_complete


def _touch(p: Path, name: str, data: bytes = b"x"):
    p.mkdir(parents=True, exist_ok=True)
    (p / name).write_bytes(data)


def test_transformers_layout_is_complete(tmp_path):
    d = tmp_path / "qwen3-8b"
    _touch(d, "config.json", b"{}")
    _touch(d, "model.safetensors")
    assert is_model_complete(d) is True


def test_diffusers_layout_is_complete(tmp_path):
    """model_index.json + a populated component directory is a usable diffusion model."""
    d = tmp_path / "flux-schnell"
    _touch(d, "model_index.json", b'{"_class_name": "FluxPipeline"}')
    _touch(d / "unet", "diffusion_pytorch_model.safetensors")
    _touch(d / "vae", "diffusion_pytorch_model.safetensors")
    assert is_model_complete(d) is True, "a diffusers directory must be recognised"


def test_speech_checkpoints_are_complete(tmp_path):
    """Kokoro ships a .pth checkpoint and faster-whisper a CTranslate2 model.bin. Both
    must count as complete — an unrecognised one is deleted as a crashed download."""
    kokoro = tmp_path / "hexgrad--Kokoro-82M"
    _touch(kokoro, "config.json", b"{}")
    _touch(kokoro, "kokoro-v1_0.pth")
    _touch(kokoro / "voices", "af_heart.pt")
    assert is_model_complete(kokoro) is True
    whisper = tmp_path / "Systran--faster-whisper-small"
    _touch(whisper, "config.json", b"{}")
    _touch(whisper, "model.bin")
    assert is_model_complete(whisper) is True


def test_config_json_alone_is_not_enough(tmp_path):
    """A config with no weights is a half-written directory, not a model."""
    d = tmp_path / "half"
    _touch(d, "config.json", b"{}")
    assert is_model_complete(d) is False


def test_model_index_alone_is_not_enough(tmp_path):
    """The diffusers descriptor without a populated component is also partial — the
    check must not rubber-stamp a directory that a crashed download left behind."""
    d = tmp_path / "half-diffusers"
    _touch(d, "model_index.json", b"{}")
    assert is_model_complete(d) is False


def test_empty_component_dir_is_not_enough(tmp_path):
    """A component directory with no weights in it does not count."""
    d = tmp_path / "half2"
    _touch(d, "model_index.json", b"{}")
    (d / "unet").mkdir()
    assert is_model_complete(d) is False


def test_missing_path_is_not_complete(tmp_path):
    assert is_model_complete(tmp_path / "nope") is False


def test_diffusers_component_with_bin_weights(tmp_path):
    """Some pipelines ship .bin rather than .safetensors; the component check keys on
    safetensors today, so document that .bin-only is treated as incomplete rather than
    silently accepted."""
    d = tmp_path / "legacy-diffusers"
    _touch(d, "model_index.json", b"{}")
    _touch(d / "unet", "diffusion_pytorch_model.bin")
    # Pinned deliberately: if support for .bin components is added, this test is the
    # place to change, and the change becomes visible.
    assert is_model_complete(d) is False


# --- fp16 variant detection (diffusion_engine._has_fp16_variant) ------------------
#
# A diffusers directory downloaded with only the fp16 weights cannot be loaded without
# variant="fp16": from_pretrained looks for diffusion_pytorch_model.safetensors, finds
# nothing, and fails. Found on real hardware with an SDXL fp16-only download.

from inference.diffusion_engine import _has_fp16_variant, _precision_kwargs


def test_fp16_only_layout_is_detected(tmp_path):
    (tmp_path / "unet").mkdir()
    (tmp_path / "unet" / "diffusion_pytorch_model.fp16.safetensors").write_bytes(b"x")
    (tmp_path / "model_index.json").write_text("{}")
    assert _has_fp16_variant(str(tmp_path))


def test_full_precision_layout_is_not_a_variant(tmp_path):
    (tmp_path / "unet").mkdir()
    (tmp_path / "unet" / "diffusion_pytorch_model.safetensors").write_bytes(b"x")
    assert not _has_fp16_variant(str(tmp_path))
    # No variant → load as shipped, and torch is never imported for the decision.
    assert _precision_kwargs(str(tmp_path)) == {}


def test_missing_directory_is_not_a_variant(tmp_path):
    assert not _has_fp16_variant(str(tmp_path / "absent"))


# --- VAE tile shrinking (diffusion_engine._shrink_vae_tiles) ----------------------
#
# SDXL ships 1024 px VAE tiles and only tiles a latent strictly larger than the
# threshold, so a 1024x1024 render never tiled and ran out of memory on a 10 GB card
# (real hardware, 2026-10-19). The threshold must be lowered.

from types import SimpleNamespace

from inference.diffusion_engine import _shrink_vae_tiles


def test_sdxl_tiles_are_shrunk_so_1024_actually_tiles():
    vae = SimpleNamespace(tile_sample_min_size=1024, tile_latent_min_size=128)
    _shrink_vae_tiles(vae, 512)
    assert vae.tile_sample_min_size == 512
    assert vae.tile_latent_min_size == 64
    # A 1024 render has latent 128; the VAE tiles when the latent EXCEEDS the threshold.
    assert 128 > vae.tile_latent_min_size


def test_smaller_tiles_are_never_enlarged():
    vae = SimpleNamespace(tile_sample_min_size=256, tile_latent_min_size=32)
    _shrink_vae_tiles(vae, 512)
    assert (vae.tile_sample_min_size, vae.tile_latent_min_size) == (256, 32)


def test_vae_without_tiling_attributes_is_left_alone():
    vae = SimpleNamespace()
    _shrink_vae_tiles(vae, 512)
    assert not hasattr(vae, "tile_sample_min_size")
    _shrink_vae_tiles(None, 512)  # no VAE at all must not raise
