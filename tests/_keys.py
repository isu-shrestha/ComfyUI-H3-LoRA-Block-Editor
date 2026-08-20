"""Shared test helpers: locate the node package and build the H3 lora key set.

Tests run against a real lora when one is available (set H3_LORA_PATH), and fall
back to an equivalent synthetic key set otherwise, so a fresh clone can run them
with no weights on disk.
"""
import json
import os
import struct
import sys

NODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NUM_BLOCKS = 50
NUM_REFINER_BLOCKS = 2
SUBLAYERS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")


def add_node_to_path():
    if NODE_DIR not in sys.path:
        sys.path.insert(0, NODE_DIR)


def _synthetic_keys():
    keys = ["diffusion_model.blocks.{}.{}.weight".format(i, s)
            for i in range(NUM_BLOCKS) for s in SUBLAYERS]
    keys += ["diffusion_model.token_refiner.blocks.{}.{}.weight".format(i, s)
             for i in range(NUM_REFINER_BLOCKS) for s in SUBLAYERS]
    return sorted(keys)


def _keys_from_safetensors(path):
    """Read the safetensors header only -- no tensor data, no torch needed."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return sorted({k[: -len(".lora_A.weight")] + ".weight"
                   for k in header if k.endswith(".lora_A.weight")})


def model_keys():
    """-> (sorted model keys, source description)"""
    path = os.environ.get("H3_LORA_PATH")
    if path and os.path.isfile(path):
        return _keys_from_safetensors(path), "real lora ({})".format(os.path.basename(path))
    return _synthetic_keys(), "synthetic key set"


def lora_state_dict(model_keys_list):
    """A stand-in lora state dict with the A/B pair for each model key."""
    sd = {}
    for key in model_keys_list:
        stem = key[: -len(".weight")]
        sd[stem + ".lora_A.weight"] = object()
        sd[stem + ".lora_B.weight"] = object()
    return sd
