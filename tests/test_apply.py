"""End-to-end test of H3LoraBlockLoader.apply() with mocked comfy internals.

Mocks mirror the real signatures read from the ComfyUI source:
  comfy.lora.model_lora_keys_unet(model, key_map) -> dict
  comfy.lora.load_lora(lora, to_load, log_missing=True) -> {model_key: adapter}
  ModelPatcher.add_patches(patches, strength_patch=1.0, strength_model=1.0) -> list of applied keys
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _keys

MODEL_KEYS, SOURCE = _keys.model_keys()
LORA_SD = _keys.lora_state_dict(MODEL_KEYS)
LORA = "test_lora.safetensors"
print("model keys: {} (from {})".format(len(MODEL_KEYS), SOURCE))

comfy = types.ModuleType("comfy")
lora_mod = types.ModuleType("comfy.lora")
convert_mod = types.ModuleType("comfy.lora_convert")
utils_mod = types.ModuleType("comfy.utils")
fp = types.ModuleType("folder_paths")

lora_mod.model_lora_keys_unet = lambda model, key_map: {k[:-len(".weight")]: k for k in MODEL_KEYS}
lora_mod.load_lora = lambda lora, to_load, log_missing=True: {v: ("lora", k) for k, v in to_load.items()}
convert_mod.convert_lora = lambda sd: sd
utils_mod.load_torch_file = lambda p, safe_load=False, return_metadata=False: (LORA_SD, {"name": "test_lora"})
fp.get_filename_list = lambda _: ["test_lora.safetensors"]
fp.get_full_path_or_raise = lambda folder, name: LORA

comfy.lora, comfy.lora_convert, comfy.utils = lora_mod, convert_mod, utils_mod
for name, mod in [("comfy", comfy), ("comfy.lora", lora_mod), ("comfy.lora_convert", convert_mod),
                  ("comfy.utils", utils_mod), ("folder_paths", fp)]:
    sys.modules[name] = mod

_keys.add_node_to_path()
import h3_lora_block as h3


class FakePatcher:
    """Stands in for ModelPatcher; records what add_patches was called with."""

    def __init__(self):
        self.model = object()
        self.calls = []
        self.attachments = {}

    def clone(self):
        c = FakePatcher()
        c.calls = self.calls
        return c

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        self.calls.append((strength_patch, sorted(patches)))
        return list(patches)

    def set_attachments(self, k, v):
        self.attachments[k] = v


failures = []


def check(label, cond, detail=""):
    print("{} {}{}".format("ok  " if cond else "FAIL", label, "" if cond else "  <- " + str(detail)))
    if not cond:
        failures.append(label)


def run(spec, strength=1.0):
    node = h3.H3LoraBlockLoader()
    src = FakePatcher()
    model, report = node.apply(src, "test_lora.safetensors", strength, spec)
    return model, report, model.calls


# --- plain full-strength application -----------------------------------------
model, report, calls = run("*: 1.0")
check("full: one add_patches call", len(calls) == 1, calls and calls[0][0])
check("full: strength 1.0", calls[0][0] == 1.0)
check("full: all 208 keys", len(calls[0][1]) == 208, len(calls[0][1]))
check("full: metadata attached", model.attachments.get("lora_metadata") == {"name": "test_lora"})

# --- global strength multiplies through ---------------------------------------
_, _, calls = run("*: 0.5", strength=0.8)
check("scaled: single group at 0.4", calls[0][0] == 0.4, calls[0][0])

# --- refiner muted -------------------------------------------------------------
_, report, calls = run("*: 1.0\nrefiner: 0.0")
check("refiner off: 200 keys patched", len(calls[0][1]) == 200, len(calls[0][1]))
check("refiner off: no token_refiner key",
      not [k for k in calls[0][1] if "token_refiner" in k])
check("refiner off: report says 8 muted", "8 muted at 0" in report, report.splitlines()[-1])

# --- several distinct strengths group correctly --------------------------------
_, _, calls = run("*: 1.0\n0-9: 0.0\n10-19: 0.25\n40-49.mlp: 0.5")
by_strength = {s: len(keys) for s, keys in calls}
check("multi: 3 nonzero groups", len(calls) == 3, by_strength)
check("multi: 0.25 group has 40 keys", by_strength.get(0.25) == 40, by_strength)
check("multi: 0.5 group has 20 keys", by_strength.get(0.5) == 20, by_strength)
check("multi: 1.0 group has 108 keys", by_strength.get(1.0) == 108, by_strength)
check("multi: total = 208 - 40 muted", sum(by_strength.values()) == 168, by_strength)

# --- everything off means no add_patches at all --------------------------------
_, report, calls = run("*: 0.0")
check("all off: no calls", calls == [], calls)
check("all off: warns", "nothing was applied" in report)

# --- global strength 0 -----------------------------------------------------------
_, _, calls = run("*: 1.0", strength=0.0)
check("strength 0: no calls", calls == [], calls)

# --- negative strength on one layer ---------------------------------------------
_, _, calls = run("*: 1.0\n25.out: -1.0")
neg = [ (s, keys) for s, keys in calls if s == -1.0 ]
check("negative: own group of 1", len(neg) == 1 and len(neg[0][1]) == 1, neg)
check("negative: correct key",
      neg[0][1] == ["diffusion_model.blocks.25.attn.out_proj.weight"], neg[0][1])

# --- lora file is cached across invocations -------------------------------------
node = h3.H3LoraBlockLoader()
loads = []
orig = utils_mod.load_torch_file
utils_mod.load_torch_file = lambda *a, **k: (loads.append(1), orig(*a, **k))[1]
node.apply(FakePatcher(), "test_lora.safetensors", 1.0, "*: 1.0")
node.apply(FakePatcher(), "test_lora.safetensors", 1.0, "*: 0.5")
check("cache: file read once over two runs", len(loads) == 1, len(loads))
utils_mod.load_torch_file = orig

# --- bad spec raises before touching the model ----------------------------------
try:
    run("0-9: nope")
    check("bad spec raises", False)
except ValueError as e:
    check("bad spec raises", True)
    print("     -> {}".format(e))

print("\n--- sample report ---")
print(run("*: 1.0\n0-9: 0.0\n40-49.mlp: 0.5\nrefiner: 0.0")[1])
print("\nFAILURES:", failures if failures else "none")
sys.exit(1 if failures else 0)
