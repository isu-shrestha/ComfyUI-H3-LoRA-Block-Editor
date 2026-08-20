"""Exercise the parser against the H3 lora key set, no ComfyUI needed."""
import os
import sys
import types
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _keys

# stub out the comfy imports so the module loads standalone
for name in ("comfy", "comfy.lora", "comfy.lora_convert", "comfy.utils", "folder_paths"):
    mod = types.ModuleType(name)
    if name == "folder_paths":
        mod.get_filename_list = lambda _: []
    sys.modules.setdefault(name, mod)
sys.modules["comfy"].lora = sys.modules["comfy.lora"]
sys.modules["comfy"].lora_convert = sys.modules["comfy.lora_convert"]
sys.modules["comfy"].utils = sys.modules["comfy.utils"]

_keys.add_node_to_path()
import h3_lora_block as h3

# model keys exactly as ComfyUI's generic key map produces them
model_keys, source = _keys.model_keys()
print("model keys: {} (from {})".format(len(model_keys), source))

parsed = [h3.parse_model_key(k) for k in model_keys]
assert not [p for p in parsed if p[0] == "other"], "unparsed keys!"
print("sections:", Counter(p[0] for p in parsed))
print("sublayers:", Counter(p[2] for p in parsed))

failures = []


def run(spec, expect_nonzero=None, label=""):
    rules = h3.parse_spec(spec)
    weights = [h3.resolve_weight(rules, *p) for p in parsed]
    nz = sum(1 for w in weights if w != 0.0)
    hist = Counter(weights)
    status = "ok "
    if expect_nonzero is not None and nz != expect_nonzero:
        status = "FAIL"
        failures.append((label, expect_nonzero, nz))
    print("{} {:<24} nonzero={:<4} {}".format(status, label, nz, dict(sorted(hist.items()))))


TOTAL = len(model_keys)          # 208
run("*: 1.0", TOTAL, "all on")
run("*: 0.0", 0, "all off")
run("*: 1.0\nrefiner: 0.0", TOTAL - 8, "refiner off")
run("*: 1.0\n0-9: 0.0", TOTAL - 40, "first 10 blocks off")
run("*: 1.0\n*.mlp: 0.0", TOTAL - 104, "attn only")
run("*: 0.0\n20-35.attn: 1.2", 32, "mid attn only")
run("*: 1.0, 12: 0.5, 40-49.fc2: 0.25", TOTAL, "inline commas + overrides")
run("*: 1.0\nblocks.49.out: -1.0", TOTAL, "negative single layer")
run("all: 1.0\ntr.0: 0.0", TOTAL - 4, "refiner block 0 off")

# override ordering: later line wins
rules = h3.parse_spec("*: 1.0\n0-49: 0.5\n25: 0.1")
assert h3.resolve_weight(rules, "block", 25, "mlp.fc1") == 0.1
assert h3.resolve_weight(rules, "block", 24, "mlp.fc1") == 0.5
assert h3.resolve_weight(rules, "refiner", 0, "mlp.fc1") == 1.0
print("ok  override ordering")

# the slider node
weights_node = h3.H3LoraBlockWeights()
spec = weights_node.build("attn", 0.0, blocks_0_9=1.0, blocks_10_19=0.5,
                          blocks_20_29=0.0, blocks_30_39=0.0, blocks_40_49=1.0)[0]
print("--- slider output ---\n" + spec + "\n---")
rules = h3.parse_spec(spec)
assert h3.resolve_weight(rules, "block", 5, "attn.qkv_proj") == 1.0
assert h3.resolve_weight(rules, "block", 5, "mlp.fc1") == 0.0
assert h3.resolve_weight(rules, "block", 15, "attn.out_proj") == 0.5
assert h3.resolve_weight(rules, "refiner", 1, "attn.qkv_proj") == 0.0
print("ok  slider node round-trip")

# error handling
for bad, why in [("0-9 1.0", "missing colon"), ("0-9: abc", "bad number"), ("0-9.wat: 1", "bad layer")]:
    try:
        h3.parse_spec(bad)
    except ValueError as e:
        print("ok  rejects {:<14} -> {}".format(why, e))
    else:
        failures.append((why, "ValueError", "no error"))

# report rendering, using a realistic resolve
rules = h3.parse_spec("*: 1.0\n0-9: 0.0\n40-49.mlp: 0.5\nrefiner: 0.0")
blocks_seen, refiner_seen, skipped = {}, {}, 0
for key, p in zip(model_keys, parsed):
    w = h3.resolve_weight(rules, *p)
    (blocks_seen if p[0] == "block" else refiner_seen).setdefault(p[1], {})[p[2]] = w
    if w == 0.0:
        skipped += 1
print("--- report ---")
print(h3.format_report("test_lora.safetensors", 1.0, TOTAL - skipped,
                       blocks_seen, refiner_seen, skipped, 0))
print("---")

print("\nFAILURES:", failures if failures else "none")
sys.exit(1 if failures else 0)
