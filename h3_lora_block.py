"""Per-block LoRA application for MiniMax H3.

H3 loras (ai-toolkit format) patch, per transformer block:
    diffusion_model.blocks.{0..49}.attn.qkv_proj
    diffusion_model.blocks.{0..49}.attn.out_proj
    diffusion_model.blocks.{0..49}.mlp.fc1
    diffusion_model.blocks.{0..49}.mlp.fc2
plus the text-side refiner:
    diffusion_model.token_refiner.blocks.{0..1}.<same four>

These land on the model unchanged through ComfyUI's generic lora key map, so
applying a per-block strength is just a matter of splitting the patch dict by
block and calling add_patches() once per distinct strength.
"""

import logging
import re

import comfy.lora
import comfy.lora_convert
import comfy.utils
import folder_paths

ALL_SUBLAYERS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")

SUBLAYER_ALIASES = {
    "attn": ("attn.qkv_proj", "attn.out_proj"),
    "qkv": ("attn.qkv_proj",),
    "qkv_proj": ("attn.qkv_proj",),
    "out": ("attn.out_proj",),
    "out_proj": ("attn.out_proj",),
    "mlp": ("mlp.fc1", "mlp.fc2"),
    "ffn": ("mlp.fc1", "mlp.fc2"),
    "fc1": ("mlp.fc1",),
    "fc2": ("mlp.fc2",),
}

REFINER_NAMES = ("refiner", "token_refiner", "tr", "text")
BLOCK_NAMES = ("block", "blocks")
WILDCARDS = ("*", "all")

KEY_RE = re.compile(r"^diffusion_model\.(token_refiner\.)?blocks\.(\d+)\.(.+)\.weight$")

DEFAULT_SPEC = """# every entry is  <selector>: <weight>   -- later lines override earlier ones
*: 1.0

# --- things to try, uncomment one at a time ---
# refiner: 0.0        # drop the text-refiner patch (best first test for lora conflicts)
# 0-9: 0.0            # mute the first 10 blocks
# 40-49: 0.5          # ease off the last 10
# *.mlp: 0.0          # attention only
# 20-35.attn: 1.2     # push mid-block attention
"""


def parse_model_key(key):
    """diffusion_model.blocks.12.attn.qkv_proj.weight -> ('block', 12, 'attn.qkv_proj')"""
    m = KEY_RE.match(key)
    if m is None:
        return ("other", -1, "")
    return ("refiner" if m.group(1) else "block", int(m.group(2)), m.group(3))


def parse_selector(selector):
    """-> (section, (lo, hi), sublayers); None in any slot means 'matches anything'."""
    tokens = [t for t in selector.strip().lower().split(".") if t]
    if not tokens:
        raise ValueError("empty selector")

    section = None
    index = None
    sublayers = None

    if tokens[0] in WILDCARDS:
        tokens = tokens[1:]
    elif tokens[0] in REFINER_NAMES:
        section = "refiner"
        tokens = tokens[1:]
    elif tokens[0] in BLOCK_NAMES:
        section = "block"
        tokens = tokens[1:]

    if tokens:
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", tokens[0])
        if m is not None:
            lo = int(m.group(1))
            hi = int(m.group(2)) if m.group(2) is not None else lo
            index = (min(lo, hi), max(lo, hi))
            if section is None:
                section = "block"
            tokens = tokens[1:]
        elif tokens[0] in WILDCARDS:
            tokens = tokens[1:]

    if tokens:
        rest = ".".join(tokens)
        if rest in SUBLAYER_ALIASES:
            sublayers = SUBLAYER_ALIASES[rest]
        elif rest in ALL_SUBLAYERS:
            sublayers = (rest,)
        else:
            raise ValueError("unknown layer '{}' (try one of {})".format(
                rest, ", ".join(sorted(SUBLAYER_ALIASES))))

    return section, index, sublayers


def parse_spec(spec):
    """Parse the block_weights text into an ordered list of (selector, weight)."""
    rules = []
    for lineno, raw in enumerate(spec.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        for chunk in line.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if ":" not in chunk:
                raise ValueError("line {}: expected '<selector>: <weight>', got '{}'".format(lineno, chunk))
            sel, _, val = chunk.partition(":")
            try:
                weight = float(val.strip())
            except ValueError:
                raise ValueError("line {}: '{}' is not a number".format(lineno, val.strip()))
            try:
                rules.append((parse_selector(sel), weight))
            except ValueError as e:
                raise ValueError("line {}: {}".format(lineno, e))
    return rules


def matches(rule_sel, section, index, sublayer):
    sec, idx, subs = rule_sel
    if sec is not None and sec != section:
        return False
    if idx is not None and not (idx[0] <= index <= idx[1]):
        return False
    if subs is not None and sublayer not in subs:
        return False
    return True


def resolve_weight(rules, section, index, sublayer):
    """Last matching rule wins; unmatched keys default to 1.0."""
    weight = 1.0
    for rule_sel, value in rules:
        if matches(rule_sel, section, index, sublayer):
            weight = value
    return weight


def format_report(lora_name, strength, applied, blocks_seen, refiner_seen, skipped, unloaded):
    """Compact table of the resolved per-block strengths, collapsing equal rows."""
    lines = ["{}  x{:.3g}".format(lora_name, strength)]

    def table(title, seen):
        if not seen:
            return
        lines.append("")
        lines.append("{:<10}{:>8}{:>8}{:>8}{:>8}".format(title, "qkv", "out", "fc1", "fc2"))
        rows = []
        for idx in sorted(seen):
            vals = tuple(seen[idx].get(s) for s in ALL_SUBLAYERS)
            if rows and rows[-1][2] == vals:
                rows[-1][1] = idx
            else:
                rows.append([idx, idx, vals])
        for lo, hi, vals in rows:
            label = str(lo) if lo == hi else "{}-{}".format(lo, hi)
            cells = "".join("{:>8}".format("-" if v is None else "{:.3g}".format(v)) for v in vals)
            lines.append("{:<10}{}".format(label, cells))

    table("block", blocks_seen)
    table("refiner", refiner_seen)

    lines.append("")
    lines.append("patched {} tensors, {} muted at 0".format(applied, skipped))
    if unloaded:
        lines.append("WARNING: {} lora keys did not match this model -- wrong architecture?".format(unloaded))
    if applied == 0:
        lines.append("WARNING: nothing was applied")
    return "\n".join(lines)


class H3LoraBlockLoader:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The H3 diffusion model to patch."}),
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The LoRA to apply."}),
                "strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01,
                                       "tooltip": "Global multiplier applied on top of every block weight."}),
                "block_weights": ("STRING", {"multiline": True, "default": DEFAULT_SPEC,
                                             "tooltip": "One '<selector>: <weight>' per line. "
                                                        "Selectors: '*', '0-9', '12', 'refiner', "
                                                        "and an optional layer suffix "
                                                        "('.attn', '.mlp', '.qkv', '.out', '.fc1', '.fc2'). "
                                                        "Later lines override earlier ones."}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    OUTPUT_TOOLTIPS = ("The patched diffusion model.", "Resolved per-block strengths, for sanity checking.")
    FUNCTION = "apply"
    CATEGORY = "loaders/h3"
    DESCRIPTION = ("Applies a LoRA to a MiniMax H3 model with a separate strength per transformer "
                   "block (and per attn/mlp sublayer). Blocks set to 0 are skipped entirely, which "
                   "is the usual fix when a LoRA refuses to share a stack with another one.")

    def apply(self, model, lora_name, strength, block_weights):
        rules = parse_spec(block_weights)

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        if self.loaded_lora is not None and self.loaded_lora[0] == lora_path:
            lora, lora_metadata = self.loaded_lora[1], self.loaded_lora[2]
        else:
            lora, lora_metadata = comfy.utils.load_torch_file(lora_path, safe_load=True, return_metadata=True)
            self.loaded_lora = (lora_path, lora, lora_metadata)

        lora = comfy.lora_convert.convert_lora(lora)
        key_map = comfy.lora.model_lora_keys_unet(model.model, {})
        patches = comfy.lora.load_lora(lora, key_map, log_missing=False)

        groups = {}
        blocks_seen = {}
        refiner_seen = {}
        skipped = 0

        for key, patch in patches.items():
            section, index, sublayer = parse_model_key(key)
            final = strength * resolve_weight(rules, section, index, sublayer)

            seen = blocks_seen if section == "block" else refiner_seen if section == "refiner" else None
            if seen is not None:
                seen.setdefault(index, {})[sublayer] = final

            if final == 0.0:
                skipped += 1
                continue
            groups.setdefault(round(final, 6), {})[key] = patch

        patched = model.clone()
        applied = 0
        for group_strength, subset in groups.items():
            applied += len(patched.add_patches(subset, group_strength))
        if lora_metadata:
            patched.set_attachments("lora_metadata", lora_metadata)

        # count distinct A/B stems rather than halving the tensor count, so loras
        # carrying alpha/dora_scale entries don't trip a spurious warning
        stems = {k.rsplit(".lora_", 1)[0] for k in lora if ".lora_" in k}
        unloaded = max(0, len(stems) - len(patches))
        report = format_report(lora_name, strength, applied, blocks_seen, refiner_seen, skipped, unloaded)
        logging.info("H3 LoRA block loader:\n%s", report)
        return (patched, report)


class H3LoraBlockWeights:
    """Slider front-end that emits a block_weights spec, for quick A/B iteration."""

    GROUPS = ((0, 9), (10, 19), (20, 29), (30, 39), (40, 49))
    TARGETS = ("all", "attn", "mlp", "qkv", "out", "fc1", "fc2")

    @classmethod
    def INPUT_TYPES(cls):
        slider = {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}
        required = {
            "target": (cls.TARGETS, {"default": "all",
                                     "tooltip": "Which sublayers the sliders drive. "
                                                "Anything other than 'all' mutes the rest of the LoRA."}),
        }
        for lo, hi in cls.GROUPS:
            required["blocks_{}_{}".format(lo, hi)] = ("FLOAT", dict(slider))
        required["token_refiner"] = ("FLOAT", dict(slider, tooltip="Strength for the text-side refiner blocks."))
        return {"required": required}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("block_weights",)
    FUNCTION = "build"
    CATEGORY = "loaders/h3"
    DESCRIPTION = "Builds a block_weights spec from sliders. Plug into H3 LoRA Block Loader."

    def build(self, target, token_refiner, **kwargs):
        suffix = "" if target == "all" else "." + target
        lines = ["*: 0.0" if suffix else "*: 1.0"]
        for lo, hi in self.GROUPS:
            value = kwargs["blocks_{}_{}".format(lo, hi)]
            lines.append("{}-{}{}: {:.3g}".format(lo, hi, suffix, value))
        lines.append("refiner{}: {:.3g}".format(suffix, token_refiner))
        return ("\n".join(lines),)


NODE_CLASS_MAPPINGS = {
    "H3LoraBlockLoader": H3LoraBlockLoader,
    "H3LoraBlockWeights": H3LoraBlockWeights,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3LoraBlockLoader": "H3 LoRA Block Loader",
    "H3LoraBlockWeights": "H3 LoRA Block Weights",
}
