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

import json
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

GRID_ROWS = ("qkv", "out", "fc1", "fc2")
NUM_BLOCKS = 50
BUCKET_SIZE = 10
NUM_BUCKETS = NUM_BLOCKS // BUCKET_SIZE


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


def build_spec(token_refiner, grid_state, block_weights):
    """Assemble the full spec: refiner widget, then the grid, then any wired text.

    Everything defaults to 1.0, so only departures from that need a rule. Later
    lines win, which is why the wired text can override anything the grid says.
    """
    parts = ["refiner: {:.4g}".format(token_refiner)]
    from_grid = spec_from_grid(grid_state)
    if from_grid:
        parts.append(from_grid)
    if block_weights and block_weights.strip():
        parts.append(block_weights)
    return "\n".join(parts)


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
                                       "tooltip": "Global multiplier applied on top of every cell in the grid."}),
                "brush": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.05,
                                    "tooltip": "What painting a cell sets it to. Clicking toggles between "
                                               "1.0 and this value, so 0 mutes and 0.5 halves."}),
                "token_refiner": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05,
                                            "tooltip": "Strength for the 2 text-side refiner blocks, which the "
                                                       "grid does not cover. Set to 0 first when a LoRA won't "
                                                       "mix -- it is only 8 of 208 tensors."}),
                "grid": ("STRING", {"default": "",
                                    "tooltip": "Painted by the grid widget. Untouched it emits nothing "
                                               "and the whole LoRA applies at full strength."}),
            },
            "optional": {
                "block_weights": ("STRING", {"forceInput": True,
                                             "tooltip": "Optional text spec, applied after the grid so it "
                                                        "overrides. Only needed for things the grid cannot "
                                                        "say, like 'refiner.0.attn: 0.5'."}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    OUTPUT_TOOLTIPS = ("The patched diffusion model.", "Resolved per-block strengths, for sanity checking.")
    FUNCTION = "apply"
    CATEGORY = "loaders/h3"
    DESCRIPTION = ("Applies a LoRA to a MiniMax H3 model with a separate strength per transformer "
                   "block and per weight matrix, painted on a 4x50 grid. Cells at 0 are skipped "
                   "entirely, which is the usual fix when a LoRA refuses to share a stack with "
                   "another one.")

    def apply(self, model, lora_name, strength, brush, token_refiner, grid="", block_weights=""):
        del brush  # the grid widget reads it in the frontend; nothing to do here
        rules = parse_spec(build_spec(token_refiner, grid, block_weights))

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


def _number(value, default=1.0):
    return float(value) if isinstance(value, (int, float)) else default


def combine(*factors):
    """The most restrictive setting wins.

    1.0 means "no opinion", so it never constrains anything. Among the factors
    that do express an opinion the lowest wins, which keeps a resolved weight
    equal to a number the user actually typed: fc1 at 0.5 crossing a bucket at
    0.5 stays 0.5 rather than compounding to 0.25. With nothing set anywhere the
    result is 1.0, and a lone boost above 1.0 survives because it is the only
    opinion on offer.
    """
    opinions = [f for f in factors if f != 1.0]
    return min(opinions) if opinions else 1.0


def grid_weights(grid_state):
    """Resolve the grid's three controls into {row: [50 weights]}.

    State is {"qkv": [50 cells], "out": [...], "fc1": [...], "fc2": [...],
              "rows": {row: multiplier}, "buckets": [5 multipliers]}.
    Each bucket covers ten blocks, matching the grid's tick marks.
    """
    if not grid_state or not grid_state.strip():
        return {}
    try:
        data = json.loads(grid_state)
    except ValueError:
        logging.warning("H3 LoRA Block Loader: grid state is not valid JSON, ignoring it")
        return {}
    if not isinstance(data, dict):
        return {}

    row_mults = data.get("rows") if isinstance(data.get("rows"), dict) else {}
    buckets = data.get("buckets") if isinstance(data.get("buckets"), list) else []

    resolved = {}
    for row in GRID_ROWS:
        cells = data.get(row)
        if not isinstance(cells, list) or not cells:
            continue
        row_mult = _number(row_mults.get(row))
        weights = []
        for index, cell in enumerate(cells):
            bucket = index // BUCKET_SIZE
            bucket_mult = _number(buckets[bucket]) if bucket < len(buckets) else 1.0
            weights.append(combine(_number(cell), row_mult, bucket_mult))
        resolved[row] = weights
    return resolved


def spec_from_grid(grid_state):
    """Compile the grid's resolved weights into spec lines.

    Only weights that differ from 1.0 emit a rule, and runs of equal weights
    collapse into ranges, so an untouched grid produces nothing at all.
    """
    lines = []
    for row, weights in grid_weights(grid_state).items():
        start = 0
        for i in range(1, len(weights) + 1):
            if i == len(weights) or weights[i] != weights[start]:
                weight = weights[start]
                if weight != 1.0:
                    lo, hi = start, i - 1
                    label = str(lo) if lo == hi else "{}-{}".format(lo, hi)
                    lines.append("{}.{}: {:.4g}".format(label, row, weight))
                start = i
    return "\n".join(lines)


NODE_CLASS_MAPPINGS = {
    "H3LoraBlockLoader": H3LoraBlockLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3LoraBlockLoader": "H3 LoRA Block Loader",
}
