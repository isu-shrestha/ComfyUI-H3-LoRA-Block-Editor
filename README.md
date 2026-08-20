# H3 LoRA Block Editor

Per-block LoRA strength control for MiniMax H3 in ComfyUI. Built for the case where a
LoRA works alone but fights other LoRAs when stacked — muting the blocks that carry the
conflict usually fixes it without giving up the parts you want.

## The node

One node, `H3 LoRA Block Loader` (`loaders/h3`):

```
model ●                                        ● model
                                               ● report
  lora_name              my_h3_lora.safetensors
  strength                                1.00
  token_refiner                           1.00

reset   mult   0-9   10-19  20-29  30-39  40-49
mult         [  1  ][  1  ][ 0.5 ][  1  ][  0  ]   <- column weights
qkv       1  [  1  ][  1  ][ 0.5 ][  1  ][  0  ]
out       1  [  1  ][  1  ][ 0.5 ][  0  ][  0  ]
fc1     0.5  [ 0.5 ][ 0.5 ][ 0.5 ][ 0.5 ][  0  ]
fc2       1  [  1  ][  1  ][ 0.5 ][  1  ][  0  ]
          ^ row weights
● block_weights
```

The weights live on the axes; the cells are on/off:

| control | what it does |
|---|---|
| **row weight** (the `mult` column) | sets that weight matrix across all 50 blocks |
| **column weight** (the `mult` row) | sets all four matrices across ten blocks |
| **cell** | switches one matrix on or off for one bucket of ten blocks |

- **click a row or column weight** to type a value
- **click a cell** to toggle it on or off; **drag** along a row to flip several
- **click `reset`** to put everything back to `1.0`

Every cell shows its resolved weight, so the grid reads as a table of the numbers that
will actually be applied.

### They do not compound

`1.0` means "no opinion". Among the controls that do have an opinion, **the most
restrictive wins**. So `fc1` at `0.5` crossing a bucket at `0.5` stays `0.5`, not `0.25` —
a resolved weight is always a number you actually typed. A lone value above `1.0` survives
for the same reason: nothing else is competing with it.

Cells shade by resolved weight: solid blue at `1.0`, dimmed for partial, grey at `0`, orange above `1.0`,
red when negative.

Ten blocks per bucket is the granularity on offer. For anything finer, the `block_weights`
socket still addresses individual blocks (`25.out: 0`), and it is applied after the grid.

`strength` scales everything, so you can sweep the whole LoRA without repainting.

`token_refiner` covers the 2 text-side refiner blocks, which the grid does **not** include
— the grid is the 50 main blocks only.

`block_weights` is an optional socket for a text spec, applied after the grid so it
overrides. Only needed for what the grid cannot say, like `refiner.0.attn: 0.5`.

## What H3 loras actually contain

Rank-16 ai-toolkit LoRAs on this architecture patch 208 tensors:

| section | count | layers |
|---|---|---|
| `blocks.0` … `blocks.49` | 50 blocks × 4 | `attn.qkv_proj`, `attn.out_proj`, `mlp.fc1`, `mlp.fc2` |
| `token_refiner.blocks.0` … `.1` | 2 blocks × 4 | same four |

`token_refiner` sits on the **text** side — it reshapes prompt embeddings before they
reach the video stack. When two LoRAs both patch it they compete over the same
conditioning signal, so `refiner: 0.0` is the cheapest first experiment.

## Spec syntax (optional `block_weights` socket)

Rarely needed — the grid covers per-block control. One `<selector>: <weight>` per line (commas
also work). **Later lines override earlier ones**, and the spec lands after the grid, so
you only write the exceptions. `#` starts a comment.

```
*: 1.0          # baseline for everything
0-9: 0.0        # then mute the first ten blocks
40-49.mlp: 0.5  # and ease off the tail MLPs
```

Selectors:

| part | forms |
|---|---|
| section | `*` / `all` (everything), `refiner` / `token_refiner` / `tr`, `blocks` |
| index | `12`, `0-9`, omitted or `*` for any |
| layer | `.attn`, `.mlp`, `.qkv`, `.out`, `.fc1`, `.fc2` — omitted means all four |

Combine them: `refiner.0.attn: 0.5`, `20-35.attn: 1.2`, `*.mlp: 0.0`, `blocks.49.out: -1.0`.

The `strength` widget multiplies every resolved weight, so you can sweep the whole LoRA
without editing the spec.

## Recipes for a LoRA that won't mix

All of these are on the one node. Try them in order:

| try | how |
|---|---|
| stop competing over prompt conditioning | `token_refiner` → `0` |
| attention only (MLPs carry most memorized content) | set the `fc1` and `fc2` row weights to `0` |
| drop early blocks (coarse motion / layout) | set the `0-9` column weight to `0` |
| drop late blocks (fine texture / detail) | set the `40-49` column weight to `0` |
| back the whole thing off, as a control | `strength` → `0.6` |

`token_refiner` → `0` is the cheapest first test: it's only 8 of 208 tensors, so you keep
~96% of the LoRA while removing the part most likely to collide.

Read the `report` output to confirm what landed:

```
my_h3_lora.safetensors  x1

block          qkv     out     fc1     fc2
0-9              0       0       0       0
10-39            1       1       1       1
40-49            1       1     0.5     0.5

refiner        qkv     out     fc1     fc2
0-1              0       0       0       0

patched 160 tensors, 48 muted at 0
```

If the report warns that lora keys did not match the model, the LoRA is not for the model
you connected.

## Install

Clone into your ComfyUI `custom_nodes` directory and restart:

```bash
git clone https://github.com/isu-shrestha/ComfyUI-H3-LoRA-Block-Editor
```

To develop it outside the ComfyUI tree instead, link it in so edits apply on the next
restart with no copying — from an elevated-free shell on Windows:

```bash
New-Item -ItemType Junction -Path "<ComfyUI>\custom_nodes\ComfyUI-H3-LoRA-Block-Editor" -Target "<repo>"
```

The LoRA itself must be under ComfyUI's `models/loras` to appear in the dropdown.

## Tests

Run without ComfyUI installed — comfy internals are stubbed, and the key set is
synthesized, so no weights are needed:

```bash
python tests/test_spec.py && python tests/test_apply.py
```

To run against real weights instead, point `H3_LORA_PATH` at a MiniMax H3 lora:

```bash
H3_LORA_PATH=/path/to/lora.safetensors python tests/test_spec.py
```

`test_spec.py` covers selector parsing and weight resolution over all 208 keys.
`test_apply.py` covers the node's `apply()` — patch grouping, zero-skipping, strength
scaling, file caching, and error handling.

## Notes and limits

- `attn.qkv_proj` is a fused Q/K/V projection, so Q, K and V cannot be weighted
  separately without splitting the `lora_B` rows. `.qkv` targets all three together.
- Applies to the diffusion model only. H3 LoRAs in this format carry no CLIP/text-encoder
  tensors, so there is nothing to patch on that side.
- The key parser keys off `diffusion_model.blocks.N.…`, which other block-structured
  architectures also use — but the sublayer names here are H3's.
