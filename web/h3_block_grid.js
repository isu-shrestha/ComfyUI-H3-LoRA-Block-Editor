import { app } from "../../scripts/app.js";

// Paints a 4 x 5 grid onto the H3 LoRA Block Loader: one row per weight matrix
// the LoRA touches, one column per bucket of ten transformer blocks. State is kept as JSON in the
// node's hidden "grid" string widget; Python compiles that into its spec DSL, so
// this is purely an editor and the node still works if the widget fails to load.
//
// Three controls decide a cell's weight: the cell itself, its row multiplier,
// and the multiplier for its bucket of ten blocks. They do not compound -- the
// most restrictive wins, and 1.0 means "no opinion" -- so a resolved weight is
// always a number that was actually set. Cells are drawn resolved, so what you
// see is what gets applied.
//
// Clicking anything toggles it between 1.0 and the node's "brush" value, so a
// brush of 0 mutes and 0.5 halves. Dragging paints across a row of cells.

const ROWS = ["qkv", "out", "fc1", "fc2"];
const N_BLOCKS = 50;
const BUCKET_SIZE = 10;
const N_BUCKETS = N_BLOCKS / BUCKET_SIZE;

const PAD = 8;
const LABEL_W = 56; // room for "fc1  0.50"
const ROW_H = 17;
const TICK_H = 12;
const BUCKET_H = 14;
const SPLIT_GAP = 4; // between the attention rows and the mlp rows

const COL_ON = "#5c9ded";
const COL_OFF = "#2b2b2b";
const COL_BOOST = "#e0a458";
const COL_NEG = "#d16a6a";
const COL_BUCKET = "#3b4c5e";
const COL_BUCKET_SET = "#4a7ba8";
const COL_TEXT = "#9a9a9a";
const COL_TEXT_HI = "#d8d8d8";

function emptyState() {
    const state = { rows: {}, buckets: new Array(N_BUCKETS).fill(1) };
    for (const row of ROWS) {
        state[row] = new Array(N_BUCKETS).fill(1);
        state.rows[row] = 1;
    }
    return state;
}

function num(value, fallback = 1) {
    const v = Number(value);
    return Number.isFinite(v) ? v : fallback;
}

function parseState(text) {
    const state = emptyState();
    if (!text) return state;
    try {
        const data = JSON.parse(text);
        if (!data || typeof data !== "object") return state;
        for (const row of ROWS) {
            const cells = data[row];
            if (Array.isArray(cells) && cells.length === N_BUCKETS) {
                state[row] = cells.map((v) => num(v));
            } else if (Array.isArray(cells) && cells.length === N_BLOCKS) {
                // older per-block state: keep the most restrictive value per bucket
                state[row] = [];
                for (let k = 0; k < N_BUCKETS; k++) {
                    const slice = cells.slice(k * BUCKET_SIZE, (k + 1) * BUCKET_SIZE).map((v) => num(v));
                    state[row].push(Math.min.apply(null, slice));
                }
            }
            if (data.rows && typeof data.rows === "object") {
                state.rows[row] = num(data.rows[row]);
            }
        }
        if (Array.isArray(data.buckets) && data.buckets.length === N_BUCKETS) {
            state.buckets = data.buckets.map((v) => num(v));
        }
    } catch (e) {
        // malformed state falls back to "everything on", which emits no rules
    }
    return state;
}

function rowTop(index) {
    return TICK_H + BUCKET_H + index * ROW_H + (index >= 2 ? SPLIT_GAP : 0);
}

function gridHeight() {
    return rowTop(ROWS.length - 1) + ROW_H + PAD;
}

function fmt(v) {
    if (v === 1) return "1";
    if (Number.isInteger(v)) return String(v);
    return v.toFixed(2).replace(/0$/, "");
}

function weightColour(v) {
    if (v > 1) return { style: COL_BOOST, alpha: 1 };
    if (v === 1) return { style: COL_ON, alpha: 1 };
    if (v < 0) return { style: COL_NEG, alpha: 1 };
    if (v === 0) return { style: COL_OFF, alpha: 1 };
    return { style: COL_ON, alpha: Math.max(0.2, v) };
}

function addGridWidget(node, stateWidget) {
    const widget = {
        type: "h3blockgrid",
        name: "block_grid",
        // state lives in the hidden string widget, so don't serialize twice
        serialize: false,
        state: parseState(stateWidget.value),
        painting: null, // "cell" | "bucket"
        paintRow: null,
        paintValue: 1,

        brush() {
            const w = node.widgets && node.widgets.find((x) => x.name === "brush");
            return num(w ? w.value : 0, 0);
        },

        // a cell's applied weight: the most restrictive of the three controls,
        // where 1.0 counts as "no opinion" (mirrors combine() in the python)
        weight(row, bucket) {
            const factors = [
                num(this.state[row][bucket]),
                num(this.state.rows[row]),
                num(this.state.buckets[bucket]),
            ].filter((v) => v !== 1);
            return factors.length ? Math.min.apply(null, factors) : 1;
        },

        computeSize(width) {
            return [width, gridHeight()];
        },

        commit() {
            stateWidget.value = JSON.stringify(this.state);
            node.setDirtyCanvas(true, true);
            if (node.graph) node.graph.change();
        },

        geometry(width) {
            const gridX = PAD + LABEL_W;
            const gridW = Math.max(1, width - gridX - PAD);
            return { gridX, gridW, cellW: gridW / N_BUCKETS };
        },

        draw(ctx, node, width, y) {
            this.lastY = y;
            this.lastWidth = width;
            const { gridX, gridW, cellW } = this.geometry(width);

            ctx.save();
            ctx.font = "9px monospace";
            ctx.textBaseline = "middle";

            // bucket range labels
            ctx.fillStyle = COL_TEXT_HI;
            ctx.textAlign = "left";
            ctx.fillText("reset", PAD, y + TICK_H * 0.5);
            ctx.fillStyle = COL_TEXT;
            ctx.textAlign = "center";
            for (let k = 0; k < N_BUCKETS; k++) {
                const lo = k * BUCKET_SIZE;
                ctx.fillText(lo + "-" + (lo + BUCKET_SIZE - 1),
                             gridX + (k + 0.5) * cellW, y + TICK_H * 0.5);
            }

            // one multiplier per bucket, covering all four rows
            const barTop = y + TICK_H;
            ctx.textAlign = "left";
            ctx.fillStyle = COL_TEXT;
            ctx.fillText("blocks", PAD, barTop + BUCKET_H * 0.5);
            for (let k = 0; k < N_BUCKETS; k++) {
                const v = num(this.state.buckets[k]);
                const x = gridX + k * cellW;
                ctx.fillStyle = v === 1 ? COL_BUCKET : COL_BUCKET_SET;
                ctx.fillRect(x, barTop + 1, cellW - 2, BUCKET_H - 3);
                ctx.fillStyle = v === 1 ? COL_TEXT : COL_TEXT_HI;
                ctx.textAlign = "center";
                ctx.fillText(fmt(v), x + cellW * 0.5, barTop + BUCKET_H * 0.5);
            }
            ctx.textAlign = "left";

            for (let r = 0; r < ROWS.length; r++) {
                const row = ROWS[r];
                const top = y + rowTop(r);
                const rowMult = num(this.state.rows[row]);

                ctx.textAlign = "left";
                ctx.fillStyle = COL_TEXT_HI;
                ctx.fillText(row, PAD, top + ROW_H * 0.5);
                ctx.textAlign = "right";
                ctx.fillStyle = rowMult === 1 ? COL_TEXT : COL_TEXT_HI;
                ctx.fillText(fmt(rowMult), gridX - 4, top + ROW_H * 0.5);
                ctx.textAlign = "left";

                for (let k = 0; k < N_BUCKETS; k++) {
                    const v = this.weight(row, k);
                    const x = gridX + k * cellW;
                    const w = cellW - 2;
                    const { style, alpha } = weightColour(v);
                    if (alpha < 1) {
                        ctx.fillStyle = COL_OFF;
                        ctx.fillRect(x, top + 1, w, ROW_H - 3);
                    }
                    ctx.fillStyle = style;
                    ctx.globalAlpha = alpha;
                    ctx.fillRect(x, top + 1, w, ROW_H - 3);
                    ctx.globalAlpha = 1;

                    ctx.fillStyle = v === 0 ? COL_TEXT : "#10151b";
                    ctx.textAlign = "center";
                    ctx.fillText(fmt(v), x + cellW * 0.5, top + ROW_H * 0.5);
                    ctx.textAlign = "left";
                }
            }

            ctx.restore();
        },

        hit(pos) {
            const x = pos[0];
            const y = pos[1] - this.lastY;
            const { gridX, cellW } = this.geometry(this.lastWidth);
            const bucket = Math.floor((x - gridX) / cellW);
            const inGrid = x >= gridX && bucket >= 0 && bucket < N_BUCKETS;

            if (y < 0 || y > gridHeight()) return null;
            if (y < TICK_H) return x < gridX ? { kind: "reset" } : null;
            if (y < TICK_H + BUCKET_H) return inGrid ? { kind: "bucket", bucket } : null;

            for (let r = 0; r < ROWS.length; r++) {
                const top = rowTop(r);
                if (y >= top && y < top + ROW_H) {
                    if (x < gridX) return { kind: "row", row: ROWS[r] };
                    if (inGrid) return { kind: "cell", row: ROWS[r], bucket };
                    return null;
                }
            }
            return null;
        },

        isPristine() {
            if (this.state.buckets.some((v) => num(v) !== 1)) return false;
            for (const r of ROWS) {
                if (num(this.state.rows[r]) !== 1) return false;
                if (this.state[r].some((v) => num(v) !== 1)) return false;
            }
            return true;
        },

        mouse(event, pos) {
            const type = event.type;

            if (type === "pointerdown" || type === "mousedown") {
                const target = this.hit(pos);
                if (!target) return false;
                const brush = this.brush();

                if (target.kind === "reset") {
                    this.state = emptyState();
                    this.commit();
                    return true;
                }
                if (target.kind === "row") {
                    const current = num(this.state.rows[target.row]);
                    this.state.rows[target.row] = current === 1 ? brush : 1;
                    this.commit();
                    return true;
                }
                if (target.kind === "bucket") {
                    this.paintValue = num(this.state.buckets[target.bucket]) === 1 ? brush : 1;
                    this.state.buckets[target.bucket] = this.paintValue;
                    this.painting = "bucket";
                    this.commit();
                    return true;
                }
                this.paintValue = num(this.state[target.row][target.bucket]) === 1 ? brush : 1;
                this.state[target.row][target.bucket] = this.paintValue;
                this.painting = "cell";
                this.paintRow = target.row;
                this.commit();
                return true;
            }

            if ((type === "pointermove" || type === "mousemove") && this.painting) {
                const target = this.hit(pos);
                if (!target) return true;
                if (this.painting === "bucket" && target.kind === "bucket") {
                    if (num(this.state.buckets[target.bucket]) !== this.paintValue) {
                        this.state.buckets[target.bucket] = this.paintValue;
                        this.commit();
                    }
                } else if (this.painting === "cell" && target.kind === "cell") {
                    // stay on the row the drag started in, so a stray vertical
                    // wobble doesn't paint a neighbouring row
                    const row = this.paintRow;
                    if (num(this.state[row][target.bucket]) !== this.paintValue) {
                        this.state[row][target.bucket] = this.paintValue;
                        this.commit();
                    }
                }
                return true;
            }

            if (type === "pointerup" || type === "mouseup") {
                if (this.painting) {
                    this.painting = null;
                    this.paintRow = null;
                    this.commit();
                    return true;
                }
            }

            return false;
        },
    };

    node.addCustomWidget(widget);
    return widget;
}

function hideWidget(widget) {
    widget.hidden = true;
    widget.computeSize = () => [0, -4];
    widget.draw = () => {};
}

app.registerExtension({
    name: "h3.lora.block.grid",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "H3LoraBlockLoader") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            const stateWidget = this.widgets && this.widgets.find((w) => w.name === "grid");
            if (!stateWidget) return result;
            hideWidget(stateWidget);

            this.h3grid = addGridWidget(this, stateWidget);
            if (this.size[0] < 440) this.size[0] = 440;
            this.setSize(this.computeSize());
            return result;
        };

        // reload the painted state when a saved workflow is restored
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            const stateWidget = this.widgets && this.widgets.find((w) => w.name === "grid");
            if (stateWidget && this.h3grid) {
                this.h3grid.state = parseState(stateWidget.value);
                this.setDirtyCanvas(true, true);
            }
            return result;
        };
    },
});
