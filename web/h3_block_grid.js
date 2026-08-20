import { app } from "../../scripts/app.js";

// Paints a 4 x 50 grid (attention + mlp weight matrices, one column per
// transformer block) onto the H3 LoRA Block Loader. State is kept as JSON in the
// node's hidden "grid" string widget; Python compiles that into its spec DSL, so
// this is purely an editor and the node still works if the widget fails to load.
//
// Clicking a cell toggles it between 1.0 and the node's "brush" value, so a brush
// of 0 mutes and 0.5 halves. Dragging paints, and the label column flips whole
// rows (or everything, from the "all" label in the header).

const ROWS = ["qkv", "out", "fc1", "fc2"];
const N_BLOCKS = 50;

const PAD = 8;
const LABEL_W = 30;
const ROW_H = 15;
const HEADER_H = 15;
const SPLIT_GAP = 4; // between the attention rows and the mlp rows

const COL_ON = "#5c9ded";
const COL_OFF = "#2b2b2b";
const COL_BOOST = "#e0a458";
const COL_NEG = "#d16a6a";
const COL_LINE = "#1a1a1a";
const COL_TEXT = "#9a9a9a";
const COL_TEXT_HI = "#d8d8d8";

function emptyState() {
    const state = {};
    for (const row of ROWS) state[row] = new Array(N_BLOCKS).fill(1);
    return state;
}

function parseState(text) {
    const state = emptyState();
    if (!text) return state;
    try {
        const data = JSON.parse(text);
        if (!data || typeof data !== "object") return state;
        for (const row of ROWS) {
            const values = data[row];
            if (Array.isArray(values) && values.length === N_BLOCKS) {
                state[row] = values.map((v) => (typeof v === "number" ? v : 1));
            }
        }
    } catch (e) {
        // malformed state falls back to "everything on", which emits no rules
    }
    return state;
}

function rowTop(index) {
    return HEADER_H + index * ROW_H + (index >= 2 ? SPLIT_GAP : 0);
}

function gridHeight() {
    return rowTop(ROWS.length - 1) + ROW_H + PAD;
}

function addGridWidget(node, stateWidget) {
    const widget = {
        type: "h3blockgrid",
        name: "block_grid",
        // state lives in the hidden string widget, so don't serialize twice
        serialize: false,
        state: parseState(stateWidget.value),
        painting: false,
        paintValue: 0,

        brush() {
            const w = node.widgets && node.widgets.find((x) => x.name === "brush");
            const v = w ? Number(w.value) : 0;
            return Number.isFinite(v) ? v : 0;
        },

        computeSize(width) {
            return [width, gridHeight()];
        },

        commit() {
            stateWidget.value = JSON.stringify(this.state);
            node.setDirtyCanvas(true, true);
            if (node.graph) node.graph.change();
        },

        draw(ctx, node, width, y) {
            this.lastY = y;
            this.lastWidth = width;

            const gridX = PAD + LABEL_W;
            const gridW = Math.max(1, width - gridX - PAD);
            const cellW = gridW / N_BLOCKS;

            ctx.save();
            ctx.font = "9px monospace";
            ctx.textBaseline = "middle";

            // block index ticks
            ctx.fillStyle = COL_TEXT;
            ctx.textAlign = "left";
            for (let b = 0; b < N_BLOCKS; b += 10) {
                ctx.fillText(String(b), gridX + b * cellW, y + HEADER_H * 0.5);
            }
            ctx.textAlign = "right";
            ctx.fillText("49", gridX + gridW, y + HEADER_H * 0.5);

            // "all" toggle in the label column of the header
            ctx.textAlign = "left";
            ctx.fillStyle = COL_TEXT_HI;
            ctx.fillText("all", PAD, y + HEADER_H * 0.5);

            for (let r = 0; r < ROWS.length; r++) {
                const row = ROWS[r];
                const values = this.state[row];
                const top = y + rowTop(r);

                ctx.fillStyle = COL_TEXT_HI;
                ctx.textAlign = "left";
                ctx.fillText(row, PAD, top + ROW_H * 0.5);

                for (let b = 0; b < N_BLOCKS; b++) {
                    const v = values[b];
                    const x = gridX + b * cellW;
                    if (v > 1) {
                        ctx.fillStyle = COL_BOOST; // above full strength
                    } else if (v === 1) {
                        ctx.fillStyle = COL_ON;
                    } else if (v < 0) {
                        ctx.fillStyle = COL_NEG; // inverted
                    } else if (v === 0) {
                        ctx.fillStyle = COL_OFF;
                    } else {
                        // partial strength draws dimmed over the off colour
                        ctx.fillStyle = COL_OFF;
                        ctx.fillRect(x, top + 1, Math.max(1, cellW - 1), ROW_H - 3);
                        ctx.fillStyle = COL_ON;
                        ctx.globalAlpha = Math.max(0.2, v);
                    }
                    ctx.fillRect(x, top + 1, Math.max(1, cellW - 1), ROW_H - 3);
                    ctx.globalAlpha = 1;
                }

                // tick separators every 10 blocks
                ctx.strokeStyle = COL_LINE;
                ctx.lineWidth = 1;
                for (let b = 10; b < N_BLOCKS; b += 10) {
                    const x = Math.round(gridX + b * cellW) + 0.5;
                    ctx.beginPath();
                    ctx.moveTo(x, top + 1);
                    ctx.lineTo(x, top + ROW_H - 2);
                    ctx.stroke();
                }
            }

            ctx.restore();
        },

        hit(pos) {
            const x = pos[0];
            const y = pos[1] - this.lastY;
            const gridX = PAD + LABEL_W;
            const gridW = Math.max(1, this.lastWidth - gridX - PAD);
            const cellW = gridW / N_BLOCKS;

            if (y < 0 || y > gridHeight()) return null;
            if (y < HEADER_H) return x < gridX ? { kind: "all" } : null;

            for (let r = 0; r < ROWS.length; r++) {
                const top = rowTop(r);
                if (y >= top && y < top + ROW_H) {
                    if (x < gridX) return { kind: "row", row: ROWS[r] };
                    const b = Math.floor((x - gridX) / cellW);
                    if (b < 0 || b >= N_BLOCKS) return null;
                    return { kind: "cell", row: ROWS[r], block: b };
                }
            }
            return null;
        },

        isPainted(values) {
            return values.some((v) => v !== 1);
        },

        mouse(event, pos, node) {
            const type = event.type;

            if (type === "pointerdown" || type === "mousedown") {
                const target = this.hit(pos);
                if (!target) return false;

                const brush = this.brush();

                if (target.kind === "all") {
                    const painted = ROWS.some((r) => this.isPainted(this.state[r]));
                    for (const r of ROWS) this.state[r] = new Array(N_BLOCKS).fill(painted ? 1 : brush);
                    this.commit();
                    return true;
                }
                if (target.kind === "row") {
                    const painted = this.isPainted(this.state[target.row]);
                    this.state[target.row] = new Array(N_BLOCKS).fill(painted ? 1 : brush);
                    this.commit();
                    return true;
                }
                // painting a cell: the first cell decides whether we paint or restore
                this.paintValue = this.state[target.row][target.block] === 1 ? brush : 1;
                this.state[target.row][target.block] = this.paintValue;
                this.painting = true;
                this.commit();
                return true;
            }

            if ((type === "pointermove" || type === "mousemove") && this.painting) {
                const target = this.hit(pos);
                if (target && target.kind === "cell") {
                    if (this.state[target.row][target.block] !== this.paintValue) {
                        this.state[target.row][target.block] = this.paintValue;
                        this.commit();
                    }
                }
                return true;
            }

            if (type === "pointerup" || type === "mouseup") {
                if (this.painting) {
                    this.painting = false;
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
            if (this.size[0] < 420) this.size[0] = 420;
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
