"""Build the two router diagrams as Excalidraw files (fixed grid, hand-placed).

  router_mechanism.excalidraw  how ZOIB and IRT-Router score a query and pick a model
  router_testing.excalidraw    data prep -> train/val training -> test-only benchmark

Labels name the files / functions and carry the formulas; update them here when the
code changes, then re-run. The layout is fixed rather than auto (dagre) because
auto-layout ignored the side-by-side lanes.

    python scripts/diagrams/build_router_diagrams.py
    python scripts/diagrams/build_router_diagrams.py --export path/to/excalidraw/export.mjs

--export runs that exporter (from the Excalidraw Claude Code plugin; needs Node) to
write .svg/.png next to each file. Without it, open the .excalidraw files in
excalidraw.com or the VS Code Excalidraw extension.
"""
import argparse
import json
import subprocess
import random
import sys
import time
from pathlib import Path

PAL = {"blue": ("#a5d8ff", "#1971c2"), "green": ("#b2f2bb", "#2f9e44"),
       "orange": ("#ffd8a8", "#e8590c"), "purple": ("#d0bfff", "#7048e8"),
       "yellow": ("#ffec99", "#f08c00"), "gray": ("#dee2e6", "#495057"),
       "ghost": ("#ffffff", "#adb5bd")}
INK = "#1e1e1e"
LINE_H = 20          # fontSize 16 * lineHeight 1.25


def _base(id_, type_, x, y, w, h):
    return {"id": id_, "type": type_, "x": x, "y": y, "width": w, "height": h, "angle": 0,
            "strokeColor": INK, "backgroundColor": "transparent", "fillStyle": "solid",
            "strokeWidth": 2, "strokeStyle": "solid", "roughness": 1, "opacity": 100,
            "seed": random.randint(1, 2**31 - 1), "version": 1,
            "versionNonce": random.randint(1, 2**31 - 1), "isDeleted": False, "groupIds": [],
            "frameId": None, "roundness": None, "boundElements": [],
            "updated": int(time.time() * 1000), "link": None, "locked": False}


class Diagram:
    def __init__(self):
        self.els, self.n = [], 0

    def _id(self, p):
        self.n += 1
        return f"{p}-{self.n}"

    def text(self, x, y, s, *, size=16, color=INK, align="left", w=None, container=None, h=None):
        lines = s.split("\n")
        w = w or max(len(l) for l in lines) * size * 0.55
        h = h or len(lines) * size * 1.25
        t = _base(self._id("text"), "text", x, y, w, h)
        t.update({"strokeColor": color, "strokeWidth": 1, "text": s, "originalText": s,
                  "fontSize": size, "fontFamily": 1, "textAlign": align,
                  "verticalAlign": "middle" if container else "top", "containerId": container,
                  "autoResize": True, "lineHeight": 1.25})
        self.els.append(t)
        return t

    def box(self, x, y, w, label, color="gray", h=None, dashed=False):
        lines = label.split("\n")
        h = h or len(lines) * LINE_H + 28
        bg, st = PAL[color]
        r = _base(self._id("rect"), "rectangle", x, y, w, h)
        r.update({"strokeColor": st, "backgroundColor": bg, "roundness": {"type": 3},
                  "strokeStyle": "dashed" if dashed else "solid"})
        self.els.append(r)
        t = self.text(x + 10, y + 6, label, color=st, align="center", w=w - 20,
                      container=r["id"], h=h - 12)
        r["boundElements"].append({"id": t["id"], "type": "text"})
        return r

    def frame(self, x, y, w, h, label):
        r = _base(self._id("frame"), "rectangle", x, y, w, h)
        r.update({"strokeColor": "#495057", "backgroundColor": "#f8f9fa",
                  "strokeStyle": "dashed", "opacity": 60, "roundness": {"type": 3}})
        self.els.insert(0, r)
        t = self.text(x + 12, y + 8, label, size=15, color="#343a40")
        self.els.remove(t)
        self.els.insert(1, t)

    def arrow(self, pts, label=None, at=None, color=INK, dashed=False, lsize=14):
        x0, y0 = pts[0]
        a = _base(self._id("arrow"), "arrow", x0, y0,
                  max(p[0] for p in pts) - min(p[0] for p in pts),
                  max(p[1] for p in pts) - min(p[1] for p in pts))
        a.update({"strokeColor": color, "roundness": {"type": 2},
                  "strokeStyle": "dashed" if dashed else "solid",
                  "points": [[px - x0, py - y0] for px, py in pts], "lastCommittedPoint": None,
                  "startBinding": None, "endBinding": None, "startArrowhead": None,
                  "endArrowhead": "arrow"})
        self.els.append(a)
        if label:
            self.text(at[0], at[1], label, size=lsize, color=color)

    def save(self, path):
        doc = {"type": "excalidraw", "version": 2, "source": "agentic-llm-router",
               "elements": self.els, "appState": {"viewBackgroundColor": "#ffffff", "gridSize": None},
               "files": {}}
        Path(path).write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")


Z, I = "#2f9e44", "#7048e8"


# =========================================================== diagram 1: mechanism
def mechanism(path):
    d = Diagram()
    ZX, IX, W = 60, 560, 360          # lane box x, width
    ZC, IC = ZX + W // 2, IX + W // 2  # lane centre lines
    d.frame(30, 10, 1110, 140, "Inputs (computed once per query; every candidate m is scored)")
    d.box(60, 46, 150, "Query text q", "gray", h=84)
    d.box(240, 46, 320, "TextEncoder\ne_q ∈ ℝ^768 (BERT, mean-pool)\nrouter/embeddings/encoder.py", "blue", h=84)
    d.arrow([(210, 88), (238, 88)])
    d.box(600, 46, 170, "Candidate LLM m\n(29-model pool)", "gray", h=84)
    d.box(810, 46, 310, "Profile text → e_m ∈ ℝ^768\nstore: model_profile__irt\n(same TextEncoder)", "gray", h=84)
    d.arrow([(770, 88), (808, 88)])

    rows = [("ability θ", 104), ("topic", 104), ("a, b heads", 164), ("logit z", 144),
            ("quality", 144), ("cost", 164)]
    ys, y = [], 220
    for _, h in rows:
        ys.append(y)
        y += h + 60
    H = dict(zip(range(6), [h for _, h in rows]))
    d.frame(10, 170, 430, ys[5] + H[5] + 30 - 170, "ZOIB NIRT router (ours)  ·  BaselineNIRT, training/nirt/baseline/model.py")
    d.frame(540, 170, 450, ys[5] + H[5] + 30 - 170, "IRT-Router (paper, K = 25)  ·  IRTRouterModel, router/nirt/model.py")
    for (lbl, h), yy in zip(rows, ys):
        d.text(445, yy + h - 26, lbl, size=14, color="#495057", align="center", w=90)

    zl = [
        "Ability  θ_m = Θ[m] ∈ ℝ^8\n(free learned row per LLM)\nBaselineNIRT.latent_ability",
        "Topic  r_q = softmax(cos(e_q, μ_c) / τ)\n(soft membership in C clusters)\ntaxonomy/relevance.py\n· relevance_from_embeddings",
        "Discrimination / difficulty\na_q = softplus(f_a(e_q))\n        ⊙ σ(W_r r_q + b_r)\nb_q = f_b(e_q)     (MLP, hidden 64)\nDiscriminationHead, DifficultyHead\nrouter/nirt/components.py",
        "Latent logit\nz = a_q·θ_m − b_q\n     + γ·g([a_q, θ_m, a_q⊙θ_m, r_q])\n(γ starts at 0 → pure IRT)\nInteractionLayer (components.py)",
        "ZOIB response head\nπ = softmax(f_π[e_q, θ_m])\nμ = σ(z),   κ = softplus(f_κ) + ε\nŶ = E[Y] = π_1 + π_c · μ\nbaseline/continuous_zoib.py\n· ZOIBResponseHead.mean",
        "Length cost head (detached input)\nP(b) = softmax(log prior_m\n             + h([e_q, θ_m]))\nĈ = p_in·L_in + p_out·Σ_b P(b)·v_b\nOutputLengthHead (components.py)\nprices: tokens.fit_token_prices\n(NNLS on train cells)",
    ]
    il = [
        "Ability  θ_m = σ(W_θ e_m) ∈ ℝ^25\n(projected from the profile)\nIRTRouterModel.latent_ability",
        None,
        "Linear heads, no topic gate\na_q = W_a e_q\nb_q = w_b · e_q\nnirt/model.py · _query_head",
        "Latent logit\nz = a_q·θ_m − b_q\n(no interaction term)\nIRTRouterModel.forward",
        "Bernoulli head\np = σ(z)\nsoft-BCE training\ntraining/nirt/train.py · fit",
        "Static cost\nC(m) = mean train cost of m\n(same value for every query)",
    ]
    for i in range(6):
        d.box(ZX, ys[i], W, zl[i], "green", h=H[i])
        if il[i] is None:
            d.box(IX, ys[i], W, "no topic signal", "ghost", h=H[i], dashed=True)
        else:
            d.box(IX, ys[i], W, il[i], "purple", h=H[i])
    mid = lambda i: ys[i] + H[i] // 2
    bot = lambda i: ys[i] + H[i]

    # ZOIB lane wiring
    d.arrow([(520, 130), (520, 160), (430, 160), (430, mid(1)), (ZX + W + 2, mid(1))], "e_q", (434, ys[1] - 26), Z)
    d.arrow([(520, 130), (520, 160), (430, 160), (430, mid(2)), (ZX + W + 2, mid(2))], "e_q", (434, ys[2] - 26), Z)
    d.arrow([(ZC, bot(1)), (ZC, ys[2] - 2)], "r_q gates a_q", (ZC + 8, bot(1) + 18), Z)
    d.arrow([(ZC, bot(2)), (ZC, ys[3] - 2)], "a_q, b_q", (ZC + 8, bot(2) + 18), Z)
    d.arrow([(ZC, bot(3)), (ZC, ys[4] - 2)], "z → μ = σ(z)", (ZC + 8, bot(3) + 18), Z)
    d.arrow([(ZX, mid(0)), (32, mid(0)), (32, mid(3)), (ZX - 2, mid(3))], "θ_m", (16, mid(0) - 24), Z)
    d.arrow([(32, mid(3)), (32, mid(5)), (ZX - 2, mid(5))], "no grad", (14, mid(5) + 6), Z, dashed=True, lsize=12)
    d.arrow([(640, 130), (640, 200), (480, 200), (480, mid(0)), (ZX + W + 2, mid(0))], "index m", (488, 206), INK)

    # IRT lane wiring
    d.arrow([(900, 130), (900, 190), (IC + 60, 190), (IC + 60, ys[0] - 2)], "e_m", (IC + 68, 196), I)
    d.arrow([(548, 130), (548, mid(2)), (IX - 2, mid(2))], "e_q", (552, mid(2) - 26), I)
    d.arrow([(IC, bot(2)), (IC, ys[3] - 2)], "a_q, b_q", (IC + 8, bot(2) + 18), I)
    d.arrow([(IC, bot(3)), (IC, ys[4] - 2)], None, None, I)
    d.arrow([(IX + W, mid(0)), (945, mid(0)), (945, mid(3)), (IX + W + 2, mid(3))], "θ_m", (950, mid(0) - 24), I)
    d.arrow([(740, 130), (740, 150), (970, 150), (970, mid(5)), (IX + W + 2, mid(5))], "per-model avg", (976, mid(5) - 28), I)

    # decision
    DY = bot(5) + 110
    d.arrow([(ZX + W, mid(4)), (450, mid(4)), (450, DY - 2)], "Ŷ_qm", (456, DY - 40), Z)
    d.arrow([(300, bot(5)), (300, DY - 2)], "Ĉ_qm", (308, DY - 40), Z)
    d.arrow([(IX, mid(4)), (530, mid(4)), (530, DY - 2)], "p_qm", (536, DY - 40), I)
    d.arrow([(680, bot(5)), (680, DY - 2)], "C(m)", (688, DY - 40), I)
    d.box(250, DY, 520, "Decide\nm̂(q) = argmax_m [ quality_qm − λ · cost_qm ]\nnon-finite cells → −∞;  exact ties → first column\nλ = 0: best predicted quality; larger λ: cheaper\nrouter/nirt/routing_decision.py\n· cost_aware_utility, routing_decision", "yellow")
    CY = DY + 6 * LINE_H + 28 + 60
    d.arrow([(510, CY - 60), (510, CY - 2)], "m̂", (518, CY - 40))
    d.box(410, CY, 200, "Send q to m̂\n(LLM call)", "gray")
    d.save(path)


# =========================================================== diagram 2: testing
def testing(path):
    d = Diagram()
    d.frame(0, 0, 1370, 210, "Data prep")
    top = [("Raw sources\nRouterBench pickles\nIRT-Router csv\nscripts/data/download_*.py", "gray"),
           ("Normalize + price\nloaders.load_routerbench /\nload_irt_router\npricing.impute_missing_usage\ncost = p_in·L_in + p_out·L_out", "blue"),
           ("responses.parquet\n1.90M rows, all sources\nbuild_response_matrix.py", "orange"),
           ("nirt_observations.parquet\n1,449,159 (q, m) cells\n29 LLMs · y ∈ [0,1] · USD cost\nnirt.build_nirt_observations", "orange"),
           ("Split by query hash\ntrain 80% · val 10%\ntest 10%\nsplits.make_splits", "yellow")]
    xs = [20, 290, 560, 830, 1100]
    for (lbl, c), x in zip(top, xs):
        d.box(x, 40, 250, lbl, c, h=150)
    for a, b in zip(xs, xs[1:]):
        d.arrow([(a + 250, 115), (b - 2, 115)])

    d.frame(0, 260, 680, 790, "Training (train + val queries)")

    # training
    fz = d.box(20, 330, 310, "Fit ZOIB\nL = mean −log p_ZOIB(y | q, m)\n   + λ_θ‖θ‖² + λ_c‖mean θ‖²\n   + λ_b‖b‖²\nAdam 1e-3 · batch 4096\nbaseline/train.py · fit", "green")
    es = d.box(20, 560, 310, "Early stop on val ZOIB NLL\npatience 8 → best epoch 15\nbaseline/train.py · _mean_nll", "green")
    pr = d.box(20, 720, 310, "Refit length cost probe\non frozen quality weights\nCE over 14 token buckets\nprices: NNLS on train cells\ntrain.py · _fit_cost_probe", "green")
    fi = d.box(350, 330, 310, "Fit IRT-Router (K = 25)\nsoft-BCE\n−[y log p + (1−y) log(1−p)]\nlr 2e-3 · batch 512\nstop on val BCE\nscripts/nirt/train_nirt.py", "purple")
    ck = d.box(20, 940, 640, "Frozen checkpoints\nartifacts/phase2/zoib_cost_length_v2\ndata/processed/nirt_runs/irt-25d-projected-29m", "orange")
    b = lambda r: r["y"] + r["height"]
    d.arrow([(1150, 190), (1150, 240), (175, 240), (175, 328)], "train 1.16M cells · val 144k (early stop only)", (185, 216), INK)
    d.arrow([(1170, 190), (1170, 250), (505, 250), (505, 328)], "train / val", (513, 276), INK)
    d.arrow([(175, b(fz)), (175, es["y"] - 2)], "each epoch", (183, b(fz) + 22), Z)
    d.arrow([(175, b(es)), (175, pr["y"] - 2)], "best state", (183, b(es) + 22), Z)
    d.arrow([(175, b(pr)), (175, ck["y"] - 2)], None, None, Z)
    d.arrow([(505, b(fi)), (505, ck["y"] - 2)], "best val epoch", (513, 700), I)

    # testing
    E = [("Dense test matrices Y[q, m], C[q, m]\nRouterBench 7,354 × 11\nIRT-Router 3,790 × 17\nrouting.eval_matrices, align\n(evaluation/nirt/routing.py)", "blue"),
         ("Predict every test cell\nZOIB: Ŷ, Ĉ   (routing_benchmark.baseline_cell_predictions)\nIRT-Router: p   (predict.predict_matrix)", "blue"),
         ("Policies for each λ ∈ {0, 10, 20, 50, 100, 200, 1000}\nrouter:   argmax_m (Ŷ − λ·Ĉ)\nGPT-4:   fixed column (gpt-4-1106 / gpt_4o)\noracle@λ:   argmax_m (Y − λ·C)\nmatch-GPT-4 oracle:  min C  s.t.  Y ≥ Y_gpt4", "yellow"),
         ("Score on realized outcomes\nquality = mean_q Y[q, m̂]\ncost per 1k = 1000 · mean_q C[q, m̂]\nΔ vs GPT-4 · share sent to GPT-4\ncombined = concat RouterBench + IRT-Router\nrouting_benchmark.source_picks, summarize\nscripts/nirt/baseline/routing_benchmark.py", "green"),
         ("Results JSON\n<ckpt>/routing_benchmark_test.json\nartifacts/phase2/zoib_vs_irt_router_test.json", "orange")]
    y = 330
    ey = []
    for lbl, c in E:
        r = d.box(730, y, 620, lbl, c)
        ey.append((y, r["height"]))
        y += r["height"] + 60
    d.frame(710, 260, 660, ey[-1][0] + ey[-1][1] + 30 - 260, "Testing (test queries only)")
    d.arrow([(1250, 190), (1250, 328)], "test 145k cells only", (1258, 276), INK)
    for (y0, h0), (y1, _), l in zip(ey, ey[1:], ["query × model grid", "", "m̂ per query", ""]):
        d.arrow([(1040, y0 + h0), (1040, y1 - 2)], l or None, (1048, y0 + h0 + 20) if l else None, INK)
    py = ey[1][0] + ey[1][1] // 2
    cy = ck["y"] + ck["height"] // 2
    d.arrow([(660, cy), (695, cy), (695, py), (728, py)])
    d.text(560, (cy + py) // 2 - 10, "frozen weights", size=14, align="right", w=128)
    d.text(20, 1080, "Leakage guards: splits are by query, so no test prompt is\nseen in training. The kNN warm-up bank, token prices and\nper-model costs C(m) come from train cells only; val only\npicks the epoch.", size=14, color="#495057")
    d.save(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="docs/diagrams")
    ap.add_argument("--export", default=None, help="path to the Excalidraw plugin's export.mjs")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / "router_mechanism.excalidraw", out / "router_testing.excalidraw"]
    mechanism(paths[0])
    testing(paths[1])
    for p in paths:
        print(f"wrote {p}")
        if args.export:
            subprocess.run(["node", args.export, str(p)], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
