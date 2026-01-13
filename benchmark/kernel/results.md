--- ModelNet10 ---
GCN - 84.7%
GraphSAGE - 82.6%
GIN - 87.9%
Set2SetNet - 87.2%
DiffPool - 87.7%
Graclus - 87.1%
TopK - 86.2%
SAGPool - 87.7%
LaCore - 90.4%
ASAP - 88.4%
EdgePool - 86.5%

--- ModelNet40 ---
GCN: 83.44% ± 1.20% (7:06)
Set2Set: 73.47% ± 2.64% (8:53)
Graclus: 83.89% ± 0.80% (8:11)
DiffPool (num_layers=4): 82.91% ± 1.07% (32:32)
TopK: 78.50% ± 4.54% (6:59)
SAG: 79.34% ± 3.14% (7:49)

--- ModelNet10 rerun ---
GCN: 88.8%
GIN: 88.5%
Set2Set: 87.7%
ASAP: 89.8%

Train epochs: 100%|#######| 300/300 [16:23<00:00,  3.28s/it]
Val Metric: 0.2840, Test Accuracy: 0.859 ± 0.000, Duration: 983.835
Train epochs: 100%|#######| 300/300 [18:06<00:00,  3.62s/it]
Val Metric: 0.2254, Test Accuracy: 0.861 ± 0.000, Duration: 1086.451
Best config: {'num_layers': 5, 'hidden': 128}
Best result - 0.861 ± 0.000

**To-do list:** (out of date, ignore this)
* Global-Attention
* GMT
* Quasi-CliquePool
* GPN
* MinCutPool
* SEP

------ ChatGPT Suggestion
If you want a **concrete, reviewer-proof appendix result**, aim for a table that answers one clean question:

> “If we keep the *same point-cloud→graph preprocessing* and the *same GNN training recipe*, does swapping the pooling/coarsening operator to LaCore help?”

That means: **one fixed backbone for the pooling methods**, plus **variance**, plus **a clearly defined evaluation protocol**.

Below is exactly what I would add, starting from the end result (the table), and then the minimum runs you need to populate it.

---

## The table you should produce (the one that actually helps your paper)

### Table A (main “generality” table): pooling comparison under a fixed backbone

Two datasets, same graph construction, same encoder capacity.

**Columns (keep it simple):**

* ModelNet40 accuracy (%, mean ± std)
* ScanObjectNN accuracy (%, mean ± std) — pick *one* canonical variant and stick to it
* Optional but nice: “Pooling params” (0 vs learned) to underline “training-free pooling”

**Rows (the methods you listed, but organized so it reads cleanly):**

* Flat/readout (no hierarchical pooling): GCN, Set2Set
* Algorithmic coarsening: Graclus, LaCore
* Learned pooling: DiffPool, TopKPool, SAGPool, ASAP, EdgePool

Here’s an exact LaTeX template you can drop into your appendix:

```latex
\begin{table*}[t]
\centering
\small
\caption{\textbf{Geometric graphs induced from point clouds.}
Accuracy (\%, mean$\pm$std over 10 seeds).
We sample 1024 points, normalize to the unit sphere, and build an undirected 16-NN graph
(\texttt{KNNGraph}(k=16, \texttt{force\_undirected=True})).
All pooling/coarsening methods are evaluated as drop-in replacements within the \emph{same}
2-layer GCN backbone and identical optimization budget.}
\label{tab:pc_generalization}
\setlength{\tabcolsep}{6pt}
\begin{tabular}{lccc}
\toprule
\textbf{Method} & \textbf{Pooling params} & \textbf{ModelNet40} & \textbf{ScanObjectNN (PB\_T50\_RS)} \\
\midrule
\multicolumn{4}{l}{\emph{Flat/readout (no hierarchical pooling)}}\\
GCN + GlobalMeanMax     & --      &  \;\;\;-- & -- \\
GCN + Set2Set           & learned &  \;\;\;-- & -- \\
\midrule
\multicolumn{4}{l}{\emph{Pooling / coarsening (inserted between GCN layers)}}\\
Graclus                 & 0       &  \;\;\;-- & -- \\
DiffPool                & learned &  \;\;\;-- & -- \\
TopKPool                & learned &  \;\;\;-- & -- \\
SAGPool                 & learned &  \;\;\;-- & -- \\
ASAP                    & learned &  \;\;\;-- & -- \\
EdgePool                & learned &  \;\;\;-- & -- \\
\textbf{LaCore (ours)}  & \textbf{0} & \textbf{--} & \textbf{--} \\
\bottomrule
\end{tabular}
\end{table*}
```

That table is “concrete” in the sense that:

* it matches the story of your paper (pooling operator),
* it isolates the effect you claim (pooling/cohesion bias),
* it doesn’t invite arguments about mixing apples/oranges.

---

## What you need to run to populate Table A (minimum, very explicit)

### 1) Fix the backbone and training recipe for **all pooling rows**

Pick one backbone (GCN is fine since it matches the main paper). Make it explicit:

* 2-layer GCN
* hidden dim = H (keep constant)
* pooling inserted between layer 1 and layer 2
* readout = concat(global mean + global max) → MLP
* identical optimizer / LR schedule / epochs / batch size

This is what makes the comparison interpretable.

### 2) Use **10 seeds** per method per dataset

Yes: **10 runs** is the right target if you want the table to look “real” instead of anecdotal.

Each entry becomes: mean ± std over seeds.

Practical detail that matters: keep the *data split fixed* across methods and seeds (or if the dataset has an official split, use that split and only vary training randomness).

### 3) Control sampling randomness (important for point clouds)

Your pipeline includes “sample 1024 points.” If that sampling is random at test time, your reported accuracy will include extra noise.

Do this:

* **Train**: random sampling each epoch is fine (augmentation).
* **Test**: use a deterministic sampling for each shape (e.g., fix the RNG seed per example, or precompute the 1024 indices once and store them).

This alone can noticeably stabilize std across seeds.

### 4) Hyperparameters: keep tuning minimal but fair

For a side appendix, you do **not** need a full grid search per method. But you do need to avoid “defaults for baselines, tuned for ours.”

A fair lightweight tuning protocol:

* Tune **one shared training grid** once (LR × dropout) on the *GCN + GlobalMeanMax* baseline.
* Then **reuse the same LR/dropout** for all pooling methods.
* For each pooling method, tune only its **one main pooling knob** on a tiny grid (e.g., pooling ratio ∈ {0.25, 0.5} or whatever is standard in your code).
* For LaCore, tune (\varepsilon) on a log-grid (like you already do), plus target_ratio/min_size if those exist in your point-cloud setup.

That’s enough to prevent “you didn’t tune baselines” complaints without ballooning work.

---

## What about GIN / GraphSAGE / “each model wants its own backbone”?

Don’t put GIN/GraphSAGE in Table A unless you’re willing to expand the table massively.

Instead, if you want to keep them (and it can be useful), make a **separate tiny context table** that does *not* mix with pooling claims.

### Table B (optional, small): encoder context (no pooling)

This answers: “How strong is the base encoder on these induced graphs?”

```latex
\begin{table}[t]
\centering
\small
\caption{\textbf{Encoder context on induced point-cloud graphs (no hierarchical pooling).}
Accuracy (\%, mean$\pm$std over 10 seeds) with the same preprocessing and readout.}
\label{tab:pc_encoder_context}
\setlength{\tabcolsep}{6pt}
\begin{tabular}{lcc}
\toprule
\textbf{Backbone (global pooling)} & \textbf{ModelNet40} & \textbf{ScanObjectNN (PB\_T50\_RS)} \\
\midrule
GCN + GlobalMeanMax      & -- & -- \\
GraphSAGE + GlobalMeanMax& -- & -- \\
GIN + GlobalMeanMax      & -- & -- \\
\bottomrule
\end{tabular}
\end{table}
```

This way:

* Table A supports your pooling claim.
* Table B satisfies your intuition that different encoders matter, without contaminating the pooling comparison.

---

## What to write in the appendix text (2–4 sentences)

Right above Table A, include something like:

* Graph construction (exactly what you already wrote: 1024 points, unit sphere, kNN16 undirected)
* “All methods use the same preprocessing and the same GCN backbone; pooling methods are drop-in replacements.”
* “Reported numbers are mean±std over 10 seeds; test uses deterministic sampling.”

That’s it.

---

## If you do only ONE thing: do this

Produce **Table A** (GCN backbone fixed, 10 seeds, mean±std) on:

* ModelNet40
* ScanObjectNN PB_T50_RS (or whichever single ScanObjectNN setting you choose)

That’s the most concrete, high-signal addition that strengthens your paper without becoming a new project.

------

## Implementation (current repo)

### Table A runner (fixed backbone)

Use `benchmark/kernel/table_a.py` to run the fixed-backbone pooling comparison on ModelNet40 + ScanObjectNN with mean±std over seeds.

Default graph construction (matches the table):

* 1024 points
* `NormalizeScale()`
* `KNNGraph(k=16, force_undirected=True)`
* `x = pos`

ScanObjectNN defaults:

* variant: `PB_T50_RS`
* split_dir: `main_split`
* resample: `irregular` (materialized once for deterministic eval)

Example command (10 seeds, all methods):

```
python benchmark/kernel/table_a.py \
  --datasets ModelNet40,ScanObjectNN \
  --methods gcn,set2set,graclus,diffpool,topk,sag,asap,edge,lacore \
  --dataset_root ./data \
  --data_seed 0
```

Notes:

* For deterministic ScanObjectNN point sampling, use `--scan_resample take_first`.
* `diffpool` runs on dense graphs and can be memory-heavy; drop it from `--methods` if needed.
