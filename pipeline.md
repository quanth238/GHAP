# Render-Surface Resampling for 3DGS Compaction (Teacher ~2M → Student K=200k–500k)

This is a **simple, effective, and theory-grounded** pipeline you can implement as a new compaction mode in the **GHAP codebase**, while keeping GHAP’s default OT/GMR compaction intact. GHAP itself is a **two-stage post-hoc compaction** method: (a) geometric compaction, then (b) appearance fine-tuning of color/opacity with fewer Gaussians. ([OpenReview][1])
We’ll **reuse GHAP’s stage-2 fine-tuning**, but replace stage-1 with **render-surface sampling + fast resampling**.

---

## 1) Problem setting

### Inputs

* A **teacher 3DGS checkpoint** with **N ≈ 2,000,000** Gaussians.
* Teacher chosen at **iter ≈ 15k**, with **densification already disabled** (so the Gaussian set is “stable”). (This matches the standard 3DGS paradigm where densification/density control is interleaved with optimization. ([arXiv][2]))
* Training cameras/images (e.g., MipNeRF360 bicycle, Tanks&Temples, DeepBlending).

### Output

* A **student** Gaussian set with **K ∈ [200k, 500k]** Gaussians that preserves rendering quality (PSNR/SSIM/LPIPS).

### Goal (what you’re trying to solve)

Typical compaction (prune/merge) can **compress the teacher’s mistakes** if the teacher contains floaters or missing coverage. Your goal is to **rebuild a better spatial support** using the *teacher’s rendered surface evidence*, then fine-tune appearance.

---

## 2) Core idea in one sentence

**Use the teacher renderer to generate a large set of weighted world-space surface samples** ((x_j, w_j)), then **solve a weighted quantization problem** to place **K centers** that represent where Gaussians *should* be, and finally **reinitialize + fine-tune**.

---

## 3) Theory-based formulation (why this isn’t just “intuition”)

### 3.1 Surface-induced target distribution

Sample training views (v \in \mathcal V) and pixels (p \in \Omega_v). From teacher rendering, obtain:

* depth (D_v(p))
* alpha/opacity (or accumulated opacity) (A_v(p))
* (optional) a texture proxy from GT image gradients (G_v(p))

Backproject a world-space surface point:
[
x(v,p) = o_v + D_v(p), r_v(p)
]
where (o_v) is camera origin and (r_v(p)) is the unit ray direction.

Define a **sample weight on the pixel-derived point**:
[
w(v,p) = A_v(p),\Big(1 + \lambda_{\text{tex}} ,\hat G_v(p)\Big)
]
((\hat G) is normalized gradient magnitude; (\lambda_{\text{tex}}\ge 0)).

This induces a **weighted empirical surface distribution**:
[
\hat \rho(x)=\frac{1}{\sum_j w_j}\sum_{j=1}^{M} w_j,\delta(x-x_j)
]
where (x_j=x(v,p)) and (w_j=w(v,p)).

**Why weight is attached to samples, not Gaussians:**
You’re not scoring existing Gaussians; you’re defining *where the surface evidence lies* according to the teacher’s visibility and image content.

---

### 3.2 Compaction as weighted vector quantization (CVT / k-means)

You want **K centers** (Z={z_k}*{k=1}^K) minimizing the weighted quantization energy:
[
E(Z)=\sum*{j=1}^{M} w_j \min_{k} |x_j - z_k|^2
]
This is the discrete analogue of the **CVT / optimal quantization objective**, where Lloyd iterations move centers toward mass centroids. ([FSU People][3])

* If you did full Lloyd/k-means at (K=500k), it’s too expensive.
* So we use a **fast approximation** (voxel aggregation) that still targets the same “centroid of mass” principle.

---

## 4) Pipeline (implementation-ready)

### Stage T0 — Teacher snapshot (your setup)

1. Train standard 3DGS to ~15k iters, stop densification (teacher has ~2M Gaussians).
   3DGS explicitly relies on densification / density control to reach high quality. ([arXiv][2])
2. Freeze teacher parameters.

---

### Stage R — Render-surface sampling → ((x_j,w_j))

**Hyperparameters**

* (V): number of training views used (e.g., 100–400)
* (P): pixels sampled per view (e.g., 5k–30k)
* (M = V\cdot P): total surface samples (target **M ≥ 4K**, preferably **8K**)

**Procedure**
For each selected training view (v):

1. Render teacher RGB and **render teacher depth** (D_v).
2. Render/compute accumulated alpha/opacity (A_v) (or approximate from alpha buffer).
3. Randomly sample (P) pixels (p).
4. For each (p):

   * compute (x_j = o_v + D_v(p),r_v(p))
   * compute (w_j = A_v(p)\big(1+\lambda_{\text{tex}}\hat G_v(p)\big))
   * **Depth gating (strongly recommended):** only keep samples with valid depth and (optionally) “front surface” consistency to reduce occlusion-bridging artifacts (a known risk in splatting due to depth ordering approximations). ([arXiv][4])

**Outputs**: arrays (X\in\mathbb R^{M\times 3}), (W\in\mathbb R^{M})

---

### Stage Q — Fast resampling to K centers (voxel-centroid “approx CVT”)

This is your **scalable replacement** for Lloyd/k-means.

**Hyperparameters**

* voxel size (s) (world units)
* optional: auto search for (s) to reach target occupied voxels (\approx K)

**Voxel hashing**

1. Compute voxel id:
   [
   b_j=\left\lfloor\frac{x_j - x_{\min}}{s}\right\rfloor \in \mathbb Z^3
   ]
2. For each voxel (b), accumulate:
   [
   m_b=\sum_{j: b_j=b} w_j,\qquad
   c_b=\frac{1}{m_b}\sum_{j:b_j=b} w_j x_j
   ]
3. Candidate centers are voxel centroids ({c_b}) with masses ({m_b}).

**Selecting exactly K centers**

* If (#{b}>K): take **top-K** voxels by (m_b).
* If (#{b}<K): decrease (s) (binary search) until close to (K).

**Why this is theory-aligned:**
Within each voxel, you are explicitly computing the **mass centroid** of samples, matching the centroid condition central to CVT. ([FSU People][3])

**Output**: (Z={z_k}*{k=1}^K) where (z_k=c*{b_k})

---

### Stage S — Student reinitialization (build K Gaussians)

You now create a student Gaussian model with exactly K Gaussians.

**Nearest-teacher attribute transfer**

* Build a KD-tree on teacher means ({\mu_i}_{i=1}^N).
* For each center (z_k), find nearest teacher Gaussian (i^*(k)).
* Initialize student attributes by copying (or kNN averaging):

  * position (\mu_k \leftarrow z_k)
  * covariance/scale/rotation from teacher (i^*)
  * opacity from teacher (i^*)
  * SH coefficients (or color params) from teacher (i^*)

**Important guardrail**: do **not** average SH aggressively across unrelated neighbors (that’s where specular/high-freq appearance smears). A safer minimal baseline is “copy nearest”, then rely on fine-tuning.

---

### Stage F — Fine-tune appearance (reuse GHAP stage-2)

GHAP explicitly uses a **second stage** to decouple and fine-tune color/opacity after geometric compaction. ([OpenReview][1])
So:

* Run GHAP’s existing fine-tuning (or standard 3DGS training loss) on the student for 2k–10k iters.
* Optional future improvement: distill student RGB from teacher renders (but not required for your first simple experiment).

---

## 5) What each stage is “doing” (your cost/meaning question)

* **Stage R**: converts *renderer evidence* into a **surface point cloud** (millions of samples) with **importance weights**.
* **Stage Q**: solves **“where should K Gaussians be placed?”** by approximate weighted quantization (voxel mass centroids).
* **Stage S**: turns those K centers into an actual **Gaussian parameter set** by copying plausible local attributes from the teacher.
* **Stage F**: recovers appearance quality, matching GHAP’s design that geometry compaction needs post appearance adjustment. ([OpenReview][1])

---

## 6) Hyperparameters (recommended starting grid)

For **K = 200k–500k**:

**Sampling**

* (V = 200) views
* (P = 10k) pixels/view
* (M = 2,000,000) samples → for K=500k, M=4K (minimum workable); better: (P=20k) → M=4M.

**Weights**

* (\lambda_{\text{tex}} = 0.5) (start)
* Reject samples with (A_v(p) < 0.05) (remove near-empty rays)
* Depth gating ON

**Voxel**

* (s): auto-search via binary search until (#\text{voxels}\approx K)

**Init**

* nearest-teacher copy (kNN=1) first
* then optionally kNN=8 average for scales/opacities only (leave SH copy)

**Fine-tune**

* 2k iters (quick check), then 10k iters (real)

---

## 7) Complexity / runtime (back-of-the-envelope)

Let:

* (N)=teacher gaussians (~2M)
* (K)=target gaussians (200–500k)
* (M)=surface samples (≈ 4K–10K)

Costs:

1. **Rendering** (V) views: (O(V \cdot T_{\text{render}})) (dominant; but you’re rendering *teacher* only a few hundred views).
2. **Voxel aggregation**: (O(M)) time, (O(#\text{voxels})) memory.
3. **KD-tree + NN**: build (O(N \log N)), query (O(K \log N)).
4. **Fine-tune**: same per-iter renderer cost but on **K** gaussians (faster than teacher).

This is generally **much cheaper than full k-means/Lloyd at K=500k**, while still reflecting CVT-style centroid placement. ([FSU People][3])

---

## 8) Notes vs GHAP / related work (to keep you “aligned”)

* GHAP’s native method frames compaction as **global Gaussian mixture reduction via OT**, with KD-tree partitioning and then appearance fine-tuning. ([OpenReview][1])
* Your method is **not OT/GMR**; it’s a **renderer-induced surface distribution + quantization** approach, closer in spirit to “reorganizing distribution” ideas (Mini-Splatting) but implemented as a very scalable voxel-centroid quantizer. ([arXiv][5])
* Depth/visibility care matters in splatting; view consistency and depth ordering approximations are nontrivial (StopThePop highlights this). ([arXiv][4])

---

If you want, paste your GHAP command structure (how it loads checkpoint + where stage-2 is called), and I can map **exactly** where “Stage R/Q/S” should be inserted (which file and what function boundaries) without changing GHAP’s default behavior.

[1]: https://openreview.net/forum?id=j1QkrVjNVF "Gaussian Herding across Pens: An Optimal Transport Perspective on Global Gaussian Reduction for 3DGS | OpenReview"
[2]: https://arxiv.org/abs/2308.04079?utm_source=chatgpt.com "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
[3]: https://people.sc.fsu.edu/~mgunzburger/files_papers/gunzburger-cvt-siamreview.pdf "35283.dvi"
[4]: https://arxiv.org/abs/2402.00525?utm_source=chatgpt.com "StopThePop: Sorted Gaussian Splatting for View-Consistent Real-time Rendering"
[5]: https://arxiv.org/abs/2403.14166?utm_source=chatgpt.com "Mini-Splatting: Representing Scenes with a Constrained Number of Gaussians"
