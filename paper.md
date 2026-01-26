%%%%%%%% ICML 2026 EXAMPLE LATEX SUBMISSION FILE %%%%%%%%%%%%%%%%%

\documentclass{article}

% Recommended, but optional, packages for figures and better typesetting:
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{subcaption}
\usepackage{booktabs} % for professional tables

% hyperref makes hyperlinks in the resulting PDF.
% If your build breaks (sometimes temporarily if a hyperlink spans a page)
% please comment out the following usepackage line and replace
% \usepackage{icml2026} with \usepackage[nohyperref]{icml2026} above.
\usepackage{hyperref}
\usepackage{algorithmic}
\usepackage{algpseudocode}

% Attempt to make hyperref and algorithmic work together better:
\newcommand{\theHalgorithm}{\arabic{algorithm}}

% Use the following line for the initial blind version submitted for review:
\usepackage{icml2026}

% For preprint, use
% \usepackage[preprint]{icml2026}

% If accepted, instead use the following line for the camera-ready submission:
% \usepackage[accepted]{icml2026}

\usepackage{multirow}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{mathtools}
\usepackage{amsthm}

\usepackage{colortbl}
\usepackage[table]{xcolor}

% Define the ranking colors (Pastel Red, Orange, Yellow)
\definecolor{best}{HTML}{FFC7CE}   % Rank 1
\definecolor{second}{HTML}{FFDCB2} % Rank 2
\definecolor{third}{HTML}{FFFACD}  % Rank 3

% if you use cleveref..
\usepackage[capitalize,noabbrev]{cleveref}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% THEOREMS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\theoremstyle{plain}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{corollary}[theorem]{Corollary}
\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}
\newtheorem{assumption}[theorem]{Assumption}
\theoremstyle{remark}
\newtheorem{remark}[theorem]{Remark}

% --- Theorem environments ---
\newcommand{\G}{\mathcal{G}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\Spp}{\mathbb{S}_{++}}
\newcommand{\tr}{\mathrm{tr}}
\newcommand{\Wtwo}{W_2}
\newcommand{\lb}{\mathrm{lb}}
\newcommand{\norm}[1]{\left\lVert #1 \right\rVert}
\newcommand{\N}{\mathcal{N}}
\newcommand{\E}{\mathbb{E}}
\newcommand{\1}{\mathbf{1}}
\newcommand{\dd}{\,\mathrm{d}}

% for comments
\newcommand{\binh}[1]{{\textbf{\color{red}{Binh: #1}}}}

% Todonotes is useful during development; simply uncomment the next line
%    and comment out the line below the next line to turn off comments
%\usepackage[disable,textsize=tiny]{todonotes}
\usepackage[textsize=tiny]{todonotes}

% The \icmltitle you define below is probably too long as a header.
% Therefore, a short form for the running title is supplied here:
\icmltitlerunning{Submission and Formatting Instructions for ICML 2026}

\begin{document}

\twocolumn[
  \icmltitle{3D Gaussian Splatting Compactation as a Discrete Resampling Problem}

  % It is OKAY to include author information, even for blind submissions: the
  % style file will automatically remove it for you unless you've provided
  % the [accepted] option to the icml2026 package.

  % List of affiliations: The first argument should be a (short) identifier you
  % will use later to specify author affiliations Academic affiliations
  % should list Department, University, City, Region, Country Industry
  % affiliations should list Company, City, Region, Country

  % You can specify symbols, otherwise they are numbered in order. Ideally, you
  % should not use this facility. Affiliations will be numbered in order of
  % appearance and this is the preferred way.
  \icmlsetsymbol{equal}{*}

  \begin{icmlauthorlist}
    \icmlauthor{Firstname1 Lastname1}{equal,yyy}
    \icmlauthor{Firstname2 Lastname2}{equal,yyy,comp}
    \icmlauthor{Firstname3 Lastname3}{comp}
    \icmlauthor{Firstname4 Lastname4}{sch}
    \icmlauthor{Firstname5 Lastname5}{yyy}
    \icmlauthor{Firstname6 Lastname6}{sch,yyy,comp}
    \icmlauthor{Firstname7 Lastname7}{comp}
    %\icmlauthor{}{sch}
    \icmlauthor{Firstname8 Lastname8}{sch}
    \icmlauthor{Firstname8 Lastname8}{yyy,comp}
    %\icmlauthor{}{sch}
    %\icmlauthor{}{sch}
  \end{icmlauthorlist}

  \icmlaffiliation{yyy}{Department of XXX, University of YYY, Location, Country}
  \icmlaffiliation{comp}{Company Name, Location, Country}
  \icmlaffiliation{sch}{School of ZZZ, Institute of WWW, Location, Country}

  \icmlcorrespondingauthor{Firstname1 Lastname1}{first1.last1@xxx.edu}
  \icmlcorrespondingauthor{Firstname2 Lastname2}{first2.last2@www.uk}

  % You may provide any keywords that you find helpful for describing your
  % paper; these are used to populate the "keywords" metadata in the PDF but
  % will not be shown in the document
  \icmlkeywords{Machine Learning, ICML}

  \vskip 0.3in
]

% this must go after the closing bracket ] following \twocolumn[ ...

% This command actually creates the footnote in the first column listing the
% affiliations and the copyright notice. The command takes one argument, which
% is text to display at the start of the footnote. The \icmlEqualContribution
% command is standard text for equal contribution. Remove it (just {}) if you
% do not need this facility.

% Use ONE of the following lines. DO NOT remove the command.
% If you have no special notice, KEEP empty braces:
\printAffiliationsAndNotice{}  % no special notice (required even if empty)
% Or, if applicable, use the standard equal contribution text:
% \printAffiliationsAndNotice{\icmlEqualContribution}

\begin{abstract}
    To be written.
\end{abstract}

\section{Introduction}
3D Gaussian Splatting (3DGS) has emerged as a practical radiance-field representation that achieves high-quality novel-view synthesis with real-time rendering by representing a scene as a set of anisotropic Gaussian primitives and a fast, visibility-aware rasterizer~\citep{kerbl2023gaussians}.
Despite its efficiency, state-of-the-art reconstructions routinely require millions of Gaussians for a single unbounded scene, which stresses memory, storage, and bandwidth for deployment and downstream applications.

A growing line of work studies \emph{compaction}, or the task of reducing the number of Gaussians while preserving rendering fidelity. Representative approaches include pruning/distillation/quantization pipelines (e.g., LightGaussian)~\citep{fan2024lightgaussian} and global Gaussian mixture reduction from an optimal-transport viewpoint (GHAP)~\citep{wang2025ghap} \binh{need a lot more citations}.
In parallel, initialization improvements such as EDGS aim to shorten training trajectories by eliminating iterative densification and preserving high-frequency details from the start~\citep{kotovenko2025edgs}.

\paragraph{Motivation: compaction is not just geometric clustering.}
Most compaction methods can be viewed as (i) pruning by an importance score or (ii) geometric merging/clustering followed by re-optimization.
However, 3DGS surfaces typically occupy a thin, highly non-linear subset of $\mathbb{R}^3$ (``on-manifold'' support): most of the visual signal is concentrated near visible surfaces, with complex multi-layer visibility and occlusion.
Centroid-based operations (e.g., voxel centroids, $k$-means centers, Gaussian merging) can drift off this thin support and create \emph{off-manifold} primitives that are difficult to repair without significant fine-tuning.
More subtly, aggressive pruning changes the \emph{sampling density} of the representation; if the primitive bandwidths (scales/covariances) are not reconfigured, the result is a \emph{bandwidth mismatch} that behaves like an unintended low-pass filter or produces holes/aliasing in high-frequency regions.
This mirrors classical bias--variance trade-offs in kernel smoothing and density estimation where the bandwidth governs the smoothness/detail preserved~\citep{silverman1986density,unser2000sampling}.

\paragraph{Our contributions: discrete subset selection under rendering statistics.}
We propose \emph{Render-Aware Importance Resampling with Adaptive Basis Reconfiguration} (RAIR-ABR), motivated by two principles:

\begin{enumerate}
    \item \textbf{On-manifold selection over centroid drift.}
    Instead of moving centers by clustering/averaging, we treat compaction as selecting a \emph{discrete subset} of the teacher Gaussians. This constrains the student support to remain on the teacher's learned manifold by construction.
    \item \textbf{Render-aware resampling with coverage regularization.}
    We estimate per-Gaussian \emph{rendering mass} from the rasterizer using the standard volumetric/raster blending weight $w=\alpha \cdot T$ (opacity times transmittance). We then perform \emph{importance resampling} on this discrete support using a dominant-per-pixel proxy and optional texture weighting.
    To prevent sample impoverishment (weight collapse onto a few ``hubs''), we enforce \emph{spatial coverage} via a greedy, adaptive octree partition that selects one representative per leaf and then fills/trim globally in a deterministic way.
\end{enumerate}

\paragraph{Contributions.}
RAIR-ABR contributes:
(i) a render-aware \emph{discrete} resampling formulation for 3DGS compaction,
(ii) a greedy octree-based selection rule that enforces spatial coverage with deterministic fill/trim, and
(iii) a lightweight basis reconfiguration rule (opacity/scale clamping) that improves stability when compressing to a small budget.

\section{Related Work}
\paragraph{3D Gaussian Splatting.}
3DGS represents a scene using anisotropic Gaussians optimized from multi-view images, coupled with a visibility-aware rasterizer that enables fast training and real-time rendering~\citep{kerbl2023gaussians}.
Densification and pruning are part of the standard training loop, which makes the final representation highly overcomplete and motivates compaction.

\paragraph{Improving initialization and training trajectories.}
EDGS argues that iterative densification can be slow and can yield suboptimal reconstructions, especially in high-frequency regions; it proposes dense triangulation-based initialization to preserve fine details and converge faster~\citep{kotovenko2025edgs}.
While EDGS primarily targets training-time efficiency rather than post-hoc compaction, it supports the broader hypothesis that \emph{distribution quality and bandwidth choices matter} for fidelity.

\paragraph{Compaction and compression for 3DGS.}
LightGaussian compresses 3DGS by pruning low-significance Gaussians and distilling/quantizing appearance parameters, reporting strong compression with limited quality loss~\citep{fan2024lightgaussian}.
GHAP frames compaction as a \emph{global Gaussian mixture reduction} problem under an optimal-transport perspective, using KD-tree partitioning to minimize a composite transport divergence and then fine-tuning appearance~\citep{wang2025ghap}.
Mini-Splatting focuses on reorganizing the spatial distribution of Gaussians and simplifying the representation under constrained budgets~\citep{fang2024minisplatting}.
These methods largely operate via pruning and/or geometric reduction, which motivates our complementary focus: \emph{render-aware discrete subset selection with explicit coverage control}.

\paragraph{Sequential Monte Carlo and resampling degeneracy.}
Our coverage regularization is inspired by classical SMC/particle filtering where weighted samples (particles) are resampled to form an equally weighted representation; naive resampling suffers from degeneracy and sample impoverishment, and stratified/residual schemes reduce variance and improve diversity \citep{arulampalam2002particle,douc2005resampling}. We use \emph{resampling} as a conceptual lens---selecting $K$ support points from a discrete importance measure---but our procedure is a \emph{deterministic, without-replacement} subset selection, not stochastic particle resampling in the classical SMC sense.

\paragraph{Bandwidth selection and sampling theory.}
Interpreting Gaussians as localized basis functions connects compaction to classical approximation and sampling: changing the number/spacing of basis functions without adjusting bandwidth leads to bias (oversmoothing) or variance/aliasing artifacts, echoing bandwidth selection in kernel methods~\citep{silverman1986density} and sampling/approximation analyses in shift-invariant spaces~\citep{unser2000sampling}.

\section{Methodology}
\subsection{Problem Setup}
We start from a trained \emph{teacher} 3DGS model with $N$ Gaussians and aim to produce a compact \emph{student} with $K\!\ll\!N$ Gaussians.
A key challenge is that intrinsic per-Gaussian parameters (e.g., a learned opacity parameter) are \emph{not} an importance signal: a Gaussian can have high opacity yet be consistently occluded, contributing negligible pixels in all training views. This motivates using \emph{render-time visibility} rather than teacher-space heuristics.

Existing compaction methods follow two broad directions.
(1) \emph{Pruning} pipelines use heuristic or sensitivity proxies (e.g., opacity/size/gradient/Hessian) and then refine the remaining set~\citep{fan2024lightgaussian,hanson2025pup3dgs,ali2024trimmingthefat,ali2025compression3dgs_survey}.
(2) \emph{Global mixture reduction} methods explicitly \emph{synthesize} a new reduced mixture by updating Gaussian positions/covariances to approximate the teacher distribution (e.g., OT/GMR-style reduction in GHAP)~\citep{wang2025ghap}, or reorganize/reinitialize Gaussian centers during training to meet a tight budget~\citep{fang2024mini_splatting}.
These approaches are powerful, but they can create \emph{off-support} primitives: averaging/relocating Gaussians can "drift" away from the thin surface manifold implicitly learned by 3DGS, and the subsequent optimizer must fight to pull them back (or else produce volumetric blur).

\textbf{Our design choice.} We perform \emph{subset selection} from the teacher: we never move a selected center at initialization. Instead, we pick teacher Gaussians that already lie on the learned geometric support and then fine-tune their appearance/opacity under the original soft renderer.
Our contribution is a simple, deterministic, coverage-regularized selection rule that (i) measures \emph{actual} teacher rendering contribution via the rasterizer weights, (ii) aggregates \emph{dominant} responsibility to suppress local redundancy from overcomplete stacks, and (iii) enforces spatial coverage via octree stratification. Formally, the teacher is
\begin{equation}
\Theta^{(t)}=\{(\boldsymbol{\mu}_g,\boldsymbol{\Sigma}_g,o_g,\mathbf{b}_g)\}_{g=1}^{N},
\end{equation}
where $\boldsymbol{\mu}_g\in\mathbb{R}^3$ is the center, $\boldsymbol{\Sigma}_g\in\mathbb{R}^{3\times 3}$ is the anisotropic covariance, $o_g$ is the learned opacity parameter, and $\mathbf{b}_g$ denotes appearance coefficients (e.g., spherical harmonics)~\citep{kerbl2023gaussians}.

% For a training image, the rasterizer projects Gaussians and composites each pixel $p$ by depth-sorting and applying front-to-back $\alpha$-compositing~\citep{zwicker2002ewa}. Let $i=1,\dots,M(p)$ index this front-to-back order. Each Gaussian contributes a per-pixel opacity $a_i(p)\in[0,1]$ (from $(\boldsymbol{\mu}_i,\boldsymbol{\Sigma}_i,\alpha_i)$ after projection) and a view-dependent color $c_i(p)$ (from $\mathbf{c}_i$). Thus, the pixel color is
% \begin{equation}
% C(p)=\sum_{i=1}^{M(p)} T_i(p)\,a_i(p)\,c_i(p),
% \label{eq:rendering}
% \end{equation}

% where $T_i(p)=\prod_{j<i}\bigl(1-a_j(p)\bigr)$ is the accumulated transmittance from previous Gaussians. Eq. \ref{eq:rendering} exposes the scalar renderer weight $w_i(p)=T_i(p)a_i(p)$, which we will reuse as our visibility-aware importance signal.

Given a target budget $K\ll N$, our goal is to construct a student set
\begin{equation}
\Theta^{(s)}=\left\{(\boldsymbol{\mu}'_k,\boldsymbol{\Sigma}'_k,\alpha'_k,\mathbf{b}'_k)\right\}_{k=1}^{K},
\end{equation}
initialized from a selected subset of teacher Gaussians, and then fine-tuned under the same differentiable rasterizer to preserve rendering quality.

% \subsection{Render-Aware Mass from the Rasterizer}

% Eq.~\eqref{eq:rendering} shows that how much Gaussian contributes to a pixel $p$ through a \emph{single scalar coefficient}.
% \begin{equation}
% w_{i}(\mathbf{p}) \coloneqq \alpha_{i}(\mathbf{p}) \, T_{i}(\mathbf{p})
% \end{equation}
% so that $C(p)=\sum_i w_i(p)\,c_i(p)$, which is exactly the multiplier of Gaussian $i$'s color in the rendered pixel. In other words, $w_{i,v}(\mathbf{p})$ measures how much Gaussian $i$ truly ``shows up'' at pixel $\mathbf{p}$ in view $v$. This weight is a visibility-aware ``responsibility'': $a_i(p)$ measures how strongly the splat covers the pixel, and $T_i(p)$ suppresses Gaussians that are behind nearer content. Our render-aware importance is built by aggregating these weights over pixels and views, so that Gaussians that consistently explain more of the observed images receive higher mass.

% \paragraph{Dominant-surface identification.}
% A pixel is generally explained by multiple Gaussians under soft alpha compositing \citep{zwicker2002ewa}. However, in practice, a converged 3DGS is often \emph{overcomplete}: densification produces stacks of overlapping, semi-transparent Gaussians around the same surface, which jointly realize a sharp edge or a solid wall. This ``volumetric'' redundancy is a known optimization artifact in 3DGS geometry and motivates surface-regularized or surfel-like variants~\citep{guedon2024sugar,huang20242dgs}. Under compaction, we cannot keep this redundancy; the question becomes \emph{which teacher primitives form the minimal ``backbone'' that actually carries visible responsibility across views}. We want a sparse \emph{anchor set} that stays on the teacher support without spending budget on near-duplicates. We therefore assign each ray (a pixel in a training view) $r=(v,p)$ to the single Gaussian with maximal visibility-aware weight:
% \begin{equation}
% i^*(r)=\arg\max_i w_i(r),
% \qquad
% w^*(r)=w_{i^*(r)}(r).
% \label{eq:dominant}
% \end{equation}

% This is a mode-seeking step on the renderer’s own weight profile along the ray: it selects the Gaussian that most directly controls the pixel under occlusion.
% Crucially, we do \emph{not} replace the renderer with hard assignment; the student is still fine-tuned with the original soft compositing, so the selected anchors can adjust opacity and appearance to absorb residual contributions from discarded neighbors.

% \paragraph{Per-Gaussian rendering mass and sampling set.}
% We estimate importance from the training views and all pixels within each selected view. Specifically, we pick $V$ views uniformly over the training set and use the full image grid per view (no stochastic pixel subsampling in teacher-space mode). We ignore empty/background rays by thresholding the accumulated opacity $A(r)=\sum_i w_i(r)$ and keeping only rays with $A(r)>\varepsilon$ (and valid dominant contributors).
% The rendering mass of Gaussian $i$ is then the total dominant credit it receives:
% \begin{equation}
% m_i=\sum_{r\in\mathcal{R}:\ i^*(r)=i} w^*(r)\cdot g(r),
% \label{eq:mass}
% \end{equation}
% where $w^*(r)$ is the maximum visibility-weighted contribution at ray $r$ and $g(r)=1+\lambda_{\text{tex}}\|\nabla I(r)\|$ is an texture weight. Intuitively, $m_i$ is large when Gaussian $i$ consistently explains visible pixels with high transmittance-weighted alpha. We use $m_i$ directly as the importance score for coverage-regularized selection.

The contribution of $g_k(r)$ to pixel $p$ is then
\begin{equation}
w_{g_k}(r)\;\coloneqq\;T_k(r)\,a_{g_k}(r)\in[0,1],
\end{equation}
i.e., the \emph{exact scalar coefficient} multiplying its color in the rasterizer output.
With an (optional) background color $c_{\mathrm{bg}}(r)$, the rendered color is
\begin{equation}
C(r) \;=\; \sum_{k=1}^{M(r)} w_{g_k}(r)\,c_{g_k}(r) \;+\; T_{M(r)+1}(r)\,c_{\mathrm{bg}}(r).
\label{eq:rendering_clean}
\end{equation}
We use the convention $w_g(r)=0$ if Gaussian $g$ does not overlap pixel $p$ on ray $r$.
Importantly, $w_g(r)$ already includes occlusion through $T_k(r)$: a Gaussian can have large local opacity $a_g(r)$ but negligible \emph{visible} effect if it is behind a strong foreground stack.

\paragraph{Accumulated opacity and empty-ray filtering.}
We define the accumulated opacity on ray $r$ as
\begin{equation}
A(r)\coloneqq \sum_{k=1}^{M(r)} w_{g_k}(r) \;=\; 1-T_{M(r)+1}(r),
\end{equation}
which measures how much of the pixel is explained by Gaussians (vs.\ background transmittance).
In all responsibility-based aggregation below, we ignore background/empty rays by restricting to
$\mathcal{R}_{\tau}\coloneqq\{r\in\mathcal{R}\mid A(r)>\tau\}$ for a small threshold $\tau$.


\subsection{Dominant-Responsibility Mass}
The weights $w_g(r)$ provide a render-aware notion of \emph{responsibility}: they are the rasterizer’s own mixing coefficients for forming $C(r)$.
However, a converged 3DGS is typically \emph{overcomplete} due to density control (cloning/splitting), producing many heavily overlapping splats on the same surface patch~\citep{kerbl2023gaussians}.
In this regime, naively accumulating soft weights $\sum_{r} w_g(r)$ tends to over-select redundant local stacks: many near-duplicates each receive moderate mass, and top-$K$ selection wastes budget on a single region while sacrificing global coverage.

\paragraph{Ray-wise consolidation via a winner-take-all anchor.}
Since our objective is subset selection (keep $K$ teacher primitives \emph{without} relocating or merging centers), we explicitly \emph{consolidate} each ray’s credit onto a single representative.
For every non-empty ray $r\in\mathcal{R}_{\tau}$, define the dominant Gaussian
\begin{equation}
g^*(r)\;=\;\operatorname*{arg\,max}_{1\le k\le M(r)} \; w_{g_k}(r).
\end{equation}
We also define the \emph{dominance ratio}
\begin{equation}
\rho(r)\;\coloneqq\;\frac{w_{g^*(r)}(r)}{A(r)} \in (0,1],
\end{equation}
which quantifies how concentrated the ray’s responsibility is on its winner.
Appendix~B.2 shows that $\rho(r)$ is rarely large in trained models, i.e., responsibility is usually shared among multiple overlapping splats.
Therefore, the hard assignment $g^*(r)$ should not be read as “high-confidence classification.”
It is a deliberate \emph{ray-wise non-maximum suppression} step that prevents importance from being fragmented across near-duplicate stacks by choosing a single anchor per ray.

\paragraph{Dominant-Responsibility Mass.}
We define the Dominant-Responsibility Mass of Gaussian $g$ as the accumulated dominant credit across training rays:
\begin{equation}
m_g \;=\; \sum_{r\in\mathcal{R}_{\tau}} \mathbb{I}\!\left[g^*(r)=g\right]\cdot w_g(r).
\end{equation}
Intuitively, each ray casts one weighted vote (of size $w_g(r)$) to the Gaussian that most strongly influences its rendered color.
Thus, $m_g$ favors Gaussians that are repeatedly the \emph{most responsible} explanation for many pixels/views, yielding a sparse anchor backbone with improved global coverage under a strict budget.

Crucially, the student is still optimized with the original soft $\alpha$-compositing.
After selecting anchors, the subsequent basis reconfiguration and fine-tuning (Sec.~3.5) re-allocates opacity/shape/color so the selected anchors can absorb residual contributions that were previously distributed across discarded neighbors.
Finally, Appendix~B.1 verifies that assignments are consistent with the rasterizer footprint: rays are only assigned to a dominant Gaussian whose projected conic actually covers the pixel (small 2D Mahalanobis distance under its screen-space ellipse).


\subsection{Dominant-Responsibility Mass}
In section 3.2, the teacher rasterizer already tells us \emph{which} Gaussians importance for each pixel: the visibility weights $w_g(r)$ are the actual coefficients used to form the rendered color on ray $r$. The problem is that a well-trained 3DGS is typically \emph{overcomplete}: due to densification, many nearby splats overlap on the same surface patch and jointly explain the same pixels~\citep{kerbl2023gaussians}. A top-$K$ selection based on accumulated soft weights tends to preserve these redundant local clusters at the expense of global coverage. Since our goal is subset selection (keep $K$ teacher primitives without relocating/merging centers), we instead extract a sparse \emph{anchor backbone} by assigning each pixel's credit to a single representative. We identify the dominant Gaussian $g^*(r)$ for each ray $r$:
$$g^*(r) = \operatorname*{arg\,max}_{1 \le k \le M(r)} w_{g_k}(r)$$
We then define the Dominant-Responsibility Mass $m_g$ as the accumulated dominant credit:
$$m_g = \sum_{r \in \mathcal{R}} \mathbb{I}[g^*(r) = g] \cdot w_{g}(r)$$
where $\mathcal{R}$ is the set of training ray.

We emphasize that the hard assignment $g^*(r)$ is not justified by “high-confidence” pixels.
Appendix~B.2 shows that responsibility is typically shared across multiple overlapping splats: only 4.9–12.6\% of the rendering mass has $\rho(r)\ge0.5$ across three scenes, confirming that converged 3DGS is highly overcomplete. Our $\arg\max$ therefore acts as a deliberate \emph{consolidation} step: it prevents the importance signal from being fragmented across near-duplicate stacks by choosing a single representative anchor per ray.
Crucially, the student is still optimized with the original soft $\alpha$-compositing, and the subsequent basis reconfiguration and fine-tuning (Sec.~3.5) re-allocates opacity/shape so the selected anchors can absorb residual contributions that were previously distributed across the local stack.

We further validate in Appendix~B.1 that the selected anchors are \emph{screen-space consistent} with the rasterizer footprint, as measured by the 2D Mahalanobis radius under the dominant splat’s conic.


\subsection{Coverage-Regularized Deterministic Selection}
Section~3.2 defines nonnegative rendering masses $\{m_i\}$ over teacher Gaussians, interpreted as a discrete measure of visual importance.
A naive importance-only subset (e.g., top-$K$ by $m_i$) tends to collapse budget into a few highly visible regions and select many near-duplicate primitives, reducing spatial coverage.
This is analogous to weight degeneracy in particle-filter resampling, where probability mass concentrates and diversity vanishes~\citep{arulampalam2002particle,douc2005resampling}.

\begin{algorithm}[htbp]
\caption{Mass-Adaptive Octree Sampling}
\label{alg:octree_sampling}
\begin{algorithmic}[1]
\REQUIRE centers $\{\boldsymbol{\mu}_i\}_{i=1}^N$, masses $\{m_i\}_{i=1}^N$, budget $K$
\ENSURE selected indices $\mathcal{S}$ with $|\mathcal{S}|=K$

\STATE Cell mass: $M(c) \coloneqq \sum_{i:\boldsymbol{\mu}_i\in c} m_i$
\STATE Valid splits: $\Phi(c) \coloneqq \{c'\in \mathrm{Oct}(c) \mid M(c')>0\}$

\STATE Initialize leaves $\mathcal{C}\leftarrow \{\mathrm{BBox}(\{\boldsymbol{\mu}_i: m_i>0\})\}$

\WHILE{$|\mathcal{C}|<K$ \AND $\exists c\in\mathcal{C} \text{ s.t. } |\Phi(c)|\ge 2$}
    % \STATE \textit{// Select heaviest split-able leaf}
    \STATE $c^\star \leftarrow \arg\max \{ M(c) \mid c \in \mathcal{C} \land |\Phi(c)| \ge 2 \}$
    \STATE $\mathcal{C} \leftarrow (\mathcal{C}\setminus\{c^\star\}) \cup \Phi(c^\star)$
\ENDWHILE

\STATE Representatives: $s(c) \coloneqq \arg\max_{i:\boldsymbol{\mu}_i\in c} m_i$
\STATE Pairs $\mathcal{P} \leftarrow \{(s(c), M(c)) \mid c\in\mathcal{C}\}$

\STATE \textbf{Rounding:} If $|\mathcal{P}| \neq K$, adjust via mass-ranking:
\STATE \quad - Trim lowest $M(c)$ if $|\mathcal{P}| > K$
\STATE \quad - Fill highest unused $m_i$ if $|\mathcal{P}| < K$

\STATE \textbf{return} $\mathcal{S} \leftarrow \{s \mid (s, \cdot) \in \mathcal{P}_{\text{adjusted}}\}$
\end{algorithmic}
\end{algorithm}

At \ref{alg:octree_sampling}, we enforce coverage by spatial stratification: partition 3D space into disjoint cells and select at most one Gaussian per occupied cell. High-mass regions can still receive more samples, but only by being subdivided into more cells, rather than by repeatedly selecting neighbors from the same volume.

\textbf{Mass-adaptive partition.} We build an axis-aligned octree over the teacher centers $\{\boldsymbol{\mu}_i\}$ (rooted at the scene bounds). For a leaf cell $c$, let
\begin{equation}
\mathcal{I}(c) \;=\; \{ i : \boldsymbol{\mu}_i \in c \},
\qquad
M(c) \;=\; \sum_{i\in\mathcal{I}(c)} m_i .
\label{eq:cellmass}
\end{equation}
We iteratively split the current leaf with the largest $M(c)$ (only if $|\mathcal{I}(c)|>1$) until the number of non-empty leaves reaches $\approx K$. Intuitively, forcing a high-$M(c)$ region to contribute only one representative discards the most mass; splitting it first greedily relaxes that constraint where it hurts most.


\paragraph{One representative per leaf.}
Given the final leaf set $\mathcal{C}$, we select one teacher Gaussian per non-empty leaf,
\begin{equation}
s(c) \;=\; \arg\max_{i\in\mathcal{I}(c)} m_i,
\qquad
\mathcal{S} \;=\; \{ s(c) \,|\, c\in\mathcal{C}\}.
\label{eq:leafrep}
\end{equation}

For a fixed partition, this choice is optimal under the one-per-cell constraint: it maximizes the total retained mass $\sum_{c\in\mathcal{C}} m_{s(c)}$.
Selecting existing teacher primitives avoids relocating centers and yields a coverage-biased initialization: local redundancy within a leaf is collapsed at initialization, while finer structure (covariances/appearance) can be reintroduced during subsequent optimization.


\paragraph{Trim/fill (edge cases).}
Because one split can create multiple non-empty children, If $|\mathcal{S}|>K$, we keep the $K$ representatives from leaves with largest $M(c)$ (dropping lowest-mass cells first).
If $|\mathcal{S}|<K$, we add remaining Gaussians by descending $m_i$; this fill introduce local redundancy, but it adds capacity on top of an already spatially distributed core set.

\subsection{Finetuning}
\begin{algorithm}[t]
\caption{RAIR-ABR: Overall Pipeline}
\label{alg:rair_abr_pipeline}
\begin{algorithmic}[1]
\STATE \textbf{Input:} Teacher 3DGS $\Theta^{(t)}=\{(\boldsymbol{\mu}_i,\boldsymbol{\Sigma}_i,\alpha_i,\mathbf{c}_i)\}_{i=1}^{N}$,
training views $\mathcal{V}$, budget $K$, fine-tune steps $T$
\STATE \textbf{Output:} Student 3DGS $\Theta^{(s)}$ with $K$ Gaussians

\STATE \textbf{Stage 1: Render-Aware Selection}
\STATE 1. \textbf{Render-aware mass:} estimate per-Gaussian masses $\{m_i\}$ from the teacher rasterizer
using visibility weights $w=\alpha\cdot T$ and dominant assignment (Sec.~3.2). \COMMENT{3DGS rasterizer}
\STATE 2. \textbf{Coverage-regularized subset:} select anchor indices
$\mathcal{S}\leftarrow \textsc{MassAdaptiveOctree}(\{\boldsymbol{\mu}_i,m_i\},K)$ (Alg.~\ref{alg:octree_sampling}).
% \COMMENT{prevents mass-degeneracy / sample impoverishment}

\STATE \textbf{Stage 2: Basis Reconfiguration + Optimization}
\STATE 1. \textbf{On-manifold init:} initialize $\Theta^{(s)}$ by copying teacher parameters at $\mathcal{S}$: $(\boldsymbol{\mu}'_k,\boldsymbol{\Sigma}'_k,\alpha'_k,\mathbf{c}'_k)\leftarrow
(\boldsymbol{\mu}_{s_k},\boldsymbol{\Sigma}_{s_k},\alpha_{s_k},\mathbf{c}_{s_k})$.
\STATE 2. \textbf{ABR init:} reset/clamp scale and opacity to match the new sampling spacing (Sec.~3.4).
\STATE 3. \textbf{Fine-tune:} optimize $\Theta^{(s)}$ with the standard 3DGS objective for $T$ steps (appearance/opacity; geometry).

\STATE \textbf{return} $\Theta^{(s)}$
\end{algorithmic}
\end{algorithm}


Subset selection changes the \emph{sampling density} of the basis: the student is sparser than the teacher.
If we keep the teacher’s original bandwidth parameters under this new spacing, we create a bandwidth mismatch:
Gaussians that were valid in an overcomplete basis can become too narrow (holes/aliasing) or too wide (blur/floaters) after compaction.
This is the same principle that links sampling rate and representable frequency content in classical sampling/approximation settings~\citep{silverman1986density,unser2000sampling}.

\paragraph{Scale reset (intuition).}
After compaction the basis is sparser, so scales that were optimal in the dense teacher can become too small (holes) or too large (blur). We therefore clamp scales to a conservative range tied to the selection grid to enforce overlap at the new spacing while preventing any single primitive from dominating multiple cells; fine adjustment is left to optimization.

\paragraph{Opacity reset (intuition).}
Opacity is a learned mixture weight, not a conserved density. After sparsification, inherited opacities are often miscalibrated or saturated, which weakens gradients and slows recovery. We therefore clamp opacities into a neutral, learnable range so optimization can quickly re-balance density after compaction.

\paragraph{Appearance transfer and fine-tuning.}
We initialize appearance by copying coefficients from the selected teacher Gaussians,
$\mathbf{c}'_k \leftarrow \mathbf{c}_{s_k}$, and then fine-tune the student with the standard 3DGS training objective on the original views~\citep{kerbl2023gaussians}. After compaction, we fine‑tune the student under the original soft renderer using a two‑phase recovery schedule. Phase‑1 increases scaling/opacity LRs to quickly restore coverage while keeping position LR modest; Phase‑2 reduces all LRs to stabilize geometry and refine appearance. We evaluate immediately after compaction, mid‑recovery, and at the final iteration.

\section{Experiments}

Our pipeline define as \ref{alg:rair_abr_pipeline}) (i) train a backbone 3DGS model to an intermediate checkpoint, (ii) apply a compaction operator once to reach a target retention ratio $\rho$, and (iii) fine-tune the student under the original renderer with a recovery schedule that allows geometry to adapt (with reduced position LR). This isolates the effect of the compaction operator while reflecting the practical need to reconfigure the remaining Gaussians.

\subsection{Experimental Setup}

\begin{table*}[t]
\centering
\caption{\textbf{Quantitative Results.} Comparison on Tanks \& Temples, Mip-NeRF 360, and Deep Blending. We separate the uncompressed reference (3DGS) from compact methods (below the line). Color coding (\colorbox{best}{1st}, \colorbox{second}{2nd}, \colorbox{third}{3rd}) is applied \textbf{only} among the compact methods to ensure fair comparison.}
\label{tab:main_results_final}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccccccccc}
\toprule
& \multicolumn{4}{c}{\textbf{Tanks \& Temples}} & \multicolumn{4}{c}{\textbf{Mip-NeRF 360}} & \multicolumn{4}{c}{\textbf{Deep Blending}} \\
\cmidrule(lr){2-5} \cmidrule(lr){6-9} \cmidrule(lr){10-13}
\textbf{Method} & \textbf{PSNR}$\uparrow$ & \textbf{SSIM}$\uparrow$ & \textbf{LPIPS}$\downarrow$ & \textbf{G(k)}$\downarrow$ & \textbf{PSNR}$\uparrow$ & \textbf{SSIM}$\uparrow$ & \textbf{LPIPS}$\downarrow$ & \textbf{G(k)}$\downarrow$ & \textbf{PSNR}$\uparrow$ & \textbf{SSIM}$\uparrow$ & \textbf{LPIPS}$\downarrow$ & \textbf{G(k)}$\downarrow$ \\
\midrule
3DGS & 23.79 & 0.853 & 0.169 & 1577 & 27.55 & 0.813 & 0.221 & 2627 & 29.82 & 0.907 & 0.238 & 2475 \\
\midrule
LightGaussian & 22.11 & 0.756 & 0.306 & 158 & 25.67 & 0.735 & 0.331 & 263 & 28.01 & 0.869 & 0.327 & 248 \\
PUP-3DGS & 21.52 & 0.767 & 0.280 & 158 & 25.33 & 0.753 & \cellcolor{third}0.309 & 262 & 29.15 & 0.895 & \cellcolor{third}0.274 & 248 \\
MesonGS & 20.71 & \cellcolor{third}0.811 & \cellcolor{second}0.208 & 157 & 24.92 & \cellcolor{third}0.773 & \cellcolor{best}0.264 & 263 & 28.69 & 0.896 & \cellcolor{best}0.264 & 248 \\
Mini-Splatting & \cellcolor{third}22.66 & 0.799 & 0.265 & \cellcolor{best}78 & \cellcolor{third}26.02 & 0.759 & 0.318 & \cellcolor{best}111 & \cellcolor{third}29.40 & 0.895 & 0.289 & \cellcolor{best}125 \\
GHAP & \cellcolor{second}23.31 & \cellcolor{best}0.818 & \cellcolor{third}0.242 & 157 & \cellcolor{second}26.40 & 0.764 & 0.314 & 263 & \cellcolor{second}29.65 & \cellcolor{best}0.905 & \cellcolor{best}0.264 & 248 \\
\textbf{Ours} & \cellcolor{best}\textbf{23.37} & \cellcolor{second}\textbf{0.817} & \cellcolor{best}\textbf{0.235} & 157 & \cellcolor{best}\textbf{26.83} & \cellcolor{best}\textbf{0.787} & \cellcolor{second}\textbf{0.274} & \textbf{263} & \cellcolor{best}\textbf{29.45} & \cellcolor{second}\textbf{0.901} & \cellcolor{second}\textbf{0.265} & \textbf{248} \\
\bottomrule
\end{tabular}%
}
\end{table*}

\paragraph{Datasets and scenes.}
We evaluate on the three real-world datasets used in GHAP:
\textbf{Tanks \& Temples} (two outdoor scenes: \emph{Train}, \emph{Truck}) \citep{knapitsch2017tanks},
\textbf{Mip-NeRF 360} (nine scenes: \emph{Bicycle, Bonsai, Counter, Flowers, Garden, Kitchen, Room, Stump, Treehill})
\citep{barron2022mipnerf360}, and
\textbf{Deep Blending} (two indoor scenes: \emph{Dr.~Johnson}, \emph{Playroom}) \citep{hedman2018deepblending}.
We report dataset averages; per-scene numbers are deferred to the appendix for space.

\paragraph{Metrics.}
We measure rendering quality using PSNR, SSIM, and LPIPS as in standard 3DGS evaluation. Higher PSNR/SSIM and lower LPIPS are better. We additionally report the final number of Gaussians (in thousands, denoted $k$).

\paragraph{Training and compaction protocol.}
Our post-training methods start from the \emph{same} backbone checkpoint trained for 15k iterations, apply compaction at iteration 15,001, and then fine-tune for another 15k iterations under identical optimization settings, totaling 30k iterations per scene. End-to-end variants are trained for 30k iterations using their default settings.
We evaluate at $\rho\in\{10\%,20\%\,50\%\}$ unless stated otherwise.
(See Appendix~\ref{app:repro} for exact commands and hyperparameters.)


\subsection{Implementation Details}
Unless stated otherwise, we use the default 3DGS optimizer in \texttt{train\_and\_prune.py}.
Compaction uses RSS‑voxel with teacher‑space selection (center\_mode=teacher) and octree partitioning (teacher\_selector=octree).
We actively rendering statistics from full views with full pixels per view (default), apply an opacity threshold $\alpha_\tau=0.02$, hit\_quantile $=0.3$, and texture weighting $\lambda_{\text{tex}}=0.5$; voxel size is chosen by search (default 8 iterations).

We adopt a two‑phase recovery schedule after compaction.
Phase‑1 runs with stronger adaptation: position LR $1.6\!\times\!10^{-4}\!\rightarrow\!2.0\!\times\!10^{-5}$ (max steps 30k), scaling LR 0.008, opacity LR 0.05, feature LR 0.003, rotation LR 0.001, and $\lambda_{\text{DSSIM}}=0.1$.
Phase‑2 uses reduced LRs to stabilize geometry: position $8\!\times\!10^{-5}\!\rightarrow\!1.0\!\times\!10^{-5}$, scaling 0.004, opacity 0.02, feature 0.002, rotation 0.001, $\lambda_{\text{DSSIM}}=0.1$.
Random seeds are fixed per run (rss\_seed) to enable mean/std aggregation.




\paragraph{Results.}




\nocite{langley00}

\bibliography{references}
\bibliographystyle{icml2026}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% APPENDIX
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newpage
\appendix
\twocolumn

\section{Render-Aware Weight Interpretation}
\label{app:weight_interpretation}
The weight $w_g(r)$ has two complementary interpretations:
\begin{enumerate}
    \item[(i)] \emph{First-order influence:} since $C(r)=\sum_g w_g(r)c_g(r)$, we have $\frac{\partial C(r)}{\partial c_g(r)}=w_g(r)$. Thus $w_g(r)$ is exactly the per-pixel sensitivity of the output color to Gaussian $g$’s color at that pixel[cite: 583].
    \item[(ii)] \emph{Visibility / ray-termination probability:} in front-to-back compositing, $T_k(r)$ is the fraction of light that survives to depth $k$, and $a_{g_k}(r)$ is the fraction absorbed by $g_k$; their product is the fraction of the pixel explained by that Gaussian under occlusion[cite: 586].
\end{enumerate}
Crucially, a single Gaussian contributes to \emph{many} rays (many pixels across many views); our goal is to aggregate these per-ray influences into a global score to identify structural anchors.

\section{Screen-Space Anchor Validation}
\label{app:dominant_anchor}



\subsection{Spatial Precision via Mahalanobis Distance}
To verify that our dominant selection $(g^{*})$ identifies valid surface anchors rather than "tail" contributors (large Gaussians centered far from the ray), we analyze the screen-space alignment. As shown in Figure \ref{fig:dominant_maha_3scenes} and Table \ref{tab:dominant_stats}, the mass-weighted distribution of the 2D Mahalanobis distance peaks at $d_{M}\approx1.2$. Table~2 shows that dominant anchors are typically aligned with the pixel center in the metric induced by the rasterizer’s screen-space conic (median $d_M\approx1.2$–$1.4$), and a majority of mass lies within $d_M\le2$ (65.6–74.5\%) and $d_M\le3$ (78.0–86.1\%). This supports our use of dominant splats as \emph{screen-space anchors} for subset selection: they are generally consistent with the renderer footprint, even when responsibility is shared across several splats.


\begin{figure}[h]
    \centering
    \includegraphics[width=1\linewidth]{figures/after compactation.jpg}
    \caption{Qualitative recovery after compaction. Left: render immediately after compaction (iter 15002, before fine‑tuning). Right: same view after fine‑tuning (iter 30000). Fine‑tuning restores sharp geometry and soft effects while preserving the recovered anchors.}
    \label{fig:placeholder}
\end{figure}

\subsection{Dominance Confidence and Stability}
A potential concern with hard assignment (\texttt{argmax}) is stability: discarding secondary contributors could theoretically cause temporal popping or aliasing at edges. We analyze this using the Dominance Confidence $\rho(r) = \max_{k}w_{g_k}(r) / \sum_{j}w_{g_j}(r)$.

\begin{figure}[htbp]
\centering
\includegraphics[width=1\columnwidth]{figures/dominant_rho_3scenes.png}
\vspace{0.5em}
\includegraphics[width=1\columnwidth]{figures/dominant_surface_3scenes.png}
\caption{\textbf{Dominant Anchor Characteristics.}
\textbf{Top:} Mass‑weighted CDF of dominance confidence $\rho$ across three scenes. Most foreground mass lies at low $\rho$, indicating strong overlap among splats and motivating consolidation into a single anchor.
\textbf{Bottom:} Mass‑weighted histogram of 2D Mahalanobis radius for the selected anchors. The distribution peaks near the expected range for well‑aligned splats, showing that the chosen anchors are spatially localized in screen space rather than tail or floater contributors.}
\label{fig:dominant_characteristics}
\end{figure}

Figure~1 (top) reports the \emph{mass-weighted} CDF of the dominance concentration
$\rho(r)=\max_k w_{g_k}(r)\,/\,\sum_j w_{g_j}(r)$.
Across three scenes, most rendering mass lies at moderate $\rho$: only 4.91–12.58\% of mass satisfies $\rho\ge0.5$, while 27.67–49.80\% satisfies $\rho\ge0.3$ (Table~2).
This indicates that a pixel is commonly explained by a small set of overlapping splats rather than a single decisive winner.
Accordingly, we interpret $\arg\max$ not as hardening the renderer, but as an \emph{initialization-time consolidation} operator that chooses one representative per local stack; the student remains a soft-composited model and fine-tuning restores smooth transitions in low-$\rho$ regions.


\textbf{Redundancy, not Transparency:} In most regions, "vote splitting" between Gaussians is an artifact of over-densification (redundancy), not true volumetric transparency. Collapsing this to the dominant anchor acts as a \emph{denoising} step rather than a loss of signal.

\textbf{Recovery of Soft Edges:} In true ambiguous regions (e.g., object edges where $\rho$ is lower), our Mass-Adaptive Octree (Alg. 1) ensures spatial coverage is preserved. While \texttt{argmax} selects a single representative initially, the subsequent \emph{Fine-tuning} stage (Sec 3.5) re-optimizes the opacity and covariance of these anchors to recover the necessary soft blending and anti-aliasing.



\begin{table}[h]
  \centering
  \caption{Mass-weighted containment statistics for dominant anchor assignment. Selected anchors are tightly clustered around the pixel center (median $d_M \approx 1.2$).}
  \label{tab:dominant_stats}
  \small
  \begin{tabular}{lrrr}
    \toprule
    Metric & flowers & treehill & room \\
    \midrule
    med $d_M$ & 1.20 & 1.25 & 1.37 \\
    p90 $d_M$ & 3.69 & 4.31 & 5.58 \\
    Mass$\le1.0$ & 40.09 & 38.90 & 35.71 \\
    Mass$\le2.0$ & 74.50 & 71.28 & 65.62 \\
    Mass$\le3.0$ & 86.09 & 83.19 & 78.04 \\
    \bottomrule
  \end{tabular}
\end{table}



\end{document}

