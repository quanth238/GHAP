\section{Methodology: Render-Aware Importance Resampling}
\subsection{Problem Setup}
Let the teacher 3DGS be a set of $N$ Gaussians
\begin{equation}
\Theta^{(t)} = \left\{(\boldsymbol{\mu}_i,\boldsymbol{\Sigma}_i,\alpha_i,\mathbf{c}_i)\right\}_{i=1}^{N},
\end{equation}
where $\boldsymbol{\mu}_i\in\mathbb{R}^3$ is the center, $\boldsymbol{\Sigma}_i\in\mathbb{R}^{3\times 3}$ is the anisotropic covariance, $\alpha_i\in\mathbb{R}_{+}$ is learned opacity parameter, and $\mathbf{c}_i$ denotes appearance coefficients (e.g., spherical harmonics)~\citep{kerbl2023gaussians}.

Consider a training view $v\in\mathcal{V}$ with pixel domain $\Omega_v$ and camera $\Pi_v$, the rasterizer projects each Gaussian and, at each pixel $p$, evaluates (i) a per-pixel alpha $a_i(p)\in[0,1]$ (from $\boldsymbol{\mu}_i,\boldsymbol{\Sigma}_i,\alpha_i$) and (ii) a view-dependent RGB color $c_i(p)$ (from $\mathbf{c}_i$).
Let $i=1,\dots,M(p)$ index the Gaussians that overlap pixel $p$, sorted front-to-back by depth. The rasterizer renders a pixel $p$ by sorting overlapping Gaussians by depth and applying front-to-back $\alpha$-compositing \cite{zwicker2002ewa}:
\begin{equation}
C(\mathbf{p}) = \sum_{i=1}^{N} T_i \alpha'_i c_i, \quad \text{where } T_i = \prod_{j=1}^{i-1}(1 - \alpha'_j).
\label{eq:rendering}
\end{equation}

where $T_i$ is the accumulated transmittance from closer Gaussians, and $a_i$ is the (view- and pixel-dependent) opacity actually used by the renderer (not the raw parameter $\alpha_i$).

Given a target budget $K\ll N$, our goal is to construct a student subset
\begin{equation}
\Theta^{(s)}=\left\{(\boldsymbol{\mu}'_k,\boldsymbol{\Sigma}'_k,\alpha'_k,\mathbf{c}'_k)\right\}_{k=1}^{K},
\end{equation}
such that rendering quality is preserved after fine-tuning under the same soft renderer, while keeping the selection simple, deterministic, and coverage-preserving.

\subsection{Render-Aware Mass from the Rasterizer}

Eq.~\eqref{eq:rendering} shows that how much Gaussian contributes to a pixel $p$ through a \emph{single scalar coefficient}.
\begin{equation}
w_{i}(\mathbf{p}) \coloneqq \alpha_{i}(\mathbf{p}) \, T_{i}(\mathbf{p})
\end{equation}
so that $C(p)=\sum_i w_i(p)\,c_i(p)$, which is exactly the multiplier of Gaussian $i$'s color in the rendered pixel. In other words, $w_{i,v}(\mathbf{p})$ measures how much Gaussian $i$ truly ``shows up'' at pixel $\mathbf{p}$ in view $v$. This weight is a visibility-aware ``responsibility'': $a_i(p)$ measures how strongly the splat covers the pixel, and $T_i(p)$ suppresses Gaussians that are behind nearer content. Our render-aware importance is built by aggregating these weights over pixels and views, so that Gaussians that consistently explain more of the observed images receive higher mass.

\paragraph{Dominant-surface identification.}
A pixel is generally explained by multiple Gaussians under soft alpha compositing \citep{zwicker2002ewa}. However, in practice, a converged 3DGS is often \emph{overcomplete}: densification produces stacks of overlapping, semi-transparent Gaussians around the same surface, which jointly realize a sharp edge or a solid wall. This ``volumetric'' redundancy is a known optimization artifact in 3DGS geometry and motivates surface-regularized or surfel-like variants~\citep{guedon2024sugar,huang20242dgs}. Under compaction, we cannot keep this redundancy; the question becomes \emph{which teacher primitives form the minimal ``backbone'' that actually carries visible responsibility across views}. We want a sparse \emph{anchor set} that stays on the teacher support without spending budget on near-duplicates. We therefore assign each ray (a pixel in a training view) $r=(v,p)$ to the single Gaussian with maximal visibility-aware weight:
\begin{equation}
i^*(r)=\arg\max_i w_i(r),
\qquad
w^*(r)=w_{i^*(r)}(r).
\label{eq:dominant}
\end{equation}

This is a mode-seeking step on the renderer’s own weight profile along the ray: it selects the Gaussian that most directly controls the pixel under occlusion.
Crucially, we do \emph{not} replace the renderer with hard assignment; the student is still fine-tuned with the original soft compositing, so the selected anchors can adjust opacity and appearance to absorb residual contributions from discarded neighbors.

\paragraph{Per-Gaussian rendering mass and sampling set.}
We estimate importance from the training views and all pixels within each selected view. Specifically, we pick $V$ views uniformly over the training set and use the full image grid per view (no stochastic pixel subsampling in teacher-space mode). We ignore empty/background rays by thresholding the accumulated opacity $A(r)=\sum_i w_i(r)$ and keeping only rays with $A(r)>\varepsilon$ (and valid dominant contributors).
The rendering mass of Gaussian $i$ is then the total dominant credit it receives:
\begin{equation}
m_i=\sum_{r\in\mathcal{R}:\ i^*(r)=i} w^*(r)\cdot g(r),
\label{eq:mass}
\end{equation}
where $w^*(r)$ is the maximum visibility-weighted contribution at ray $r$ and $g(r)=1+\lambda_{\text{tex}}\|\nabla I(r)\|$ is an texture weight. Intuitively, $m_i$ is large when Gaussian $i$ consistently explains visible pixels with high transmittance-weighted alpha. We use $m_i$ directly as the importance score for coverage-regularized selection.