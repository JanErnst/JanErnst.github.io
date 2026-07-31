---
title: 'MiDi: Mixed Graph and 3D Denoising Diffusion for Molecule Generation'
date: 2026-07-31
permalink: /posts/2026/07/midi-molecule-generation/
tags:
  - Diffusion
  - Drug Molecule Generation
---


Introduction
======

Designing new molecules is a central challenge in modern drug discovery. A useful generative model should not only propose atoms in 3D space, but also understand how those atoms are connected by chemical bonds. Both views matter: the molecular graph tells us which atoms are bonded and what functional groups are present, while the 3D conformation determines how the molecule can interact with proteins and other biological targets.

Many previous approaches treat these two aspects separately. Some models generate only the 2D molecular graph, ignoring the spatial arrangement of atoms. Others generate a 3D point cloud first and then recover the bond structure afterward using hand-crafted rules or external chemistry software. This separation is limiting: the resulting pipeline is no longer fully differentiable, and errors in the post-processing step can easily turn a promising 3D structure into an invalid molecule.

Motivation
======
The paper MiDi: Mixed Graph and 3D Denoising Diffusion for Molecule Generation addresses this problem with a single diffusion model that generates both parts of a molecule together. MiDi represents a molecule as a graph embedded in 3D space, with atom types, formal charges, bond types, and coordinates all denoised jointly. The key idea is simple but powerful: instead of asking the model to learn geometry first and chemistry later, train it to learn the relationship between geometry and chemistry throughout the entire generation process.

In this blog post, we will build up the motivation behind MiDi, explain why molecule generation needs both graph and 3D information, and look at how the model combines discrete diffusion for chemical structure with Gaussian diffusion for spatial coordinates. Finally, we will discuss why this joint approach leads to much more stable molecules on challenging drug-like datasets.

 
Background knowledge
======

To fully understand the process and architecture, we will now cover the background knowledge needed to understand the function, achitecture and specialties of MiDi.


Graph structure
------ 

Previous works on molecule generation have already demonstrated the efficiency of graph structures. MiDi is no different in this regard and also represents its molecules as a graph. A graph generally consists of nodes and edges connecting them. Nodes represent the atom of a molecule, the edges are the bonds between them and both are described by their features. In the case of MiDi each molecule buld from $n$ atoms is represented as the graph $G = (\mathbf{X},\mathbf{C},\mathbf{R},\mathbf{Y})$ with $\mathbf{X}$ as the atom type and $\mathbf{C}$ as their formal charge, which are both one hot encoded vector of length $n$. $\mathbf(R)\in\mathbb{R}^{3 \times n}$ are the three dimensional coordiates all $n$ atoms and $\mathbf(Y)$ are the bond types between them, saved as a $n \times n$ matrix. Notably, having no bond type is also considerd to be a bond type. Looking at figure ... it becomes obvious why this representation fits a molecule well and intuaticly.


Equivariance:
------
When generating any structure in 3D space we always have respect its dynamic nature. Molecules do not have a general order or fixed sequence and can undergo translation and rotation in space. This does not effect their functionality, type or features except the coordinates of each atom when refrences to the same origin point. A molecule generator should therefore be able to generate the same molecule if the input is moved in space. This is what the word equivariance comes into play. Equivariance holds when any transformation of the input results in the same transformation of the output. So wehen we rotate the input graph $G_1$ from figure ... to the graph $G_2$, their correspondung outputs rotate equally. This also eliminated the need for randomly augmented data for training, as all those samples then become redundant.

Diffusion:
------ 

As the paper title states, MiDi is a denoising diffusion model, which is build from the noise and the dennoising model, where only the denoising model is used during inference. However, it is still crucial to understand the complete pipeline to see how MiDi works. 


Noise model:
------

The sole purpose of the noise model is to incrementally apply noise to the input $x$ so that each step is a more corrupted version of the previous step. This results in a Markovian trajectory in the form of:

$$
q\!\left(G^{1},\ldots,G^{T}\mid G^{0}\right)
=
\prod_{t=1}^{T}
q\!\left(G^{t}\mid G^{t-1}\right),
$$

This proess results in the a complete noised picture like in figure ... .

As already explained in previous sections, MiDi generates a graph described by its edges and node features. The 3D coordinates are continuous values while the remaining features are discrete values. The corresponding noise is therefore either continuous Gaussian noise or discrete noise, resulting in the following combined noise at each step:

| Quantity | Gaussian diffusion | Discrete diffusion |
|---|---|---|
| One-step forward transition | $$q(z_t\mid z_{t-1})=\mathcal{N}\!\left(z_t;\alpha_t z_{t-1},\sigma_t^2I\right)$$ | $$q(z_t\mid z_{t-1})=\mathcal{C}\!\left(z_t;z_{t-1}Q_t\right)$$ |
| Direct transition from clean data | $$q(z_t\mid x)=\mathcal{N}\!\left(z_t;\bar{\alpha}_t x,\bar{\sigma}_t^2I\right)$$ | $$q(z_t\mid x)=\mathcal{C}\!\left(z_t;x\bar{Q}_t\right)$$ |
| Reverse transition after marginalizing the clean prediction | $$\int p_\theta(z_{t-1}\mid x,z_t)\,dp_\theta(x\mid z_t)=\mathcal{N}\!\left(z_{t-1};\mu_t\hat{x}+\nu_tz_t,\tilde{\sigma}_t^2I\right)$$ | $$\sum_x p_\theta(x\mid z_t)\,q(z_{t-1}\mid z_t,x)\propto\sum_xp_\theta(x\mid z_t)\left(z_tQ_t^\top\odot x\bar{Q}_{t-1}\right)$$ |

$$
\begin{aligned}
q\!\left(G^{t}\mid G^{t-1}\right)
={}&
\mathcal{N}_{\mathrm{CoM}}
\!\left(
R^{t};
\alpha_{t}^{r}R^{t-1},
(\sigma_{t}^{r})^{2}I
\right)
\\
&\times
\mathcal{C}
\!\left(
X^{t};
X^{t-1}Q_{x}^{t}
\right)
\\
&\times
\mathcal{C}
\!\left(
C^{t};
C^{t-1}Q_{c}^{t}
\right)
\\
&\times
\mathcal{C}
\!\left(
Y^{t};
Y^{t-1}Q_{y}^{t}
\right).
\end{aligned}
$$


Atom type $\mathbf{X}$, bond type $\mathbf{Y}$ and formal charge $\mathbf{C}$ get discrete noise that is defined by the transition matrix Q that has been derived from the marginal distribution of the training sets for each feature. To ensure that the generated molecule stays centered in the subspace the total amount of applied gaussian noise has to always add up to 0. This ensures that the input and output are roto-translation equivariant during training and inference.  

 
The parameter α defines how much of the input remains in a single noise step and is changed during training according to the newly introduced adaptive noise schedule. Even though MiDi must balance all features of the graph to generate a stable molecule, atom coordinates and bond types have less flexibility in this regard and cannot be derived from atom type and formal charge. This is why during the noise application α for coordinates and bond type decreases slower than for atom type and formal charge. In the next chapter we will discuss what this means for the inference step, after exploring the architecture of the denoising model.

Denoising model
------

The denoising model is the main component of MiDi that is trained to generate mixed molecule graphs. In general, the second part of a diffusion model takes noisy data as input and returns a clean output. It achieves this by learning to reverse the noise that the noise model applied to the original training data and then applying this reversal over the noisy input graph. To do so, MiDi combines three architectures and a mixed loss function, which we will now explore further.

Firstly, we are going to explore the architecture. At its core, the denoising model of MiDi is a multi-layer transformer graph neural network enclosed by MLPs. 

To understand what happens herer we will follow a sample throught the network and discuss all parts of the denoising model, from input over architecture to loss.

The denoising model takes the noisy graph features and graph-level features. The graph-level features are not part of the output but help the network retain and process global information throughout the generation process.

Architecture
------

MiDi has a transformer architecture that is build up from successive self-attention modules, following normalization layers and feedforward networks. The complete transformer architecture is encapseld in two MLP and shown in the following picture. We will now discuss each part in more detail to fully understand what here happens.

As we have already discussed the inputs lets focus first on the separated and parallel MLPs for each feature. At first glance it appears to be standard MLPs, which it is, apart from the MLP of the 3D coordinates. For the molecule generation it is important to hold the E(3) equivariance and therefore the MLP was adapted to hold this condition.

Here possibly the definition of what has happend with the MLP and NOrma layers

After the first MLP we start with the core of the network. We can see multiple E(3)+ Graph Transdormer layers that are all build equally. It starts with the so called Update Block, that updates all features in sequence, starting with extractiong the 3D informations from the coordinates. This is done with the novel Relaxed Equivariant Graph Neural Networks (rEGNNs) that are based on Equivariant Graph Neural Networks (EGNN). EGNNs are effective and affordable layer for processing 3D coordinates while maintaining the E(3)equivariane. These layers recursivly update each coordinate of the graph using only rotation-invariant arguments from the node and edge features.

$$
[\Delta r]_{ij}
=
\operatorname{cat}
\left(
\left\|r_i-r_j\right\|_2,\;
\left\|r_i\right\|_2,\;
\left\|r_j\right\|_2,\;
\cos(r_i,r_j)
\right).
$$

In the case of MiDi we can exploit the fact that our inputs are all centered around their zero point of mass (CoM) after the noise process. Therefore the original formulation was *relaxed* to ignore translation and only uses rotation invariant descriptors (e.g $\cos(r_i,r_j) \text{ or } ||r_j||_2$). The new 3D information $||r_i-r_j||_2,\, ||r_i||_2,||r_j||_2$ are all calculated realtive to the center of mass $0$ and are rotation invariant. Combined with the term $\left(r_i-r_j\right)$ they guarantedd rotation equivariance.

$$
\begin{aligned}
[\Delta_r]_{ij}
\leftarrow
\left[
\lVert r_i\rVert,\,
\lVert r_j\rVert,\,
\lVert r_i-r_j\rVert,\,
\cos(r_i,r_j)
\right] && \text{(Extract 3D information)} \\
y_{ij}
\leftarrow
\phi_y\!\left(
y_{ij},\,
x_i\odot x_j,\,
[\Delta_r]_{ij},\,
w
\right) && \text{(Edge Embeddings)} \\
\alpha_{ij}
\leftarrow
\operatorname{softmax}\!\left(
\phi_\alpha\!\left(
y_{ij},\,
W_{\mathrm{key}}^{T}x_i,\,
W_{\mathrm{query}}^{T}x_j,\,
[\Delta_r]_{ij}
\right)
\right) && \text{(Attention coefficient)} \\
x_i
\leftarrow
\phi_X\!\left(
\sum_j
\alpha_{ij}W_{\mathrm{val}}^{T}x_j,\,
\operatorname{PNA}(Y),\,
\operatorname{PNA}(\Delta_r),\,
w
\right) && \text{(Node Embeddings)} \\
r_i
\leftarrow
r_i
+
\sum_j
\phi_m\!\left(
[\Delta_r]_{ij},\,
y_{ij}
\right)
(r_j-r_i) && \text{(rGNN Update)} \\
w
\leftarrow
\phi_w\!\left(
w,\,
\operatorname{PNA}(X),\,
\operatorname{PNA}(\Delta_r),\,
\operatorname{PNA}(Y)
\right) && \text{(Graph Embeddings)} \\
\end{aligned}
$$

The complete update process extracts the 3D information from the current graph and then uses the $\Delta_r$, node, and global features to update the edge embeddings

noise s
loss 

$$
\mathcal{L}(G,\hat{p}^{G}) =
\lambda_r \left\|\hat{R}-R\right\|^2
+ \lambda_x\,\mathrm{CE}\!\left(X,p_{\theta}^{X}\right)
+ \lambda_c\,\mathrm{CE}\!\left(C,p_{\theta}^{C}\right)
+ \lambda_y\,\mathrm{CE}\!\left(Y,p_{\theta}^{Y}\right)
$$

Experiments:
======

After introducing the model, the most important question is whether generating the molecular graph and the 3D conformation together actually helps. MiDi is evaluated on unconditional molecule generation: the model is not asked to optimize a specific property or binding pocket, but simply to sample realistic molecules from the learned distribution.

This setting is useful because it isolates the core ability of the model. A good molecule generator should not only place atoms in plausible 3D positions, but also assign chemically meaningful bonds, atom types, and formal charges. In previous 3D diffusion models, the graph is usually reconstructed after generation, for example by looking at interatomic distances or by using chemistry software such as OpenBabel. MiDi instead predicts the graph and the conformation during the same denoising process.

Experimental setup
------

The authors compare MiDi against the Equivariant Diffusion Model (EDM) followed by a separate bond prediction step. EDM is also a equivariant diffusion model for 3D molecule generation. Since EDM only generates atom types and coordinates, bonds must be added afterwards. The paper considers two variants: EDM with a simple distance-based lookup table, and EDM followed by OpenBabel, which tries to optimize bond orders to make the molecule chemically valid. 

The experiments are performed with explicit hydrogen atoms, which makes the task harder because the model has to place and connect all atoms, not only the heavy atoms. The authors evaluate on two datasets, depicted in table ... :

QM9 has a lot less samples and contains small molecules with up to 9 heavy atoms. The GEOM-DRUGS data set on the other side is much more challenging and closer to realistic drug discovery. It contains around 430,000 drug-sized molecules, with an average of 44 atoms and up to 181 atoms. The evaluation combines graph-based and geometry-based metrics. 

For the graph structure, the paper reports standard molecule generation metrics such as validity, uniqueness, novelty, and connectedness. Validity is measured using RDKit sanitization. The authors also report atom stability and molecule stability, which check whether valency constraints are satisfied without adding implicit hydrogens. 

For the distribution of generated molecules, they compare atom types, bond types, and valencies between generated samples and the test set. Smaller distances mean that the generated molecules look more like the data distribution.

Finally, MiDi is also evaluated as a 3D generator. The authors compare bond length and bond angle distributions between generated molecules and real molecules. This is important because a model could generate a valid graph but still place atoms in unrealistic 3D arrangements.

Results
------

On the smaller QM9 dataset, most methods already perform quite well. This is not too surprising: QM9 molecules are small ($), and their bonds can often be recovered from atom distances without much ambiguity. Even here, MiDi improves over the base EDM model on graph-based metrics. With the adaptive noise schedule, MiDi reaches 97.5% molecular stability and 97.9% validity, while EDM reaches 90.7% molecular stability and 91.7% validity. OpenBabel remains very strong on this dataset, because the molecules are simple enough for a rule-based bond-reconstruction step to work reliably.

The more interesting test is GEOM-DRUGS. This dataset contains larger, drug-like molecules with many more atoms and more complicated structures. Here, the limitations of the two-step approach become obvious. EDM can generate reasonable-looking 3D point clouds, but when bonds are inferred afterwards, only 5.5% of the resulting molecules are molecularly stable. Adding OpenBabel helps, increasing molecular stability to 40.3%, but this still leaves most generated molecules chemically problematic.

MiDi performs much better on this harder benchmark. The adaptive version generates 91.6% molecularly stable molecules, while also keeping atom stability close to the data distribution at 99.8%. This is the main result of the paper: jointly denoising the graph and the coordinates allows the model to learn chemistry and geometry as a coupled object, rather than treating bond prediction as an afterthought.

The 3D metrics show the same trend. On GEOM-DRUGS, MiDi produces much more realistic bond angles than EDM or EDM followed by OpenBabel. The bond-angle Wasserstein distance drops from around 6 degrees for the baselines to 1.07 degrees for adaptive MiDi. The valency distribution is also much closer to the test set, which means the generated atoms satisfy chemical bonding rules more consistently.

The adaptive noise schedule is another important part of the result. Compared to the uniform schedule, it improves molecular stability on GEOM-DRUGS from 89.9% to 91.6%, and it also improves the valency, bond-angle, and bond-length metrics. Intuitively, this supports the authors’ design choice: during sampling, it is useful to first form a rough 3D structure and bond layout, and only later refine atom types and formal charges.

Overall, the results suggest that MiDi’s advantage does not come from simply adding more machinery to a diffusion model. Its strength comes from matching the generative process to the object being generated. A molecule is not just a point cloud and not just a graph. It is both at the same time. MiDi benefits from treating these two views as one coupled representation throughout the denoising process.
