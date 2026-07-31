---
title: 'MiDi: Mixed Graph and 3D Denoising Diffusion for Molecule Generation'
date: 2026-07-31
permalink: /posts/2026/07/midi-molecule-generation/
tags:
  - Diffusion
  - Drug Molecule Generation
  - Molecule Graph
---


Introduction to MiDi
======
When designing a new drug, scientists often have to generate a new molecule that is not only stable but also satisfies every condition necessary for the desired effect. Since the explosive rise of generative models, it is no wonder that these topics have been combined in research. The authors Vignac, Osman, Toni & Frossard wanted to tackle this task too, and presented their model MiDi in their paper *MiDi: **Mi**xed Graph and 3D Denoising **Di**ffusion for Molecule Generation* (2023). In this blog post, I want to recap the architecture, novelties, and results of MiDi so even without deep AI knowledge you can understand what makes it special.

We will explore the motivation behind MiDi, explain why molecule generation needs both graph and 3D information, and look at how the model combines discrete diffusion for chemical properties with Gaussian diffusion for spatial coordinates. Finally, we will discuss why this approach leads to much more stable molecules on challenging drug-like datasets compared to similar models.

To do so, we will first discuss why it even makes sense to use neural networks for molecule generation, and cover the most important basics needed to understand how MiDi works and what is important when creating a molecule.

Why even consider deep learning?
======

When we think about generating molecules, we can reformulate it as a search problem with a new dimension for each feature of a molecule. The features that interest us here are the number of atoms, type of atoms, type of bond between all atoms, and three-dimensional structure. When we consider an average-sized drug of $\approx40$ atoms, we can already see how this results in an enormous search space of different combinations. Even though only a small number of all elements are actually used in drug synthesis, the possible structures in 3D space are near limitless. This search space would increase even further if we included special atom groups. So at this point we can clearly say that randomly selecting a possible molecule will most likely not result in a viable new drug.

Through the aquired chemical knowledge, we know that not all combinations are possible and can therefore reduce the space by applying rules and constraints that apply to stable molecules. This opens the possibility of creating heuristic rule sets or algorithms to generate a molecule, right? Yes, but that really only works for small molecules. The larger a molecule gets, the more complex its configuration becomes, resulting in increasingly difficult prediction of its attributes and properties. Every new change can destabilize the molecule, and there is no single fixed order to follow when creating one. This is where even advanced algorithms start to fail.

To summarize, we have a huge search space combined with complex relations between the features. I do not know about you, but this already sounds like a solution where we should definitely try to apply deep learning, and this is what we will explore with MiDi.

What is special about MiDi?
======
Before I can answer this question, I have to tell you something important about molecules. Molecules are often depicted as two-dimensional graphs. These representations define the chemical properties, bond and atom types, and functional groups that are essential for drugs. They are also used by scientists to derive the synthetic path of the molecule. But this is not enough: given a fixed 2D graph, we can create multiple fitting 3D arrangements, called conformers. The conformer defines how the molecule interacts with other molecules, controls its biological activity, and determines its binding affinity to proteins. The following figure 1 shows you an example of a two-dimensional graph and a fitting three-dimensional conformer.

<figure>
  <img src="/images/mixed_graph.jpg" alt="mixed graph" style="display:block; margin:auto; width:80%"/>
  <figcaption style="text-align:center">Figure 1: Example for 2D and single 3D structure of a molecule. Source: <a href="https://www.drugdiscoverytrends.com/figuring-out-the-3-d-shape-of-molecules-with-a-push-of-a-button/​">Drug Discovery & Development</a>
  </figcaption>
</figure>

Generating only the 2D graph is a research field that has already been explored thoroughly, and it is possible to generate a conformer using an additional generator or algorithm. Even though 3D-structure molecule generators are not as well researched and provide no information on bond types, the reverse process is also possible. One example of a 3D-only generator is the Equivariant Diffusion Model (EDM). When combining it with the chemical software OpenBabel, which predicts the resulting 2D graph, we get both representations — just not in a single step, but across multiple steps. This is often undesirable as it introduces more room for errors across the chain of steps and is less practical. However, before MiDi, EDM+OpenBabel could be considered state-of-the-art when it comes to molecule generation.

This is the point where MiDi enters the stage. Unlike before, MiDi is the first model able to generate both the 2D and 3D structure of a molecule given only the number of atoms. It is therefore the first end-to-end differentiable model in its field and can optimize the entire task jointly. This allows the model to learn all properties of a molecule while balancing the different effects of the 2D and 3D representations.

<figure>
  <img src="/images/selected_samples1.png" alt="selected_samples1" style="display:block; margin:auto; width:80%"/>
  <figcaption style="text-align:center">Figure 2: Example output from MiDi.</figcaption>
</figure>

In the figure 2 above, I give you a first glimpse at the generated molecules from MiDi, so you can imagine the 2D and 3D structures we are talking about in this post. As a side note: this is where the *Mixed* (Mi) part in the name comes from.


We have to talk about some basics now
======

To fully grasp the process and architecture, we will now cover the background knowledge needed to understand MiDi's function, architecture, and novelties.

Molecules as graphs
------

Previous papers on molecule generation have already demonstrated the efficiency of graph structures. MiDi is no different in this regard and also represents its molecules as a graph. A graph generally consists of nodes and edges connecting them. Nodes represent the atoms of a molecule, the edges are the bonds between them, and both are described by their features. In the case of MiDi, each molecule built from $n$ atoms is represented as the graph $G = (\mathbf{X},\mathbf{C},\mathbf{R},\mathbf{Y})$ with $\mathbf{X}$ as the atom type and $\mathbf{C}$ as their formal charge, which are both one-hot encoded vectors of length $n$. $\mathbf{R}\in\mathbb{R}^{3 \times n}$ are the three-dimensional coordinates of all $n$ atoms and $\mathbf{Y}$ are the bond types between them, saved as a $n \times n$ matrix. Notably, having no bond is also considered to be a bond type. Atom type, formal charge and coordinates relate to an atom and are therefore node features, while the bond type is a edge feature, connecting two nodes. Looking at Figure 1 again, it becomes obvious why this representation fits a molecule so naturally and intuitively.

E(3)-Equivariance
------
When generating any structure in space, we must respect its dynamic nature. Molecules do not have a general order or fixed sequence and can undergo translation and rotation in 3D space. This does not affect their functionality, type, or features — except the coordinates of each atom when referenced to the same origin point. A molecule generator should therefore be able to generate the same molecule regardless of how the input is positioned in space. This is where the concept of equivariance comes into play. Equivariance holds when any transformation of the input results in the same transformation of the output. So when we rotate the first input graph from Figure 3, the corresponding outputs rotate equally. This also eliminates the need for randomly augmented training data, as all only rotated samples become redundant. For MiDi we do consider E(3)-equivariance, which is equivariance in three-dimensional space.

<figure>
  <img src="/images/equivariance.jpg" alt="Diagram showing that rotating the input graph by a transformation S_g and then passing it through the network gives the same result as first passing the original input through the network and then rotating the output by the same transformation." style="display:block; margin:auto; width:80%"/>
  <figcaption style="text-align:center">Figure 3: The original input graph (top-left) is rotated by $S_g$ to produce a second input graph (top-right). Both graphs are processed by the neural network $\phi$. The key insight is that the second output (bottom-right) can be derived directly from the first output (bottom-left) by applying the same rotation $T_g = S_g$. Source: <a href="https://arxiv.org/abs/2106.09645">E(n) Equivariant Graph Neural Networks (Satorras et al., 2021)</a>
  </figcaption>
</figure>

Transformers
------

Transformers are already well established in the world of neural networks. MiDi is no different from most other modern models and benefits from the stable processing over multiple layers. Most people in the world of AI have probably already heard of them and are at least somewhat familiar with them. Nonetheless, we will do a short sidetrack on what a transformer is and does.

Transformers are a class of neural networks that process sets of elements by letting every element attend to every other element simultaneously. At its core lies the self-attention mechanism: given a set of input vectors (representing a part of the input, also called a token), each one is projected into a query $q$, a key $k$, and a value $v$. The attention weight between element $i$ and element $j$ is computed as $\alpha_{ij} = \text{softmax}(q_i^\top k_j / \sqrt{d})$, and the new representation of element $i$ is the weighted sum $\sum_j \alpha_{ij} v_j$. Intuitively, this allows the network to decide how much each element should "attend to" every other element when updating its own representation. Because self-attention has no notion of order, positional or structural information (such as edges in a graph) must be injected explicitly. Transformers are stacked in multiple layers, each followed by a normalization step and a feedforward network, enabling the model to build increasingly abstract representations of the input. In MiDi, this mechanism is applied over the atoms of a molecule: each atom can attend to all other atoms and their bonds, allowing the network to capture long-range chemical dependencies.

Diffusion:
------

As the paper title states, MiDi is a denoising **di**ffusion model, which is the combination of a noise model and a denoising model. Only the denoising model is actually used during inference. However, it is still crucial to understand the complete pipeline to see how MiDi operates.

Noise model:
------

The sole purpose of the noise model is to incrementally apply noise to the input $x$ so that each step is a more corrupted version of the previous step. This results in a Markovian trajectory in the form of:

$$
q\!\left(G^{1},\ldots,G^{T}\mid G^{0}\right)
=
\prod_{t=1}^{T}
q\!\left(G^{t}\mid G^{t-1}\right),
$$

With each step the original input gets more and more corrupted until it is, at least to us mere humans, no longer recognizable. Figure 4 visualises this process on a picture.

<figure>
  <img src="/images/Diffusion.png" alt="A sequence of images showing a cat being progressively corrupted by Gaussian noise over T timesteps until the original image is completely unrecognizable." style="display:block; margin:auto; width:80%"/>
  <figcaption style="text-align:center">Figure 4: The forward noise process illustrated on an image. Starting from the clean input $G^0$ on the left, noise is added step by step until the original structure is completely destroyed at step $T$ on the right.</figcaption>
</figure>

The purpose of the noise model lies in creating a noisy graph so that the denoising model can learn to reverse the noise. Generally, the denoising model can either predict the noise and reverse it in the same style as the noise model or predict the clean output directly. MiDi does the former. As already explained in previous sections, MiDi generates a clean graph described by its edges and node features. The 3D coordinates are continuous values while the remaining features are discrete values. The corresponding noise is therefore either continuous Gaussian for coordinates noise or discrete noise for the remaining features, resulting in the following combined noise at each step on each training sample:

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
&& \text{(Gaussian noise on coordinates)}
\\
&\times
\mathcal{C}
\!\left(
X^{t};
X^{t-1}Q_{x}^{t}
\right)
&& \text{(Discrete noise on atom types)}
\\
&\times
\mathcal{C}
\!\left(
C^{t};
C^{t-1}Q_{c}^{t}
\right)
&& \text{(Discrete noise on formal charge)}
\\
&\times
\mathcal{C}
\!\left(
Y^{t};
Y^{t-1}Q_{y}^{t}
\right)
&& \text{(Discrete noise on bond types)}.
\end{aligned}
$$

Atom type $\mathbf{X}$, bond type $\mathbf{Y}$, and formal charge $\mathbf{C}$ receive discrete noise defined by the transition matrix $Q^t$, which has been derived from the marginal distribution of the training set for each feature. All transition matrix $Q$ define the probablity of a feature to transfer from one class to another. To ensure that the generated molecule stays centered in the subspace, the total amount of applied Gaussian noise $\epsilon$ must always sum to zero $\sum_{i=1}^{n} \epsilon_i = 0$. This ensures that the input and output are roto-translation equivariant during training and inference.

The parameter $\alpha$ defines how much of the input remains after a single noise step and is adjusted during training according to the newly introduced adaptive noise schedule in figure 5. Even though MiDi must balance all features of the graph to generate a stable molecule, atom coordinates and bond types have less flexibility in this regard, as they cannot be derived from atom type and formal charge alone. This is why, during noise application, $\alpha$ for coordinates and bond type decreases more slowly than for atom type and formal charge. If the denoising model then tries to reverse this problem, it will first generate a possible 3D structure and bond types and then fit them with atom types and formal charges to stabilize everything.

<figure>
  <img src="/images/adaptive_cosine_schedule-1.png" alt="Plot of the adaptive cosine noise schedule showing that alpha decreases more slowly for atom coordinates and bond types than for atom types and formal charges across the T diffusion timesteps." style="display:block; margin:auto; width:80%"/>
  <figcaption style="text-align:center">Figure 5: The adaptive cosine noise schedule. Atom coordinates and bond types (blue) retain more of their original signal for longer than atom types and formal charges (orange). This reflects the fact that coordinates and connectivity are harder to recover and should therefore be corrupted more gradually.</figcaption>
</figure>

Denoising model
------

The denoising model is the main component of MiDi that is trained to generate clean mixed molecule graphs. In general, the second part of a diffusion model takes noisy data as input and returns a clean output. MiDi achieves this by learning to reverse the noise that has been added to the original training data, and then applying this reversal to the noisy input graph. To do so, MiDi combines a transformer architecture with a mixed loss function, which we will now explore further.

First, we will look at the architecture from figure 6. At its core, the denoising model of MiDi is a multi-layer transformer graph neural network enclosed by multilayer perceptrons (MLPs). To understand what happens there, we will follow a sample through the network and discuss all parts of the denoising model, from input over hidden layers to loss.

The denoising model takes a noisy graph of $n$ atoms with noisy features and graph-level features $\omega$ sampled from the distributions of the training data as input. The graph-level features $\omega$ are not part of the output but help the network retain and process global information throughout the generation process.
The following section will contain a lot of math. To understand what MiDi does you do not have to understand each equation, but I included them anyway for the math enthusiasts or just for a deeper understanding of what happens.

Architecture
------

MiDi has a transformer architecture that is built up from successive self-attention modules, followed by normalization layers and feedforward networks. The complete transformer architecture is encapsulated within two MLPs, as shown in the following figure 6. We will now discuss each part in more detail to fully understand what happens here.

<figure>
  <img src="/images/architecture.png" alt="Architecture of MiDI" style="display:block; margin:auto; width:80%"/>
  <figcaption style="text-align:center">Figure 6: Architecture of MiDi.</figcaption>
</figure>

Having already discussed the inputs, let us focus first on the separated, parallel MLPs for each feature. At first glance these appear to be standard MLPs, which they are — except for the MLP of the 3D coordinates. For molecule generation it is important to maintain E(3) equivariance, and therefore this MLP was adapted to fullfill that condition. To preserve E(3) equivariance, MiDi replaces the ordinary coordinate-wise MLP with a geometric PosMLP:

$$
\operatorname{PosMLP}(R)
=
\Pi_{\mathrm{CoM}}
\left(
\operatorname{MLP}(\lVert R\rVert)
\frac{R}{\lVert R\rVert+\delta}
\right)
\in \mathbb{R}^{n\times 3}
$$

it applies an MLP only to each atom’s rotation-invariant distance from the origin $\|\mathbf{R}\| \in \mathbb{R}^{n\times 1}$, then uses the result to rescale the atom’s direction vector and re-centers all coordinates to keep the molecular center of mass at zero.

After the first MLP we enter the core of the network. We can see multiple E(3)+Graph Transformer layers that are all built the same way. Each begins with the so-called Update Block, which updates all features in sequence, starting with extracting the 3D information from the coordinates. This is done for the novel relaxed Equivariant Graph Neural Networks (rEGNNs), which are based on Equivariant Graph Neural Networks (EGNNs). EGNNs are effective and efficient layers for processing 3D coordinates while maintaining E(3) equivariance. These layers recursively update each coordinate of the graph using only rotation-invariant arguments from the node and edge features.

In the case of MiDi, we can exploit the fact that our inputs are all centered around their center of mass (CoM) after the noise process. Therefore, the original formulation was *relaxed* to ignore translation and only use rotation-invariant descriptors (e.g., $\cos(r_i,r_j)$ or $\|r_j\|_2$). The new 3D information $\|r_i-r_j\|_2,\, \|r_i\|_2, \|r_j\|_2$ are all calculated relative to the zero center of mass and are rotation-invariant. Combined with the term $(r_i-r_j)$, they guarantee rotation equivariance.

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
w
\leftarrow
\phi_w\!\left(
w,\,
\operatorname{PNA}(X),\,
\operatorname{PNA}(\Delta_r),\,
\operatorname{PNA}(Y)
\right) && \text{(Graph Embeddings)} \\
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
\end{aligned}
$$

The complete update process extracts the 3D information from the current graph and then uses $\Delta_r$, node $\mathbf{X}$, and global features $\omega$ to update the edge embeddings $\mathbf{Y}$. Using transformer attention, MiDi updates the node embeddings $\mathbf{X}$ with the previously calculated attention coefficients $\alpha$. The flattened attention heads are then used to calculate final graph embedding. To do this, MiDi uses PNA layers for pooling pairwise features ($\mathbf{Y}$ and $\Delta_r$). In the last step the afore mentioned rEGNN layer updates the coordinates to keep E(3) equivariance.

After the updates are complete, MiDi applies dropout for better generalization and then normalizes all features except the graph features. Similarly to the first MLP layer, the normalization layer for the 3D coordinates is again modified to maintain E(3) equivariance:

$$
\operatorname{E3Norm}(R)
=
\gamma\,
\frac{\lVert R\rVert}{\bar{n}+\delta}
\frac{R}{\lVert R\rVert}
=
\gamma\,
\frac{R}{\bar{n}+\delta},
\qquad
\bar{n}
=
\sqrt{\frac{1}{n}\sum_{i=1}^{n}\lVert r_i\rVert_2^2}
$$

Finally, after repeating this process multiple times, MiDi applies a final MLP layer and returns its output values. 3D coordinates are returned as pointwise estimates, while the atom type, formal charge, and bond type are predicted as distributions over all classes. The corresponding loss during training is therefore a combination of regression loss for coordinates and cross-entropy (CE) loss for all others.

$$
\begin{aligned}
\mathcal{L}(G,\hat{p}^{G}) ={}&
\lambda_r \left\|\hat{R}-R\right\|^2
&& \text{(Coordinate regression loss)} \\
&+ \lambda_x\,\mathrm{CE}\!\left(X,p_{\theta}^{X}\right)
&& \text{(CE atom type loss)} \\
&+ \lambda_c\,\mathrm{CE}\!\left(C,p_{\theta}^{C}\right)
&& \text{(CE formal charge loss)} \\
&+ \lambda_y\,\mathrm{CE}\!\left(Y,p_{\theta}^{Y}\right)
&& \text{(CE bond type loss)}
\end{aligned}
$$

That was a lot of math. But we did it and all you have to remember is that MiDi is an E(3)+Graph Transformer that takes a noisy graph as input, attends each feature to each other multiple times while maintaining equivariance through targeted modifications to finally return a clean and stable graph after repeatedly propagating a combined loss back through the network.

How was MiDi tested?
======

After introducing the model itself, the most important question for us should be whether generating the molecular graph and the 3D conformation together actually helps. To find that out, MiDi will now be asked to generate realistic molecules from the learned distribution after being trained on two different datasets. A good molecule generator should not only place atoms in plausible 3D positions, but also assign chemically meaningful bonds, atom types, and formal charges.

Experimental setup
------

MiDi will compete against EDM followed by a separate bond prediction step. The paper considers two variants: EDM with a simple distance-based lookup table, and EDM followed by OpenBabel, which tries to optimize bond orders to make the molecule chemically valid.

The experiments are performed with explicit hydrogen atoms, which makes the task harder because the model has to place and connect all atoms, not only the heavy atoms. The authors evaluate on two datasets, depicted in the following table:

| Information | QM9 | GEOM-DRUGS |
|---|---|---|
| **Dataset type** | Standard benchmark of small molecules | Large dataset of drug-sized, drug-like molecules |
| **Number of molecules** | **133,000** molecules | **430,000** molecules |
| **Molecule size** | Up to **9 heavy atoms** per molecule | Average of **44 atoms**, with up to **181 atoms** |

QM9 has far fewer samples and contains small molecules with up to 9 heavy atoms, which are all atoms except hydrogen atoms. The evaluation for MiDi was done with full molecules, including the additional hydrogen atoms. The GEOM-DRUGS dataset, on the other hand, is much more challenging and closer to realistic drug molecules. It contains 430,000 drug-sized molecules, with an average of 44 atoms and up to 181 atoms. The evaluation combines metrics for molecule structure and the general distribution of generated molecules compared to the training data. The following table explains in more detail what each metric means, first the six molecule metrics and then the five distribution metrics.

| Metric | Description |
|---|---|
| **Molecule stability** | The molecule resists changing into other substances and does not break apart on its own. |
| **Atom stability** | The atom's nucleus and electron arrangement are balanced and will not spontaneously change or break apart. |
| **Validity** | Success rate of the RDKit sanitization pipeline over 10000 molecules. This defines whether the molecule is chemically valid. |
| **Uniqueness** | Proportion of valid generated molecules with different canonical SMILES. SMILES is a way to write a molecule as a simple line of text and therefore reflects the diversity of the generated molecules. This includes molecules that are already present in the training data. |
| **Novelty** | Similar to Uniqueness, with the only difference being that the SMILES are not allowed to appear in the training data, therefore showing the model's ability to generate new molecules. |
| **Connected** | All training molecules have a single connected compound, meaning every atom can be reached from any other atom by following the graph structure through its edges. |
| **Valency\[$e^{-2}$\]** | Valency is the capacity of an atom to bind, measured as the number of electrons it loses, gains, or shares when bonding. |
| **Atom\[$e^{-2}$\]** | Atom types that occurred in the generated molecules. |
| **Bond\[$e^{-2}$\]** | Bond types that occurred in the generated molecules. |
| **Angles\[$°$\]** | Angles between two bonds. |
| **Bond Lengths\[$e^{-2}\text{Å}$\]** | Length of each occurring bond type. |

For general molecule generation properties, the paper reports standard metrics such as molecule stability, atom stability, validity, uniqueness, novelty, and connectedness. The reported results show the proportion of generated molecules that satisfy the above-mentioned criteria from the generated samples. The metrics for valency, atom, and bond focus on comparing the distribution of the training dataset with the distribution of the generated dataset using the Wasserstein distance.

Additionally, MiDi is evaluated as a 3D generator. The authors compare bond length and bond angle distributions between generated molecules and real molecules. This is important because a model could generate a valid graph but still place atoms in unrealistic 3D arrangements. Similarly to the previous metrics, these are also compared using the Wasserstein distance.

Does MiDi work?
======
In short: yes — but how well exactly? Now that we know how MiDi works and have established a basis for evaluation, let's first look at the smaller, simpler QM9 dataset. The following table gives a good picture for comparing the different results. Remember that EDM only generates 3D structures with atoms, but no bond types. We will focus on the EDM model with lookup table, EDM assisted by OpenBabel to generate fitting bonds, and the MiDi model with the adaptive noise schedule. The following results always show the comparison between the test set and the generated set. The data section of the results compares the test set with the training set to show the generalization ability of MiDi.

So far not a huge improvement
------

<figure>
  <img src="/images/qm9_molecule.png" alt="Table of molecule generation metrics on QM9 comparing EDM, EDM with OpenBabel, and MiDi with adaptive noise schedule across stability, validity, uniqueness, novelty, and connectedness." style="display:block; margin:auto; width:80%"/>
  <figcaption style="text-align:center">Figure 7: Molecule generation metrics on QM9. MiDi with the adaptive noise schedule improves over the base EDM model on stability, validity, and connectedness. OpenBabel-assisted EDM remains a strong competitor on this simpler dataset, where bond types can often be inferred from atom distances alone.</figcaption>
</figure>

On the smaller QM9 dataset, the baseline EDM networks already perform quite well, both with and without their assistive software. This is not too surprising: QM9 molecules are small and their bonds can often be recovered from atom distances without much ambiguity. Still, MiDi improves over the base EDM model on molecule stability, atom stability, validity, and connectedness, though only by a small margin. With the adaptive noise schedule, MiDi reaches $97.5\%$ molecular stability and $97.9\%$ validity, while EDM reaches $90.7\%$ molecular stability and $91.7\%$ validity. OpenBabel further improves EDM's performance, which leads to an even smaller margin.

<figure>
  <img src="/images/distribution_qm9.png" alt="Bar chart comparing the Wasserstein distances for valency, atom type, bond type, bond angle, and bond length distributions on QM9 between EDM, EDM with OpenBabel, and MiDi." style="display:block; margin:auto; width:80%"/>
  <figcaption style="text-align:center">Figure 8: Wasserstein distances of the marginal distributions on QM9 (lower is better). MiDi improves on valency, atom, and bond type distributions, while EDM still produces slightly more realistic bond angles and lengths for these small molecules.</figcaption>
</figure>

Looking at the Wasserstein distances of the marginal distributions, we can see a similar pattern — remember that we want a small distance between the distributions. Valency, atom, and bond diversity have already improved when using MiDi, but this changes when we look at the bond angles and lengths. Apparently, when generating small molecules, EDM generates more realistic 3D information, which is not surprising as EDM's main task is to generate coordinates and not the complete 2D and 3D structure of a molecule. If we stopped here, we could already say that MiDi is able to compete with state-of-the-art models, even if it does not always surpass them — especially when they are combined with additional software. But this is already a significant finding, because MiDi is able to generate both 2D and 3D stable structures while being end-to-end differentiable.

Let's make it more difficult
------
Now that we have seen the results on the small dataset, I am sure you are as excited as I am to look at the realistic GEOM-DRUGS dataset.

<figure>
  <img src="/images/geom_molecule.png" alt="Table of molecule generation metrics on GEOM-DRUGS comparing EDM, EDM with OpenBabel, and MiDi with adaptive noise schedule, showing a dramatic improvement in molecular stability for MiDi." style="display:block; margin:auto; width:80%"/>
  <figcaption style="text-align:center">Figure 9: Molecule generation metrics on GEOM-DRUGS. The advantage of MiDi's joint generation becomes clear here: while EDM alone achieves only $5.5\%$ molecular stability, MiDi reaches $91.6\%$ — showing that generating graph and 3D structure together is essential for drug-like molecules.</figcaption>
</figure>

This dataset contains larger, drug-like molecules with many more atoms and more complex structures. Here, the limitations of the two-step approach become obvious. EDM can generate reasonable-looking 3D point clouds, but when bonds are inferred afterwards, only $5.5\%$ of the resulting molecules are molecularly stable. Adding OpenBabel helps, increasing molecular stability to $40.3\%$, but this still leaves most generated molecules chemically problematic. MiDi performs much better on this harder benchmark. The adaptive version generates $91.6\%$ molecularly stable molecules, while also keeping atom stability high at $99.8\%$. This is the main result of the paper: jointly denoising the graph and the coordinates allows the model to learn chemistry and geometry as a coupled object, rather than treating bond prediction as an afterthought. In the end, MiDi is able to outperform EDM with and without OpenBabel on every molecule metric except Validity, where MiDi only reaches $77.8\%$.

<figure>
  <img src="/images/distribution_qm9.png" alt="Bar chart comparing the Wasserstein distances for valency, atom type, bond type, bond angle, and bond length distributions on GEOM-DRUGS between EDM, EDM with OpenBabel, and MiDi." style="display:block; margin:auto; width:80%"/>
  <figcaption style="text-align:center">Figure 10: Wasserstein distances of the marginal distributions on GEOM-DRUGS (lower is better). MiDi produces substantially more realistic bond angles and valency distributions than both EDM variants, confirming that jointly learning graph structure and 3D geometry leads to chemically more consistent molecules.</figcaption>
</figure>

The 3D metrics show the same trend. On GEOM-DRUGS, MiDi produces much more realistic bond angles than EDM or EDM followed by OpenBabel. The bond-angle Wasserstein distance drops from around $6°$ for the baselines to $1.07°$ for adaptive MiDi. The valency distribution is also much closer to the test set, which means the generated atoms satisfy atom bonding rules more consistently.

Overall, the results suggest that MiDi's advantage does not come from simply adding more machinery to a diffusion model. Its strength comes from matching the generative process to the object being generated. A molecule is not just a point cloud and not just a graph — it is both at the same time. MiDi benefits from treating these two views as one coupled representation throughout the denoising process.

Surely MiDi cannot do everything?
======
Of course not. Beyond the obvious drawback of diffusion models being computationally expensive, MiDi also requires a large and high-quality dataset to truly capture the vast search space and the chemical complexity. With GEOM we already have a good dataset for drug-sized molecules, but this, unfortunately, does not hold for all molecule sizes. Even if we had a dataset for each size, MiDi itself has size limitations. This may be harder to believe when we look at AlphaFold, which is able to generate proteins with thousands of amino acids — but proteins are sequences and can therefore also be generated sequentially. Molecules must be generated as a whole to have a true end-to-end solution. The biggest drawback for MiDi, however, is at the same time its future goal. MiDi is a so-called unconditional generator, meaning it generates a molecule without any further information other than its atom count. So we as users get a stable molecule, but we cannot define whether it needs a specific functional group or must have a specific structural feature to bind properly in our body. Returning to the motivation section, where we talked about the vast search space, you can imagine that MiDi does not deliver our desired molecule, but instead greatly narrows the search space by only returning stable molecules. This sounds worse than it is — we have to remember what a significant leap this represents compared to previous methods. And you always have to take the first step in order to tackle the second, and eventually reach the goal of conditionally generating targeted drug molecules.

Conclusion
======

MiDi makes a compelling case for a simple but important principle: the representation should match the object. A molecule is simultaneously a chemical graph and a 3D structure, and treating these two views as inseparable throughout the generative process leads to measurably better results — especially when the molecules get large and chemically complex.

We started from first principles: how to represent a molecule as a graph $G = (\mathbf{X}, \mathbf{C}, \mathbf{R}, \mathbf{Y})$, why equivariance matters when working in 3D space, and how diffusion models gradually corrupt and then reconstruct data. Building on that foundation, we saw how MiDi applies separate but coupled noise processes to continuous coordinates and discrete features, and how the rEGNN-based transformer denoising model jointly predicts all of them in a single forward pass.

The experiments confirm that this joint approach pays off. On the small QM9 benchmark, MiDi competes with strong baselines. On the much harder GEOM-DRUGS dataset — closer to real drug discovery — it decisively outperforms the two-step pipeline, raising molecular stability to $91.6\%$ — up from $5.5\%$ (EDM alone) and $40.3\%$ (EDM + OpenBabel) — while also producing more realistic bond angles and valency distributions.

What makes MiDi particularly appealing from an engineering perspective is that the entire pipeline remains end-to-end differentiable. There is no chemistry post-processing step that could silently corrupt a promising generated structure. The model learns what makes a valid molecule directly from data, rather than relying on hand-crafted rules.

Of course, MiDi is not the final word. It still struggles with validity on GEOM-DRUGS compared to OpenBabel-assisted baselines, and generating very large molecules remains a challenge for diffusion models in general. Future work will likely push further by conditioning generation on binding pockets, improving sampling efficiency, or scaling to even larger chemical spaces. But as a demonstration that geometry and chemistry can — and should — be learned together, MiDi sets a strong precedent.