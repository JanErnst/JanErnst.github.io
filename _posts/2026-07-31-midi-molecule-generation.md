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
When designing a new drug scientist often have to generate a new molecule that not only is stable but also does cross every consition necessary for the effect and since the explosive rise of generative model it is now wonder that those topics have been combined in reserch. The authors Vignac, Osman, Toni & Frossard wanted to tackle this task to and presented us in their paper *MiDi: **Mi**xed Graph and 3D Denoising **Di**ffusion for Molecule Generation* their model MiDi. In this Blogpost I want to recasulate on the architecture, novelties and results that MiDi has. 

In this blog post, we will build up the motivation behind MiDi, explain why molecule generation needs both graph and 3D information, and look at how the model combines discrete diffusion for chemical structure with Gaussian diffusion for spatial coordinates. Finally, we will discuss why this joint approach leads to much more stable molecules on challenging drug-like datasets.

To do so, we will first discuss why it even makes sense to use neural networks for molecule generation, and repeat the most important basics to understand how MiDi works and what is important when creating a molecule. Afterwards we take a deeper look into MiDis architecture and new features, to then compare it with, at the time, state of the art models that try to do the same.

Why even consider deep learning?
======

When we think about generating molecules we can reformulate it to a search problem with a new dimension for each feature of a molecule. The features that interest us here are number of atoms, type of atoms, type of bond between atoms, 3 dimentional structure. When we now take a average sized drug of $40$ atoms we can already see how this results in an enourmous search space of different combinations. Even if only a small amount of all elements are actually used for drug synthesizing the possible structe in our 3D space is near limitless. This search space would increase even further if we include special groups of atoms, but we will not talk about that further. So at this point we can clearly say that randomly taking a possible molecule will most likely not result in a new drug.

From all the chemical knowledge we, of course, already know that not all combinations are possible and can therefore reduce the space by applying rules and constraints that apply to stable molecules. This opens the possibility of creating heuristic rulesets or algorithms to generate a molecule, right? Yes, but that really only works for small molecules. The larger a molecule gets the more complex its configuration becomes resulting in harder and harder prediction of its attributes and properties. Every new change can destabilize the atom and there is no one order to follow when creating a molecule.


What is special about MiDi?
======
Before I can answer this questions, I have to tell you something important about molecules. Molecules are often depicted as two dimensional graphs like in figure 1. These representations define the chemical properties, bond and atom types, and functional groups that are essential for drugs. They are also important, because scientist can derive the synthetuc path from this graph. But this is not enough: If we now have a fixed 2D graph we can create multiple fitting 3D arrangements, called conformers. The three dimensional structre defines how the molecule interacts with other molecules, controls its biological activity and binding affinity to proteins.

<figure>
  <img src="/images/mixed_graph.jpg" alt="mixed graph" style="display:block; margin:auto; width:80%"/>
  <figcaption style="text-align:center">Figure 1: Example for 2D and single 3D structue of a molecule <a href="https://www.drugdiscoverytrends.com/figuring-out-the-3-d-shape-of-molecules-with-a-push-of-a-button/​">.</figcaption>
</figure>

Generating only the 2D graph is a research field that has already been explored well and it is possible to generate the a conformer with an additional generator or algorithm. Even though 3D structure molecule generators are not researched as well and deliver now information on the bond type, the reverse process is also possible. One example for a 3D-only generator is the Equivariant Diffusion Model (EDM). When combining it with the chemical software OpenBabel, that predicts the resulting 2d graph, we get both pictures, just not with a single but with multiple steps. This is often not desirable as it gives more room for errors accros the chain of events and is less practical.

This is the point where MiDi comes on the stage. Unlike before MiDi is the first model that is able to generate both the 2d and 3d struture of an molecule given only the number of atoms. It is therefore the first model that is end-to-end differentiable in this field and can optimize the entire task. This allows the model to learn all properties of a molecule while balancing the diffent effects of the 2 and 3 dimensional arrangement. 

<figure>
  <img src="/images/selected_samples1.png" alt="selected_samples1" style="display:block; margin:auto; width:80%"/>
  <figcaption style="text-align:center">Figure 1: Example output from MiDi.</figcaption>
</figure>

In the figure above I give you a first glance at the fenerated molecules from MiDi, so ypu can imagin the 2D and 3D structures we are taklking about in this post. Little side note: This is where the *Mixed* (Mi) in the name comes from.


We have to talk about some basics now.
======

To fully understand the process and architecture, we will now cover the background knowledge needed to understand the function, achitecture and novelties of MiDi.

Molecules as graphs
------

Previous works on molecule generation have already demonstrated the efficiency of graph structures. MiDi is no different in this regard and also represents its molecules as a graph. A graph generally consists of nodes and edges connecting them. Nodes represent the atom of a molecule, the edges are the bonds between them and both are described by their features. In the case of MiDi each molecule buld from $n$ atoms is represented as the graph $G = (\mathbf{X},\mathbf{C},\mathbf{R},\mathbf{Y})$ with $\mathbf{X}$ as the atom type and $\mathbf{C}$ as their formal charge, which are both one hot encoded vector of length $n$. $\mathbf(R)\in\mathbb{R}^{3 \times n}$ are the three dimensional coordiates all $n$ atoms and $\mathbf(Y)$ are the bond types between them, saved as a $n \times n$ matrix. Notably, having no bond type is also considerd to be a bond type. Looking at figure ... it becomes obvious why this representation fits a molecule well and intuaticly.

Equivariance
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

This proess results in the a complete noised picture like in figure .. .

<figure>
  <img src="/images/Diffusion.png" alt="Diffusion" style="display:block; margin:auto; width:80%"/>
  <figcaption style="text-align:center">Figure 1: Noise Process on cat.</figcaption>
</figure>

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

The parameter α defines how much of the input remains in a single noise step and is changed during training according to the newly introduced adaptive noise schedule. Even though MiDi must balance all features of the graph to generate a stable molecule, atom coordinates and bond types have less flexibility in this regard and cannot be derived from atom type and formal charge. This is why during the noise application α for coordinates and bond type decreases slower than for atom type and formal charge.

<figure>
  <img src="/images/adaptive_cosine_schedule-1.png" alt="Adaptive cosine noise schedule" style="display:block; margin:auto; width:80%"/>
  <figcaption style="text-align:center">Figure 1: Adaptive cosine noise schedule.</figcaption>
</figure>

In the next chapter we will discuss what this means for the inference step, after exploring the architecture of the denoising model.

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

The complete update process extracts the 3D information from the current graph and then uses the $\Delta_r$, node, and global features to update the edge embeddings. Using more transformer architecture MiDi updates the node embeddings with the previously calculated attention coefficient. the flattend attention heads are then used to calculate the rGNN update and the final graph embedding. To do this MiDi uses PNA layers that use pooling to resize the embedding.

After the updates have been concluded MiDi uses dropout for better Generalization and then normalizes all exept the graph features. Similar to MLP layer the normalization layer for the 3D coordinates is again modified to hold the E(3) equivariance.

Finally, after repeating this proccess multiple times the MiDi uses a final MLP layer and returns its output values. 3D coordinates are returend as pointwise estimates while the atom type, formal charge and bond type are predicted distributions over all classes. The corresponding loss is therefore also a combination of regression loss for coordinations and cross-entropy loss for all other.

$$
\mathcal{L}(G,\hat{p}^{G}) =
\lambda_r \left\|\hat{R}-R\right\|^2
+ \lambda_x\,\mathrm{CE}\!\left(X,p_{\theta}^{X}\right)
+ \lambda_c\,\mathrm{CE}\!\left(C,p_{\theta}^{C}\right)
+ \lambda_y\,\mathrm{CE}\!\left(Y,p_{\theta}^{Y}\right)
$$


How was MiDi tested?
======

After introducing the model, the most important question is whether generating the molecular graph and the 3D conformation together actually helps. MiDi is evaluated on unconditional molecule generation: the model is not asked to optimize a specific property or binding pocket, but simply to sample realistic molecules from the learned distribution.

This setting is useful because it isolates the core ability of the model. A good molecule generator should not only place atoms in plausible 3D positions, but also assign chemically meaningful bonds, atom types, and formal charges. In previous 3D diffusion models, the graph is usually reconstructed after generation, for example by looking at interatomic distances or by using chemistry software such as OpenBabel. MiDi instead predicts the graph and the conformation during the same denoising process.

Experimental setup
------

The authors compare MiDi against the Equivariant Diffusion Model (EDM) followed by a separate bond prediction step. EDM is also a equivariant diffusion model for 3D molecule generation. Since EDM only generates atom types and 3D coordinates, bonds must be added afterwards. The paper considers two variants: EDM with a simple distance-based lookup table, and EDM followed by OpenBabel, which tries to optimize bond orders to make the molecule chemically valid.

The experiments are performed with explicit hydrogen atoms, which makes the task harder because the model has to place and connect all atoms, not only the heavy atoms. The authors evaluate on two datasets, depicted in the following table:

| Information | QM9 | GEOM-DRUGS |
|---|---|---|
| **Dataset type** | Standard benchmark of small molecules | Large dataset of drug-sized, drug-like molecules |
| **Number of molecules** | **133,000** molecules | **430,000** molecules |
| **Molecule size** | Up to **9 heavy atoms** per molecule | Average of **44 atoms**, with up to **181 atoms** |

QM9 has a lot less samples and contains small molecules with up to 9 heavy atoms, which are all atoms except hydrogen. The evaluation of for MiDi was done with full molecules, so with the additional heavy atoms. The GEOM-DRUGS data set on the other side is much more challenging and closer to realistic drug molecules. It contains around 430,000 drug-sized molecules, with an average of 44 atoms and up to 181 atoms. The evaluation combines metrics for the molecule structue and general dsitribution of the generated molecules compared to the training data. The following table explains a bit more in detail what each metric means, first the six molecule metricies and then the five generated distributions.

| Metric | Description |
|---|---|
| **Molecule stability** | The molecule resitis changing into other substances and does not break apart on its own. |
| **Atom stability** | The atoms nucleus and electron arrangement are balanced and will not spontaneously change or break apart. |
| **Validity** | Success rate of the RDKit sanitization pipeline over $10{,}000$ molecules. This definesif the molecule is chemically valid. |
| **Uniqueness** | Proportion of valid generates molecules with different canonical SMILES. SMILES is a way to write a molecule in a simple line of text and therefore shows the diversity of the generated molecules. This inludes molecules that are already presetn in the training data. |
| **Novelty** | Similar to Uniqueness with the only diffrence, that the SMILES are not allowed to appear in the training data and therefore shows the models ability to generate new molecules. |
| **Connected** | All training molecules have a single connected compound, meaning every atom can be reached from any other atom if you follow the graph structure through its edges. |
| **Valency\[$e^{-2}$\]** | Valency is the capacity of an atom to bin and measured as the number of electrons it loses, gains or shares when bonding. |
| **Atom\[$e^{-2}$\]** | Atom types that occured in the generated molecules. |
| **Bond\[$e^{-2}$\]** | Bond types that occured in the generated molecules. |
| **Angles\[$°$\]** | Angles between to bonds. |
| **Bond Lengths\[$e^{-2}\text{Å}$\]** | Length of each occuring bond type. |

For general molecule generation properties, the paper reports standard metrics such as molecule stability, atom stability, validity, uniqueness, novelty, and connectedness. The reported results schow the proportion of generated molecules that satisfy the above mentioned crteria. The matricies valency, atom, and bond focus more on comparing the distribution of the training dataset with the distribution of the generated dataset using the Wasserstein Distance. This is a famous measure to define the similarity of two distributions. The results show the distance between the marginal disribution of the training and the generated dataset.

Finally, MiDi is also evaluated as a 3D generator. The authors compare bond length and bond angle distributions between generated molecules and real molecules. This is important because a model could generate a valid graph but still place atoms in unrealistic 3D arrangements. Similar to the previous paragraph, these metricies are also compared with the Wasserstein distance.

Does MiDi work?
======
Shortly, Yes! We now know how MiDi works and establised a foundation to test its performance. Lets first have a look on the smaller, simpler dataset. The following table give a good picture to compare the diffrent results. Remember, that EDM does only generate 3D structures with atoms, but no bond types. We will focus on the EMD model without additional software, EDM with help from OpenBabel to generate fitting bonds and the MiDi model with adaptive noise schedule. These results always show the comparison between the test and the generated set. The data part of the results compare the test with the training set to show the generalization ability of MiDi.

So far not a real improvement
------

<figure>
  <img src="/images/qm9_molecule.png" alt="qm9_molecule" style="display:block; margin:auto; width:80%"/>
  <figcaption style="text-align:center">Figure 2: Results for molecule metrics on QM9.</figcaption>
</figure>

On the smaller QM9 dataset, the comparison EDM networks already perform quite well with and without their assisitve software. This is not too surprising: QM9 molecules are small and their bonds can often be recovered from atom distances without much ambiguity. Still, MiDi improves over the base EDM model on molecule stability, atom stability, Validity, and connectedness, but just by a small margin. With the adaptive noise schedule, MiDi reaches $97.5%$ molecular stability and $97.9%$ validity, while EDM reaches $90.7%$ molecular stability and $91.7%$ validity. Openbabal improves EDMs performance even more which leads to a even smaller margin. 

<figure>
  <img src="/images/distribution_qm9.png" alt="distribution_qm9" style="display:block; margin:auto; width:80%"/>
  <figcaption style="text-align:center">Figure 1: Wasserstein distance on QM9.</figcaption>
</figure>

Looking at the Wasserstein disances of the marginal distributions we can see a similar pattern. just remember that we want a small distance between the distributions. Valency, Atom, and Bond diversity have already improved when using MiDi but this changes when we look at the bond angles and length. Apparently when generating small molecules EDM is generates more realisitc 3D information, which is not strange as EDMs main task is to generate the coordinates and not the complete 2D and 3D structure of a molecule. If we would stop at this point we could already say that MiDi is able to compete with their state of the art models, even though it is not always able to surpase them, even less if they are combined with additional software. But this is already great information, because MiDi is able to generate both 2D and 3D stable structures while beeing end-to-end differentiable.

Lets make it more difficult
------
Now that we have seen the results on the smal dataset, I am sure you are as excited as I am to look at the realistic GEOM-Drugs dataset.

<figure>
  <img src="/images/geom_molecule.png" alt="geom_molecule" style="display:block; margin:auto; width:80%"/>
  <figcaption style="text-align:center">Figure 1: Results for molecule metrics on GEOM-Drugs.</figcaption>
</figure>

This dataset contains larger, drug-like molecules with many more atoms and more complicated structures. Here, the limitations of the two-step approach become obvious. EDM can generate reasonable-looking 3D point clouds, but when bonds are inferred afterwards, only $5.5%$ of the resulting molecules are molecularly stable. Adding OpenBabel helps, increasing molecular stability to $40.3%$, but this still leaves most generated molecules chemically problematic. MiDi performs much better on this harder benchmark. The adaptive version generates $91.6%$ molecularly stable molecules, while also keeping atom stability close to the data distribution at $99.8%$. This is the main result of the paper: jointly denoising the graph and the coordinates allows the model to learn chemistry and geometry as a coupled object, rather than treating bond prediction as an afterthought. In the end MiDi is able to outperfom EDM with and without OpenBabel at every molecule metric except Validity, where MiDi only reaches $77.8%$.

<figure>
  <img src="/images/distribution_qm9.png" alt="distribution_qm9" style="display:block; margin:auto; width:80%"/>
  <figcaption style="text-align:center">Figure 1: Wasserstein distance on QM9.</figcaption>
</figure>

The 3D metrics show the same trend. On GEOM-DRUGS, MiDi produces much more realistic bond angles than EDM or EDM followed by OpenBabel. The bond-angle Wasserstein distance drops from around $6°$ for the baselines to $1.07°$ for adaptive MiDi. The valency distribution is also much closer to the test set, which means the generated atoms satisfy atom bonding rules more consistently.

Overall, the results suggest that MiDi’s advantage does not come from simply adding more machinery to a diffusion model. Its strength comes from matching the generative process to the object being generated. A molecule is not just a point cloud and not just a graph. It is both at the same time. MiDi benefits from treating these two views as one coupled representation throughout the denoising process.

Conclusion
======

MiDi makes a compelling case for a simple but important principle: the representation should match the object. A molecule is simultaneously a chemical graph and a 3D structure, and treating these two views as inseparable throughout the generative process leads to measurably better results — especially when the molecules get large and chemically complex.

We started from first principles: how to represent a molecule as a graph $G = (\mathbf{X}, \mathbf{C}, \mathbf{R}, \mathbf{Y})$, why equivariance matters when working in 3D space, and how diffusion models gradually corrupt and then reconstruct data. Building on that foundation, we saw how MiDi applies separate but coupled noise processes to continuous coordinates and discrete features, and how the rEGNN-based transformer denoising model jointly predicts all of them in a single forward pass.

The experiments confirm that this joint approach pays off. On the small QM9 benchmark, MiDi competes with strong baselines. On the much harder GEOM-DRUGS dataset — closer to real drug discovery — it decisively outperforms the two-step pipeline, raising molecular stability from $5.5\%$ (EDM alone) and $40.3\%$ (EDM + OpenBabel) to $91.6\%$, while also producing more realistic bond angles and valency distributions.

What makes MiDi particularly appealing from an engineering perspective is that the entire pipeline remains end-to-end differentiable. There is no chemistry post-processing step that could silently corrupt a promising generated structure. The model learns what makes a valid molecule directly from data, rather than relying on hand-crafted rules.

Of course, MiDi is not the final word. It still struggles with validity on GEOM-DRUGS compared to OpenBabel-assisted baselines, and generating very large molecules remains a challenge for diffusion models in general. Future work will likely push further by conditioning generation on binding pockets, improving sampling efficiency, or scaling to even larger chemical spaces. But as a demonstration that geometry and chemistry can — and should — be learned together, MiDi sets a strong precedent.