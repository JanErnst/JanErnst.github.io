---
title: 'MiDi: Mixed Graph and 3D Denoising Diffusion for Molecule Generation'
date: 2026-07-31
permalink: /posts/2026/07/midi-molecule-generation/
tags:
  - Diffiusion
  - category1
  - category2
---


Introduction
======

Designing new molecules is a central challenge in modern drug discovery. A useful generative model should not only propose atoms in 3D space, but also understand how those atoms are connected by chemical bonds. Both views matter: the molecular graph tells us which atoms are bonded and what functional groups are present, while the 3D conformation determines how the molecule can interact with proteins and other biological targets. 

Many previous approaches treat these two aspects separately. Some models generate only the 2D molecular graph, ignoring the spatial arrangement of atoms. Others generate a 3D point cloud first and then recover the bond structure afterward using hand-crafted rules or external chemistry software. This separation is limiting: the resulting pipeline is no longer fully differentiable, and errors in the post-processing step can easily turn a promising 3D structure into an invalid molecule. 

The paper MiDi: Mixed Graph and 3D Denoising Diffusion for Molecule Generation addresses this problem with a single diffusion model that generates both parts of a molecule together. MiDi represents a molecule as a graph embedded in 3D space, with atom types, formal charges, bond types, and coordinates all denoised jointly. The key idea is simple but powerful: instead of asking the model to learn geometry first and chemistry later, train it to learn the relationship between geometry and chemistry throughout the entire generation process. 

In this blog post, we will build up the motivation behind MiDi, explain why molecule generation needs both graph and 3D information, and look at how the model combines discrete diffusion for chemical structure with Gaussian diffusion for spatial coordinates. Finally, we will discuss why this joint approach leads to much more stable molecules on challenging drug-like datasets. 

 
Motivation
====== 

Background knowledge 
======

To fully understand the process and architecture we wil now venture into the basic background knowledge needed to build MiDi. 

Graph Molecule Generation: 

Graph structure 

Previous works on the generation of molecules have already proven the efficency of graph structures. MiDi is no diffrence in that regard and does also represent its molecuels as a graph. A graph consists of nodes and edges between the nodes. Both are described by their features. In the case of MiDi each molecule is a graph G with 0 


Equivariance:
------

Diffusion:
------ 

As the paper title states MiDi is a denoising diffusion model, which does resamble the structure of a decoder encoder model, where only the decoder is used for inference. However, it is still crucial to understand the complete pipeline to to get behind the how MiDi does what it does. The first part of a Denoising Diffusion model is Diffusion, which is refrenced as the noise model, and the second part is the Denoising model that generates the mixed graph molecule. 

Architecture
======
Noise model:

The sole purpose of a noise model is to incrementally apply noise to  the input x so that each step is a more currupted version of the previous step. This results in a markovian trajectory in the form of: 

[Equation] 

As already explained in previous sections, MiDi generates a graph that is described by their edges and node features. The 3D coodrinates are continuous values and the remaining features are discrete values. The corresponding noise is therefore also either continuous gaussian or discrete noise and results in the following complete noise at each step: 

“Insert complete noise formular” 

Atom type X, bond type Y and formal charge C get discrete noise that is defined by the transition matrix Q that has been derived from the marginal distribution of the training sets for each feature. To ensure that the generated molecule stays centered in the subspace the total amount of applied gaussian noise has to always add up to 0. This ensures that the input and outpur are roto-translation equivariant during training and inference.  

 
The parameter α defines how much of the input remains in a single noise step and is changed during training according to the newly introduced adaptive noise schedule. Even tough MiDi must balance all features of the graph to generate a stable molecule, atom coordinates and bond types have less flexibility in that regard and cannot be derived from atom type and formal charge. This is why during the noise application α for coordinates and bond type decreases slower than for atom type and formal charge. In the next chapter we will discuss what this means for the inference step, after exploring the architecture of the denoising model. 

Denoising model: 

The Denoising model is the main component of MiDi that gets trained and generates the mixed molecule graphs. In general, the second part of a diffusion model takes a noise as input and returns a clean output. It does that by learning to reverse the noise that the noise model added to the original training data input and then reversing it over the noisy input graph. To do so the MiDi combines three architectures and a mixed loss that we will now explore further. 

Firstly, we are going to explore the architecture. On the first level the denoising model of MiDi is a multi layer transformer graph neural network enclosed by MLPs. As input it takes the noise graph features and graph-level features. These are not part of the output but still help the network to remember and process graph-level information througout the generation process. 

    Update block 

    REGNN 

    Equivariance in the transformer architecture 

    loss 

Experiments:
======

After introducing the model, the most important question is whether generating the molecular graph and the 3D conformation together actually helps. MiDi is evaluated on unconditional molecule generation: the model is not asked to optimize a specific property or binding pocket, but simply to sample realistic molecules from the learned distribution. 

This setting is useful because it isolates the core ability of the model. A good molecule generator should not only place atoms in plausible 3D positions, but also assign chemically meaningful bonds, atom types, and formal charges. In previous 3D diffusion models, the graph is usually reconstructed after generation, for example by looking at interatomic distances or by using chemistry software such as OpenBabel. MiDi instead predicts the graph and the conformation during the same denoising process. 

Experimental setup
------

The authors compare MiDi against 3D molecule generators followed by a separate bond prediction step. The main baseline is EDM, a previous equivariant diffusion model for 3D molecule generation. Since EDM only generates atom types and coordinates, bonds must be added afterwards. The paper considers two variants: EDM with a simple distance-based lookup table, and EDM followed by OpenBabel, which tries to optimize bond orders to make the molecule chemically valid. 

The experiments are performed with explicit hydrogen atoms, which makes the task harder because the model has to place and connect all atoms, not only the heavy atoms. The authors evaluate on two datasets: 

QM9 is the simpler benchmark. It contains small molecules with up to 9 heavy atoms. The dataset is split into 100k training molecules, 20k validation molecules, and 13k test molecules. 

GEOM-DRUGS is much more challenging and closer to realistic drug discovery. It contains around 430,000 drug-sized molecules, with an average of 44 atoms and up to 181 atoms. The authors use an 80/10/10 train-validation-test split and extract the five lowest-energy conformations for each molecule. 

What is measured? 

The evaluation combines graph-based and geometry-based metrics. 

For the graph structure, the paper reports standard molecule generation metrics such as validity, uniqueness, novelty, and connectedness. Validity is measured using RDKit sanitization. The authors also report atom stability and molecule stability, which check whether valency constraints are satisfied without adding implicit hydrogens. 

For the distribution of generated molecules, they compare atom types, bond types, and valencies between generated samples and the test set. Smaller distances mean that the generated molecules look more like the data distribution. 

Finally, MiDi is also evaluated as a 3D generator. The authors compare bond length and bond angle distributions between generated molecules and real molecules. This is important because a model could generate a valid graph but still place atoms in unrealistic 3D arrangements. 

 

Results
------

The central question of the experiments is: does it actually help to generate the molecular graph and the 3D conformation together? 

The answer is yes, especially once the molecules become large and chemically complex. 

On the smaller QM9 dataset, most methods already perform quite well. This is not too surprising: QM9 molecules are small, and their bonds can often be recovered from atom distances without much ambiguity. Even here, MiDi improves over the base EDM model on graph-based metrics. With the adaptive noise schedule, MiDi reaches 97.5% molecular stability and 97.9% validity, while EDM reaches 90.7% molecular stability and 91.7% validity. OpenBabel remains very strong on this dataset, because the molecules are simple enough for a rule-based bond-reconstruction step to work reliably. 

The more interesting test is GEOM-DRUGS. This dataset contains larger, drug-like molecules with many more atoms and more complicated structures. Here, the limitations of the two-step approach become obvious. EDM can generate reasonable-looking 3D point clouds, but when bonds are inferred afterwards, only 5.5% of the resulting molecules are molecularly stable. Adding OpenBabel helps, increasing molecular stability to 40.3%, but this still leaves most generated molecules chemically problematic. 

MiDi performs much better on this harder benchmark. The adaptive version generates 91.6% molecularly stable molecules, while also keeping atom stability close to the data distribution at 99.8%. This is the main result of the paper: jointly denoising the graph and the coordinates allows the model to learn chemistry and geometry as a coupled object, rather than treating bond prediction as an afterthought. 

The 3D metrics show the same trend. On GEOM-DRUGS, MiDi produces much more realistic bond angles than EDM or EDM followed by OpenBabel. The bond-angle Wasserstein distance drops from around 6 degrees for the baselines to 1.07 degrees for adaptive MiDi. The valency distribution is also much closer to the test set, which means the generated atoms satisfy chemical bonding rules more consistently. 

The adaptive noise schedule is another important part of the result. Compared to the uniform schedule, it improves molecular stability on GEOM-DRUGS from 89.9% to 91.6%, and it also improves the valency, bond-angle, and bond-length metrics. Intuitively, this supports the authors’ design choice: during sampling, it is useful to first form a rough 3D structure and bond layout, and only later refine atom types and formal charges. 

Overall, the results suggest that MiDi’s advantage does not come from simply adding more machinery to a diffusion model. Its strength comes from matching the generative process to the object being generated. A molecule is not just a point cloud and not just a graph. It is both at the same time. MiDi benefits from treating these two views as one coupled representation throughout the denoising process. 