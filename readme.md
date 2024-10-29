# UTango: Untangling Commits with Context-aware, Graph-based, Code Change Clustering Learning Model

<p aligh="center"> This repository contains the code for <b>UTango: Untangling Commits with Context-aware, Graph-based, Code Change Clustering Learning Model</b>.</p>

# Source Code: https://github.com/Commit-Untangling/commit-untangling#Instruction_to_Run_UTango
# Demo: https://github.com/Commit-Untangling/commit-untangling#Demo


## Contents
1. [Introduction](#Introduction)
2. [Dataset](#Dataset)
3. [Requirement](#Requirement)
4. [Instruction_to_Run_UTango](#Instruction_to_Run_UTango)
5. [Demo](#Demo)

## Introduction

During software evolution, developers make several changes and
commit them into the repositories. Unfortunately, many of them
tangle different purposes, both hampering program comprehension
and reducing separation of concerns. Automated approaches with
deterministic solutions have been proposed to untangle commits.
In this work, we present UTango, a machine learning (ML)-
based approach that learns to untangle the changes in a commit.We
develop a novel code change clustering learning model that learns to
cluster the code changes, represented by the embeddings, into different
groups with different concerns. We adapt the agglomerative
clustering algorithm into a supervised-learning clustering model
operating on the learned code change embeddings via trainable parameters
and a loss function in comparing the predicted clusters and
the correct ones during training. To facilitate our clustering learning
model, we develop a context-aware, graph-based, code change
representation learning model, leveraging Label, Graph-based Convolution
Network to produce the contextualized embeddings for code
changes, that integrates program dependencies and the surrounding
contexts of the changes. The contexts and cloned code are also
explicitly represented, helping UTango distinguish their concerns.
Our empirical evaluation on a C# dataset with 1,612 tangled commits
shows that it achieves the accuracy of 28.6%–462.5%, relatively
higher than the state-of-the-art approaches in clustering the
changed code. We evaluated UTango in a Java dataset with +14k
tangled commits. The result shows that it achieves 13.3%–100.0%
relatively higher accuracy than the state-of-the-art approaches.

## Dataset

### Preprocessed Dataset

We published our processed dataset at https://drive.google.com/file/d/1WdP9SI4T8H_8Tb-zAKIat_o70sQuUbTl/view?usp=sharing

Please create a folder named ```data``` under the root folder of UTango, download the dataset, unzip it and put all files in ```./data``` folder.

### Use your own dataset

If you want to use your own dataset, please prepare the data as follow:

1. The data are stored in ```data_n.pt``` and ordered based on the tangled commit last modified time.

2. Each ```data_n.pt``` include a set of ```Data``` object from ```torch_geometric``` with one extra element for the ground truth labels. Each ```Data``` represent a method in a tangled commit:
	
	1> ```Data.x = Node_feature_vector```
	

	2> ```Data.y = Context_info```

	
	3> ```Data.edge_index = edge_list```
	
Where ```Node_feature_vector``` is ```N*R``` sized torch tensors that represent the node features in the graph, ```N``` is the number of nodes on the graph and ```R``` is the representation vector length.

```edge_list``` is  the matrixs to represent the graph edges for each method. Please refer to ```torch_geometric``` package for more details.

```Context_info``` is an array that has the same length as the node numbers on the graph. Each element inside represent one node. If it is not a code change node, the value for that position is 0. If it is the code change node, the element is an array that include all index for the context nodes.

## Installation

Use Python 3.9.20. You can use [pyenv](https://github.com/pyenv/pyenv) to run different Python versions at the same time.

Install `torch` first with:
```bash
pip install torch==1.13.1
```

Then, run following:

```shell
pip install -r requirements.txt
```


## Train own model

Put your data into `data` folder. Then run following:

```bash
python -m UTango train <model_path> <dataset_folder> <num_data_files>
```

## Run demo

```bash
python -m UTango demo
```
