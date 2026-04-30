# The illusion of interpretability in biologically informed neural networks

This repository contains the code, results, and figures for the paper:

> **The illusion of interpretability in biologically informed neural networks**

The project investigates whether embedding biological structure (e.g., pathways) into neural networks truly leads to meaningful interpretability, or whether this assumption can be misleading.

---

## Graphical Abstract

![Graphical Abstract](Figures/Graphical_Abstract.png)

---

##  Repository Structure

```text
.
├── Code/
│   └── pathway_tasks_complete.py
├── Figures/
│   ├── graphical_abstract.png
│   └── (all figures used in the paper)
├── Results_Parameter_Level/
│   ├── (metrics for all tasks)
│   └── (scripts to reproduce paper figures)
├── Results_Activation_Level/
│   ├── (activation-level interpretability results)
│   └── (scripts to reproduce paper figures)
```

###  Code

The `Code/` folder contains the main script:

* `pathway_tasks_complete.py`
  → Runs all experiments across all tasks and configurations.

###  Figures

Contains all figures used in the paper, including the graphical abstract.

###  Results_Parameter_Level

* Interpretability analysis at the **parameter level**
* Includes:

  * Metrics across all tasks
  * Code to reproduce figures in the paper

###  Results_Activation_Level

* Interpretability analysis at the **activation node level**
* Includes:

  * Metrics across all tasks
  * Code to reproduce figures in the paper

---

##  What the Code Does

The main script implements a **Teacher–Student framework** using biologically informed neural networks.

It supports four main tasks:

* ✅ Binary classification
* ✅ Multiclass classification
* ✅ Regression
* ✅ Survival analysis

The default configuration uses a biologically inspired structure:

* Number of genes (`G`)
* Number of pathways (`P`)
* Pathway sizes
* Overlap between pathways

These parameters can be modified via command line to reproduce all experiments in the paper, including:

* Variations in network depth
* Biological configuration sweeps:

  * Number of genes (`G`)
  * Number of pathways (`P`)
  * Pathway size range
  * Pathway overlap

---

## ⚙️ Installation

You can install the environment using either **Conda** or **pip**.

---

###  Option 1 — Conda (Recommended)

#### CPU version

```bash
conda create -n pathway-ts python=3.10
conda activate pathway-ts

conda install numpy pandas pytorch cpuonly -c pytorch -c conda-forge
```

#### GPU version (NVIDIA)

```bash
conda create -n pathway-ts python=3.10
conda activate pathway-ts

conda install pytorch pytorch-cuda=12.1 numpy pandas -c pytorch -c nvidia -c conda-forge
```

> Adjust CUDA version according to your system.

---

###  Option 2 — pip

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
# venv\Scripts\activate    # Windows

pip install numpy pandas
```

#### Install PyTorch

 CPU:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

 GPU:
Go to https://pytorch.org/get-started/locally/ and select your configuration.

---

##  Usage

Run all tasks:

```bash
python Code/pathway_tasks_complete.py --task all
```

Run a specific task:

```bash
python Code/pathway_tasks_complete.py --task survival
python Code/pathway_tasks_complete.py --task regression
python Code/pathway_tasks_complete.py --task binary
python Code/pathway_tasks_complete.py --task multiclass
```

---

##  Key Arguments

You can customize both biological structure and model parameters:

```bash
--G 400              # Number of genes
--P 60               # Number of pathways
--size_min 14        # Minimum pathway size
--size_max 22        # Maximum pathway size
--overlap 0.55       # Overlap between pathways

--hidden 64          # Hidden layer size
--epochs 120
--batch 512
```

Example:

```bash
python Code/pathway_tasks_complete.py \
  --task survival \
  --G 800 \
  --P 100 \
  --overlap 0.7
```

---

##  Outputs

The script generates `.csv` files containing:

### Performance metrics

* Accuracy (classification)
* MSE / R² (regression)
* Concordance index (survival)

### Knowledge distillation metrics

* KL divergence
* MSE between teacher and student outputs

### Interpretability metrics

* Weight recovery (parameter-level)
* Activation alignment (node-level)

Results are saved in:

```text
results_new/
```

---

##  Reproducing Paper Results

To fully reproduce the experiments:

1. Run the main script with different configurations
2. Use the scripts provided in:

   * `Results_Parameter_Level/`
   * `Results_Activation_Level/`

These folders include:

* Precomputed results
* Figure generation scripts used in the paper

---

##  Key Idea

This work challenges a common assumption:

> Embedding biological structure into neural networks does **not necessarily guarantee interpretability**

We show that:

* Models can match predictions very closely (high agreement)
* While failing to recover the underlying biological structure
* At both:

  * Parameter level
  * Activation level


---
