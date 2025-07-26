# ASkDAgger: Active Skill-level Data Aggregation for Interactive Imitation Learning

This repository contains the code for the CLIPort experiments from the paper *ASkDAgger: Active Skill-level Data Aggregation for Interactive Imitation Learning*.

## Overview of ASkDAgger

<img src=figures/askdagger_overview.png width="75%">

**Figure 1**: The Active Skill-level Data Aggregation (ASkDAgger) framework consists of three main components: S-Aware Gating (SAG), Foresight Interactive Experience Replay (FIER), and Prioritized Interactive Experience Replay (PIER).
In this interactive imitation learning framework, we allow the novice to say: "*I plan to do this, but I am uncertain.*"
The uncertainty gating threshold is set by SAG to track a user-specified metric: sensitivity, specificity, or minimum system success rate.
Teacher feedback is obtained with FIER, enabling demonstrations through validation, relabeling, or teacher demonstrations.
Lastly, PIER prioritizes replay based on novice success, uncertainty, and demonstration age.

## Installation Instructions

### Prerequisites: install `uv`

It is adviced to use uv to install the dependencies of `askdagger_cliport` package.
Please make sure `uv` is installed according to the [installation instructions](https://github.com/astral-sh/uv?tab=readme-ov-file#installation).

### Install `askdagger_cliport`

Clone `askdagger_cliport`, go to the folder and:

```bash
git clone git@github.com:askdagger/askdagger_cliport.git
cd askdagger_cliport
export ASKDAGGER_ROOT=$(pwd)
echo "export ASKDAGGER_ROOT=$(pwd)" >> ~/.bashrc
```

Create a virtual environment:

```bash
uv venv
```

Source the virtual environment:

```bash
source .venv/bin/activate
```

Install the `askdagger_cliport` package`:

```bash
uv pip install -e .
```

Download the Google objects:

```bash
./scripts/google_objects_download.sh
```


## Validate the installation

You can validate the installation by performing interactive training with ASkDAgger with the `train_interactive.py` script, which requires 8GB of GPU memory:


```bash
python src/askdagger_cliport/train_interactive.py interactive_demos=3 save_every=3 disp=True exp_folder=exps_test train_interactive.batch_size=1
```

After training with ASkDAgger, you can evaluate the policy:

```bash
python src/askdagger_cliport/eval.py interactive_demos=3 n_demos=5 disp=True exp_folder=exps_test
```

The policy will probably fail as it needs more demonstrations/training steps to converge, but the code should run without any errors.

It is also possible to run a set of tests to confirm all is working well:
```bash
pytest tests
```

## Download and plot results from paper

You can download the results from the paper as follows:

```bash
python scripts/results_download.py
```

Next, you can plot the results:

```bash
python figures/demo_types.py
python figures/domain_shift.py
python figures/training_scenario1.py
python figures/training_scenario2.py
python figures/training_scenario3.py
python figures/sensitivity.py
python figures/evaluation.py
python figures/real.py
```

The figures should appear in the `figures` directory.

## Train the models yourself

#### Reproduction of results in the paper
To reproduce the results from the paper, you will need about 40GB of GPU memory.
A set of batch scripts for SLURM jobs is available under `scripts`.
These serve as templates and should be updated based on the specifications of your own system.
If done properly, the models can be trained and evaluated as follows:

Run twice the following, for `ASKDAGGER` is `True` and `ASKDAGGER` is `False`:
```bash
sbatch --array=0-39 scripts/train_interactive.sh
```
Run twice the following, for `ASKDAGGER` is `True` and `ASKDAGGER` is `False`:
```bash
sbatch --array=0-239 scripts/eval.sh
```

Run twice the following, for `ASKDAGGER` is `True` and `ASKDAGGER` is `False`:
```bash
sbatch --array=0-239 scripts/eval_unseen.sh
```

Run thrice the following, for PIER and FIER is `True`, for PIER is `True` and FIER is `False`, and for PIER is `False` and FIER is `True` :
```bash
sbatch --array=0-9 scripts/train_interactive_domain_shift.sh
```

#### Interactive training of a single model

A single model can be trained with the following command:
```bash
python src/askdagger_cliport/train_interactive.py
```
The available arguments for training and evaluation can be found in `src/askdagger_cliport/cfg`.
The model can be evaluated with:
```bash
python /src/askdagger_cliport/eval.py
```

#### Interactive training after BC pretraining
It is also possible to first perform offline Behavioral Cloning (BC) training and then continue with interactive training.
For this example, we will do pretraining with 50 offline demos (25 train, 25 val).
For this, you will first have to create demonstrations:
```bash
python src/askdagger_cliport/demos.py mode=train n=25
python src/askdagger_cliport/demos.py mode=val n=25
```

Next, you can perform offline BC training:
```bash
python src/askdagger_cliport/train.py train.n_demos=25 train.n_val=25 train.n_steps=200 train.save_steps=[200]
```
Afterwards, you can continue training interactively using ASkDAgger:
```bash
python src/askdagger_cliport/train_interactive.py train_demos=25 train_steps=200
```
Finally, the model can be evaluated as follows:
```bash
python src/askdagger_cliport/eval.py train_demos=25
```

## Notebook and Colab

We have prepared a Jypyter Notebook for getting acquainted with the code.
This will walk you through the interactive training procedure and visualizes the novice's actions and the demonstrations.
You can open the notebook by starting Jupyter-Lab:
```bash
jupyter-lab $ASKDAGGER_ROOT
```
and then in Jupyter-Lab you can open `askdagger_cliport.ipynb` in the folder `notebooks`.

The notebook is also available for [Colab](https://colab.research.google.com/github/askdagger/askdagger_cliport/blob/colab/notebooks/askdagger_cliport.ipynb).

## Credits

This work uses code from the following open-source projects and datasets:

#### CLIPort

Original:  [https://github.com/cliport/cliport](https://github.com/cliport/cliport)  
License: [Apache 2.0](https://github.com/cliport/cliport/blob/master/LICENSE)    
Changes: The code under `src` is mainly based on the codebase of [CLIPort](https://github.com/cliport/cliport).
We created new files for interactive training, such as `interactive_agent.py`, `pier.py`, `sag.py`, `train_interactive.py`, `uncertainty_quantification.py` and `train_interactive.py`.
In `clip_lingunet_lat.py` and `resnet_lat.py` some layers are removed to reduce the GPU memory footprint.
Furthermore, we replaced the `ReLU` activations in CLIPort for `LeakyReLU`, since we experienced some problems with vanishing gradients during interactive training.
For the rest, minor changes have been made to facilitate interactive training, relabeling demonstrations and to allow for prioritization with PIER.


#### CLIPort-Batchify

Original: [https://github.com/ChenWu98/cliport-batchify](https://github.com/ChenWu98/cliport-batchify)  
License: [Apache 2.0](https://github.com/ChenWu98/cliport-batchify/blob/master/LICENSE)    
Changes: We implemented batch training for CLIPort following the changes in this repo.

#### Google Ravens (TransporterNets)
Original:  [https://github.com/google-research/ravens](https://github.com/google-research/ravens)  
License: [Apache 2.0](https://github.com/google-research/ravens/blob/master/LICENSE)    
Changes: We use the tasks as adapted for [CLIPort](https://github.com/cliport/cliport) to include unseen objects as distractor objects.
Also created a `packing-seen-shapes` and `packing-unseen-shapes` task rather than only a `packing-shapes` task.

#### OpenAI CLIP

Original: [https://github.com/openai/CLIP](https://github.com/openai/CLIP)  
License: [MIT](https://github.com/openai/CLIP/blob/main/LICENSE)  
Changes: We used CLIP as adapted for [CLIPort](https://github.com/cliport/cliport), with minor bug fixes.

#### Google Scanned Objects

Original: [Dataset](https://app.ignitionrobotics.org/GoogleResearch/fuel/collections/Google%20Scanned%20Objects)  
License: [Creative Commons BY 4.0](https://creativecommons.org/licenses/by/4.0/)  
Changes: We use the objects as adapted for [CLIPort](https://github.com/cliport/cliport) with fixed center-of-mass (COM) to be geometric-center for selected objects.

#### U-Net 

Original: [https://github.com/milesial/Pytorch-UNet/](https://github.com/milesial/Pytorch-UNet/)  
License: [GPL 3.0](https://github.com/milesial/Pytorch-UNet/)  
Changes: Used as is in [unet.py](cliport/models/core/unet.py). Note: This part of the code is GPL 3.0.  