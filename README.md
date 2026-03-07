## Get Data

```
$ kaggle competitions download -c jaguar-re-id
```

## Cluster Access

```
$ ssh sci
```

start gpu job

```
$ salloc --partition=gpu-interactive --account=sci-demelo-computer-vision --cpus-per-task=8 --mem=32G --time=08:00:00 --no-shell --job-name=vs-code --gpus=1 --constraint="GPU_SKU:A100"
```

start cpu job

```
$ salloc --partition=cpu-interactive --account=sci-demelo-computer-vision --cpus-per-task=8 --mem=32G --time=01:00:00 --no-shell --job-name=vs-code
```

## Experiments

All experiments were run on the HPI SCI cluster using an NVIDIA A100 40GB GPU.

||Experiment|Type||||
|--|:--|:--|--|--|--|
|01|Backbone Comparison|Leaderboard|[Documentation](LEADERBOARD_EXPERIMENTS.md#experiment-1---backbone-comparison)|[Notebook](notebooks/01_backbones.ipynb)|[W&B Run Group](https://wandb.ai/juggling-jaguars/jaguar-reid-jugglingjaguars/groups/Experiment-1-Backbones)|
|02|Loss Function Comparison|Leaderboard|[Documentation](LEADERBOARD_EXPERIMENTS.md#experiment-2---loss-function-comparison)|[Notebook](notebooks/02_loss_functions.ipynb)|[W&B Run Group](https://wandb.ai/juggling-jaguars/jaguar-reid-jugglingjaguars/groups/Experiment-2-LossFunctions)|
|03|Handling Data Imbalance with weighted Sampling|EDA|[Documentation](EDA_EXPERIMENTS.md#experiment-3---weighted-sampling)|[Notebook](notebooks/03_weighted_sampling.ipynb)|[W&B Run Group](https://wandb.ai/juggling-jaguars/jaguar-reid-jugglingjaguars/groups/Experiment-3-WeightedSampling)|
|04|K-Reciprocal Re-Ranking|Leaderboard||||
|05|Optimizer / Scheduler Comparison|EDA||||
|06|Training Stability across different random seeds|EDA||||
|07|Background vs. no Background|EDA||||
|08|Interpretability Visualization|EDA||||
|09|Hyperparameter Sweep|Leaderboard||||
|10|||||||
|||||||
