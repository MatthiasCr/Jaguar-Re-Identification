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

||Experiment|Type||||Experiment Points|
|--|:--|:--|--|--|--|--:|
|01|Backbone Comparison|Leaderboard|[Documentation](LEADERBOARD_EXPERIMENTS.md#experiment-1---backbone-comparison)|[Notebook](notebooks/01_backbones.ipynb)|[W&B Run Group](https://wandb.ai/juggling-jaguars/jaguar-reid-jugglingjaguars/groups/Experiment-1-Backbones)|2|
|02|Loss Function Comparison|Leaderboard|[Documentation](LEADERBOARD_EXPERIMENTS.md#experiment-2---loss-function-comparison)|[Notebook](notebooks/02_loss_functions.ipynb)|[W&B Run Group](https://wandb.ai/juggling-jaguars/jaguar-reid-jugglingjaguars/groups/Experiment-2-LossFunctions)|2|
|03|Handling Data Imbalance with weighted Sampling|EDA|[Documentation](EDA_EXPERIMENTS.md#experiment-3---weighted-sampling)|[Notebook](notebooks/03_weighted_sampling.ipynb)|[W&B Run Group](https://wandb.ai/juggling-jaguars/jaguar-reid-jugglingjaguars/groups/Experiment-3-WeightedSampling)|1|
|04|Backbone Freezing vs Fine Tuning|Leaderboard|[Documentation](LEADERBOARD_EXPERIMENTS.md#experiment-4---backbone-fine-tuning)|[Notebook](notebooks/04_backbone_finetuning.ipynb)|[W&B Run Group](https://wandb.ai/juggling-jaguars/jaguar-reid-jugglingjaguars/groups/Experiment-4-BackboneFinetuning)|1|
|05|Hyperparameter Sweep|Leaderboard|[Documentation](LEADERBOARD_EXPERIMENTS.md#experiment-5---hyperparameter-search)|[Notebook](notebooks/05_hyperparamter_search.ipynb)|[W&B Run Group](https://wandb.ai/juggling-jaguars/jaguar-reid-jugglingjaguars/groups/Experiment-5-HyperparameterSearch)|1|
|06|K-Reciprocal Re-Ranking|Leaderboard|[Documentation](LEADERBOARD_EXPERIMENTS.md#experiment-6---k-reciprocal-re-ranking)|[Notebook](notebooks/06_k_reciprocal_re_ranking.ipynb)|[W&B Project](https://wandb.ai/juggling-jaguars/jaguar-reid-jugglingjaguars/groups/Experiment-6-KReciprocalReRanking)|1|
|07|GeM Pooling|Leaderboard|[Documentation](LEADERBOARD_EXPERIMENTS.md#experiment-7---gem-pooling)|[Notebook](notebooks/07_gem_pooling.ipynb)|[W&B Run Group](https://wandb.ai/juggling-jaguars/jaguar-reid-jugglingjaguars/groups/Experiment-7-GeMPooling)|1|
|08|Test-Time Agumentation|Leaderboard|[Documentation](LEADERBOARD_EXPERIMENTS.md#experiment-8---test-time-augmentation)|[Notebook](notebooks/08_tta_comparison.ipynb)|[W&B Run Group](https://wandb.ai/juggling-jaguars/jaguar-reid-jugglingjaguars/groups/Experiment-8-TestTimeAugmentation)|1|
|09|Training Stability across different random seeds|EDA|[Documentation](EDA_EXPERIMENTS.md#experiment-9---random-seed-comparison)|[Notebook](notebooks/09_seed_comparison.ipynb)|[W&B Run Group](https://wandb.ai/juggling-jaguars/jaguar-reid-jugglingjaguars/groups/Experiment-9-RandomSeeds)|1|
|10|With Background vs. Without Background|EDA|[Documentation](LEADERBOARD_EXPERIMENTS.md#experiment-10---background-comparison)|[Notebook](notebooks/10_background.ipynb)|[W&B Run Group](https://wandb.ai/juggling-jaguars/jaguar-reid-jugglingjaguars/groups/Experiment-10-Background/)|1|
|11|Interpretability Visualization with Integrated Gradients|EDA|[Documentation](EDA_EXPERIMENTS.md#experiment-11---interpretability-with-integrated-gradients)|[Notebook](notebooks/11_interpretability.ipynb)|[W&B Run Group](https://wandb.ai/juggling-jaguars/jaguar-reid-jugglingjaguars/groups/Experiment-11-Interpretability)|1|
|**Total**||||||**13**|
