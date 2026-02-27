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

### EDA

- Interpretability to visualize which regions the model uses
- Comparison of Optimizers and Schedulers


### Leaderboard

- Backbone Comparison (5 Backbones = 2 credits)
- Loss Function Comparison (2 Losses = 1 credit, 4 Losses = 2 credits)
- k-reciprocal re-ranking
- Extensive Hyperparameter Sweep
- Same Config on 10 different Seeds
- Appy same Model to both Competition Rounds (with/without background)