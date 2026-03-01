## Experiment 1 - Backbone Comparison

| [Notebook](notebooks/exp_backbones.ipynb) | 
[W&B Run Group](https://wandb.ai/juggling-jaguars/jaguar-reid-jugglingjaguars/groups/Experiment-1-Backbones) | 
Kaggle Submission: best model Eva-02, Score: 0.871 | 

The backbone creates the base embeddings and is therefore one of the most important parts of the model. Different backbones can have different architectures, different input and output sizes, and different number of parameters, so the decision can have important implications on performance but also on efficiency. In this experiment we want to compare 5 different backbones to see which one results the best mAP. 

### Setup

We will compare 5 backbone models, among them traditional and modern CNNs, medium and large sized ViTs, and supervised and self-supervised pretrained models:

- **MegaDescriptor-L-384** - This is the competitions baseline
- **ResNet50** - A traditional CNN and often a reference model in literature
- **EfficientNetB3** - A more modern, very parameter-efficient CNN
- **DINOv3 ViT-Base** - Large, self-supervised ViT
- **EVA-02 Large** - Large, pretrained ViT foundation model

**Research Questions:**
Which architecture returns the best mAP? How do parameter-efficient CNNs perform in comparison to large vision transfomers?

The initial embeddings created by the backbones will get projected to 256 dimensional embeddings using two linear layers, followed by an ArcFace Loss layer. The backbone itself will not be trained. For the image preprocessing transforms we resize the images to whichever input size the backbone requires. After that we will apply a normalization transform using the mean and std that is provided by the backbone. We will not use any random augmentations.

All other hyperparameters and the training procedure will be identical for all runs. We use a batch size of 32, dropout of 0.3, AdamW as optimizer, and ReduceLROnPlateau as scheduler. We will train 100 epochs with a patience of 10 epochs, that means if for 10 consecutive epochs we can not beat the currently best validation loss we will stop training early. In the result table below we report how many epochs each model trained. The experiments were run on the HPI SCI cluster using an NVIDIA A100 40GB GPU.

### Results

In the table below we report some essential metrics of the runs. More metrics such as losses, learning rates, etc are logged in W&B.

|backbone|architecture|parameter|input size|epochs (best) |best val mAP|kaggle public score|
|--|--|--:|--|--|--|--|
|MegaDescriptor|Swin Transformer|195,198,516|384|84 (74)|0.784|0.754|
|ResNet50|CNN|23,508,032|288|91 (81)|0.817|0.728|
|EfficientNetB3|CNN|10,696,232|300|100 (90)|0.841|0.759|
|DINOv3|ViT|303,079,424|256|99 (89)|0.867|0.841|
|Eva02|ViT|304,055,232|448|84 (74)|0.862|0.871|

DINOv3 gives the strongest validation mAP in this setup, with EVA-02 close behind. On the test set in Kaggle, EVA-02 scored an even better public score of 0.871, while DINOv3 got only 0.841. We conclude that large ViT backbones are currently the best choice for this task. 
EfficientNetB3 also performs surprisingly well for its tiny size in comparison and even outperforms ResNet50 and the much larger MegaDescriptor. That makes EfficientNetB3 a very space and compute efficient option. However, we only focus on achieving the best performance, so for leaderboard submissions we will prioritize DINOv3/EVA-02.
