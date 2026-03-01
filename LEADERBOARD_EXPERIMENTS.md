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

All other hyperparameters and the training procedure will we identical for all runs. We use a batch size of 32, dropout of 0.3, AdamW as optimizer, and ReduceLROnPlateau as scheduler. We will train 100 epochs with a patience of 10 epochs, that means if for 10 consecutive epochs we can not beat the currently best valiation loss we will stop training early. In the result table below we report how many epochs each model trained.

### Results

In the table below we report some essential metrics of the runs. More metrics such as losses, learning rates, etc are logged in W&B.

|backbone|architecture|parameter|input size|epochs (best) |best val mAP|
|--|--|--:|--|--|--|
|MegaDescriptor|Swin Transformer|195,198,516|384|84 (74)|0.784|
|ResNet50|CNN|23,508,032|288|91 (81)|0.817|
|EfficientNetB3|CNN|10,696,232|300|100 (90)|0.841|
|DinoV3|ViT|303,079,424|256|99 (89)|0.867|
|Eva02|ViT|304,055,232|448|84 (74)|0.862|

[TODO: Result Interpretation. Answers for research questions]