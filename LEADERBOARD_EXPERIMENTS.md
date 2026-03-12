## Experiment 1 - Backbone Comparison

| [Notebook](notebooks/01_backbones.ipynb) | 
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

All other hyperparameters and the training procedure will be identical for all runs. We use a batch size of 32, dropout of 0.3, AdamW as optimizer, and ReduceLROnPlateau as scheduler. We will train 100 epochs with a patience of 10 epochs, that means if for 10 consecutive epochs we can not beat the currently best validation loss we will stop training early. In the result table below we report how many epochs each model trained. After training we always restore the best checkpoint (by validation loss) to compute metrics and create the kaggle submission.

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


## Experiment 2 - Loss Function Comparison

| [Notebook](notebooks/02_loss_functions.ipynb) | 
[W&B Run Group](https://wandb.ai/juggling-jaguars/jaguar-reid-jugglingjaguars/groups/Experiment-2-LossFunctions) | 
no improvment so no new kaggle sumbission | 

In Experiment 1 we fixed the loss to the metric learning loss ArcFace. Here we want to understand ArcFace and metric learning in general better by comparing it to plain CrossEntropy and the very similar alternatives SphereFace and CosFace. Compared to plain classification which only optimizes correct class assignment, metric learning explicitly shapes the embedding space by pulling samples of the same identity closer and pushing different identities farther apart. The goal of this experiment is to answer whether margin-based metric learning improves mAP performance and which margin variant works best in our setup.

### Setup

We use the best performing backbone from experiment 1, which is EVA-02. All other settings are fixed and identical to experiment 1 (same split, same optimizer/scheduler, same training budget and patience). We only change the loss/head formulation.

The compared losses are:
- **CrossEntropy** - standard for classification tasks and therefore an important baseline. Useful to quantify how much margin-based metric learning really helps
- **SphereFace** - first major angular-margin softmax formulation (2017)
- **CosFace** - additive cosine margin
- **ArcFace** - additive angular margin and common modern default for re-id tasks.

Because each method applies margin in a different way (additive angular, additive cosine, multiplicative angular) the margin values are not comparable (i.e. we can not use the same margin m for each loss). Therefore we use estalished defaults for each method for a fair comparison:  **ArcFace m=0.5**, **CosFace m=0.35**, and **SphereFace m=1.35**.

**Research questions:** Do margin-based metric learning losses improve mAP over plain CrossEntropy? Which method performs best for this dataset?

### Results

|loss |margin type|margin|epochs (best) |best val acc|best val mAP|
|--|--|--|--|--|--|
|CrossEntropy|-|-|66 (56)|0.955|0.835|
|SphereFace|multiplicative angular|1.35|84 (74)|0.918|0.841|
|CosFace|additive cosine|0.35|85 (75)|0.913|0.858|
|ArcFace|additive angular|0.5|85 (75)|0.902|0.860|

![](images/e2_wandb_graphs.png)

CrossEntropy achives the highest validation accuracy but clearly underperforms on mAP. This is expected because CE optimizes class prediction rather than embedding alignment which is needed for good similarities. The strong class imbalance likely amplifies this effect: accuracy and CE are dominated by frequent identities, while our mAP is identity-balanced and gives equal importance to rare identities. 

From the learning curves we can also observe that CE shows much steeper early loss and accuracy improvements and converges earlier than the other losses. This is consistent with CE being an easier optimization objective, while ArcFace/CosFace spend more time on enforcing stricter embedding clustering/separation.

ArcFace and CosFace perform very similarly and both achieve substantially better mAP than CrossEntropy. This shows that explicit margin constraints improve embedding separability for retrieval. SphereFace also improves over CrossEntropy but remains below ArcFace/CosFace. For the next experiments we will therefore continue with ArcFace.


## Experiment 4 - Backbone Fine-Tuning

In the last experiments we always froze the backbone and just trained the a few linear layers as embedding projection and the ArcFace head model. In this experiment we want to evaluate if fine-tuning the backbone during can achieve a higher mAP.

### Setup


### Results


## Experiment 5 - Hyperparameter Search

In the last four experiments we already found setup (backbone, loss function, backbone freezing) that achieves good scores on kaggle (TODO best kaggle score). 