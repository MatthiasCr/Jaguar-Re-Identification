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

| [Notebook](notebooks/04_backbone_finetuning.ipynb) | 
[W&B Run Group](https://wandb.ai/juggling-jaguars/jaguar-reid-jugglingjaguars/groups/Experiment-4-BackboneFinetuning/) | 
Kaggle Submission: best run fine-tune all, Score: 0.907 | 

In the last experiments we always froze the backbone and just trained a few linear layers as embedding projection and the ArcFace head model. However, the backbone is pretrained on huge amounts of general image data with the training goal of general image classification. The backbone is therefore not specialized on our specific task of jaguar re-identification. Fine-tuning the last few layers or even the entire backbone can often help the model adapt to a specific task and dataset. In this experiment we want to evaluate if fine-tuning the backbone during training can achieve a higher identity balanced mAP.

### Setup

We use the previously best performing backbone which is Eva-02 Large. This model is a vision transformer and consists of **24 identical transfomer-encoder blocks** (self-attention, SwiGlu, RoPE, MLP). Each of these blocks have around 12.6 million parameters which sums up to around 305 million parameters in total. 
We evaluate different levels of backbone fine-tuning/freezing:

- **Backbone completely frozen** - the backbone is not retrained at all. This is the baseline.
- **Fine-Tune last 2 Blocks** - only last 2 transformer-encoder blocks are trainable
- **Fine-Tune last 4 Blocks**
- **Fine-Tune last 8 Blocks**
- **Fine-Tune the entire backbone** - this makes all backbone parameters trainable

Since the Eva-02 model is large and our data is relativly small, backpropagation and weight updates on the backbone comes with risks of changing the backbone to much and thus "destroying" the pretrained weights. Therefore we will use a smaller learning rate for the backbone parameters: The backbone learning rate will be `1e-5` which is ten times smaller than the learning rate for the ArcFace head (`1e-4`).

For the run with the fully frozen backbone we precompute the backbone embeddings for all training and validation data once and cache them to speed up training (same how we did it in the first three experiments). The other runs are trained end-to-end, so all images are processed through the backbone again in every forward pass, which leads to much longe training times. 

All other hyperparameters will be fixed for each run. All runs get a budget of 100 epochs with a patience of 8 for early stopping. The learning rate for both head and backbone is scheduled using `ReduceLROnPlateau` with a patience of 2.

### Results

|run|backbone trainable params|epochs trained|time per epoch|best val mAP|kaggle public score|
|--|--:|--:|--:|--:|--|
|freeze all|0|60| 2.9 s|0.854|-|
|train last 2|12,600,000|39|6.00 min|0.874|-|
|train last 4|25,200,000|28|6.14 min|0.881|-|
|train last 8|50,400,000|19|5.51 min|0.886|-|
|train all|304,055,232|16|5.71 min|0.902|0.907|

![](/images/e4_wandb_dashboard.png)

## Experiment 5 - Hyperparameter Search

| [Notebook](notebooks/05_hyperparamter_search.ipynb) | 
[W&B Run Group](https://wandb.ai/juggling-jaguars/jaguar-reid-jugglingjaguars/groups/Experiment-5-HyperparameterSearch) | 
Kaggle Submission: Score: 0.912 | 

In the previous experiments we fixed the general architecture: EVA-02 Large backbone, ArcFace loss, and full backbone fine-tuning. This already achieved a strong Kaggle score of `0.907`, but many training hyperparameters were still chosen manually. The goal of this experiment was therefore to identify a better combination of learning rates, regularization, head width, and training augmentation.

### Setup

We keep the overall model architecture fixed:

- **Backbone:** `eva02_large_patch14_448.mim_m38m_ft_in22k_in1k`
- **Input size:** `448`
- **Training mode:** full backbone fine-tuning (`freeze_backbone=False`)
- **Loss/head:** ArcFace with `margin=0.5`, `scale=64`
- **Optimizer:** AdamW
- **Scheduler:** ReduceLROnPlateau with patience `2`
- **Validation split:** `0.2`
- **Seed:** `42`
- **Reranking during validation:** enabled with `k1=20`, `k2=6`, `lambda=0.3`

For each sampled run we train a fresh model and select the best checkpoint by **validation rerank mAP**.

### Search Space

We perform a random search over the following parameters:

|parameter|possible values|
|--|--|
|head learning rate|`3e-5`, `1e-4`, `3e-4`|
|backbone learning rate|`3e-6`, `1e-5`, `3e-5`|
|weight decay|`1e-5`, `1e-4`, `5e-4`|
|dropout|`0.2`, `0.3`, `0.4`|
|train augmentation|`True`, `False`|
|batch size|`16`, `32`|
|embedding dimension|`256`, `384`|
|hidden dimension|`512`, `768`|


### Results

In total we trained 48 configurations and logged them to W&B. Below we report the strongest configurations.

|W&B run id|head lr|back-bone lr|weight decay|drop-out|aug|batch|embed|hidden|best val mAP|best val mAP rerank|best val loss|best epoch|
|--|--:|--:|--:|--:|--|--:|--:|--:|--:|--:|--:|--:|
|[qriyulso](https://wandb.ai/juggling-jaguars/jaguar-reid-jugglingjaguars/groups/Experiment-5-HyperparameterSearch/runs/qriyulso)|1e-4|1e-5|1e-5|0.2|on|16|384|768|0.9170|**0.9365**|2.2664|18|
|[qp8fg51n](https://wandb.ai/juggling-jaguars/jaguar-reid-jugglingjaguars/groups/Experiment-5-HyperparameterSearch/runs/qp8fg51n)|3e-4|3e-5|1e-4|0.3|off|16|256|512|**0.9355**|0.9336|2.4942|10|
|[k2pi28gh](https://wandb.ai/juggling-jaguars/jaguar-reid-jugglingjaguars/groups/Experiment-5-HyperparameterSearch/runs/k2pi28gh)|3e-5|3e-5|1e-4|0.2|off|16|384|768|0.9095|0.9238|**1.8927**|21|
|[v67i2spa](https://wandb.ai/juggling-jaguars/jaguar-reid-jugglingjaguars/groups/Experiment-5-HyperparameterSearch/runs/v67i2spa)|3e-4|1e-5|5e-4|0.2|on|32|384|768|0.8989|0.9226|2.1571|8|
|[b40hr4rt](https://wandb.ai/juggling-jaguars/jaguar-reid-jugglingjaguars/groups/Experiment-5-HyperparameterSearch/runs/b40hr4rt)|3e-5|1e-5|5e-4|0.3|off|16|384|768|0.9127|0.9201|1.8237|25|

Several useful patterns emerge from the search:

- **Batch size 16** dominates the top runs. All five best runs use batch size `16`.
- **384 / 768** is a strong head size. Most top rerank results use `embedding_dim=384` and `hidden_dim=768`.
- **Low dropout helps.** The strongest runs use `0.2` or `0.3`; `0.4` appears less competitive.
- **Both augmentation settings can work.** The best reranked run uses augmentation, but several other top runs perform best without it.
- **Reranked ranking and plain mAP do not always choose the same winner.** One run achieves the best plain validation mAP (`0.9355`) but is slightly behind the best reranked validation mAP (`0.9365`).

The best run of the search is therefore: [qriyulso](https://wandb.ai/juggling-jaguars/jaguar-reid-jugglingjaguars/groups/Experiment-5-HyperparameterSearch/runs/qriyulso)
- best validation mAP: **0.9170**
- best validation rerank mAP: **0.9365**

This run became the new default checkpoint for later experiments and improved the Kaggle public score from `0.907` to `0.912`.

## Experiment 6 - K-Reciprocal Re-Ranking

| [Notebook](notebooks/06_k_reciprocal_re_ranking.ipynb) | 
[W&B Project](https://wandb.ai/juggling-jaguars/jaguar-reid-jugglingjaguars) | 
No new Kaggle submission | 

We wanted to test whether the default k-reciprocal reranking parameters were already good enough for our current validation setup, or whether a small validation-only sweep over `k1` and `lambda_value` could still improve retrieval.

### Setup

We keep the model fixed and only tune reranking. The selected checkpoint comes from the hyperparameter search.

- **Checkpoint:** `eva_unfrozen_rs_08_hlr3e-05_blr3e-05_wd1e-04_do0.2_aug0_bs16`
- **Backbone:** EVA-02 Large fine-tuned end-to-end
- **Head learning rate:** `3e-5`
- **Backbone learning rate:** `3e-5`
- **Weight decay:** `1e-4`
- **Dropout:** `0.2`
- **Train augmentation:** off
- **Batch size:** `16`
- **Embedding / hidden dim:** `384 / 768`
- **Best checkpoint epoch:** `21`

We reconstruct the validation split for this checkpoint, extract the validation embeddings once, and then run a small grid search over reranking:

- **`k1`** in `{10, 15, 20, 25, 30, 35, 40}`
- **`lambda_value`** in `{0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6}`
- **`k2`** fixed at `6`

For each parameter pair we compute validation rerank mAP and compare it against both the no-rerank baseline and the checkpoint's default rerank setting.

### Results

|setting|val mAP rerank|
|--|--:|
|no reranking|0.9095|
|default rerank (`k1=20`, `k2=6`, `lambda=0.3`)|**0.9237**|
|best grid-search result|**0.9237**|

The validation sweep did **not** improve on the default reranking setup. The best searched configuration was again:

- **`k1=20`**
- **`k2=6`**
- **`lambda_value=0.3`**

This means the tuned search gained:

- **`+0.0142`** over no reranking
- **`+0.0000`** over the default reranking parameters

So the practical conclusion is modest but useful: **for this checkpoint and this validation split, the existing rerank defaults were already as good as the tested alternatives**.

### Limitations

This experiment has several important limitations, so the result should be interpreted carefully:

- It uses **one specific checkpoint**, not all strong models. In particular, the selected model comes from the available `incomplete_random_search_results.csv` snapshot and is **not the final strongest reranked model from Experiment 5**.
- The chosen checkpoint was already evaluated and selected with the **same default reranking parameters** (`k1=20`, `k2=6`, `lambda=0.3`) on the same validation setup. That biases the search toward rediscovering the defaults.
- The observed optimum depends on the **training configuration** of this checkpoint. Learning rates, dropout, augmentation, head width, and the resulting embedding geometry can all change which reranking parameters work best.
- We only tune **`k1` and `lambda_value`**, while **`k2` stays fixed at `6`**. A broader search might still find different trade-offs.
- The sweep is done on a **single validation split**, so it tells us that the defaults are robust for this split, but not that they are universally optimal for every model or for Kaggle test performance.

We therefore treat this experiment mainly as a **sanity check**: it supports keeping the standard rerank parameters for later experiments, but it does **not** prove that reranking can no longer be improved in general.


## Experiment 7 - GeM Pooling

| [Notebook](notebooks/07_gem_pooling.ipynb) | 
[W&B Run Group](https://wandb.ai/juggling-jaguars/jaguar-reid-jugglingjaguars/groups/Experiment-7-GeMPooling) | 
Kaggle Submission: Score: 0.903 | 

In this experiment we tested whether replacing the backbone's default global pooling with **GeM pooling** can improve jaguar re-identification. The motivation is that re-ID often depends on a few highly discriminative local fur patterns. GeM is more selective than plain average pooling and can emphasize strong local activations instead of smoothing them away too aggressively.

### Setup

We keep the full training recipe fixed and change only the pooling layer inside the backbone:

- **Backbone:** `eva02_large_patch14_448.mim_m38m_ft_in22k_in1k`
- **Training mode:** full backbone fine-tuning (`freeze_backbone=False`)
- **Input size:** `448`
- **Head:** ArcFace with `margin=0.5`, `scale=64`
- **Embedding / hidden dim:** `256 / 512`
- **Dropout:** `0.3`
- **Batch size:** `32`
- **Head learning rate:** `1e-4`
- **Backbone learning rate:** `1e-5`
- **Weight decay:** `1e-4`
- **Train augmentation:** off
- **Validation split:** `0.2`
- **Reranking during validation:** `k1=20`, `k2=6`, `lambda=0.3`

The two compared runs are:

- **Default pooling**: standard backbone pooling as provided by the EVA model
- **GeM pooling**: replace the default pooling with GeM using `p=3.0` and `eps=1e-6`

So this is a very targeted architectural test: if GeM helps here, the gain should come from better spatial aggregation rather than from a different optimizer, schedule, augmentation policy, or backbone.

### Results

We repeated the comparison for three training seeds (`42`, `43`, `44`) to avoid over-interpreting a single lucky run.

|variant|seeds|val mAP mean (std)|val mAP rerank mean (std) |
|--|--:|--:|--:|
|Default pooling|3|0.917 (0.018)|**0.922** (0.021) 
|GeM pooling|3|0.915 (0.028)|0.921 (0.0289)|

Seed-wise results:

|seed|default val mAP|GeM val mAP|delta GeM-default|default val mAP rerank|GeM val mAP rerank|delta GeM-default rerank|
|--:|--:|--:|--:|--:|--:|--:|
|42|0.9063|0.8964|-0.0099|0.9110|0.9055|-0.0055|
|43|0.9372|0.9468|+0.0096|0.9454|0.9537|+0.0083|
|44|0.9068|0.9008|-0.0060|0.9079|0.9039|-0.0040|

Across the paired seeds, the mean GeM-minus-default difference is:

- **`-0.0021`** in plain validation mAP
- **`-0.0004`** in reranked validation mAP

So GeM does not show a consistent improvement in this experiment. It helps on seed `43`, but loses on seeds `42` and `44`, and the average reranked result is effectively unchanged.

### What We Wanted To Learn

This experiment is mainly meant to answer two questions:

- Does GeM improve retrieval quality over the default pooling for this EVA-based re-ID model?
- If there is a gain, is it large enough to justify the added architectural complexity and to carry forward into later experiments?

With the current seed sweep, the answer is: **not convincingly**. The GeM variant is competitive, but there is no reliable mean improvement over default pooling, so we do not treat GeM as a clearly better replacement in this setup.


## Experiment 8 - Test-Time Augmentation

| [Notebook](notebooks/08_tta_comparison.ipynb) | 
[W&B Run Group](https://wandb.ai/juggling-jaguars/jaguar-reid-jugglingjaguars/groups/Experiment-8-TestTimeAugmentation) | 
No new Kaggle submission | 

After establishing a strong EVA-02 fine-tuned baseline, we tested whether deterministic test-time augmentation can improve retrieval performance at inference time. The central question was simple: does TTA help our current model enough to justify the additional inference cost?

### Setup

We keep the model fixed and only change inference-time augmentation. The evaluated checkpoint is `EVA02_Large_finetune.pth` (`eva02_finetune_seed42`). We compare four deterministic TTA presets:

- **none** - only the default resized image
- **light** - base view plus a centered 95% crop
- **medium** - base view plus centered 95% and 90% crops
- **heavy** - medium plus off-center 90% crops (`top_left`, `bottom_right`)

For each TTA preset we extract embeddings for all views, average the embeddings, normalize them again, and compute both plain validation mAP and reranked validation mAP.

### Results

|TTA mode|views|val mAP|val mAP rerank|
|--|--:|--:|--:|
|none|1|0.8903|0.8930|
|light|2|0.8919|0.8920|
|medium|3|0.8922|0.8923|
|heavy|5|0.8915|0.8910|

The differences are marginal. While `light` and `medium` slightly improve the plain validation mAP, none of the deterministic TTA presets improves the reranked validation mAP over the no-TTA baseline. In fact, the best reranked score is obtained with **no TTA at all** (`0.8930`), while the larger TTA presets are slightly worse.

We therefore conclude that **deterministic crop-based TTA does not provide a meaningful benefit for this model in our setup**. Given the extra inference cost, we do not continue with TTA for leaderboard submissions.


## Experiment 9 - Random Seed Comparison

| [Notebook](notebooks/09_seed_comparison.ipynb) | 
[W&B Run Group](https://wandb.ai/juggling-jaguars/jaguar-reid-jugglingjaguars/groups/Experiment-9-RandomSeeds) | 
No new Kaggle submission | 

After fixing the training parameters, we wanted to measure how much variance remains purely from the random seed. This is important because if seed-to-seed variance is large, then small differences between experimental tweaks can be misleading unless runs are repeated.

### Setup

We keep the full training configuration fixed and only vary the random seed. The compared model is the EVA unfrozen ArcFace setup with the following fixed hyperparameters:

- head learning rate `1e-4`
- backbone learning rate `1e-5`
- weight decay `1e-5`
- dropout `0.2`
- training augmentation enabled
- batch size `16`
- reranking enabled with `k1=20`, `k2=6`, `lambda=0.3`

We run the same experiment for 10 seeds (`42` to `51`) and compare the best validation metrics of each run.

### Results

|seed|best val mAP|best val mAP rerank|best val loss|best epoch|epochs trained|
|--:|--:|--:|--:|--:|--:|
|43|0.9308|0.9381|2.1995|15|23|
|48|0.9246|0.9313|2.2370|10|18|
|46|0.9249|0.9228|2.0425|17|25|
|51|0.9176|0.9213|2.5083|10|18|
|49|0.9116|0.9112|2.6716|22|25|
|50|0.8978|0.9036|2.8094|17|25|
|42|0.8989|0.8997|2.3413|18|25|
|45|0.8985|0.8960|2.2614|18|25|
|47|0.8880|0.8929|2.4787|9|17|
|44|0.8866|0.8920|2.8812|21|25|

Across the 10 seeds, the mean reranked validation mAP is **0.9109 ± 0.0166**. The best run reaches **0.9381** (seed 43), while the weakest run reaches **0.8920** (seed 44). This spread is substantial and larger than many of the marginal effects observed in later experiments such as deterministic TTA.

The conclusion is that **training seed has a meaningful impact on final retrieval performance in this setup**. Therefore, single-run comparisons should be interpreted carefully, and strong results should ideally be confirmed across multiple seeds or at least by repeating promising configurations.


## Experiment 10 - Background Comparison

| [Notebook](notebooks/10_background.ipynb) | 
[Results CSV](checkpoints/e15_dataset_source_comparison/dataset_source_results.csv) | 
No new Kaggle submission | 

After discovering that the original `data` images still contain RGB values in transparent regions while `data_background` removes them, we wanted to test whether those hidden background pixels materially affect training and retrieval.

### Setup

We keep the model configuration fixed and compare two data sources:

- **`data`**: original images, including RGB values hidden behind the alpha mask
- **`data_background`**: same images, but hidden RGB values removed

To make the comparison fair, the notebook:

- uses the same EVA checkpoint configuration for both runs: `eva_unfrozen_rs_04_hlr1e-04_blr1e-05_wd1e-05_do0.2_aug1_bs16`
- enforces a **shared validation split** across both sources
- trains a **fresh model on each source**
- runs a **2x2 cross-evaluation**:
  - train on `data`, evaluate on `data`
  - train on `data`, evaluate on `data_background`
  - train on `data_background`, evaluate on `data`
  - train on `data_background`, evaluate on `data_background`

This isolates whether the background treatment changes the learned representation, rather than just changing the train/val split.

### Results

Training each source on its own version of the validation set gives:

|train source|eval source|val mAP|val mAP rerank|
|--|--|--:|--:|
|`data`|`data`|0.8921|**0.9122**|
|`data_background`|`data_background`|**0.8984**|0.9067|

Cross-evaluation shows a stronger effect:

|train source|eval source|val mAP|val mAP rerank|
|--|--|--:|--:|
|`data`|`data_background`|0.6021|0.6085|
|`data_background`|`data`|0.8187|0.8309|

### Interpretation

The result is asymmetric:

- The model trained on original `data` performs best on original `data`.
- The model trained on `data_background` performs slightly better in plain mAP on its own source, but still trails the `data` model in reranked mAP on its own source.
- When evaluated on the opposite source, both models degrade sharply, especially the model trained on `data` and evaluated on `data_background`.

This suggests that the hidden RGB values are not just harmless noise. They appear to create a real domain shift that the model learns to rely on. Removing them changes the image distribution enough that embeddings no longer transfer cleanly between the two sources.

The practical takeaway is that **background handling matters a lot in this project**. Any final training or submission pipeline should stick to one consistent image source and avoid mixing `data` and `data_background` without retraining.
