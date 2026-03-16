## Experiment 3 - Weighted Sampling

| [Notebook](notebooks/03_weighted_sampling.ipynb) | 
[W&B Run Group](https://wandb.ai/juggling-jaguars/jaguar-reid-jugglingjaguars/groups/Experiment-3-WeightedSampling) |

In the first two experiments we compared model backbones and loss functions evaluating what achieves the highest identity balanced mAP. In this experiment we want to not only look at the final identity balanced mAP but also on the performance of the different identities individually. The dataset is very imbalanced: Some of the 31 identities have very many images while others have only a few. Marcela has with 183 the highest number of images while Bernard has with only 13 images the fewest.

![](images/data_counts_per_identity.png)

The goal of this experiment is to test whether the class imbalance is a bottleneck for retrieval quality.  Our metric (identity-balanced mAP) gives each identity equal importance, but the training distribution does not: frequent identities contribute many more updates than rare identities. Using weighted sampling shows rare identites more often which could increase their individual average precision (AP) and also the overall identity-balanced mAP.

**Research Question:** Does using weighted sampling improve average precision for low frequency identites and does it improve overall identity-balanced mAP?

### Setup

**Baseline:** standard shuffled sampling (`DataLoader(..., shuffle=True)`) on the embedding dataset. In this baseline, each training image appears exactly once per epoch, but in a random order that changes each epoch.

**Intervention:** weighted sampling with replacement (`WeightedRandomSampler`).
This means an epoch still has the same number of samples, but now samples are drawn probabilistically: some images can appear multiple times and some may not appear in a given epoch. As weights we use the inverse of the square root of the class frequencies ($w_c = 1/\sqrt{n_c}$). This means sampling will not be exactly balanced as it would be when using the full inverse frequency ($1 / n_c$), but it still increases exposure for rare identites. Using the full inverse frequency often over-corrects in small datasets so using the square root is a tempered alternative which tends to be more stable and less destructive for higher frequency identities.

The backbone (EVA-02) and loss function (ArcFace) as well the data split (train/val) and all other settings and hyperparameter will be fixed. We run each alternative **five times** using a different random seed each time (10, 38, 56, 102, 2024) and compute averages and standard deviations for all metrics. This way we strengthen the evidence even if the variation is high.

### Results

If we look at the overall identity-balanced mAP for the validation data we only see a **very small average improvement of 0.001** when using a weighted sampler, as shown in the following table:

|variant|num runs|val mAP mean|val mAP std|
|--|--|--:|--:|
|weighted sampler|5|0.8611|0.0076|
|random sampler|5|0.8599|0.0071|

Now we look at the individual identities. The following table shows the top movers: Three identites which AP increased the most and the three identities where it decreased the most. All shown APs are averages of the results across the five random seeds.

|identity|train samples|AP baseline|AP weighted sampler|difference|
|--|--|--|--|--|
|Oxum|14|0.820|0.850|+ 0.030|
|Madalena|22|0.976|0.998|+ 0.0224|
|Tomas|50|0.910|0.921|+ 0.011|
||
|Bororo|18|0.522|0.507|- 0.015|
|Marcela|147|0.897|0.892|- 0.005|
|Kwang|90|0.997|0.992|- 0.005|

In the following figure we visualize the AP differences per identity sorted by class frequency. We can see that there are only a few identites with a substantial AP delta. The greatest changes, positive and negative, happend for lower frequency identites, which is expected. The highest frequency identites tend to slightly decrease in AP which is also expected as they were given less importance to than in the baseline. 

![](images/e3_result_fig_ap_delta_with_sample_count.png)

Especially Oxum and Madalena benefit from the weighted sampling while Bororo decreases the most. We have inspected the images of Oxum and Bororo: Both have only little data but Oxum's images are almost all of very high quality. Bororo on the other hand, while also having *some* high quality images, is more noisy: In many images the Jaguar is mostly covered by vegetation and is hardly visible. We therefore conclude that the benefit of weighted sampling depends on the quality of the data. If many samples are of low quality, oversampling them can also worsen performance. 

### Conclusion

Weighted sampling does improve average precision for some identites, but can have the opposite effect if the data is low quality. This makes the overall mAP improvement in our dataset extremely small (+ 0.001). For better results it would be important to focus on data quality.