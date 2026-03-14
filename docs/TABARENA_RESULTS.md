---
title: TabArena Results
nav_order: 5
---

# TabArena Benchmark Results

> **Context:** [TabArena](https://arxiv.org/abs/2506.08828) is a NeurIPS 2025 benchmark suite of 38 classification datasets within a larger living tabular benchmark. This page positions the current tabnetics `general` run against the current TabArena leaderboard using the same core scoring model: task-weighted pairwise battles, MLE Elo with bootstrap confidence intervals, average rank, win-rate, MRR, and normalized score.

> **Note:** this is an informational snapshot showing how tabnetics competes on general tabular data. General-tabular optimization is not currently the focus of the library, and this page should be read as a reference point rather than as a formal leaderboard submission.

## General profile

### Run configuration

| Parameter             | Value       |
|:----------------------|:------------|
| Profile               | general     |
| Seeds                 | 42          |
| Max training samples  | 50000       |
| Task timeout          | 3600 s      |
| Workers               | 12          |
| Classifier oracle     | MNPO hybrid |
| Leaderboard bootstrap | 200 rounds  |

### Current run snapshot

On the 38 classification datasets currently covered by the merged `general` profile run, `tabnetics (general)` receives **Elo 1012.1** in the overall leaderboard-style comparison, with **normalized score 0.105**.

The corresponding binary and multiclass leaderboard rows are shown below, followed by the dataset-level results from this run against the current official benchmark table.

### Tabnetics row

| method              |    Elo | Elo 95% CI   |   Score |   Rank |   Winrate |   MRR |
|:--------------------|-------:|:-------------|--------:|-------:|----------:|------:|
| tabnetics (general) | 1012.1 | +105/-121    |   0.105 |  32.56 |     0.283 | 0.116 |

### Nearby overall leaderboard rows

| method                |    Elo |   Score |   Rank |   Winrate |
|:----------------------|-------:|--------:|-------:|----------:|
| NN_TORCH (default)    | 1079.2 |   0.017 |  29.68 |     0.348 |
| FASTAI (default)      | 1040.5 |   0.038 |  31.39 |     0.309 |
| tabnetics (general)   | 1012.1 |   0.105 |  32.56 |     0.283 |
| RF (default)          | 1000   |   0.022 |  33.04 |     0.272 |
| LR (tuned + ensemble) |  980.6 |   0.027 |  33.78 |     0.255 |

### Binary leaderboard row

| method              |    Elo | Elo 95% CI   |   Score |   Rank |   Winrate |   MRR |
|:--------------------|-------:|:-------------|--------:|-------:|----------:|------:|
| tabnetics (general) | 1008.3 | +135/-151    |   0.076 |  32.84 |     0.276 | 0.099 |

### Multiclass leaderboard row

| method              |    Elo | Elo 95% CI   |   Score |   Rank |   Winrate |   MRR |
|:--------------------|-------:|:-------------|--------:|-------:|----------:|------:|
| tabnetics (general) | 1027.3 | +269/-481    |   0.213 |  31.49 |     0.307 | 0.176 |

### Per-dataset comparison against the current official best method

| Dataset                                    | problem_type   |   Bal. Acc. |   Tabnetics metric_error |   Best official metric_error | Best official method        |   Delta vs best |   Dataset rank | Selected model      |
|:-------------------------------------------|:---------------|------------:|-------------------------:|-----------------------------:|:----------------------------|----------------:|---------------:|:--------------------|
| APSFailure                                 | binary         |       0.957 |                   0.0139 |                       0.0071 | TABICL (default)            |          0.0069 |             42 | mnpo_lr             |
| Amazon_employee_access                     | binary         |       0.756 |                   0.1437 |                       0.1168 | CAT (tuned)                 |          0.0269 |             11 | mnpo_rf             |
| Bank_Customer_Churn                        | binary         |       0.766 |                   0.1589 |                       0.1256 | TABPFNV2 (tuned)            |          0.0333 |             39 | mnpo_lr             |
| Bioresponse                                | binary         |       0.788 |                   0.1459 |                       0.1243 | XGB (tuned + ensemble)      |          0.0216 |             36 | mnpo_rf             |
| Diabetes130US                              | binary         |       0.573 |                   0.39   |                       0.3277 | GBM (tuned + ensemble)      |          0.0623 |             41 | mnpo_lr             |
| E-CommereShippingData                      | binary         |       0.688 |                   0.2715 |                       0.2557 | TABPFNV2 (default)          |          0.0159 |             42 | mnpo_lr             |
| Fitness_Club                               | binary         |       0.735 |                   0.1911 |                       0.1781 | TABPFNV2 (default)          |          0.013  |             32 | mnpo_lr             |
| GiveMeSomeCredit                           | binary         |       0.71  |                   0.2281 |                       0.1329 | TABM (tuned + ensemble)     |          0.0953 |             42 | mnpo_lr             |
| HR_Analytics_Job_Change_of_Data_Scientists | binary         |       0.73  |                   0.2208 |                       0.1947 | TABICL (default)            |          0.0261 |             42 | mnpo_rf             |
| Is-this-a-good-customer                    | binary         |       0.678 |                   0.298  |                       0.2495 | EBM (default)               |          0.0485 |             41 | mnpo_lr             |
| Marketing_Campaign                         | binary         |       0.784 |                   0.1342 |                       0.0806 | TABPFNV2 (tuned + ensemble) |          0.0536 |             42 | mnpo_lr             |
| NATICUSdroid                               | binary         |       0.931 |                   0.0199 |                       0.0126 | TABICL (default)            |          0.0074 |             40 | mnpo_lr             |
| bank-marketing                             | binary         |       0.697 |                   0.2395 |                       0.2344 | CAT (default)               |          0.0051 |             27 | mnpo_lr             |
| blood-transfusion-service-center           | binary         |       0.734 |                   0.2189 |                       0.2445 | FASTAI (tuned + ensemble)   |         -0.0256 |              1 | mnpo_lr             |
| churn                                      | binary         |       0.856 |                   0.0934 |                       0.0695 | MNCA (default)              |          0.0238 |             38 | mnpo_xgb            |
| coil2000_insurance_policies                | binary         |       0.632 |                   0.3008 |                       0.2268 | TABPFNV2 (tuned + ensemble) |          0.0739 |             40 | mnpo_lr             |
| credit-g                                   | binary         |       0.661 |                   0.2452 |                       0.2037 | GBM (tuned + ensemble)      |          0.0416 |             42 | mnpo_lr             |
| credit_card_clients_default                | binary         |       0.695 |                   0.271  |                       0.2121 | TABICL (default)            |          0.0589 |             42 | mnpo_nb             |
| customer_satisfaction_in_airline           | binary         |       0.939 |                   0.0138 |                       0.0049 | REALMLP (tuned + ensemble)  |          0.0089 |             36 | mnpo_rf             |
| diabetes                                   | binary         |       0.805 |                   0.1137 |                       0.1556 | TABPFNV2 (default)          |         -0.0419 |              1 | mnpo_lr             |
| hazelnut-spread-contaminant-detection      | binary         |       0.927 |                   0.0244 |                       0.0076 | TABDPT (default)            |          0.0168 |             22 | mnpo_lgbm           |
| heloc                                      | binary         |       0.715 |                   0.2051 |                       0.1987 | TABPFNV2 (tuned + ensemble) |          0.0064 |             27 | mnpo_lr             |
| in_vehicle_coupon_recommendation           | binary         |       0.75  |                   0.1748 |                       0.1483 | TABM (tuned + ensemble)     |          0.0265 |             23 | mnpo_lgbm           |
| jm1                                        | binary         |       0.654 |                   0.2904 |                       0.2239 | TABICL (default)            |          0.0665 |             44 | mnpo_lr             |
| kddcup09_appetency                         | binary         |       0.74  |                   0.1867 |                       0.1542 | CAT (default)               |          0.0325 |             25 | mnpo_lr             |
| online_shoppers_intention                  | binary         |       0.821 |                   0.0996 |                       0.0627 | TABPFNV2 (tuned + ensemble) |          0.0369 |             42 | mnpo_lr             |
| polish_companies_bankruptcy                | binary         |       0.777 |                   0.0681 |                       0.0187 | TABPFNV2 (tuned + ensemble) |          0.0494 |             31 | mnpo_lgbm           |
| qsar-biodeg                                | binary         |       0.865 |                   0.0822 |                       0.0615 | TABICL (default)            |          0.0207 |             40 | mnpo_lr             |
| seismic-bumps                              | binary         |       0.651 |                   0.2824 |                       0.2166 | TABICL (default)            |          0.0658 |             42 | mnpo_lr             |
| taiwanese_bankruptcy_prediction            | binary         |       0.837 |                   0.0993 |                       0.0547 | REALMLP (tuned + ensemble)  |          0.0446 |             42 | mnpo_lr             |
| MIC                                        | multiclass     |       0.373 |                   2.2882 |                       0.4303 | TABM (tuned + ensemble)     |          1.858  |             45 | mnpo_elastic_net_lr |
| SDSS17                                     | multiclass     |       0.964 |                   0.1221 |                       0.0723 | RF (tuned + ensemble)       |          0.0498 |             37 | mnpo_rf             |
| anneal                                     | multiclass     |       0.894 |                   0.1004 |                       0.0156 | TABPFNV2 (default)          |          0.0847 |             42 | mnpo_lr             |
| hiva_agnostic                              | multiclass     |       0.349 |                   1.43   |                       0.1738 | RF (tuned)                  |          1.2562 |             45 | mnpo_knn            |
| maternal_health_risk                       | multiclass     |       0.84  |                   0.3628 |                       0.4048 | TABDPT (default)            |         -0.0419 |              1 | mnpo_xgb            |
| splice                                     | multiclass     |       0.975 |                   0.1048 |                       0.0993 | TABPFNV2 (tuned + ensemble) |          0.0056 |              5 | mnpo_xgb            |
| students_dropout_and_academic_success      | multiclass     |       0.693 |                   0.6439 |                       0.5266 | TABPFNV2 (tuned + ensemble) |          0.1173 |             42 | mnpo_lr             |
| website_phishing                           | multiclass     |       0.885 |                   0.2802 |                       0.2215 | TABPFNV2 (tuned + ensemble) |          0.0587 |             32 | mnpo_rf             |

## Interpretation

This page is intended as an informational view of the current tabnetics run against the current TabArena leaderboard. It provides a general-tabular reference point while the library remains focused primarily on HDLSS problems.

---

> Documentation and webpages on this site are generated from authoritative internal sources using a combination of deterministic rules and generative AI. Errors are possible. Please report issues via [GitHub Discussions](https://github.com/klokedm/tabnetics-public/discussions) or email [marko@tabnetics.org](mailto:marko@tabnetics.org).
