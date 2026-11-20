# Data

The challenge data is **not stored in this repository**. Get it from one of these:

- **Zindi:** the **Data** tab of the
  [challenge page](https://zindi.world/competitions/climate-risk-health-prediction-cmu-hackathon). This needs a
  Zindi account and the challenge rules accepted.
- **Kaggle mirror:** [moanlobago/climate-health-risk-hackathon-data](https://www.kaggle.com/datasets/moanlobago/climate-health-risk-hackathon-data).
  - On Kaggle, attach it to your notebook with **Add Input**.
  - Elsewhere, run: `kaggle datasets download moanlobago/climate-health-risk-hackathon-data -p data --unzip`

Put the files in this folder:

```
data/
├── Train.csv
├── Test.csv
├── climate_features.csv
└── SampleSubmission.csv
```

The baseline notebook looks in `./data` by itself.

**Licence:** CC BY-SA 4.0, from the Zindi *Climate Risk and Health Prediction Challenge*. Keep this attribution if
you share the data or anything derived from it.
