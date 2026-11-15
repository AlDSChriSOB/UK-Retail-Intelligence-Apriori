# Execution verification

Original Online Retail.xlsx (541,909 rows) processed in this environment. Cleaning yielded 485,123 UK positive-sale rows, 18,019 invoices and 3,916 codes. Chronological 70/30 benchmark and 10,000-resample paired bootstrap ran successfully. The notebook Jupyter kernel itself failed to complete, so the RESULTS notebook contains independently computed and checked outputs, **not** an authenticated end-to-end Jupyter-kernel run.

## Benchmark

{
  "protocol": "chronological 70/30 invoice split; one seeded hidden item per eligible test basket",
  "seed": 42,
  "train_invoices": 12613,
  "test_invoices": 5406,
  "eligible_test_invoices": 4997,
  "min_support": 0.01,
  "min_confidence": 0.2,
  "metrics": {
    "popularity": {
      "5": {
        "hits": 102,
        "precision_at_k": 0.004082449469681809,
        "recall_at_k": 0.020412247348409046,
        "hit_rate_at_k": 0.020412247348409046
      },
      "10": {
        "hits": 189,
        "precision_at_k": 0.0037822693616169703,
        "recall_at_k": 0.0378226936161697,
        "hit_rate_at_k": 0.0378226936161697
      }
    },
    "cooccurrence": {
      "5": {
        "hits": 381,
        "precision_at_k": 0.015249149489693817,
        "recall_at_k": 0.07624574744846908,
        "hit_rate_at_k": 0.07624574744846908
      },
      "10": {
        "hits": 564,
        "precision_at_k": 0.011286772063237943,
        "recall_at_k": 0.11286772063237943,
        "hit_rate_at_k": 0.11286772063237943
      }
    },
    "apriori": {
      "5": {
        "hits": 435,
        "precision_at_k": 0.017410446267760657,
        "recall_at_k": 0.08705223133880328,
        "hit_rate_at_k": 0.08705223133880328
      },
      "10": {
        "hits": 593,
        "precision_at_k": 0.011867120272163297,
        "recall_at_k": 0.11867120272163298,
        "hit_rate_at_k": 0.11867120272163298
      }
    }
  }
}

## Bootstrap

{
  "n": 4997,
  "hit_rate_a": 0.11867120272163298,
  "hit_rate_b": 0.11286772063237943,
  "difference": 0.005803482089253552,
  "ci95_percentile": [
    -0.0018010806483890335,
    0.013207924754852912
  ],
  "a_only": 199,
  "b_only": 170,
  "both_hit": 394,
  "bootstrap_replicates": 10000,
  "bootstrap_seed": 20261111
}