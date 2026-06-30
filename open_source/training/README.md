# Training Code

This directory contains the model-training lineages used by the release.

- `intersection_intention_legacy/`: original intersection intention pipeline,
  including explicit/implicit gesture classification and maneuver prediction.
- `bus_stop_v5_public/`: public V5 bus-stop behavior pipeline and combined
  maneuver model training code.

The intersection explicit/implicit default checkpoint directories currently do
not contain the expected `best_model_fold_*.pkl` files; see the runtime notes in
`../inference/unified_prediction/README.md`.
