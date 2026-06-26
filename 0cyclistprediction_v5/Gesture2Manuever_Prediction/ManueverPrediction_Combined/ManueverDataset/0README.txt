This directory is a placeholder for generated local Stage2 data.

Source-derived split CSV files and PKL windows are intentionally excluded from
the public Git repository. Generate anonymized fold PKL files from the public
Hugging Face release with:

    python tools/prepare_hf_stage2.py --fold 1

The generated files are written under release_data/ManueverDataset by default
and remain ignored by Git.
