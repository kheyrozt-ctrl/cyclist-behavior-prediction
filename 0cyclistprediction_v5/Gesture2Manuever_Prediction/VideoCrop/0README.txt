This directory contains legacy local-source data-construction scripts.

Generated fold lists, source label tables, blocker lists, original filenames,
recording dates, and internal paths are intentionally excluded from the public
Git repository. They are private inputs to the authorized dataset-release
process and are not required for public Stage2 data preparation.

For public reproduction, use the anonymized Hugging Face dataset and run:

    python tools/prepare_hf_stage2.py --fold 1

See docs/PUBLIC_TRAINING.md for the supported public workflow and its explicit
Stage1 checkpoint limitation.
