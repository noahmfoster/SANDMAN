# The Datasets and Instructions:

- Mice (MAP): DANDI:000363/0.230822.0128
- Mice (IBL): DANDI:000032/0.220706.1533
- Monkeys: https://dandiarchive.org/dandiset/000128 Download from DANDI:000128/0.220113.0400
- Synthetic: Used ChaoticRNN from https://github.com/tinaxia2016/NeuroPaint/tree/main to generate synthetic data.

The Mice and Synthetic datasets are multi-site recordings for training and testing the inpainting. The Monkeys dataset is a single-site recording from the original LDNS paper. We use this to test the generalization of our inpainting model to new organisms for non-inpainting tasks such as neural activity prediction.

These will all be downloaded automatically when running the setup.sh script with the --download-data flag.
