# Spiking Activity in Neurons with Diffusion Modeling Across Neural Regions

This repository contains code and resources for Noah Fosters course project for STAT 8201 Statistical Analysis of Neural Data and STAT 6701 Probabilistic Models and Machine Learning at Columbia University for professor Liam Paninski and David Blei respectively.

This project applies diffusion models (mostly Latent Diffusion Models inspired by LDNS: https://arxiv.org/abs/2407.08751) to the Spike Train Infill Problem (as tackled by NeuroPaint: https://arxiv.org/abs/2510.11924).

**Proper Credit:** Much of the codebase is adapted from the NeuroPaint and LDNS repositories. Code from NeuroPaint is primarily used in the data loading (for all but the monkey data).

**Compatibility:** I've done a little testing of cold start installs on my own machine, but that does not guarantee that it will work on yours. Good luck. Also worth noting that I'm using a Mac with an M4 chip. I try to include CUDA/CPU compatibility where possible, but I have not tested on an NVIDIA GPU machine nor on CPU-only machines. Again, good luck.
