# diffusion-bridge-downscaling
Code to recreate results from "Unpaired Downscaling of Fluid Flows with Diffusion Bridges", Bischoff &amp; Deck (2023).

First, navigate to `CliMAgen.jl/examples/context2dturbulence`.
- Training:
Run `julia --project training.jl Experiment.toml`

- Computing and saving statistics of generated images:
In order to generate statistics from 32 batches of 25 images, using 2000 pixels from each image, for the wavenumber 16 dataset, run `julia --project analyze_gen_by_batch.jl 32 2000 16 Experiment.toml`

- Computing and saving statistics of training images:
In order to generate statistics from all of the training data, using 2000 pixels from each image, for the wavenumber 16 dataset, run `julia --project analyze_gen_by_batch.jl 2000 16 Experiment.toml`. The model is not used in this process, but the Experiment.toml specifies where the preprocessing parameter file is stored. By default, our dataloader preprocesses the data, so the inverse transformation is required before computing statistics.

- Computing and saving statistics of downscaled images:
In order to generate statistics for 32 batches of 25 downscaled images, using 2000 pixels from each image, for the wavenumber 16 dataset, run `julia --project analyze_downscaled_by_batch.jl 32 2000 16 LowResolution.toml Experiment.toml`.
Again, no model is required for the low-resolution dataset, but the LowResolution.toml specifies where the preprocessing parameter file is stored. By default, our dataloader preprocesses the data, so the inverse transformation is required before computing statistics.

This generates approximately the same number of samples in the training statistics, generated statistics, and downscaled statistics. While the code does run on the CPU, we recommend running on a GPU. Parameter choices for training, for the diffusion model, and for sampling are in the experiment toml files.
