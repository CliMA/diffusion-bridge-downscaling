# Diffusion-bridge-downscaling
Code repository for the preprint "Unpaired Downscaling of Fluid Flows with Diffusion Bridges", Bischoff &amp; Deck (2023).
In order to train model, follow the the steps below.

1. Navigate to `CliMAgen.jl/examples/context2dturbulence`.
2. Setup Julia Project via:</br>
`julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.build(); Pkg.precompile'`

3. Model training:</br>
`julia --project training.jl Experiment_512x512.toml`

4. Computing and saving statistics of training images:
In order to generate statistics from all of the training data, using 2000 pixels from each image, for the wavenumber 16 dataset,</br>
`julia --project analyze_train_by_batch.jl 2000 16 Experiment_512x512.toml`.</br>
The model is not used in this process, but the Experiment.toml specifies where the preprocessing parameter file is stored. By default, our dataloader preprocesses the data, so the inverse transformation is required before computing statistics.

5. Computing and saving statistics of generated images:
In order to generate statistics from 32 batches of 25 images, using 2000 pixels from each image, for the wavenumber 16 dataset,</br>
`julia --project analyze_gen_by_batch.jl 32 2000 16 Experiment_512x512.toml`

6. Computing and saving statistics of downscaled images:
In order to generate statistics for 32 batches of 25 downscaled images, using 2000 pixels from each image, for the wavenumber 16 dataset,</br>
`julia --project analyze_downscaled_by_batch.jl 32 2000 16 Experiment_64x64.toml Experiment_512x512.toml`</br>
Again, no model is required for the low-resolution dataset, but the Experiment_64x64.toml specifies where the preprocessing parameter file is stored. By default, our dataloader preprocesses the data, so the inverse transformation is required before computing statistics.

This generates approximately the same number of samples in the training statistics, generated statistics, and downscaled statistics. While the code does run on the CPU, we recommend running on a GPU. Parameter choices for training, for the diffusion model, and for sampling are in the experiment TOML files.
