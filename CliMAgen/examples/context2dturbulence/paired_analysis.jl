using BSON
using CUDA
## Script for computing metrics of interest on the training data ##

using Flux
using ProgressMeter
using Random
using Statistics
using TOML
using DelimitedFiles
using StatsBase
using HDF5

using CliMAgen
package_dir = pkgdir(CliMAgen)
include(joinpath(package_dir,"examples/utils_data.jl"))
include(joinpath(package_dir,"examples/utils_analysis.jl"))

function unpack_experiment(experiment_toml, wavenumber; batchsize=batchsize, fraction=fraction, device=Flux.gpu, FT=Float32)
    params = TOML.parsefile(experiment_toml)
    params = CliMAgen.dict2nt(params)
    # unpack params
    savedir = params.experiment.savedir
    resolution = params.data.resolution
    standard_scaling  = params.data.standard_scaling
    preprocess_params_file = joinpath(savedir, "preprocessing_standard_scaling_$standard_scaling.jld2")
    scaling = JLD2.load_object(preprocess_params_file)
    noised_channels = params.model.noised_channels
    context_channels = params.model.context_channels
    dl, _ =  get_data_context2dturbulence(
        batchsize;
        resolution = resolution,
        wavenumber = wavenumber,
        fraction = fraction,
        standard_scaling = standard_scaling,
        FT=FT,
        read=true,
        preprocess_params_file=preprocess_params_file
    )

    train = first(dl)
    xtrain = train[:,:,1:noised_channels,:] |> device
    ctrain = train[:,:,(noised_channels+1):(noised_channels+context_channels),:] |> device

    # set up model
    checkpoint_path = joinpath(savedir, "checkpoint.bson")
    BSON.@load checkpoint_path model model_smooth opt opt_smooth
    model = device(model)
    return model, xtrain, ctrain, scaling
end

"""
    generate_samples!(samples, init_x, model, context, time_steps, Δt; forward = false)

Generate and fill `samples` in place using Euler-Maruyama sampling, with `init_x` as initial
conditions, `model` as the score-based diffusion model, `context` as the contextual conditioning
fields, `time_steps` as an array of times to step to, and `Δt` the timestep.

The keyword arg `forward` indicates if the integration is carried out in the noising (forward) 
or reverse (denoising) direction.
"""
function generate_samples!(samples, init_x, model, context, time_steps, Δt; forward = false)
    samples .= Euler_Maruyama_sampler(model, init_x, time_steps, Δt; c=context, forward=forward)
    return samples
end

"""
    lowpass_filter(x, k)

Lowpass filters the data `x`, assumed to be structured as
dxdxCxB, where d is the number of spatial pixels, C is the number
of channels, and B is the number of batch members, such that
all power above wavenumber `k` = kx = ky is set to zero.
"""
function lowpass_filter(x, k)
    d = size(x)[1]
    if iseven(k)
        k_ny = Int64(k/2+1)
    else
        k_ny = Int64((k+1)/2)
    end
    FT = eltype(x)
    y = Complex{FT}.(x)
    fft!(y, (1,2));
    # Filter. The indexing here is specific to how `fft!` stores the 
    # Fourier transformed image.
    y[:,k_ny:(d-k_ny),:,:] .= Complex{FT}(0);
    y[k_ny:(d-k_ny),:,:,:] .= Complex{FT}(0);
    ifft!(y, (1,2))
    return real(y)
end

"""
    main(wavenumber; target_toml = "Experiment_512x512.toml", source_toml = "Experiment_64x64.toml", FT=Float32)

Carries out an analysis where high-resolution (target) data samples are low-pass filtered to create a paired 
target-source data set. Plots are created showing examples of the pairs, as well as plotting their power spectral
densities. At that point, we restrict to a single pair, and carry out the downscaling procedure 100x. The 100 downscaled
examples, as well as the true high-resolution/low-resolution pair, are saved to HDF5.
"""
function main(wavenumber; target_toml = "Experiment_512x512.toml", source_toml = "Experiment_64x64.toml", FT=Float32)
    # Some basic setup of params, directories
    stats_savedir = string("stats/paired_analysis/")
    !ispath(stats_savedir) && mkpath(stats_savedir)
    params = TOML.parsefile(target_toml)
    params = CliMAgen.dict2nt(params)
    nsteps = params.sampling.nsteps
    imgsize = params.sampling.imgsize
    context_channels = params.model.context_channels
    noised_channels = params.model.noised_channels
    nbatches = 5
    batchsize = 20
    # Set up device
    nogpu = params.experiment.nogpu
    if !nogpu && CUDA.has_cuda()
        device = Flux.gpu
        @info "Sampling on GPU"
    else
        device = Flux.cpu
        @info "Sampling on CPU"
    end
    
    # Load the model, data, context, and scaling used for preprocessing
    # Note that not all of the data is required for this task, so we don't need to load it all (via `fraction`).
    reverse_model, xtarget, ctarget, scaling_target = unpack_experiment(target_toml, wavenumber; batchsize = batchsize, fraction = 0.125, device = device,FT=FT)
    # create ``paired data`` x_source by low pass filtering the target
    xsource =  lowpass_filter(xtarget, 16)
    # The source data has flat context
    csource = similar(ctarget).*FT(0)

    # We found it helpful to look at some of the images of the target and source pairs.
    # This creates heatmaps of 10 of each, for each channel
    heatmap_grid(cpu(xtarget)[:, :, :, 1:10], 1, stats_savedir, "real_target_images_ch1_$(wavenumber).png"; clims = extrema(xtarget[:,:,[1],:]))
    heatmap_grid(cpu(xsource)[:, :, :, 1:10], 1, stats_savedir, "lowpass_target_images_ch1_$(wavenumber).png"; clims = extrema(xtarget[:,:,[1],:]))
    heatmap_grid(cpu(xtarget)[:, :, :, 1:10], 2, stats_savedir, "real_target_images_ch2_$(wavenumber).png"; clims = extrema(xtarget[:,:,[2],:]))
    heatmap_grid(cpu(xsource)[:, :, :, 1:10], 2, stats_savedir, "lowpass_target_images_ch2_$(wavenumber).png"; clims = extrema(xtarget[:,:,[2],:]))
    
    # Create the mean spectra by channel for the target and source images
    # As in the analyze_downscale_by_batch script, these are used to determine t* (see paper for details)
    target_spectra, k = batch_spectra((xtarget |> cpu))
    source_spectra, k = batch_spectra((xsource |> cpu))
    d = FT(size(xtarget)[1])
    cutoff_idx = 3 # This value was chosen because it is suitable for all contexts except kx=ky=2.
    k_cutoff = FT(k[cutoff_idx])
    target_power_at_cutoff = FT(mean(target_spectra[cutoff_idx,:,:]))
    reverse_t_end = FT(t_cutoff(target_power_at_cutoff, k_cutoff, d, reverse_model.σ_max, reverse_model.σ_min))
    forward_t_end = reverse_t_end
    # We can assess where this lies with respect to the spectra by plotting them
    Plots.plot(log.(k) ./ log(2), source_spectra[:,1,1] , label = "Blurry Truth, Ch 1")
    Plots.plot!(log.(k) ./ log(2), target_spectra[:,1,1], label = "High-res Truth, Ch 1")
    Plots.plot!(log.(k) ./ log(2), source_spectra[:,2,1] , label = "Blurry Truth, Ch 2")
    Plots.plot!(log.(k) ./ log(2), target_spectra[:,2,1], label = "High-res Truth, Ch 2")
    Plots.plot!(xlabel = "n, s.t. k = 2^n", ylabel = "PSD", yaxis = :log10, margin = 10Plots.mm, ylim = [1e-10, 1])
    Plots.plot!(log(k_cutoff)/log(2) .+ [0,0], [1e-10, 1], label = "Cutoff")
    Plots.savefig(joinpath(stats_savedir, "psd_$(wavenumber).png"))


    # We've been taking nsteps for the entire (0,1] timespan, so scale
    # accordingly, since we will be integrating for less time now.
    nsteps =  Int64.(round(forward_t_end*nsteps))

    # Allocate memory for the samples
    nsamples = batchsize
    target_samples= zeros(FT, (imgsize, imgsize, context_channels+noised_channels, nsamples)) |> device
    source_samples= zeros(FT, (imgsize, imgsize, context_channels+noised_channels, nsamples)) |> device
    real_target_samples= zeros(FT, (imgsize, imgsize, context_channels+noised_channels, nsamples)) |> device

    init_x_reverse =  zeros(FT, (imgsize, imgsize, noised_channels, nsamples)) |> device
    # Set up timesteps for both forward and reverse
    t_forward = zeros(FT, nsamples) .+ forward_t_end |> device
    time_steps_forward = LinRange(FT(1.0f-5),FT(forward_t_end), nsteps)
    Δt_forward = time_steps_forward[2] - time_steps_forward[1]

    t_reverse = zeros(FT, nsamples) .+ reverse_t_end |> device
    time_steps_reverse = LinRange(FT(reverse_t_end), FT(1.0f-5), nsteps)
    Δt_reverse = time_steps_reverse[1] - time_steps_reverse[2]


    # In this exercise, we want multiple downscaled examples of the same source image,
    # so here we replace xsource by multiple copies of the first example.
    # We replace the target context by multiple copies of the first example as well.
    # That means we will be downscaling the low-pass version of xtarget[:,:,:,1] = xsource[:,:,:,1]
    # using ctarget[:,:,:,1] multiple times.
    xsource = cat([xsource[:,:,:,[1]] for i in 1:nsamples]..., dims = 4)
    ctarget =  cat([ctarget[:,:,:,[1]] for i in 1:nsamples]..., dims = 4)

    samples_file = "samples_$(wavenumber).hdf5"
    hdf5_path = joinpath(stats_savedir, samples_file)
    fid = HDF5.h5open(hdf5_path, "w")
    for batch in 1:nbatches
        # Integrate forwards to fill init_x_reverse in place
        # The IC for this step are xsource[:,:,1:noised_channels,:]
        # Context is passed in as a separate field.
        init_x_reverse .= generate_samples!(init_x_reverse,
                                            xsource[:,:,1:noised_channels,:], #forward IC
                                            reverse_model,
                                            csource[:,:,:,:],# forward context
                                            time_steps_forward,
                                            Δt_forward;
                                            forward = true);
        
        # Integrate backwards to fill the noised channels of target_samples in place.
        # Since we do this by wavenumber, all the target context are the same
        target_samples[:,:,1:noised_channels,:] .= generate_samples!(target_samples[:,:,1:noised_channels,:],
                                                                init_x_reverse,
                                                                reverse_model,
                                                                ctarget[:,:,:,1:nsamples],
                                                                time_steps_reverse,
                                                                Δt_reverse;
                                                                forward = false);
        if batch == 1
            # Save some examples to look at
            heatmap_grid(cpu(target_samples)[:, :, :, 1:10], 1, stats_savedir, "fake_target_images_ch1_$(wavenumber).png"; clims = extrema(xtarget[:,:,[1],:]))
            heatmap_grid(cpu(target_samples)[:, :, :, 1:10], 2, stats_savedir, "fake_target_images_ch2_$(wavenumber).png"; clims = extrema(xtarget[:,:,[2],:]))
        end

        # Carry out the inverse preprocessing transform to go back to real space
        # Preprocessing acts on both noised and context channels
        target_samples[:,:,(noised_channels+1):(noised_channels+context_channels),:] .= ctarget;
        # Save to HDF5
        fid["downscaled_samples_$(batch)"] = invert_preprocessing(cpu(target_samples), scaling_target)[:,:,1:2,:]
    end
    # Carry out the inverse preprocessing transform to go back to real space
    # Preprocessing acts on both noised and context channels
    source_samples[:,:,1:noised_channels,:] .= xsource[:,:,1:noised_channels,:];
    source_samples[:,:,(noised_channels+1):(noised_channels+context_channels),:] .= csource;
    source_samples = invert_preprocessing(cpu(source_samples), scaling_target)

    real_target_samples[:,:,1:noised_channels,:] .= xtarget[:,:,1:noised_channels,:];
    real_target_samples[:,:,(noised_channels+1):(noised_channels+context_channels),:] .= ctarget;
    real_target_samples = invert_preprocessing(cpu(real_target_samples), scaling_target)

    # In this particular analysis, these are all the same, so just save the first example of each
    fid["original_samples"] = real_target_samples[:,:,1:2,[1]]
    fid["fake_lowres_samples"] = source_samples[:,:,1:2,[1]]
    close(fid)
    
end

