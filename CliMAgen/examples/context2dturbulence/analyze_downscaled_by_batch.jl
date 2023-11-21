using BSON
using CUDA
using Flux
using ProgressMeter
using Random
using Statistics
using TOML
using DelimitedFiles

using CliMAgen
package_dir = pkgdir(CliMAgen)
include(joinpath(package_dir,"examples/utils_data.jl"))
include(joinpath(package_dir,"examples/utils_analysis.jl"))

function unpack_experiment(experiment_toml, wavenumber; device = Flux.gpu, FT=Float32)
    params = TOML.parsefile(experiment_toml)
    params = CliMAgen.dict2nt(params)

    # unpack params
    savedir = params.experiment.savedir
    batchsize = params.data.batchsize
    resolution = params.data.resolution
    fraction::FT = params.data.fraction
    standard_scaling  = params.data.standard_scaling
    preprocess_params_file = joinpath(savedir, "preprocessing_standard_scaling_$standard_scaling.jld2")
    scaling = JLD2.load_object(preprocess_params_file)
    noised_channels = params.model.noised_channels
    context_channels = params.model.context_channels

    if resolution == 64
        # Our coarse res data has flat context. If we did have nonzero context fields,
        # we'd have to be careful to get the right wavenumber
        # so that the diffusion bridge is done for the same context in both directions.
        dl, _ =  get_data_context2dturbulence(
            batchsize;
            resolution = resolution,
            wavenumber = 0,
            fraction = fraction,
            standard_scaling = standard_scaling,
            FT=FT,
            save=true,
            preprocess_params_file=preprocess_params_file
        )
    else
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
    end
    train = cat([x for x in dl]..., dims=4)

    xtrain = train[:,:,1:noised_channels,:] |> device
    ctrain = train[:,:,(noised_channels+1):(noised_channels+context_channels),:] |> device

    # set up model
    checkpoint_path = joinpath(savedir, "checkpoint.bson")
    BSON.@load checkpoint_path model model_smooth opt opt_smooth
    model = device(model)
    return model, xtrain, ctrain, scaling
end

function generate_samples!(samples, init_x, model, context, time_steps, Δt; forward = false)
    samples .= Euler_Maruyama_sampler(model, init_x, time_steps, Δt; c=context, forward=forward)
    return samples
end

function hard_filter(x, k)
    d = size(x)[1]
    if iseven(k)
        k_ny = Int64(k/2+1)
    else
        k_ny = Int64((k+1)/2)
    end
    FT = eltype(x)
    y = Complex{FT}.(x)
    # Filter.
    fft!(y, (1,2));
    y[:,k_ny:(d-k_ny),:,:] .= Complex{FT}(0);
    y[k_ny:(d-k_ny),:,:,:] .= Complex{FT}(0);
    ifft!(y, (1,2))
    return real(y)
end

function main(nbatches, npixels, wavenumber; source_toml = "Experiment_64x64.toml",
                                             target_toml = "Experiment_512x512.toml",
                                             FT=Float32,
                                             mean_std_datafile = "./train_means_stds.csv")
    stats_savedir = string("stats/512x512/downscale_gen")
    !ispath(stats_savedir) && mkpath(stats_savedir)
    
    params = TOML.parsefile(target_toml)
    params = CliMAgen.dict2nt(params)
    nsamples = params.sampling.nsamples
    nsteps = params.sampling.nsteps
    imgsize = params.sampling.imgsize
    context_channels = params.model.context_channels
    noised_channels = params.model.noised_channels
    # set up device
    nogpu = params.experiment.nogpu
    if !nogpu && CUDA.has_cuda()
        device = Flux.gpu
        @info "Sampling on GPU"
    else
        device = Flux.cpu
        @info "Sampling on CPU"
    end
    
    forward_model, xsource, csource, scaling_source = unpack_experiment(source_toml, wavenumber; device = device, FT=FT)
    reverse_model, xtarget, ctarget, scaling_target = unpack_experiment(target_toml, wavenumber; device = device,FT=FT)

    # Determine the turnaround time for the diffusion bridge,
    # corresponding to the power spectral density at which
    # the two sets of images begin to differ from each other
    source_spectra, k = batch_spectra((xsource |> cpu)[:,:,:,1:20])
    target_spectra, k = batch_spectra((xtarget |> cpu)[:,:,:,1:20])
    N = FT(size(xsource)[1])
    cutoff_idx = min(3,Int64(floor(sqrt(2)*wavenumber-1)))
    if cutoff_idx > 0
        k_cutoff = FT(k[cutoff_idx])
    	source_power_at_cutoff = FT(mean(source_spectra[cutoff_idx,:,:]))
    	target_power_at_cutoff = FT(mean(target_spectra[cutoff_idx,:,:]))
        reverse_t_end = FT(t_cutoff(target_power_at_cutoff, k_cutoff, N, reverse_model.σ_max, reverse_model.σ_min))
        forward_t_end = FT(t_cutoff(source_power_at_cutoff, k_cutoff, N, reverse_model.σ_max, reverse_model.σ_min))
    else
	reverse_t_end = FT(1)
	forward_t_end = FT(1)
    end
    @show forward_t_end
    @show reverse_t_end
    # We've been taking nsteps for the entire (0,1] timespan, so scale
    # accordingly, since we will be integrating for less time now.
    nsteps =  Int64.(round(forward_t_end*nsteps))

    # Allocate memory for the samples and their pixel values
    target_samples= zeros(FT, (imgsize, imgsize, context_channels+noised_channels, nsamples)) |> device
    lo_res_target_samples = zeros(FT, (imgsize, imgsize, context_channels+noised_channels, nsamples)) |> device
    sample_pixels = reshape(target_samples[:,:, 1:noised_channels, :], (prod(size(target_samples)[1:2]), noised_channels, nsamples))
    source_samples = zeros(FT, (imgsize, imgsize, context_channels+noised_channels, nsamples)) |> device
    corr =  zeros(FT, (imgsize, imgsize, 1, nsamples))
    # Allocate memory for the noised channels at t = t_end
    init_x_reverse =  zeros(FT, (imgsize, imgsize, noised_channels, nsamples)) |> device

    # Set up timesteps for both forward and reverse
    t_forward = zeros(FT, nsamples) .+ forward_t_end |> device
    time_steps_forward = LinRange(FT(1.0f-5),FT(forward_t_end), nsteps)
    Δt_forward = time_steps_forward[2] - time_steps_forward[1]

    t_reverse = zeros(FT, nsamples) .+ reverse_t_end |> device
    time_steps_reverse = LinRange(FT(reverse_t_end), FT(1.0f-5), nsteps)
    Δt_reverse = time_steps_reverse[1] - time_steps_reverse[2]

    # Filenames for saving
    filenames = [joinpath(stats_savedir, "downscale_gen_statistics_ch1_$wavenumber.csv"),
                 joinpath(stats_savedir, "downscale_gen_statistics_ch2_$wavenumber.csv")]
    corr_filename = joinpath(stats_savedir, "corr_spectrum_$wavenumber.csv")

    pixel_filenames = [joinpath(stats_savedir, "downscale_gen_pixels_ch1_$wavenumber.csv"),joinpath(stats_savedir, "downscale_gen_pixels_ch2_$wavenumber.csv")]
    L2_filenames = [joinpath(stats_savedir, "downscale_gen_L2_ch1_$wavenumber.csv"),joinpath(stats_savedir, "downscale_gen_L2_ch2_$wavenumber.csv")]
    # Read in mean and std of data from training data; used in correlation statistics
    means_stds = readdlm(mean_std_datafile, ',')
    means_stds = means_stds[(means_stds[:,2] .==wavenumber) .& (means_stds[:,1] .==512),:]
    μ1 = means_stds[3]
    μ2 = means_stds[4]
    σ1 = means_stds[5]
    σ2 = means_stds[6]

    # Obtain the indices of the source data.
    # We'll randomly sample from this for the initial conditions of the forward model.
    indices = 1:1:size(csource)[end]
    for batch in 1:nbatches
        @info batch
        selection = StatsBase.sample(indices, nsamples)
        # Integrate forwards to fill init_x_reverse in place
        # The IC for this step are xsource[:,:,1:noised_channels,selection]
        # Context is passed in as a separate field.
        init_x_reverse .= generate_samples!(init_x_reverse,
                                            xsource[:,:,1:noised_channels,selection], #forward IC
                                            forward_model,
                                            csource[:,:,:,selection],# forward context
                                            time_steps_forward,
                                            Δt_forward;
                                            forward = true);

        # If the models use different sigma schedules or if the times are different, we need to adjust
        adapt_x!(init_x_reverse, forward_model, reverse_model, forward_t_end, reverse_t_end);
        
        # Integrate backwards to fill the noised channels of target_samples in place.
        # Since we do this by wavenumber, all the target context are the same
        target_samples[:,:,1:noised_channels,:] .= generate_samples!(target_samples[:,:,1:noised_channels,:],
                                                              init_x_reverse,
                                                              reverse_model,
                                                              ctarget[:,:,:,1:nsamples],
                                                              time_steps_reverse,
                                                              Δt_reverse;
                                                              forward = false);
        

        # Carry out the inverse preprocessing transform to go back to real space
        # Preprocessing acts on both noised and context channels
        target_samples[:,:,(noised_channels+1):(noised_channels+context_channels),:] .= ctarget[:,:,:,1:nsamples];
        target_samples = invert_preprocessing(cpu(target_samples), scaling_target) |> device;

        source_samples[:,:,1:noised_channels,:] .= xsource[:,:,1:noised_channels,selection];
        source_samples[:,:,(noised_channels+1):(noised_channels+context_channels),:] .= csource[:,:,:,selection];
        source_samples = invert_preprocessing(cpu(source_samples), scaling_source) |> device

        # compute metrics of interest
        sample_means =  mapslices(Statistics.mean, cpu(target_samples), dims=[1, 2])
        sample_κ2 = Statistics.var(cpu(target_samples), dims = (1,2))
        sample_κ3 = mapslices(x -> StatsBase.cumulant(x[:],3), cpu(target_samples), dims=[1, 2])
        sample_κ4 = mapslices(x -> StatsBase.cumulant(x[:],4), cpu(target_samples), dims=[1, 2])
        sample_spectra = mapslices(x -> hcat(power_spectrum2d(x)[1]), cpu(target_samples), dims =[1,2])
        # Difference between filtered high res and true low res source
        # Since the lo res is biased, especially in the tracer, we cant
        # just compare hi res to low res. so first we filter & normalize.
        # We assume the statistics over all pixels in 10 images is sufficient.
        μ_target = Statistics.mean(target_samples, dims = (1,2,4))
        μ_source = Statistics.mean(source_samples, dims = (1,2,4))
        σ_target = Statistics.std(target_samples, dims = (1,2,4))
        σ_source = Statistics.std(source_samples, dims = (1,2,4))
        lo_res_target_samples .= hard_filter((target_samples .- μ_target) ./ σ_target , k[cutoff_idx])
        source_samples .= hard_filter((source_samples .- μ_source) ./ σ_source, k[cutoff_idx])

        L2_hi_true_lo =sqrt.(Statistics.mean(cpu((source_samples[:,:,1:noised_channels,:] .- lo_res_target_samples[:,:,1:noised_channels,:]).^2), dims = [1,2]));
        # Difference between filtered high res and random low res
        L2_hi_random_lo =sqrt.(Statistics.mean(cpu((source_samples[:,:,1:noised_channels,randcycle(nsamples)] .- lo_res_target_samples[:,:,1:noised_channels,:]).^2), dims = [1,2]));

        # average instant condensation rate
        sample_icr = make_icr(cpu(target_samples))

        # samples pixels
        sample_pixels .= reshape(target_samples[:,:, 1:noised_channels, :], (prod(size(target_samples)[1:2]), noised_channels, nsamples))
        pixel_indices = StatsBase.sample(1:1:size(sample_pixels)[1], npixels)

        # pixelwise correlation
        # demean using values from training data
        # correlation field
        corr .= (cpu(target_samples)[:,:,[1],:] .- μ1).*(cpu(target_samples)[:,:,[2],:] .- μ2) ./ (σ1*σ2)
        corr_spectra = mapslices(x -> hcat(power_spectrum2d(x)[1]), corr, dims =[1,2])
        #save the metrics
        open(corr_filename, "a") do io
            writedlm(io, transpose(corr_spectra[:,1,1,:]), ',')
        end
        for ch in 1:noised_channels
            # write pixel vaues to pixel file
            open(pixel_filenames[ch],"a") do io
                writedlm(io, transpose(cpu(sample_pixels)[pixel_indices, ch, :]), ',')
            end
            # write L2 vaues to L2 file
            open(L2_filenames[ch],"a") do io
                output = hcat(L2_hi_true_lo[1,1,ch,:], L2_hi_random_lo[1,1,ch,:])
                writedlm(io, output, ',')
            end

            if ch == 1
                output = hcat(sample_means[1,1,ch,:],sample_κ2[1,1,ch,:], sample_κ3[1,1,ch,:],sample_κ4[1,1,ch,:], transpose(sample_spectra[:,1,ch,:]), sample_icr[1,1,ch,:])
            else
                output = hcat(sample_means[1,1,ch,:],sample_κ2[1,1,ch,:], sample_κ3[1,1,ch,:],sample_κ4[1,1,ch,:], transpose(sample_spectra[:,1,ch,:]))
            end
            open(filenames[ch], "a") do io
                writedlm(io, output, ',')
            end
        end
    end
    
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(parse(Int64, ARGS[1]), parse(Int64, ARGS[2]), parse(Float32, ARGS[3]))
end
