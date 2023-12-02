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

using CliMAgen
package_dir = pkgdir(CliMAgen)
include(joinpath(package_dir,"examples/utils_data.jl"))
include(joinpath(package_dir,"examples/utils_analysis.jl"))

function obtain_train_dl(params, wavenumber, FT)
    # unpack params
    savedir = params.experiment.savedir
    batchsize = params.data.batchsize
    resolution = params.data.resolution
    fraction::FT = params.data.fraction
    standard_scaling  = params.data.standard_scaling
    preprocess_params_file = joinpath(savedir, "preprocessing_standard_scaling_$standard_scaling.jld2")

    noised_channels = params.model.noised_channels
    context_channels = params.model.context_channels
    
    # set up dataset - we need this in order to get the context
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
    
    return dl
end

function main(npixels, wavenumber, experiment_toml; FT=Float32, mean_std_datafile = "./train_means_stds.csv")    
    # read experiment parameters from file
    params = TOML.parsefile(experiment_toml)
    params = CliMAgen.dict2nt(params)
    batchsize = params.data.batchsize
    rngseed = params.experiment.rngseed

    # set up rng
    rngseed > 0 && Random.seed!(rngseed)

    dl  = obtain_train_dl(params, wavenumber, FT)
    resolution = params.data.resolution
    noised_channels = params.model.noised_channels
    savedir = params.experiment.savedir
    standard_scaling  = params.data.standard_scaling
    preprocess_params_file = joinpath(savedir, "preprocessing_standard_scaling_$standard_scaling.jld2")
    scaling = JLD2.load_object(preprocess_params_file)
    imgsize = params.sampling.imgsize

    # set up directory for saving stats, and filenames
    stats_savedir = string("stats/",string(resolution),"x", string(resolution),"/train")
    !ispath(stats_savedir) && mkpath(stats_savedir)
    filenames = [joinpath(stats_savedir, "train_statistics_ch1_$wavenumber.csv"),joinpath(stats_savedir, "train_statistics_ch2_$wavenumber.csv")]
    corr_filename = joinpath(stats_savedir, "corr_spectrum_$wavenumber.csv")
    pixel_filenames = [joinpath(stats_savedir, "train_pixels_ch1_$wavenumber.csv"),joinpath(stats_savedir, "train_pixels_ch2_$wavenumber.csv")]
    train_pixels = zeros(FT,(imgsize*imgsize, noised_channels, batchsize))

    # Read in mean and std of data from training data; used in correlation statistics
    means_stds = readdlm(mean_std_datafile, ',')
    means_stds = means_stds[(means_stds[:,2] .==wavenumber) .& (means_stds[:,1] .==resolution),:]
    μ1 = means_stds[3]
    μ2 = means_stds[4]
    σ1 = means_stds[5]
    σ2 = means_stds[6]
    for batch in dl
        # revert to real space using the inverse preprocessing step
        batch .= invert_preprocessing(batch, scaling)

        # compute metrics of interest
        train_means =  mapslices(Statistics.mean, batch, dims=[1, 2])
        train_κ2 = Statistics.var(batch, dims = (1,2))
        train_κ3 = mapslices(x -> StatsBase.cumulant(x[:],3), batch, dims=[1, 2])
        train_κ4 = mapslices(x -> StatsBase.cumulant(x[:],4), batch, dims=[1, 2])
        train_spectra = mapslices(x -> hcat(power_spectrum2d(x)[1]), batch, dims =[1,2])

        # correlation field
        corr = (batch[:,:,[1],:] .- μ1).*(batch[:,:,[2],:] .- μ2) ./ (σ1*σ2)
        corr_spectra = mapslices(x -> hcat(power_spectrum2d(x)[1]), corr, dims =[1,2])

        # average instant condensation rate
        train_icr = make_icr(batch)

        # batch is imgsize x imgsize x nchannels x batchsize, except possibly on the last batch which might be smaller.
        current_batchsize = size(batch)[end]
        train_pixels[:,:,1:current_batchsize] .= reshape(batch[:,:, 1:noised_channels, :], (prod(size(batch)[1:2]), noised_channels, current_batchsize))
        pixel_indices = StatsBase.sample(1:1:size(train_pixels)[1], npixels)

        open(corr_filename, "a") do io
            writedlm(io, transpose(corr_spectra[:,1,1,:]), ',')
        end
        for ch in 1:noised_channels
            # write pixel vaues to other file
            open(pixel_filenames[ch],"a") do io
                writedlm(io, transpose(train_pixels[pixel_indices, ch, 1:current_batchsize]), ',')
            end

            if ch == 1
                output = hcat(train_means[1,1,ch,:],train_κ2[1,1,ch,:], train_κ3[1,1,ch,:],train_κ4[1,1,ch,:], transpose(train_spectra[:,1,ch,:]), train_icr[1,1,ch,:])
            else
                output = hcat(train_means[1,1,ch,:],train_κ2[1,1,ch,:], train_κ3[1,1,ch,:],train_κ4[1,1,ch,:], transpose(train_spectra[:,1,ch,:]))
            end
            open(filenames[ch], "a") do io
                writedlm(io, output, ',')
            end
        end
    end
    
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(parse(Int64, ARGS[1]), parse(Float32, ARGS[2]), ARGS[3])
end
