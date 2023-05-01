using JLD2
using FFTW
using MLDatasets
using MLUtils
using DataLoaders
using Statistics
using CliMADatasets
using CliMAgen: expand_dims
using Random

"""
    get_data_context2dturbulence(batchsize;
                                 rng=Random.GLOBAL_RNG,
                                 resolution = 512,
                                 wavenumber = 0.0,
                                 fraction = 1.0,
                                 standard_scaling = false,
                                 FT = Float32,
                                 read = false,
                                 save = false,
                                 preprocess_params_file)

Obtains the raw data from the 2D turbulence with context dataset,
carries out a scaling of the data, and loads the data into train and test
dataloders, which are returned.

The user can pick:
- resolution:       (64 or 512)
- wavenumber:       (0 = all wavenumbers, supported for both resolutions
                    or, 2,4,8,16, supported only for 512 resolution.)
- fraction:         the amount of the data to use. Must be of the form 1/integer.
- standard_scaling: boolean indicating if standard minmax scaling is used
                    or if minmax scaling of the mean and spatial variations
                    are both implemented.
- FT:               the float type of the model
- read:             a boolean indicating if the preprocessing parameters should be read
- save:             a boolean indicating if the preprocessing parameters should be
                    computed and read.
- preprocess_params:filename where preprocessing parameters are stored or read from.

If a resolution of 64 is chosen, the raw data is upsampled to 512x512 using
nearest-neighbors, and then low-pass filtered.
"""
function get_data_context2dturbulence(batchsize;
                                      rng=Random.GLOBAL_RNG,
                                      resolution=512,
                                      wavenumber=0.0,
                                      fraction = 1.0,
                                      standard_scaling = false,
                                      FT=Float32,
                                      read = false,
                                      save = false,
                                      preprocess_params_file)
    @assert xor(read, save)
    @assert resolution ∈ [512, 64]
    if resolution == 512
        @assert wavenumber ∈ FT.([0, 1, 2, 4, 8, 16])
    elseif resolution == 64
        @assert wavenumber ∈ FT.([0, 1])
    end

    if wavenumber == FT(0) # Returns all the data, for every wavenumber
        xtrain = CliMADatasets.Turbulence2DContext(:train; fraction = fraction, resolution=resolution, wavenumber = :all, Tx=FT,)[:]
        xtest = CliMADatasets.Turbulence2DContext(:test; fraction = fraction, resolution=resolution, wavenumber = :all, Tx=FT,)[:]
    else # Returns data for a specific wavenumber only
        xtrain = CliMADatasets.Turbulence2DContext(:train; fraction = fraction, resolution=resolution, wavenumber = wavenumber, Tx=FT,)[:]
        xtest = CliMADatasets.Turbulence2DContext(:test; fraction = fraction, resolution=resolution, wavenumber = wavenumber, Tx=FT,)[:]
    end

    if resolution == 64
        # Upsampling
        upsample = Flux.Upsample(8, :nearest)
        xtrain_upsampled = Complex{FT}.(upsample(xtrain))
        xtest_upsampled = Complex{FT}.(upsample(xtest));
        # Upsampling produces artifacts at high frequencies, so now
        # we filter.
        fft!(xtrain_upsampled, (1,2));
        xtrain_upsampled[:,33:479,:,:] .= Complex{FT}(0);
        xtrain_upsampled[33:479,:,:,:] .= Complex{FT}(0);
        ifft!(xtrain_upsampled, (1,2))
        xtrain = real(xtrain_upsampled)

        fft!(xtest_upsampled, (1,2));
        xtest_upsampled[:,33:479,:,:] .= Complex{FT}(0);
        xtest_upsampled[33:479,:,:,:] .= Complex{FT}(0);
        ifft!(xtest_upsampled, (1,2))
        xtest = real(xtest_upsampled)
    end
    
    if save
        if standard_scaling
            maxtrain = maximum(xtrain, dims=(1, 2, 4))
            mintrain = minimum(xtrain, dims=(1, 2, 4))
            Δ = maxtrain .- mintrain
            # To prevent dividing by zero
            Δ[Δ .== 0] .= FT(1)
            scaling = StandardScaling{FT}(mintrain, Δ)
        else
            #scale means and spatial variations separately
            x̄ = mean(xtrain, dims=(1, 2))
            maxtrain_mean = maximum(x̄, dims=4)
            mintrain_mean = minimum(x̄, dims=4)
            Δ̄ = maxtrain_mean .- mintrain_mean
            xp = xtrain .- x̄
            maxtrain_p = maximum(xp, dims=(1, 2, 4))
            mintrain_p = minimum(xp, dims=(1, 2, 4))
            Δp = maxtrain_p .- mintrain_p

            # To prevent dividing by zero
            Δ̄[Δ̄ .== 0] .= FT(1)
            Δp[Δp .== 0] .= FT(1)
            scaling = MeanSpatialScaling{FT}(mintrain_mean, Δ̄, mintrain_p, Δp)
        end
        JLD2.save_object(preprocess_params_file, scaling)
    elseif read
        scaling = JLD2.load_object(preprocess_params_file)
    end
    xtrain .= apply_preprocessing(xtrain, scaling)
    # apply the same rescaler as on training set
    xtest .= apply_preprocessing(xtest, scaling)

    xtrain = MLUtils.shuffleobs(rng, xtrain)
    loader_train = DataLoaders.DataLoader(xtrain, batchsize)
    loader_test = DataLoaders.DataLoader(xtest, batchsize)

    return (; loader_train, loader_test)
end
