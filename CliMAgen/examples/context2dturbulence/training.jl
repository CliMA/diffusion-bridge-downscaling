using BSON
using CUDA
using Flux
using Random
using TOML

using CliMAgen
using CliMAgen: dict2nt
using CliMAgen: VarianceExplodingSDE, NoiseConditionalScoreNetwork
using CliMAgen: score_matching_loss
using CliMAgen: WarmupSchedule, ExponentialMovingAverage
using CliMAgen: train!, load_model_and_optimizer

package_dir = pkgdir(CliMAgen)
include(joinpath(package_dir,"examples/utils_data.jl")) # for data loading

function run_training(params; FT=Float32)
    # unpack params
    savedir = params.experiment.savedir
    rngseed = params.experiment.rngseed
    nogpu = params.experiment.nogpu

    batchsize = params.data.batchsize
    resolution = params.data.resolution
    wavenumber::FT = params.data.wavenumber
    fraction::FT = params.data.fraction
    standard_scaling  = params.data.standard_scaling
    preprocess_params_file = joinpath(savedir, "preprocessing_standard_scaling_$standard_scaling.jld2")

    sigma_min::FT = params.model.sigma_min
    sigma_max::FT = params.model.sigma_max
    dropout_p::FT = params.model.dropout_p
    noised_channels = params.model.noised_channels
    context_channels = params.model.context_channels
    shift_input = params.model.shift_input
    shift_output = params.model.shift_output
    mean_bypass = params.model.mean_bypass
    scale_mean_bypass = params.model.scale_mean_bypass
    gnorm = params.model.gnorm
    proj_kernelsize = params.model.proj_kernelsize
    outer_kernelsize = params.model.outer_kernelsize
    middle_kernelsize = params.model.middle_kernelsize
    inner_kernelsize = params.model.inner_kernelsize

    nwarmup = params.optimizer.nwarmup
    gradnorm::FT = params.optimizer.gradnorm
    learning_rate::FT = params.optimizer.learning_rate
    beta_1::FT = params.optimizer.beta_1
    beta_2::FT = params.optimizer.beta_2
    epsilon::FT = params.optimizer.epsilon
    ema_rate::FT = params.optimizer.ema_rate

    nepochs = params.training.nepochs
    freq_chckpt = params.training.freq_chckpt

    # set up rng
    rngseed > 0 && Random.seed!(rngseed)

    # set up device
    if !nogpu && CUDA.has_cuda()
        device = Flux.gpu
        @info "Training on GPU"
    else
        device = Flux.cpu
        @info "Training on CPU"
    end

    # set up dataset
    dataloaders = get_data_context2dturbulence(
        batchsize;
        resolution = resolution,
        wavenumber = wavenumber,
        fraction = fraction,
        standard_scaling = standard_scaling,
        FT=FT,
        save=true,
        preprocess_params_file=preprocess_params_file
    )

    # set up model and optimizers
    checkpoint_path = joinpath(savedir, "checkpoint.bson")
    loss_file = joinpath(savedir, "losses.txt")
    
    if isfile(checkpoint_path) && isfile(loss_file)
        BSON.@load checkpoint_path model model_smooth opt opt_smooth
        model = device(model)
        model_smooth = device(model_smooth)
        loss_data = DelimitedFiles.readdlm(loss_file, ',', skipstart = 1)
        start_epoch = loss_data[end,1]+1
    else
        net = NoiseConditionalScoreNetwork(;
                                           context = true,
                                           dropout_p = dropout_p,
                                           noised_channels = noised_channels,
                                           context_channels = context_channels,
                                           shift_input = shift_input,
                                           shift_output = shift_output,
                                           mean_bypass = mean_bypass,
                                           scale_mean_bypass = scale_mean_bypass,
                                           gnorm = gnorm,
                                           proj_kernelsize = proj_kernelsize,
                                           outer_kernelsize = outer_kernelsize,
                                           middle_kernelsize = middle_kernelsize,
                                           inner_kernelsize = inner_kernelsize
                                           )
        model = VarianceExplodingSDE(sigma_max, sigma_min, net)
        model = device(model)
        model_smooth = deepcopy(model)
    
        opt = Flux.Optimise.Optimiser(
            WarmupSchedule{FT}(
                nwarmup 
            ),
            Flux.Optimise.ClipNorm(gradnorm),
            Flux.Optimise.Adam(
                learning_rate,
                (beta_1, beta_2),
                epsilon
            )
        )
        opt_smooth = ExponentialMovingAverage(ema_rate)

        # set up loss file
        loss_names = reshape(["#Epoch", "Mean Train", "Spatial Train","Mean Test","Spatial Test"], (1,5))
        open(loss_file,"w") do io
             DelimitedFiles.writedlm(io, loss_names,',')
        end

        start_epoch=1
    end

    # set up loss function
    function lossfn(y; noised_channels = noised_channels, context_channels=context_channels)
        x = y[:,:,1:noised_channels,:]
        c = y[:,:,(noised_channels+1):(noised_channels+context_channels),:]
        return score_matching_loss(model, x; c = c)
    end

    # train the model
    train!(
        model,
        model_smooth,
        lossfn,
        dataloaders,
        opt,
        opt_smooth,
        nepochs,
        device;
        start_epoch = start_epoch,
        savedir = savedir,
        freq_chckpt = freq_chckpt,
    )
end

function main("Experiment.toml")
    FT = Float32

    # read experiment parameters from file
    params = TOML.parsefile(experiment_toml)
    params = CliMAgen.dict2nt(params)

    # set up directory for saving checkpoints
    !ispath(params.experiment.savedir) && mkpath(params.experiment.savedir)

    run_training(params; FT=FT)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS[1])
end
