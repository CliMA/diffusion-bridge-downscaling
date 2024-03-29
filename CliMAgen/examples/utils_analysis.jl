using StatsBase
using Statistics
using CUDA
using Random
using FFTW
using Plots
using CliMAgen


"""
    batch_spectra(data)

Computes and returns the mean azimuthally averaged power 
spectrum for the data, where the mean is taken
over the batch dimension,
but not over the channel dimension.
"""
function batch_spectra(data)
    statistics = x -> hcat(power_spectrum2d(x)...)
    data = data |> Flux.cpu
    results = mapslices(statistics, data, dims=[1, 2])
    k = results[:, 2, 1, 1]
    results = results[:, 1, :, :]
    spectrum = mean(results, dims=3)
    return spectrum, k
end

"""
    power_spectrum2d(img)

Adapted from https://github.com/CliMA/ClimateMachine.jl/blob/master/src/Common/Spectra/power_spectrum_les.jl
for two spatial dimensions.

Inputs need to be equi-spaced and the domain is assumed to be the same size and
have the same number of points in all directions.

# Arguments
- img: a 2 dimension matrix of size (N, N).

# Returns
 - spectrum, wavenumber
"""
function power_spectrum2d(img)
    @assert size(img)[1] == size(img)[2]
    dim = size(img)[1]
    img_fft = abs.(fft(img .- mean(img)))
    m = Array(img_fft / size(img_fft, 1)^2)
    if mod(dim, 2) == 0
        rx = range(0, stop=dim - 1, step=1) .- dim / 2 .+ 1
        ry = range(0, stop=dim - 1, step=1) .- dim / 2 .+ 1
        R_x = circshift(rx', (1, dim / 2 + 1))
        R_y = circshift(ry', (1, dim / 2 + 1))
        k_nyq = dim / 2
    else
        rx = range(0, stop=dim - 1, step=1) .- (dim - 1) / 2
        ry = range(0, stop=dim - 1, step=1) .- (dim - 1) / 2
        R_x = circshift(rx', (1, (dim + 1) / 2))
        R_y = circshift(ry', (1, (dim + 1) / 2))
        k_nyq = (dim - 1) / 2
    end
    r = zeros(size(rx, 1), size(ry, 1))
    for i in 1:size(rx, 1), j in 1:size(ry, 1)
        r[i, j] = sqrt(R_x[i]^2 + R_y[j]^2)
    end
    k = range(1, stop=k_nyq, step=1)
    endk = size(k, 1)
    contribution = zeros(endk)
    spectrum = zeros(endk)
    for N in 2:Int64(k_nyq - 1)
        for i in 1:size(rx, 1), j in 1:size(ry, 1)
            if (r[i, j] <= (k'[N+1] + k'[N]) / 2) &&
               (r[i, j] > (k'[N] + k'[N-1]) / 2)
                spectrum[N] =
                    spectrum[N] + m[i, j]^2
                contribution[N] = contribution[N] + 1
            end
        end
    end
    for i in 1:size(rx, 1), j in 1:size(ry, 1)
        if (r[i, j] <= (k'[2] + k'[1]) / 2)
            spectrum[1] =
                spectrum[1] + m[i, j]^2
            contribution[1] = contribution[1] + 1
        end
    end
    for i in 1:size(rx, 1), j in 1:size(ry, 1)
        if (r[i, j] <= k'[endk]) &&
           (r[i, j] > (k'[endk] + k'[endk-1]) / 2)
            spectrum[endk] =
                spectrum[endk] + m[i, j]^2
            contribution[endk] = contribution[endk] + 1
        end
    end
    spectrum = spectrum ./ contribution

    return spectrum, k
end

"""
    t_cutoff(power::FT, k::FT, N::FT, σ_max::FT, σ_min::FT) where {FT}

Computes and returns the time `t` at which the power of 
the radially averaged Fourier spectrum of white noise of size NxN, 
with variance σ_min^2(σ_max/σ_min)^(2t), at wavenumber `k`,
is equal to `power`.
"""
function t_cutoff(power::FT, k::FT, N::FT, σ_max::FT, σ_min::FT) where {FT}
    return 1/2*log(power*N^2/σ_min^2)/log(σ_max/σ_min)
end


"""
    adapt_x!(x,
            forward_model::CliMAgen.VarianceExplodingSDE,
            reverse_model::CliMAgen.VarianceExplodingSDE,
            forward_t_end::FT,
            reverse_t_end::FT) where{FT}

Adapts the state `x` produced by diffusion to `forward_t_end`
from `t=0`, using the `forward_model` to an equivalent state produced by
`reverse_model`` after integrating to `reverse_t_end` from `t=1`.

Useful for diffusion bridges between datasets generated by 
Variance Exploding SDE models with different values of
σ_max and σ_min.
"""
function adapt_x!(x,
                 forward_model::CliMAgen.VarianceExplodingSDE,
                 reverse_model::CliMAgen.VarianceExplodingSDE,
                 forward_t_end::FT,
                 reverse_t_end::FT) where{FT}
    _, forward_σ_end = CliMAgen.marginal_prob(forward_model, x, FT(forward_t_end)) # x only affects the mean, which we dont use
    _, reverse_σ_end = CliMAgen.marginal_prob(reverse_model, x, FT(reverse_t_end)) # x only affects the mean, which we dont use
    @. x = x * reverse_σ_end / forward_σ_end
end

"""
    make_icr(batch)

Computes and returns the mean condensation rate of the data `batch`.
"""
function make_icr(batch)
    τ = 1e-2 # condensation time scale which was set in the fluid simulations
    cond = @. batch * (batch > 0) / τ
    return  mean(cond, dims=(1,2))
end


## The following are useful plots for model assessment during training, but not for publication.

"""
    heatmap_grid(samples, ch, savepath, plotname; ncolumns = 10,FT=Float32, logger=nothing)

Creates a grid of images with `ncolumns` using the data `samples`. 
Saves the resulting plot at joinpath(savepath,plotname).

"""
function heatmap_grid(samples, ch, savepath, plotname; clims = nothing, ncolumns = 5,FT=Float32, logger=nothing)
    batchsize = size(samples)[end]
    ncolumns = min(batchsize, ncolumns)
    # We want either an even number of images per row
    nrows = div(batchsize, ncolumns)
    nimages = nrows*ncolumns
    if clims isa Nothing
        clims = (minimum(samples), maximum(samples))
    end
    plts = []
    for img in 1:nimages
        push!(plts, Plots.heatmap(samples[:,:,ch,img], aspect_ratio=:equal, clims = clims, border = :box, legend = :none, axis=([], false)))
    end
    Plots.plot(plts..., layout = (nrows, ncolumns), size = (ncolumns*200, nrows*200))
    Plots.savefig(joinpath(savepath, plotname))

    if !(logger isa Nothing)
        CliMAgen.log_artifact(logger, joinpath(savepath, plotname); name=plotname, type="PNG-file")
    end
end

"""
    loss_plot(savepath::String, plotname::String; xlog::Bool=false, ylog::Bool=true)

Creates and saves a plot of the training and test loss values, for both the spatial
and mean loss terms; creates a saves a plot of the training and test loss values
for the total loss, if using the vanilla loss function. Which option is carried out
depends on the number of columns in the data file: 5 for the split loss function, and 3
for the vanilla loss function.

Whether or not the axes are linear or logarithmic is controlled
by the `xlog` and `ylog` boolean keyword arguments. The saved plot can be found at `joinpath(savepath,plotname)`.
"""
function loss_plot(savepath::String, plotname::String; xlog::Bool=false, ylog::Bool=true)
    path = joinpath(savepath,plotname)
    filename = joinpath(savepath, "losses.txt")
    data = DelimitedFiles.readdlm(filename, ',', skipstart = 1)
    
    if size(data)[2] == 5
        plt1 = plot(left_margin = 20Plots.mm, ylabel = "Log10(Mean Loss)")
	plt2 = plot(bottom_margin = 10Plots.mm, left_margin = 20Plots.mm,xlabel = "Epoch", ylabel = "Log10(Spatial Loss)")
	plot!(plt1, data[:,1], data[:,2], label = "Train", linecolor = :black)
    	plot!(plt1, data[:,1], data[:,4], label = "Test", linecolor = :red)
    	plot!(plt2, data[:,1], data[:,3], label = "", linecolor = :black)
    	plot!(plt2, data[:,1], data[:,5], label = "", linecolor = :red)
    	if xlog
           plot!(plt1, xaxis=:log)
           plot!(plt2, xaxis=:log)
    	end
    	if ylog
           plot!(plt1, yaxis=:log)
           plot!(plt2, yaxis=:log)
        end
	plot(plt1, plt2, layout =(2,1))
	savefig(path)
    elseif size(data)[2] == 3
        plt1 = plot(left_margin = 20Plots.mm, ylabel = "Log10(Loss)")
	plot!(plt1, data[:,1], data[:,2], label = "Train", linecolor = :black)
    	plot!(plt1, data[:,1], data[:,3], label = "Test", linecolor = :red)
    	if xlog
           plot!(plt1, xaxis=:log)
    	end
    	if ylog
           plot!(plt1, yaxis=:log)
        end
	savefig(path)
    else
        @info "Loss CSV file has incorrect number of columns"
    end
end


"""
    spatial_mean_plot(data, gen, savepath, plotname; FT=Float32)

Creates and saves histogram plots of the spatial means of `data` and `gen`;
the plot is saved at joinpath(savepath, plotname). Both `data` and `gen`
are assumed to be of size (Nx, Ny, Nchannels, Nbatch).
"""
function spatial_mean_plot(data, gen, savepath, plotname; FT=Float32)
    inchannels = size(data)[end-1]

    gen = gen |> Flux.cpu
    gen_results = mapslices(Statistics.mean, gen, dims=[1, 2])
    gen_results = gen_results[1,1,:,:]

    data_results = mapslices(Statistics.mean, data, dims=[1, 2])
    data_results = data_results[1,1,:,:]
    plot_array = []
    for channel in 1:inchannels
        plt = plot(xlabel = "Spatial Mean", ylabel = "Probability density", title = string("Ch:",string(channel)))
        plot!(plt, data_results[channel,:], seriestype=:stephist, label = "data", norm = true, color = :red)
        plot!(plt, gen_results[channel,:],  seriestype=:stephist, label ="generated", norm = true, color = :black)
        push!(plot_array, plt)
    end
    
    plot(plot_array..., layout=(1, inchannels))
    Plots.savefig(joinpath(savepath, plotname))

end

"""
    qq_plot(data, gen, savepath, plotname; FT=Float32)

Creates and saves qq plots of the higher order cumulants of `data` and `gen`;
the plot is saved at joinpath(savepath, plotname). Both `data` and `gen`
are assumed to be of size (Nx, Ny, Nchannels, Nbatch).
"""
function qq_plot(data, gen, savepath, plotname; FT=Float32)
    statistics = (Statistics.var, x -> StatsBase.cumulant(x[:], 3), x -> StatsBase.cumulant(x[:], 4))
    statistic_names = ["σ²", "κ₃", "κ₄"]
    inchannels = size(data)[end-1]

    gen = gen |> Flux.cpu
    gen_results = mapslices.(statistics, Ref(gen), dims=[1, 2])
    gen_results = cat(gen_results..., dims=ndims(gen) - 2)
    sort!(gen_results, dims=ndims(gen_results)) # CDF of the generated data for each channel and each statistics


    data_results = mapslices.(statistics, Ref(data), dims=[1, 2])
    data_results = cat(data_results..., dims=ndims(data) - 2)
    sort!(data_results, dims=ndims(data_results)) # CDF of the  data for each channel and each statistics
    plot_array = []
    for channel in 1:inchannels
        for stat in 1:length(statistics)
            data_cdf = data_results[1, stat, channel, :]
            gen_cdf = gen_results[1, stat, channel, :]
            plt = plot(gen_cdf, data_cdf, color=:red, label="")
            plot!(plt, data_cdf, data_cdf, color=:black, linestyle=:dot, label="")
            plot!(plt,
                xlabel="Gen",
                ylabel="Data",
                title=string("Ch:", string(channel), ", ", statistic_names[stat]),
                tickfontsize=4)
            push!(plot_array, plt)
        end
    end

    plot(plot_array..., layout=(inchannels, length(statistics)), aspect_ratio=:equal)
    Plots.savefig(joinpath(savepath, plotname))

end

"""
Helper function to make a spectrum plot.
"""
function spectrum_plot(data, gen, savepath, plotname; FT=Float32) 
    statistics = x -> hcat(power_spectrum2d(x)...)
    inchannels = size(data)[end-1]

    data_results = mapslices(statistics, data, dims=[1, 2])
    k = data_results[:, 2, 1, 1]
    data_results = data_results[:, 1, :, :]

    gen = gen |> Flux.cpu
    gen_results = mapslices(statistics, gen, dims=[1, 2])
    gen_results = gen_results[:, 1, :, :]

    plot_array = []
    for channel in 1:inchannels
        data_spectrum = mean(data_results[:, channel, :], dims=2)
        lower_data_spectrum = mapslices(x -> percentile(x[:], 10), data_results[:, channel, :], dims=2)
        upper_data_spectrum = mapslices(x -> percentile(x[:], 90), data_results[:, channel, :], dims=2)
        data_confidence = (data_spectrum .- lower_data_spectrum, upper_data_spectrum .- data_spectrum)
        gen_spectrum = mean(gen_results[:, channel, :], dims=2)
        lower_gen_spectrum = mapslices(x -> percentile(x[:], 10), gen_results[:, channel, :], dims=2)
        upper_gen_spectrum = mapslices(x -> percentile(x[:], 90), gen_results[:, channel, :], dims=2)
        gen_confidence = (gen_spectrum .- lower_gen_spectrum, upper_gen_spectrum .- gen_spectrum)
        plt = plot(k, data_spectrum, ribbon = data_confidence, color=:red, label="", yaxis=:log, xaxis=:log)
        plot!(plt, k, gen_spectrum, ribbon = gen_confidence, color=:blue, label="")
        plot!(plt, ylim = (1e-10, 1e-1))
        plot!(plt,
            xlabel="Log(k)",
            ylabel="Log(Power)",
            title=string("Ch:", string(channel)),
            tickfontsize=4)
        push!(plot_array, plt)
    end

    plot(plot_array..., layout=(inchannels, 1))
    Plots.savefig(joinpath(savepath, plotname))
end
