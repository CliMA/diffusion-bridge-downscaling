# To be run from plotting_scripts/
using Bootstrap
using CairoMakie
using DataFrames
using DelimitedFiles: readdlm, writedlm
using Interpolations
using KernelDensity
using Statistics
using Random
using HDF5

# Where the data is
basedir = dirname(pwd()) # full path to /diffusion-bridge-downscaling/CliMAgen/examples/context2dturbulence/
lr_stats_basedir = joinpath(basedir, "stats/64x64")
hr_stats_basedir = joinpath(basedir, "stats/512x512")

# Plotting utils
include(joinpath(basedir, "plotting_scripts/utils.jl"))

# Make correlation plot - Figure 12
include(joinpath(basedir, "plotting_scripts/correlation_script.jl"))

# Compute KS statistics for pixel distributions - Table 1
include(joinpath(basedir, "plotting_scripts/ks_pixel_statistics.jl"))

# Paired analysis plots: Appendix G
include(joinpath(basedir, "plotting_scripts/paired_analysis_plots.jl"))

## loading statistics into dataframes for channel-wise spectra, spatial means.
types = [:train, :downscale_gen]
channels = [1, 2]
wavenumbers = [0.0, 2.0, 4.0, 8.0, 16.0]

data = []
# extract hi-res data from files
for wn in wavenumbers
    for ch in channels
        for type in types
            if type == :downscale_gen && wn == 0.0
                # There is no downscaled data for wn = 0.0 (lo res data)
                nothing
            else
                stats_basedir = wn == 0.0 ? lr_stats_basedir : hr_stats_basedir
                filename = joinpath(stats_basedir, "$(type)/$(type)_statistics_ch$(ch)_$(wn).csv")
                df = DataFrame(readdlm(filename, ',', Float32, '\n'), :auto)
                # no condensation rate for vorticity
                if ch == 2
                    df.x261 .= nothing
                end
                df.isreal .= type == :downscale_gen ? false : true
                df.channel .= ch
                df.wavenumber .= wn
                # adjust number of observations, they are different for the different sets
                if type == :train
                    df = df[end-99:end, :]
                end
                push!(data, df)
            end
        end
    end
end
data = vcat(data...)

# headers
hd_stats = [:mean, :variance, :skewness, :kurtosis]
hd_spec = []
hd_cond = [:cond_rate]
rename!(data, Dict(Symbol("x$i") => s for (i, s) in enumerate(hd_stats)))
rename!(data, Dict(Symbol("x$i") => Symbol("s$(i-4)") for i in 5:260))
rename!(data, Dict(:x261 => hd_cond...))

# split it up!
stats = data[:, vcat(hd_stats, [:isreal, :channel, :wavenumber])]
spectra = data[:, vcat([Symbol("s$i") for i in 1:256], [:isreal, :channel, :wavenumber])]
cond = data[:, vcat(hd_cond, [:isreal, :channel, :wavenumber])]

## plotting spatial means and spectra for each channel - Figures 7, 8, 9, 10
ch = 1
include("plot_mean.jl")
include("plot_spectra.jl")

ch = 2
include("plot_mean.jl")
include("plot_spectra.jl")

## Repeat for pixel level results - channel-wise pixel value distributions, and condensation rate.
## Figures 5, 6, 13

# extract pixels from files
pixels = []
# extract hi-res data from files
for wn in wavenumbers
    for ch in channels
        for type in types
            if type == :downscale_gen && wn == 0.0
                # There is no downscaled data for wn = 0.0 (lo res data)
                nothing
            else
                stats_basedir = wn == 0.0 ? lr_stats_basedir : hr_stats_basedir
                filename = joinpath(stats_basedir, "$(type)/$(type)_pixels_ch$(ch)_$(wn).csv")
                df = DataFrame(readdlm(filename, ',', Float32, '\n')[:][1:1600000,:]', :auto)
                df.isreal .= type == :downscale_gen ? false : true
                df.channel .= ch
                df.wavenumber .= wn
                push!(pixels, df)
            end
        end
    end
end
pixels = vcat(pixels...)

ch = 1
include("plot_pdfs.jl")
include("plot_cond_rate.jl")
ch = 2
include("plot_pdfs.jl")