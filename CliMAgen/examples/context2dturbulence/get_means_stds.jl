using Statistics
using CliMADatasets
using DelimitedFiles
resolutions = [64,512]
wavenumber_sets = [[:all,], [2.0f0, 4.0f0, 8.0f0, 16.0f0]]
output = []
FT = Float32

for i in 1:2
    resolution = resolutions[i]
    wavenumbers = wavenumber_sets[i]
    for wn in wavenumbers
        fraction = resolution == 64 ? 1.0f0 : 0.04f0
        xtrain = CliMADatasets.Turbulence2DContext(:train; fraction = fraction, resolution=resolution, wavenumber = wn, Tx=FT,)[:];
        means = Statistics.mean(xtrain, dims = (1,2,4))[1:2]
        stds = Statistics.std(xtrain, dims = (1,2,4))[1:2]
        wn = wn == :all ? 0.0f0 : wn
        push!(output, [resolution, wn, means..., stds...])
        @info [resolution, wn, means..., stds...]
    end
end

open("train_means_stds.csv", "w") do io
    writedlm(io, output, ',')
end;
