"""
    AbstractPreprocessing{FT<:AbstractFloat}

An abstract type for preprocessing transformations.
"""
abstract type AbstractPreprocessing{FT<:AbstractFloat} end

"""
    StandardScaling{FT} <: AbstractPreprocessing{FT}

A struct that holds the minimum and range by channel, 
where the minimum and range are taken over pixels and 
over all samples of the training data.
"""
struct StandardScaling{FT} <: AbstractPreprocessing{FT}
    mintrain::Array{FT}
    Δ::Array{FT}
end

"""
    MeanSpatialScaling{FT} <: AbstractPreprocessing{FT}

A struct that holds the minimum and range by spatial 
(denoted `p`) and mean component (denoted with overbars),
 and by channel, where the minimum and range are taken over 
pixels and over all samples of the training data.
"""
struct MeanSpatialScaling{FT} <: AbstractPreprocessing{FT}
    mintrain_mean::Array{FT}
    Δ̄::Array{FT}
    mintrain_p::Array{FT}
    Δp::Array{FT}
end

"""
    invert_preprocessing(x̃, scaling::MeanSpatialScaling)

Computes the inverse preprocess transform of the data, given a
scaling of type `MeanSpatialScaling`.
"""
function invert_preprocessing(x̃, scaling::MeanSpatialScaling)
    (; mintrain_mean, Δ̄, mintrain_p, Δp) = scaling
    tmp = @. (x̃ + 2) / 2 * Δp + mintrain_p
    xp = tmp .- Statistics.mean(tmp, dims = (1,2))
    x̄ = @. (tmp - xp) / Δp * Δ̄ + mintrain_mean
    return xp .+ x̄
end

"""
    invert_preprocessing(x̃, scaling::StandardScaling)

Computes the inverse preprocess transform of the data, given a
scaling of type `StandardScaling`.
"""
function invert_preprocessing(x̃, scaling::StandardScaling)
    (; mintrain, Δ) = scaling
    return @. (x̃ + 1) / 2 * Δ + mintrain
end

"""
    apply_preprocessing(x, scaling::StandardScaling)

Preprocesses the data x, given a scaling of type `StandardScaling`.
"""
function apply_preprocessing(x, scaling::StandardScaling)
    (; mintrain, Δ) = scaling
    return @. 2 * (x - mintrain) / Δ - 1
end

"""
    apply_preprocessing(x, scaling::MeanSpatialScaling)


Preprocesses the data x, given a scaling of type `MeanSpatialScaling`.
"""
function apply_preprocessing(x, scaling::MeanSpatialScaling)
    (; mintrain_mean, Δ̄, mintrain_p, Δp) = scaling
    x̄ = Statistics.mean(x, dims=(1, 2))
    xp = x .- x̄
    x̄̃ = @. 2(x̄ -  mintrain_mean) / Δ̄ - 1
    x̃p = @. 2(xp -  mintrain_p) / Δp - 1
    return  x̄̃ .+ x̃p
end
