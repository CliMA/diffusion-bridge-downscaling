abstract type AbstractDiffusionModel end

"""
    CliMAgen.drift

An extendable function which returns the drift term of
the diffusion model forward SDE.
"""
function drift end

"""
    CliMAgen.diffusion

An extendable function which returns the diffusion term of
the diffusion model forward SDE.
"""
function diffusion end


"""
    CliMAgen.marginal_prob

An extendable function which returns mean and standard
deviation of the marginal probability P(x(t)).

"""
function marginal_prob end

"""
    CliMAgen.score

Returns the score(m, x(t), t, c) given the diffusion model `m`,
state `x`, time `t`, and context `c`.
"""
function score(m::AbstractDiffusionModel, x, t; c = nothing)
    _, σ_t = marginal_prob(m, x, t)
    return m.net(x, c, t) ./ σ_t
end

"""
    CliMAgen.forward_sde

Returns the drift (f) and diffusion (g) terms
for the forward SDE as functions which are amenable for
use with DifferentialEquations.jl.

Note: The DifferentialEquations.jl API expects the input 
`t` to be a scalar and requires the argument `p`, which is 
not used in our tendency terms.
"""
function forward_sde(m::AbstractDiffusionModel)
    function f(x, p, t)
        t = fill!(similar(x, size(x)[end]), 1) .* t
        expand_dims(drift(m, t), ndims(x) - 1)
    end
    function g(x, p, t)
        t = fill!(similar(x, size(x)[end]), 1) .* t
        expand_dims(diffusion(m, t), ndims(x) - 1)
    end

    return f, g
end

"""
    CliMAgen.reverse_sde

Returns the drift (f) and diffusion (g) terms 
for the reverse SDE as functions which are amenable for
use with DifferentialEquations.jl.

Note: The DifferentialEquations.jl API expects the input 
`t` to be a scalar and requires the argument `p`, which is 
not used in our tendency terms.
"""
function reverse_sde(m::AbstractDiffusionModel)
    function f(x, p, t) 
        t = fill!(similar(x, size(x)[end]), 1) .* t
        expand_dims(drift(m, t), ndims(x) - 1) .- expand_dims(diffusion(m, t), ndims(x) - 1) .^ 2 .* score(m, x, t; c=p)
    end
    function g(x, p, t)
        t = fill!(similar(x, size(x)[end]), 1) .* t
        expand_dims(diffusion(m, t), ndims(x) - 1)
    end
    return f, g
end

"""
    Flux.params(m::AbstractDiffusionModel)

Returns the trainable parameters of the diffusion model.
"""
Flux.params(m::AbstractDiffusionModel) = Flux.params(m.net)

"""
    Base.deepcopy(m::M)
Carries out a deepcopy of `m` which has abstract type
`AbstractDiffusionModel`.
"""
Base.deepcopy(m::M) where {M <: AbstractDiffusionModel} = 
    M((deepcopy(getfield(m, f)) for f in fieldnames(M))...)

"""
    CliMAgen.VarianceExplodingSDE

A concrete type of AbstractDiffusionModel with a
prescribed variance schedule of
`σ(t) = σ_min (σ_max/σ_min)^t.

# References
Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution.
https://arxiv.org/abs/1907.05600
"""
Base.@kwdef struct VarianceExplodingSDE{FT,N} <: AbstractDiffusionModel
    "The value of σ(t=1)"
    σ_max::FT
    "The value of σ(t=0)"
    σ_min::FT
    "The convolutional neural network"
    net::N
end

@functor VarianceExplodingSDE

"""
    CliMAgen.drift(::VarianceExplodingSDE,t)

Returns the drift term of the VarianceExplodingSDE
diffusion model(`m`)'s forward SDE at time `t`.
"""
function drift(::VarianceExplodingSDE, t)
    # Note: similar(t) .* 0 occasionally results
    # in NaN, so this is a workaround.
    return t .* 0
end

"""
    CliMAgen.diffusion(::VarianceExplodingSDE,t)

Returns the diffusion term of the VarianceExplodingSDE
diffusion model(`m`)'s forward SDE at time `t`.
"""
function diffusion(m::VarianceExplodingSDE, t)
    return @. m.σ_min * (m.σ_max/m.σ_min)^t*sqrt(2*log(m.σ_max/m.σ_min))
end

"""
    CliMAgen.marginal_prob(m::VarianceExplodingSDE, x_0, t)

Returns the mean and standard deviation of the marginal probability
P(x(t)) for the VarianceExplodingSDE diffusion model `m`, given
`x_0 = x(t=0)` and t.
"""
function marginal_prob(m::VarianceExplodingSDE, x_0, t)
    μ_t = x_0
    σ_t = @. m.σ_min * (m.σ_max/m.σ_min)^t
    return μ_t, expand_dims(σ_t, ndims(μ_t) - 1)
end
