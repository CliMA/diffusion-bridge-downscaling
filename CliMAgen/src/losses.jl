"""
    CliMAgen.vanilla_score_matching_loss(model::AbstractDiffusionModel, x_0; ϵ=1.0f-5, c=nothing)

Computes and returns the score matching loss given
a model `model` and batch `x_0`.

# Notes
Denoising score matching objective:
```julia
min wrt. θ (
    𝔼 wrt. 𝘵 ∼ 𝒰(0, 𝘛)[
        λ(𝘵) * 𝔼 wrt. 𝘹(0) ∼ 𝒫₀(𝘹) [
            𝔼 wrt. 𝘹(t) ∼ 𝒫₀ₜ(𝘹(𝘵)|𝘹(0)) [
                (||s₀(𝘹(𝘵), 𝘵) - ∇ log [𝒫₀ₜ(𝘹(𝘵) | 𝘹(0))] ||₂)²
            ]
        ]
    ]
)
``` 
Where 𝒫₀ₜ(𝘹(𝘵) | 𝘹(0)) and λ(𝘵), are available analytically and
s₀(𝘹(𝘵), 𝘵) is estimated by a U-Net architecture.

# References:
https://arxiv.org/abs/2011.13456
https://arxiv.org/abs/1907.05600
"""
function vanilla_score_matching_loss(model::AbstractDiffusionModel, x_0; ϵ=1.0f-5, c=nothing)
    # sample times
    t = rand!(similar(x_0, size(x_0)[end])) .* (1 - ϵ) .+ ϵ

    # sample from normal marginal
    z = randn!(similar(x_0))
    μ_t, σ_t = marginal_prob(model, x_0, t)
    x_t = @. μ_t + σ_t * z

    # evaluate model score s₀(𝘹(𝘵), 𝘵)
    s_t = score(model, x_t, t; c= c)

    # Assume that λ(t) = σ(t)² and pull it into L₂-norm
    # Below, z / σ_t = -∇ log [𝒫₀ₜ(𝘹(𝘵) | 𝘹(0))
    loss = @. (z + σ_t * s_t)^2 # squared deviations from real score
    loss = Statistics.mean(loss, dims=1:(ndims(x_0)-1)) # L₂-norm
    loss = Statistics.mean(loss) # mean over samples/batches

    return loss
end


"""
    CliMAgen.score_matching_loss(model::AbstractDiffusionModel, x_0; ϵ=1.0f-5, c= nothing)

Computes and returns the score matching loss for the
spatial mean score and for the variations about the spatial mean score
given a model `model` and batch `x_0`.

Here, `ϵ` is the lower bound on the expectation over time, and `c` is the optional
contextual input.

# Notes
Splits the vanilla score matching loss into a contribution
from the mean score and from the spatial variations in the score
about the mean. This assumes that the two are uncorrelated
which should hold when the model is well-trained.

The vanilla score matching loss is given by
```julia
min wrt. θ (
    𝔼 wrt. 𝘵 ∼ 𝒰(0, 𝘛)[
        λ(𝘵) * 𝔼 wrt. 𝘹(0) ∼ 𝒫₀(𝘹) [
            𝔼 wrt. 𝘹(t) ∼ 𝒫₀ₜ(𝘹(𝘵)|𝘹(0)) [
                (||s₀(𝘹(𝘵), 𝘵) - ∇ log [𝒫₀ₜ(𝘹(𝘵) | 𝘹(0))] ||₂)²
            ]
        ]
    ]
)
``` 
Where 𝒫₀ₜ(𝘹(𝘵) | 𝘹(0)) and λ(𝘵), are available analytically and
s₀(𝘹(𝘵), 𝘵) is estimated by a U-Net architecture.

# References:
https://arxiv.org/abs/2011.13456
https://arxiv.org/abs/1907.05600
"""
function score_matching_loss(model::AbstractDiffusionModel, x_0; ϵ=1.0f-5, c= nothing)
    # sample times
    t = rand!(similar(x_0, size(x_0)[end])) .* (1 - ϵ) .+ ϵ

    # sample from normal marginal
    z = randn!(similar(x_0))
    μ_t, σ_t = marginal_prob(model, x_0, t)
    x_t = @. μ_t + σ_t * z

    # evaluate model score s₀(𝘹(𝘵), 𝘵)
    s_t = score(model, x_t, t; c=c)

    # split into spatial averages and deviations
    nspatial = ndims(x_0)-2
    n = prod(size(z)[1:nspatial])
    √n = convert(eltype(x_0), sqrt(n))
    z_avg = Statistics.mean(z, dims=1:nspatial)
    z_dev = @. z - z_avg
    z_star = @. √n * z_avg
    s_t_avg = Statistics.mean(s_t, dims=1:nspatial)
    s_t_dev = @. s_t - s_t_avg

    # spatial average component of loss
    # We have retained the 1/n in front of the avg loss in order to compare with original net
    loss_avg = @. (z_star + √n * σ_t * s_t_avg)^2 # squared deviations from real score
    loss_avg = mean(loss_avg, dims=1:(ndims(x_0)-1)) # spatial & channel mean 
    loss_avg = 1/n * Statistics.mean(loss_avg) # mean over samples/batches

    # spatial deviation component of loss
    loss_dev = @. (z_dev + σ_t * s_t_dev)^2 # squared deviations from real score
    loss_dev = mean(loss_dev, dims=1:(ndims(x_0)-1)) # spatial & channel mean 
    loss_dev = Statistics.mean(loss_dev) # mean over samples/batches    

    return [loss_avg, loss_dev]
end

