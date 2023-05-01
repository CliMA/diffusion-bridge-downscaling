"""
    Euler_Maruyama_sampler(model::CliMAgen.AbstractDiffusionModel,
                           init_x::A,
                           time_steps,
                           Δt;
                           c=nothing,
                           forward = false
                           )::A where {A}

Generate a sample from a diffusion model using the Euler-Maruyama method,
with 
- `model` the diffusion model,
- `init_x` as the initial condition,
- `time_steps` the vector of times at which a solution is computed,
   which should advance in ascending order for the forward SDE
   and descending order for the reverse SDE,
- `Δt` the absolute value of the timestep,
- `c` the contextual fields,
- `forward` a boolean indicating if the forward or reverse SDE is used.
# References
https://arxiv.org/abs/1505.04597
"""
function Euler_Maruyama_sampler(model::CliMAgen.AbstractDiffusionModel,
                                init_x::A,
                                time_steps,
                                Δt;
                                c=nothing,
                                forward = false
                                )::A where {A}
    x = mean_x = init_x

    @showprogress "Euler-Maruyama Sampling" for time_step in time_steps
        batch_time_step = fill!(similar(init_x, size(init_x)[end]), 1) .* time_step
        g = CliMAgen.diffusion(model, batch_time_step)
        if forward
            x = x .+ sqrt(Δt) .* CliMAgen.expand_dims(g, 3) .* randn!(similar(x))
        else
        score = CliMAgen.score(model, x, batch_time_step; c=c)
        mean_x = x .+ CliMAgen.expand_dims(g, 3) .^ 2 .* score .* Δt
        x = mean_x .+ sqrt(Δt) .* CliMAgen.expand_dims(g, 3) .* randn!(similar(x))
        end
    end
    return x
end
