@testset "Models" begin
    σ_max = 4.66f0
    σ_min = 0.466f0
    net=(x, c, t)->x
    m = CliMAgen.VarianceExplodingSDE(σ_max, σ_min, net)

    # test constructor
    @test m.σ_max == 4.66f0
    @test m.σ_min == 0.466f0

    # test drift
    t = rand(3, 10)
    @test CliMAgen.drift(m, t) == zero(t)

    # test diffusion
    t = rand(3, 10)

    @test CliMAgen.diffusion(m, t) ≈ @. m.σ_min * (m.σ_max/m.σ_min)^t*sqrt(2*log(m.σ_max/m.σ_min))

    # test marginal_prob
    x, t = rand(3, 10), rand(10)
    μ, σ = CliMAgen.marginal_prob(m, x, t)
    @test μ ≈ x
    @test σ ≈ CliMAgen.expand_dims(sqrt.(m.σ_min^2 .* (m.σ_max/m.σ_min).^(2 .*t)), ndims(x) -1 )
  
    # test dummy score
    x, t = rand(3, 3), rand()
    _, σ = CliMAgen.marginal_prob(m, x, t)  
    @test CliMAgen.score(m, x, t) == x ./ σ
    x, t = rand(11), rand(11)
    _, σ = CliMAgen.marginal_prob(m, x, t)  
    @test CliMAgen.score(m, x, t) == x ./ σ
    x, t = rand(3, 3, 4, 11), rand(11)
    _, σ = CliMAgen.marginal_prob(m, x, t)  
    @test CliMAgen.score(m, x, t) == x ./ σ

    # test forward_sde
    f, g = CliMAgen.forward_sde(m)
    x, t = rand(11), rand(11)
    @test all(f(x, nothing, t) .== CliMAgen.drift(m, t))
    @test all(g(x, nothing, t) .== CliMAgen.diffusion(m, t))

    # test reverse_sde
    f, g = CliMAgen.reverse_sde(m)
    x, t = rand(3), rand(3)
    @test f(x, nothing, t) == CliMAgen.drift(m, t) .- CliMAgen.diffusion(m, t) .^ 2 .* CliMAgen.score(m, x, t)
    @test all(g(x, nothing, t) .== CliMAgen.diffusion(m, t))

    # test reverse_ode
    f = CliMAgen.probability_flow_ode(m)
    x, t = rand(3), rand(3)
    @test f(x, nothing, t) == CliMAgen.drift(m, t) .- CliMAgen.diffusion(m, t) .^ 2 .* CliMAgen.score(m, x, t) ./ 2
end

@testset "Losses" begin
    σ_max = 4.66f0
    σ_min = 0.466f0
    net=(x, c, t)->x
    m = CliMAgen.VarianceExplodingSDE(σ_max, σ_min, net)
    x = rand(3, 7, 8, 3, 25)
    @test CliMAgen.score_matching_loss(m, x)[1] isa Real
    @test CliMAgen.score_matching_loss(m, x)[2] isa Real
    @test CliMAgen.vanilla_score_matching_loss(m, x) isa Real
end
