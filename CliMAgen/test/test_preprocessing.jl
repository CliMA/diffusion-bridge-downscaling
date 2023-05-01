@testset "Preprocessing" begin
    @testset "Scaling transforms" begin
        FT = Float32
        x = randn(FT,(64,64,3,100))
        min_values = minimum(x, dims = (1,2,4))
        max_values = maximum(x, dims = (1,2,4))
        Δ = max_values .- min_values
        scaling = StandardScaling{FT}(min_values, Δ)
        @test x ≈ invert_preprocessing(apply_preprocessing(x, scaling), scaling)

        x̄ = mean(x, dims=(1, 2))
        max_mean = maximum(x̄, dims=4)
        min_mean = minimum(x̄, dims=4)
        Δ̄ = max_mean .- min_mean
        xp = x .- x̄
        max_p = maximum(xp, dims=(1, 2, 4))
        min_p = minimum(xp, dims=(1, 2, 4))
        Δp = max_p .- min_p
        scaling = MeanSpatialScaling{FT}(min_mean, Δ̄, min_p, Δp)
        @test x ≈ invert_preprocessing(apply_preprocessing(x, scaling), scaling)


        # Test in case where a field is constant
        x[:,:,1,:] .= FT(0)
        # zero spatial mean
        x[:,:,2,:] .=  x[:,:,2,:] .- Statistics.mean(x[:,:,2,:], dims = (1,2))
        # zero deviations
        x[:,:,3,1:50] .=  FT(2)
        x[:,:,3,50:100] .=  FT(3)

        min_values = minimum(x, dims = (1,2,4))
        max_values = maximum(x, dims = (1,2,4))
        Δ[Δ .== 0] .= FT(1)
        scaling = StandardScaling{FT}(min_values, Δ)
        @test x ≈ invert_preprocessing(apply_preprocessing(x, scaling), scaling)

        x̄ = mean(x, dims=(1, 2))
        max_mean = maximum(x̄, dims=4)
        min_mean = minimum(x̄, dims=4)
        Δ̄ = max_mean .- min_mean
        Δ̄[Δ̄ .== 0] .= FT(1)

        xp = x .- x̄
        max_p = maximum(xp, dims=(1, 2, 4))
        min_p = minimum(xp, dims=(1, 2, 4))
        Δp = max_p .- min_p
        Δp[Δp .== 0] .= FT(1)
        scaling = MeanSpatialScaling{FT}(min_mean, Δ̄, min_p, Δp)
        @test x ≈ invert_preprocessing(apply_preprocessing(x, scaling), scaling)

    end
end
