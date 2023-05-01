@testset "Utils" begin
    @testset "Training utils" begin
        @test CliMAgen.struct2dict(0) == 0
        @test CliMAgen.struct2dict(true) == true
        @test CliMAgen.struct2dict("lol") == "lol"
        @test CliMAgen.struct2dict((; a = 1, b = true)) == Dict(:a => 1, :b => true)
    end
end