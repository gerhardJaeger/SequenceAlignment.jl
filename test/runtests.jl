using SequenceAlignment
using Test

@testset "SequenceAlignment.jl" begin
    @test my_f(1,1) == 5
    @test my_f(0,0) == 0
end
