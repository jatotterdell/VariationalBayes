@testset "Testing utils" begin
    # vec⁻¹
    @test VariationalBayes.vec⁻¹(vec(ones(2, 2))) == ones(2, 2)
    @test VariationalBayes.vec⁻¹(vec(ones(Int, 2, 2))) == ones(Int, 2, 2)

    # vech
    @test VariationalBayes.vech(ones(2, 2)) == ones(3)
    @test VariationalBayes.vech(ones(Int, 2, 2)) == ones(Int, 3)
    @test VariationalBayes.vech(ones(3, 3)) == ones(6)
    @test VariationalBayes.vech(diagm(ones(3))) == [1.0, 0.0, 0.0, 1.0, 0.0, 1.0]

    # vech⁻¹
    @test VariationalBayes.vech⁻¹(VariationalBayes.vech(ones(2, 2))) == ones(2, 2)
    @test VariationalBayes.vech⁻¹(VariationalBayes.vech(ones(Int, 2, 2))) == ones(Int, 2, 2)
end