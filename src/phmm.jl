mutable struct Phmm # all probabilities are log-transformed
    alphabet::Vector{Char}
    α::Vector{Float64} # initial state probabilities
    τ::Vector{Float64} # transition probs into final state
    lt::Matrix{Float64} # transition matrix between M, X and Y
    lp::DefaultDict{Tuple{Char,Char},Float64,Float64} # emission probs in state M
    lq::DefaultDict{Char,Float64,Float64} # emission probs in states X, Y
    Phmm(alphabet, α, τ, lt, lp, lq) =
        assert_phmm(alphabet, α, τ, lt, lp, lq) &&
        new(alphabet, α, τ, lt, lp, lq)
end

import LinearAlgebra:issymmetric
function issymmetric(
    lp::DefaultDict{Tuple{S,S},T,U},
) where {S<:Any,T<:Any,U<:Any}
    all([lp[x] ≈ lp[reverse(x)] for x in keys(lp)])
end


function assert_phmm(alphabet, α, τ, lt, lp, lq)
    nSymbols = length(alphabet)
    @argcheck nSymbols > 1
    @argcheck size(α) == (3,)
    @argcheck size(τ) == (3,)
    @argcheck size(lt) == (3, 3)
    @argcheck size(collect(lp)) == (nSymbols^2,)
    @argcheck size(collect(lq)) == (nSymbols,)
    @argcheck isprobvec(exp.(α))
    @argcheck isprobvec(vcat(exp.(lt[1, :]), exp(τ[1])))
    @argcheck isprobvec(vcat(exp.(lt[2, :]), exp(τ[2])))
    @argcheck isprobvec(vcat(exp.(lt[3, :]), exp(τ[2])))
    @argcheck isprobvec(exp.(collect(values(lp))))
    @argcheck issymmetric(lp)
    @argcheck isprobvec(exp.(collect(values(lq))))
    return true
end

function Phmm(alphabet, δ, ϵ, λ, τ, lp, lq)
    α = log.([1 - 2δ, δ, δ])
    lt = log.([
        1 - 2δ - exp(τ[1]) δ δ
        1 - ϵ - λ - exp(τ[2]) ϵ λ
        1 - ϵ - λ - exp(τ[2]) λ ϵ
    ])
    Phmm(alphabet, α, τ, lt, lp, lq)
end
