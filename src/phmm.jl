mutable struct Phmm # all probabilities are log-transformed
    alphabet::Vector{Char}
    α::Vector{Float64} # initial state probabilities
    τ::Vector{Float64} # transition probs into final state
    lt::Matrix{Float64} # transition matrix between M, X and Y
    lp::DefaultDict{Tuple{Char,Char},Float64,Float64} # emission probs in state M
    lq::DefaultDict{Char,Float64,Float64} # emission probs in states X, Y
    # Phmm(alphabet, α, τ, lt, lp, lq) =
    #     assert_phmm(alphabet, α, τ, lt, lp, lq) &&
    #     new(alphabet, α, τ, lt, lp, lq)
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

import Base:print
import Base:write
import Base:read
import Base:isequal
import Base:(==)
import Base:isapprox

function print(io::IO, ph::Phmm)
    print(replace(replace(json(ph, 1), "\"" => ""), "null" => "-Inf"))
end

function write(fn::AbstractString, ph::Phmm)
    write(fn, json(ph, 1))
end

function read(fn::AbstractString, ::Type{Phmm})
    s = read(fn, String)
    j = JSON.Parser.parse(s)
    α = j["α"]
    τ = j["τ"]
    alphabet = first.(j["alphabet"])
    lp = DefaultDict(Float64(-Inf), Dict{Tuple{Char,Char},Float64}())
    llq = DefaultDict(Float64(-Inf), Dict{Char,Float64}())
    for s1 in alphabet, s2 in alphabet
        v = j["lp"]["('$s1', '$s2')"]
        lp[s1, s2] = isnothing(v) ? -Inf : v
    end
    for s in alphabet
        v = j["lq"][join(s)]
        llq[s] = isnothing(v) ? -Inf : v
    end
    lt = convert(Matrix{Float64}, hcat(replace.(j["lt"], nothing => -Inf)...))
    Phmm(alphabet, α, τ, lt, lp, llq)
end


function isequal(a::Phmm, b::Phmm)
    all([getfield(a, fn) == getfield(b, fn) for fn in fieldnames(Phmm)])
end

function (==)(a::Phmm, b::Phmm)
    isequal(a, b)
end

function isapprox(a::Phmm, b::Phmm)
    if !isapprox(a.α, b.α)
        return false
    end
    if !isapprox(a.τ, b.τ)
        return false
    end
    if !isapprox(a.lt, b.lt)
        return false
    end
    if !isapprox(a.lp.d.default, a.lp.d.default)
        return false
    end
    if keys(a.lp) != keys(b.lp)
        return false
    end
    for k in keys(a.lp)
        if !isapprox(a.lp[k], b.lp[k])
            return false
        end
    end
    if !isapprox(a.lq.d.default, a.lq.d.default)
        return false
    end
    if keys(a.lq) != keys(b.lq)
        return false
    end
    for k in keys(a.lq)
        if !isapprox(a.lq[k], b.lq[k])
            return false
        end
    end
    return true
end
