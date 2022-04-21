mutable struct Phmm{T} # all probabilities are log-transformed
    alphabet::Vector{T}
    lt::Matrix{Float64} # transition matrix
    lp::Matrix{Float64} # emission probs in state M
    lq::Vector{Float64} # emission probs in states X, Y
    Phmm{T}(alphabet, lt, lp, lq)  where T =
        assert_phmm(alphabet, lt, lp, lq) &&
        new{T}(alphabet, lt, lp, lq)
end


function assert_phmm(alphabet, lt, lp, lq)
    nSymbols = length(alphabet)
    @argcheck nSymbols > 1
    @argcheck size(lt) == (4, 5)
    @argcheck size(lp) == (nSymbols,nSymbols)
    @argcheck size(lq) == (nSymbols,)
    @argcheck all([isprobvec(exp.(r)) for r in eachrow(lt)])
    @argcheck isprobvec(exp.(vec(lp)))
    @argcheck issymmetric(lp)
    @argcheck isprobvec(exp.(lq))
    return true
end

function Phmm(
    alphabet::Vector{T},
    δ::Float64,
    ϵ::Float64,
    τ::Float64,
    lp::Matrix{Float64},
    lq::Vector{Float64},
) where {T}
    lt = log.([
        0 1-2δ-τ δ δ τ
        0 1-2δ-τ δ δ τ
        0 1-ϵ-τ ϵ 0 τ
        0 1-ϵ-τ 0 ϵ τ
    ])
    Phmm{T}(alphabet, lt, lp, lq)
end

import Base:print
import Base:write
import Base:read
import Base:isequal
import Base:(==)
import Base:isapprox
import Base:copy

function print(io::IO, ph::Phmm)
    print(replace(replace(json(ph, 1), "\"" => ""), "null" => "-Inf"))
end

function write(fn::AbstractString, ph::Phmm)
    write(fn, json(ph, 1))
end


# TODO: returns from type of Phmm
function read(fn::AbstractString, ::Type{Phmm})
    s = read(fn, String)
    j = JSON.Parser.parse(s)
    alphabet = j["alphabet"]
    lp = Matrix{Float64}(hcat(j["lp"]...))
    lq = Vector{Float64}(j["lq"])
    lt = convert(Matrix{Float64}, hcat(replace.(j["lt"], nothing => -Inf)...))
    Phmm{eltype(alphabet)}(alphabet, lt, lp, lq)
end


function isequal(a::Phmm, b::Phmm)
    all([getfield(a, fn) == getfield(b, fn) for fn in fieldnames(Phmm)])
end

function (==)(a::Phmm, b::Phmm)
    isequal(a, b)
end

function isapprox(a::Phmm, b::Phmm)
    a.alphabet == b.alphabet && all([
        getfield(a, fn) ≈ getfield(b, fn) for fn in fieldnames(Phmm)[2:end]
    ])
end



function randomPhmm(alphabet::Vector{T}) where {T}
    nSymbols = length(alphabet)
    pr = reshape(rand(Dirichlet(nSymbols^2, 1)), (nSymbols, nSymbols))
    pr = (pr .+ pr') / 2
    qr = rand(Dirichlet(nSymbols, 1))
    lp = log.(pr)
    lq = log.(qr)
    tm = zeros(4,5)
    a,b,τ = rand(Dirichlet(3,1))
    tm[1,:] = [0, a, b/2, b/2, τ]
    tm[2,:] = [0, a, b/2, b/2, τ]
    a,b = rand(Dirichlet(2,1))
    tm[3,:] = [0, a*(1-τ), b*(1-τ), 0, τ]
    tm[4,:] = tm[3,[1,2,4,3,5]]
    lt = log.(tm)
    Phmm{T}(alphabet, lt, lp, lq)
end

function uniformPhmm(alphabet::Vector{T}) where {T}
    nSymbols = length(alphabet)
    pr = ones(nSymbols, nSymbols) ./ (nSymbols^2)
    qr = ones(nSymbols) ./ nSymbols
    lp = log.(pr)
    lq = log.(qr)
    tm = [
        0 1/4 1/4 1/4 1/4
        0 1/4 1/4 1/4 1/4
        0 3/8 3/8 0   1/4
        0 3/8 0   3/8 1/4
    ]
    lt = log.(tm)
    Phmm{T}(alphabet, lt, lp, lq)
end


function levPhmm(alphabet::Vector{T}) where {T}
    nSymbols = length(alphabet)
    pr = ones(nSymbols, nSymbols)
    for i in 1:nSymbols
        pr[i,i] += nSymbols
    end
    pr /= sum(pr)
    qr = ones(nSymbols) ./ nSymbols
    lp = log.(pr)
    lq = log.(qr)
    tm = [
        0 1/4 1/4 1/4 1/4
        0 1/4 1/4 1/4 1/4
        0 3/8 3/8 0 1/4
        0 3/8 0 3/8 1/4
    ]
    lt = log.(tm)
    Phmm{T}(alphabet, lt, lp, lq)
end



function copy(a::Phmm)
    Phmm(
        copy(a.alphabet),
        copy(a.lt),
        copy(a.lp),
        copy(a.lq)
    )
end
