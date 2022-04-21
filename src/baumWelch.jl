

mutable struct phmmExpectations{T}
    alphabet::Vector{T}
    α::Vector{Float64}
    τ::Vector{Float64}
    p::Matrix{Float64}
    q::Vector{Float64}
    tm::Matrix{Float64}
end


import Base: +
function (+)(a::phmmExpectations{T}, b::phmmExpectations{T}) where T
    @argcheck a.alphabet == b.alphabet
    phmmExpectations(
        a.alphabet,
        a.α .+ b.α,
        a.τ .+ b.τ,
        a.p .+ b.p,
        a.q .+ b.q,
        a.tm .+ b.tm,
    )
end

function phmmExpectations(ph::Phmm{T}, count::Float64 = 1.0) where T
    nSymbols = length(ph.alphabet)
    phmmExpectations{T}(
        ph.alphabet,
        ones(Float64, 3) .* count,
        ones(Float64, 3) .* count,
        ones(Float64, nSymbols, nSymbols) .* count,
        ones(Float64, nSymbols) .* count,
        [
            1.0 1.0 1.0
            1.0 1.0 0.0
            1.0 0.0 1
        ] .* count,
    )
end


function Phmm(e::phmmExpectations{T}) where T
    transitions = zeros(Float64, 4, 5)
    transitions[1, 2:4] = e.α
    transitions[2:4, 2:4] = e.tm
    transitions[2:4, 5] = e.τ

    transitions[1, 3] =
        transitions[1, 4] = (transitions[1, 3] + transitions[1, 4]) / 2
    transitions[2, 3] =
        transitions[2, 4] = (transitions[2, 3] + transitions[2, 4]) / 2
    transitions[3, 3] =
        transitions[4, 4] = (transitions[3, 3] + transitions[4, 4]) / 2
    transitions[3, 4] =
        transitions[4, 3] = (transitions[3, 4] + transitions[4, 3]) / 2
    transitions[3, 5] =
        transitions[4, 5] = (transitions[3, 5] + transitions[4, 5]) / 2
    transitions[3, 2] =
        transitions[4, 2] = (transitions[3, 2] + transitions[4, 2]) / 2

    tn = mapslices(x -> x ./ sum(x), transitions, dims = 2)
    lt = log.(tn)

    lp = log.((e.p + e.p')/(2*sum(e.p)))

    lq = log.(e.q/sum(e.q))
    Phmm{T}(e.alphabet, lt, lp, lq)
end

function baumWelch(
    w1::Vector{T},
    w2::Vector{T},
    ph::Phmm{T}
) where T
    v1 = indexin(w1, ph.alphabet)
    v2 = indexin(w2, ph.alphabet)
    n, m = length(w1), length(w2)
    fdp = zeros(Float64, n + 1, m + 1, 3)
    bdp = zeros(Float64, n + 1, m + 1, 3)
    ll = SequenceAlignment.forward!(fdp, w1, w2, ph)
    SequenceAlignment.backward!(bdp, w1, w2, ph)
    nSymbols = length(ph.alphabet)
    expectedP = zeros(Float64, nSymbols, nSymbols)
    expectedQ = zeros(Float64, nSymbols)
    expectedT = zeros(Float64, 3, 3)
    expectedAlpha = zeros(Float64, 3)
    expectedTau = zeros(Float64, 3)
    ev1 = exp(ph.lt[1, 2] + bdp[2, 2, 1] + ph.lp[v1[1], v2[1]] - ll)
    expectedAlpha[1] += ev1
    expectedP[v1[1], v2[1]] += ev1
    ev1 = exp(ph.lt[1, 3] + bdp[2, 1, 2] + ph.lq[v1[1]] - ll)
    expectedAlpha[2] += ev1
    expectedQ[v1[1]] += ev1
    ev1 = exp(ph.lt[14] + bdp[1, 2, 3] + ph.lq[v2[1]] - ll)
    expectedAlpha[3] += ev1
    expectedQ[v2[1]] += ev1
    expectedTau = exp.(fdp[n+1, m+1, :] + ph.lt[2:4, 5] .- ll)
    for j = 1:(m+1), i = 1:(n+1)
        if i > 1 && j > 1 && (i, j) != (2, 2)
            ev =
                exp.(
                    fdp[i-1, j-1, :] + ph.lt[2:4, 2] .+ bdp[i, j, 1] .+
                    ph.lp[v1[i-1], v2[j-1]] .- ll,
                )
            expectedT[:, 1] += ev
            expectedP[v1[i-1], v2[j-1]] += sum(ev)
        end
        if i > 1 && (i, j) != (2, 1)
            ev =
                exp.(
                    fdp[i-1, j, :] + ph.lt[2:4, 3] .+ bdp[i, j, 2] .+
                    ph.lq[v1[i-1]] .- ll,
                )
            expectedT[:, 2] += ev
            expectedQ[v1[i-1]] += sum(ev)
        end
        if j > 1 && (i, j) != (1, 2)
            ev =
                exp.(
                    fdp[i, j-1, :] + ph.lt[2:4, 4] .+ bdp[i, j, 3] .+
                    ph.lq[v2[j-1]] .- ll,
                )
            expectedT[:, 3] += ev
            expectedQ[v2[j-1]] += sum(ev)
        end
    end
    ll,
    phmmExpectations(
        ph.alphabet,
        expectedAlpha,
        expectedTau,
        expectedP,
        expectedQ,
        expectedT,
    )
end
