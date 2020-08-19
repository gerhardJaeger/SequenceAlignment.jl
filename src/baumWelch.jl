

mutable struct phmmExpectations
    alphabet::Vector{Char}
    α::Vector{Float64}
    τ::Vector{Float64}
    p::Matrix{Float64}
    q::Vector{Float64}
    tm::Matrix{Float64}
end


import Base:+
function (+)(a::phmmExpectations, b::phmmExpectations)
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

function phmmExpectations(ph::Phmm, count::Float64=1.0)
    nSymbols = length(ph.alphabet)
    phmmExpectations(
        ph.alphabet,
        ones(3) .* count,
        ones(3) .* count,
        ones(nSymbols, nSymbols) .* count,
        ones(nSymbols) .* count,
        ones(3,3) .* count,
    )
end


function Phmm(e::phmmExpectations)
    transitions = zeros(5,5)
    transitions[1,2:4] = e.α
    transitions[2:4,2:4] = e.tm
    transitions[2:4,5] = e.τ

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



    tn = mapslices(x -> x ./ sum(x), transitions, dims=2)
    α = log.(tn[1,2:4])
    τ = log.(tn[2:4,5])
    lt = log.(tn[2:4, 2:4])


    lp = Dict{Tuple{Char, Char}, Float64}()
    for (i,s1) in enumerate(e.alphabet), (j, s2) in enumerate(e.alphabet)
        lp[s1, s2] = log(e.p[i,j] + e.p[j,i]) - log(sum(e.p)) - log(2)
    end
    lp = DefaultDict(-Inf, lp)

    lq = Dict{Char, Float64}()
    for (i, s) in enumerate(e.alphabet)
        lq[s] = log(e.q[i]) - log(sum(e.q))
    end
    lq = DefaultDict(-Inf, lq)
    Phmm(e.alphabet, α, τ, lt, lp, lq)
end

function baumWelch(w1::String, w2::String, levP::Phmm)
    n, m = length(w1), length(w2)
    fdp = zeros(n + 1, m + 1, 3)
    bdp = zeros(n + 1, m + 1, 3)
    i1 = Vector{Int}(indexin(w1, levP.alphabet))
    i2 = Vector{Int}(indexin(w2, levP.alphabet))
    ll = SequenceAlignment.forward!(fdp, w1, w2, levP)
    SequenceAlignment.backward!(bdp, w1, w2, levP)
    nSymbols = length(levP.alphabet)
    expectedP = zeros(nSymbols, nSymbols)
    expectedQ = zeros(nSymbols)
    expectedT = zeros(Float64, 3, 3)
    expectedAlpha = zeros(3)
    expectedTau = zeros(3)
    ev1 = exp(levP.α[1] + bdp[2, 2, 1] + levP.lp[w1[1], w2[1]] - ll)
    expectedAlpha[1] += ev1
    expectedP[i1[1], i2[1]] += ev1
    ev1 = exp(levP.α[2] + bdp[2, 1, 2] + levP.lq[w1[1]] - ll)
    expectedAlpha[2] += ev1
    expectedQ[i1[1]] += ev1
    ev1 = exp(levP.α[3] + bdp[1, 2, 3] + levP.lq[w2[1]] - ll)
    expectedAlpha[3] += ev1
    expectedQ[i2[1]] += ev1
    expectedTau = exp.(fdp[n+1, m+1, :] + levP.τ .- ll)
    for j = 1:(m+1), i = 1:(n+1)
        if i > 1 && j > 1 && (i, j) != (2, 2)
            ev =
                exp.(
                    fdp[i-1, j-1, :] + levP.lt[:, 1] .+ bdp[i, j, 1] .+
                    levP.lp[w1[i-1], w2[j-1]] .- ll,
                )
            expectedT[:, 1] += ev
            expectedP[i1[i-1], i2[j-1]] += sum(ev)

        end
        if i > 1 && (i, j) != (2, 1)
            ev =
                exp.(
                    fdp[i-1, j, :] + levP.lt[:, 2] .+ bdp[i, j, 2] .+
                    levP.lq[w1[i-1]] .- ll,
                )
            expectedT[:, 2] += ev
            expectedQ[i1[i-1]] += sum(ev)
        end
        if j > 1 && (i, j) != (1, 2)
            ev =
                exp.(
                    fdp[i, j-1, :] + levP.lt[:, 3] .+ bdp[i, j, 3] .+
                    levP.lq[w2[j-1]] .- ll,
                )
            expectedT[:, 3] += ev
            expectedQ[i2[j-1]] += sum(ev)
        end
    end
    phmmExpectations(
        levP.alphabet,
        expectedAlpha,
        expectedTau,
        expectedP,
        expectedQ,
        expectedT,
    )
end
