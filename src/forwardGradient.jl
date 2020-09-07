function mylogsumexp(x::Vector{Float64})
    r = 0.0
    for y in x
        if y > -Inf
            r += exp(y)
        end
    end
    log(r)
end

#---

function forwardD(
    v1::Vector{Int},
    v2::Vector{Int},
    lt::Matrix{Float64},
    lp::Matrix{Float64},
    lq::Vector{Float64},
)
    n, m = length(v1), length(v2)
    dp = Dict{Tuple{Int,Int,Int},Float64}()
    for k = 1:3, j = 1:(m+1), i = 1:(n+1)
        dp[i, j, k] = -Inf
    end
    dp[2, 2, 1] = lt[1, 2] + lp[v1[1], v2[1]]
    dp[2, 1, 2] = lt[1, 3] + lq[v1[1]]
    dp[1, 2, 3] = lt[1, 4] + lq[v2[1]]

    for j = 1:(m+1), i = 1:(n+1)
        if i > 1 && j > 1 && (i, j) != (2, 2)
            dp[i, j, 1] =
                mylogsumexp([
                    dp[i-1, j-1, 1] + lt[2, 2],
                    dp[i-1, j-1, 2] + lt[3, 2],
                    dp[i-1, j-1, 3] + lt[4, 2],
                ]) + lp[v1[i-1], v2[j-1]]
        end
        if i > 1 && (i, j) != (2, 1)
            dp[i, j, 2] =
                mylogsumexp([
                    dp[i-1, j, 1] + lt[2, 3],
                    dp[i-1, j, 2] + lt[3, 3],
                ]) + lq[v1[i-1]]
        end
        if j > 1 && (i, j) != (1, 2)
            dp[i, j, 3] =
                mylogsumexp([
                    dp[i, j-1, 1] + lt[2, 4],
                    dp[i, j-1, 3] + lt[4, 4],
                ]) + lq[v2[j-1]]
        end
    end
    mylogsumexp([
        dp[n+1, m+1, 1] + lt[2, 5],
        dp[n+1, m+1, 2] + lt[3, 5],
        dp[n+1, m+1, 3] + lt[4, 5],
    ])
end
#---

function forwardD0(
    v1::Vector{Int},
    v2::Vector{Int},
    lt::Matrix{Float64},
    lp::Matrix{Float64},
    lq::Vector{Float64},
)
    n, m = length(v1), length(v2)
    dp = Dict{Tuple{Int,Int,Int},Float64}()
    for k = 1:3, j = 1:(m+1), i = 1:(n+1)
        dp[i, j, k] = -Inf
    end
    dp[2, 2, 1] = lt[1, 2]
    dp[2, 1, 2] = lt[1, 3]
    dp[1, 2, 3] = lt[1, 4]

    for j = 1:(m+1), i = 1:(n+1)
        if i > 1 && j > 1 && (i, j) != (2, 2)
            dp[i, j, 1] =
                mylogsumexp([
                    dp[i-1, j-1, 1] + lt[2, 2],
                    dp[i-1, j-1, 2] + lt[3, 2],
                    dp[i-1, j-1, 3] + lt[4, 2],
                ])
        end
        if i > 1 && (i, j) != (2, 1)
            dp[i, j, 2] =
                mylogsumexp([
                    dp[i-1, j, 1] + lt[2, 3],
                    dp[i-1, j, 2] + lt[3, 3],
                ])
        end
        if j > 1 && (i, j) != (1, 2)
            dp[i, j, 3] =
                mylogsumexp([
                    dp[i, j-1, 1] + lt[2, 4],
                    dp[i, j-1, 3] + lt[4, 4],
                ])
        end
    end
    mylogsumexp([
        dp[n+1, m+1, 1] + lt[2, 5],
        dp[n+1, m+1, 2] + lt[3, 5],
        dp[n+1, m+1, 3] + lt[4, 5],
    ])
end

#---
function auxArrays(nSymbols::Int)
    pArr = zeros(Int, nSymbols, nSymbols)
    cnt = 1
    for i = 1:nSymbols, j = 1:nSymbols
        if i == j
            pArr[i, j] = cnt
            cnt += 1
        elseif i > j
            pArr[i, j] = pArr[j, i] = cnt
            cnt += 1
        end
    end
    pM = (Matrix(I, nSymbols, nSymbols) .+ ones(nSymbols,nSymbols))/2
    pArr, pM
end

#---
function transformParameters(
    p::Phmm,
)
    pArr, pM = auxArrays(length(p.alphabet))
    nSymbols = length(p.alphabet)
    nP = (nSymbols * (nSymbols+1)) ÷ 2 - 1
    pProbs = [sum(exp.(p.lp[pArr.==i])) for i = 1:(nP+1)]::Vector{Float64}
    xp = (log.(pProbs).-log(pProbs[1]))[2:end]
    xq = (p.lq.-p.lq[1])[2:end]
    δ2 = 2 * exp(p.lt[1, 3])
    τ = exp(p.lt[1, 5])
    eps = exp(p.lt[3, 2])
    xt = zeros(3)
    xt[1:2] .= log.([δ2, τ]) .- log(1 - δ2 - τ)
    xt[3] = logit(eps / (1 - τ))
    vcat(xp, xq, xt)
end


#---
function transformBack(
    x::Vector{Float64},
    pArr::Matrix{Int},
    pM::Matrix{Float64},
)
    nSymbols = size(pArr, 1)
    nP = (nSymbols * (nSymbols + 1)) ÷ 2 - 1
    nQ = nSymbols - 1
    nT = 3
    xp = x[1:nP]
    pProbs = Flux.softmax(vcat(0, xp))
    lp = log.(pProbs[pArr] .* pM)
    xq = x[nP.+(1:nQ)]
    lq = log.(Flux.softmax(vcat(0, xq)))

    xt = x[nP+nQ.+(1:nT)]
    tm = zeros(4, 5)
    _, δ2, τ = Flux.softmax([0, xt[1], xt[2]])
    δ = δ2 / 2
    eps = logistic(xt[3]) * (1 - τ)
    tm = [
        0.0 1 - 2δ - τ δ δ τ
        0.0 1 - 2δ - τ δ δ τ
        0.0 eps 1 - eps - τ 0 τ
        0 eps 0 1 - eps - τ τ
    ]
    lt = log.(tm)
    lt, lp, lq
end

#---

function ∇forward(
    w1::String,
    w2::String,
    x::Vector{Float64},
    alphabet::Vector{Char},
    pArr::Matrix{Int},
    pM::Matrix{Float64},
)
    v1 = Vector{Int}(indexin(w1, alphabet))
    v2 = Vector{Int}(indexin(w2, alphabet))
    gradient(x -> forwardD(
        v1, v2,
        transformBack(x, pArr, pM)...),
        x)[1]
end

#---

function ∇conditionalLL(
    w1::String,
    w2::String,
    x::Vector{Float64},
    alphabet::Vector{Char},
    pArr::Matrix{Int},
    pM::Matrix{Float64},
)
    v1 = Vector{Int}(indexin(w1, alphabet))
    v2 = Vector{Int}(indexin(w2, alphabet))
    gradient(
        x ->
            forwardD(v1, v2, transformBack(x, pArr, pM)...) -
            forwardD0(v1, v2, transformBack(x, pArr, pM)...),
        x,
    )[1]
end
