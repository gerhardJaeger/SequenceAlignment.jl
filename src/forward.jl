function forward!(dp::Array{Float64,3}, w1::String, w2::String, p::Phmm)
    @argcheck all([s ∈ p.alphabet for s in w1])
    @argcheck all([s ∈ p.alphabet for s in w2])
    n, m = length(w1), length(w2)
    @argcheck size(dp) <= (n + 1, m + 1, 3)
    fill!(dp, -Inf)

    dp[2, 2, 1] = p.α[1] + p.lp[w1[1], w2[1]]
    dp[2, 1, 2] = p.α[2] + p.lq[w1[1]]
    dp[1, 2, 3] = p.α[3] + p.lq[w2[1]]

    for j = 1:(m+1), i = 1:(n+1)
        if i > 1 && j > 1 && (i, j) != (2, 2)
            dp[i, j, 1] = logsumexp(dp[i-1, j-1, :] + p.lt[:, 1])
            dp[i, j, 1] += p.lp[w1[i-1], w2[j-1]]
        end
        if i > 1 && (i, j) != (2, 1)
            dp[i, j, 2] = logsumexp(dp[i-1, j, :] + p.lt[:, 2])
            dp[i, j, 2] += p.lq[w1[i-1]]
        end
        if j > 1 && (i, j) != (1, 2)
            dp[i, j, 3] = logsumexp(dp[i, j-1, :] + p.lt[:, 3])
            dp[i, j, 3] += p.lq[w2[j-1]]
        end
    end
    logsumexp(dp[n+1, m+1, :] + (p.τ))
end

function forward(w1::String, w2::String, p::Phmm)
    dp = Array{Float64,3}(undef, length(w1) + 1, length(w2) + 1, 3)
    forward!(dp, w1, w2, p)
end


#---

function forward0!(dp::Array{Float64,3}, n::Int, m::Int, p::Phmm)
    @argcheck size(dp) <= (n + 1, m + 1, 3)
    fill!(dp, -Inf)

    dp[2, 2, 1] = p.α[1]
    dp[2, 1, 2] = p.α[2]
    dp[1, 2, 3] = p.α[3]

    for j = 1:(m+1), i = 1:(n+1)
        if i > 1 && j > 1 && (i, j) != (2, 2)
            dp[i, j, 1] = logsumexp(dp[i-1, j-1, :] + p.lt[:, 1])
        end
        if i > 1 && (i, j) != (2, 1)
            dp[i, j, 2] = logsumexp(dp[i-1, j, :] + p.lt[:, 2])
        end
        if j > 1 && (i, j) != (1, 2)
            dp[i, j, 3] = logsumexp(dp[i, j-1, :] + p.lt[:, 3])
        end
    end
    logsumexp(dp[n+1, m+1, :] + (p.τ))
end

@memoize function forward0(n::Int, m::Int, p::Phmm)
    dp = Array{Float64,3}(undef, n + 1, m + 1, 3)
    forward0!(dp, n, m, p)
end
