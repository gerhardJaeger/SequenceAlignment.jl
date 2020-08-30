function forward!(dp::Array{Float64,3}, w1::String, w2::String, p::Phmm)
#    @argcheck all([s ∈ p.alphabet for s in w1])
#    @argcheck all([s ∈ p.alphabet for s in w2])
    n, m = length(w1), length(w2)
#    @argcheck size(dp) <= (n + 1, m + 1, 3)
    fill!(dp, -Inf)
    v1 = indexin(w1, p.alphabet)
    v2 = indexin(w2, p.alphabet)

    dp[2, 2, 1] = p.lt[1, 2] + p.lp[v1[1], v2[1]]
    dp[2, 1, 2] = p.lt[1, 3] + p.lq[v1[1]]
    dp[1, 2, 3] = p.lt[1, 4] + p.lq[v2[1]]

    for j = 1:(m+1), i = 1:(n+1)
        if i > 1 && j > 1 && (i, j) != (2, 2)
            dp[i, j, 1] =
                logsumexp([dp[i-1, j-1, k] + p.lt[k+1, 2]
                            for k = 1:3]) + p.lp[v1[i-1], v2[j-1]]
        end
        if i > 1 && (i, j) != (2, 1)
            dp[i, j, 2] =
                logsumexp([dp[i-1, j, k] + p.lt[k+1, 3]
                            for k = 1:3]) + p.lq[v1[i-1]]
        end
        if j > 1 && (i, j) != (1, 2)
            dp[i, j, 3] =
                logsumexp([dp[i, j-1, k] + p.lt[k+1, 4]
                            for k = 1:3]) + p.lq[v2[j-1]]
        end
    end
    logsumexp([dp[n+1, m+1, k] + p.lt[k+1, 5] for k in 1:3])
end

function forward(w1::String, w2::String, p::Phmm)
    n,m = length.([w1, w2])
    dp = Array{Float64,3}(undef, n + 1, m + 1, 3)
    forward!(dp, w1, w2, p)
end


#---

function forward0!(dp::Array{Float64,3}, n::Int, m::Int, p::Phmm)
    @argcheck size(dp) <= (n + 1, m + 1, 3)
    fill!(dp, -Inf)

    dp[2, 2, 1] = p.lt[1,2]
    dp[2, 1, 2] = p.lt[1,3]
    dp[1, 2, 3] = p.lt[1,4]

    for j = 1:(m+1), i = 1:(n+1)
        if i > 1 && j > 1 && (i, j) != (2, 2)
            dp[i, j, 1] = logsumexp(dp[i-1, j-1, :] + p.lt[2:4, 2])
        end
        if i > 1 && (i, j) != (2, 1)
            dp[i, j, 2] = logsumexp(dp[i-1, j, :] + p.lt[2:4, 3])
        end
        if j > 1 && (i, j) != (1, 2)
            dp[i, j, 3] = logsumexp(dp[i, j-1, :] + p.lt[2:4, 4])
        end
    end
    logsumexp(dp[n+1, m+1, :] + p.lt[2:4,5])
end

@memoize function forward0(n::Int, m::Int, p::Phmm)
    dp = Array{Float64,3}(undef, n + 1, m + 1, 3)
    forward0!(dp, n, m, p)
end
