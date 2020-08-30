function backward!(dp::Array{Float64,3}, w1::String, w2::String, p::Phmm)
    n, m = length(w1), length(w2)
    v1 = indexin(w1, p.alphabet)
    v2 = indexin(w2, p.alphabet)
    fill!(dp, Float64(-Inf))
    dp[n+1, m+1, :] = p.lt[2:4,5]
    for j = (m+1):-1:1, i = (n+1):-1:1
        for s = 1:3
            if i <= n && j <= m
                dp[i, j, s] = logaddexp(
                    dp[i, j, s],
                    dp[i+1, j+1, 1] + p.lp[v1[i],v2[j]] + p.lt[s+1, 2],
                )
            end
            if i <= n
                dp[i, j, s] = logaddexp(
                    dp[i, j, s],
                    dp[i+1, j, 2] + p.lq[v1[i]] + p.lt[s+1, 3],
                )
            end
            if j <= m
                dp[i, j, s] = logaddexp(
                    dp[i, j, s],
                    dp[i, j+1, 3] + p.lq[v2[j]] + p.lt[s+1, 4],
                )
            end
        end
    end
    logProb = logsumexp([
        dp[2, 2, 1] + p.lt[1,2] + p.lp[v1[1], v2[1]],
        dp[2, 1, 2] + p.lt[1,3] + p.lq[v1[1]],
        dp[1, 2, 3] + p.lt[1,4] + p.lq[v2[1]],
    ])

end


function backward(w1::String, w2::String, p::Phmm)
    dp = Array{Float64,3}(undef, length(w1) + 1, length(w2) + 1, 3)
    backward!(dp, w1, w2, p)
end
