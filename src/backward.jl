function backward!(dp::Array{Float64,3}, w1::String, w2::String, p::Phmm)
    n, m = length(w1), length(w2)
    fill!(dp, Float64(-Inf))
    dp[n+1, m+1, :] = p.τ
    for j = (m+1):-1:1, i = (n+1):-1:1
        for s = 1:3
            if i <= n && j <= m
                dp[i, j, s] = logaddexp(
                    dp[i, j, s],
                    dp[i+1, j+1, 1] + p.lp[w1[i],w2[j]] + p.lt[s, 1],
                )
            end
            if i <= n
                dp[i, j, s] = logaddexp(
                    dp[i, j, s],
                    dp[i+1, j, 2] + p.lq[w1[i]] + p.lt[s, 2],
                )
            end
            if j <= m
                dp[i, j, s] = logaddexp(
                    dp[i, j, s],
                    dp[i, j+1, 3] + p.lq[w2[j]] + p.lt[s, 3],
                )
            end
        end
    end
    logProb = logsumexp([
        dp[2, 2, 1] + p.α[1] + p.lp[w1[1], w2[1]],
        dp[2, 1, 2] + p.α[2] + p.lq[w1[1]],
        dp[1, 2, 3] + p.α[3] + p.lq[w2[1]],
    ])

end


function backward(w1::String, w2::String, p::Phmm)
    dp = Array{Float64,3}(undef, length(w1) + 1, length(w2) + 1, 3)
    forward!(dp, w1, w2, p)
end
