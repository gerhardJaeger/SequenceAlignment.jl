function viterbi!(
    dp::Array{Float64,3},
    pt::Array{Int,3},
    w1::String,
    w2::String,
    p::Phmm,
)
    @argcheck all([s ∈ p.alphabet for s in w1])
    @argcheck all([s ∈ p.alphabet for s in w2])
    n, m = length(w1), length(w2)
    @argcheck size(dp) <= (n + 1, m + 1, 3)
    @argcheck size(pt) <= (n + 1, m + 1, 3)
    fill!(dp, -Inf)
    fill!(pt, 0)

    dp[2, 2, 1] = p.α[1] + p.lp[w1[1], w2[1]]
    dp[2, 1, 2] = p.α[2] + p.lq[w1[1]]
    dp[1, 2, 3] = p.α[3] + p.lq[w2[1]]
    pt[2, 2, 1] = -1
    pt[2, 1, 2] = -1
    pt[1, 2, 3] = -1

    for j = 1:(m+1), i = 1:(n+1)
        if i > 1 && j > 1 && (i, j) != (2, 2)
            dp[i, j, 1], pt[i, j, 1] = findmax(dp[i-1, j-1, :] + p.lt[:, 1])
            dp[i, j, 1] += p.lp[w1[i-1], w2[j-1]]
        end
        if i > 1 && (i, j) != (2, 1)
            dp[i, j, 2], pt[i, j, 2] = findmax(dp[i-1, j, :] + p.lt[:, 2])
            dp[i, j, 2] += p.lq[w1[i-1]]
        end
        if j > 1 && (i, j) != (1, 2)
            dp[i, j, 3], pt[i, j, 3] = findmax(dp[i, j-1, :] + p.lt[:, 3])
            dp[i, j, 3] += p.lq[w2[j-1]]
        end
    end
    llMax, finalpt = findmax(dp[n+1, m+1, :] + (p.τ))
    increments = [
        1 1
        1 0
        0 1
    ]
    path = Int[]
    let (i, j) = size(pt)
        current = finalpt
        while current != -1
            pushfirst!(path, current)
            (i, j, current) = (
                i - increments[current, 1],
                j - increments[current, 2],
                pt[i, j, current],
            )
        end
    end

    a = eltype(w1)[]
    b = eltype(w2)[]
    let (i, j) = (1, 1)
        for x in path
            if x == 1
                push!(a, w1[i])
                i += 1
                push!(b, w2[j])
                j += 1
            elseif x == 2
                push!(a, w1[i])
                i += 1
                push!(b, '-')
            else
                push!(a, '-')
                push!(b, w2[j])
                j += 1
            end
        end
    end
    return (alignment = [a b]::Matrix{Char}, logprob = llMax::Float64)
end

function viterbi(w1::String, w2::String, p::Phmm)
    dp = Array{Float64,3}(undef, length(w1) + 1, length(w2) + 1, 3)
    pt = Array{Int,3}(undef, length(w1) + 1, length(w2) + 1, 3)
    viterbi!(dp, pt, w1, w2, p)
end
