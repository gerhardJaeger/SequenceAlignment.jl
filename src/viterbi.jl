function viterbi!(
    dp::Array{Float64,3},
    pt::Array{Int,3},
    w1::Union{AbstractString,T},
    w2::Union{AbstractString,T},
    p::Phmm{T},
) where T
    @argcheck all([s ∈ p.alphabet for s in w1])
    @argcheck all([s ∈ p.alphabet for s in w2])
    v1 = indexin(w1, p.alphabet)
    v2 = indexin(w2, p.alphabet)
    n, m = length(w1), length(w2)
    @argcheck size(dp) <= (n + 1, m + 1, 3)
    @argcheck size(pt) <= (n + 1, m + 1, 3)
    fill!(dp, -Inf)
    fill!(pt, 0)

    dp[2, 2, 1] = p.lt[1, 2] + p.lp[v1[1], v2[1]]
    dp[2, 1, 2] = p.lt[1, 3] + p.lq[v1[1]]
    dp[1, 2, 3] = p.lt[1, 4] + p.lq[v2[1]]
    pt[2, 2, 1] = -1
    pt[2, 1, 2] = -1
    pt[1, 2, 3] = -1

    for j = 1:(m+1), i = 1:(n+1)
        if i > 1 && j > 1 && (i, j) != (2, 2)
            dp[i, j, 1], pt[i, j, 1] = findmax(dp[i-1, j-1, :] + p.lt[2:4, 2])
            dp[i, j, 1] += p.lp[v1[i-1], v2[j-1]]
        end
        if i > 1 && (i, j) != (2, 1)
            dp[i, j, 2], pt[i, j, 2] = findmax(dp[i-1, j, :] + p.lt[2:4, 3])
            dp[i, j, 2] += p.lq[v1[i-1]]
        end
        if j > 1 && (i, j) != (1, 2)
            dp[i, j, 3], pt[i, j, 3] = findmax(dp[i, j-1, :] + p.lt[2:4, 4])
            dp[i, j, 3] += p.lq[v2[j-1]]
        end
    end
    llMax, finalpt = findmax(dp[n+1, m+1, :] + (p.lt[2:4, 5]))
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
            (i, j, current) =
                (i - increments[current, 1], j - increments[current, 2], pt[i, j, current])
        end
    end

    a = Union{T,Missing}[]
    b = Union{T,Missing}[]
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
                push!(b, missing)
            else
                push!(a, missing)
                push!(b, w2[j])
                j += 1
            end
        end
    end
    return (alignment = [a b], logprob = llMax::Float64)
end

function viterbi(
    w1::Union{AbstractString, T},
    w2::Union{AbstractString, T},
    p::Phmm{T}
) where {T}
    dp = Array{Float64,3}(undef, length(w1) + 1, length(w2) + 1, 3)
    pt = Array{Int,3}(undef, length(w1) + 1, length(w2) + 1, 3)
    viterbi!(dp, pt, w1, w2, p)
end
