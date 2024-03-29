mutable struct NW{T}
    alphabet::Vector{T}
    s::Dict{Tuple{T,T},Float64}
    gp1::Float64 # must be non-negative!
    gp2::Float64 # must be non-negative!
end


function nwAlign!(
    dp::Array{Float64,3},
    pt::Array{Int,3},
    w1::Union{AbstractString, Vector{T}},
    w2::Union{AbstractString, Vector{T}},
    p::NW{T},
) where {T}
    @argcheck all([s ∈ p.alphabet for s in w1])
    @argcheck all([s ∈ p.alphabet for s in w2])
    n, m = length(w1), length(w2)
    @argcheck size(dp) <= (n + 1, m + 1, 3)
    @argcheck size(pt) <= (n + 1, m + 1, 3)
    fill!(dp, -Inf)
    fill!(pt, 0)

    dp[2, 2, 1] = p.s[w1[1], w2[1]]
    dp[2, 1, 2] = -p.gp1
    dp[1, 2, 3] = -p.gp1
    pt[2, 2, 1] = -1
    pt[2, 1, 2] = -1
    pt[1, 2, 3] = -1

    for j = 1:(m+1), i = 1:(n+1)
        if i > 1 && j > 1 && (i, j) != (2, 2)
            dp[i, j, 1], pt[i, j, 1] =
                findmax(dp[i-1, j-1, :])
            dp[i, j, 1] += p.s[w1[i-1], w2[j-1]]
        end
        if i > 1 && (i, j) != (2, 1)
            dp[i, j, 2], pt[i, j, 2] =
                findmax(dp[i-1, j, :] + [-p.gp1, -p.gp2, -Inf])
        end
        if j > 1 && (i, j) != (1, 2)
            dp[i, j, 3], pt[i, j, 3] =
                findmax(dp[i, j-1, :] + [-p.gp1, -Inf, -p.gp2])
        end
    end
    llMax, finalpt = findmax(dp[n+1, m+1, :])
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

    a = Union{T, Missing}[]
    b = Union{T, Missing}[]
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
    return (alignment = convert(Matrix{Union{Missing, T}},[a b]), score = llMax::Float64)
end

#---

function nwAlign(
    w1::Union{AbstractString, Vector{T}},
    w2::Union{AbstractString, Vector{T}},
    p::NW{T},
) where {T}
    dp = Array{Float64,3}(undef, length(w1) + 1, length(w2) + 1, 3)
    pt = Array{Int,3}(undef, length(w1) + 1, length(w2) + 1, 3)
    nwAlign!(dp, pt, w1, w2, p)
end

#---

function nw!(
    dp::Array{Float64,3},
    w1::Union{AbstractString, Vector},
    w2::Union{AbstractString, Vector},
    p::NW,
)
    @argcheck all([s ∈ p.alphabet for s in w1])
    @argcheck all([s ∈ p.alphabet for s in w2])
    n, m = length(w1), length(w2)
    @argcheck size(dp) <= (n + 1, m + 1, 3)
    fill!(dp, -Inf)

    dp[2, 2, 1] = p.s[w1[1], w2[1]]
    dp[2, 1, 2] = -p.gp1
    dp[1, 2, 3] = -p.gp1

    for j = 1:(m+1), i = 1:(n+1)
        if i > 1 && j > 1 && (i, j) != (2, 2)
            dp[i, j, 1] =
                maximum(dp[i-1, j-1, :])
            dp[i, j, 1] += p.s[w1[i-1], w2[j-1]]
        end
        if i > 1 && (i, j) != (2, 1)
            dp[i, j, 2] =
                maximum(dp[i-1, j, :] + [-p.gp1, -p.gp2, -Inf])
        end
        if j > 1 && (i, j) != (1, 2)
            dp[i, j, 3] =
                maximum(dp[i, j-1, :] + [-p.gp1, -Inf, -p.gp2])
        end
    end
    maximum(dp[n+1, m+1, :])
end

function nw(
    w1::Union{AbstractString, Vector{T}},
    w2::Union{AbstractString, Vector{T}},
    p::NW{T}
) where {T}
    dp = Array{Float64,3}(undef, length(w1) + 1, length(w2) + 1, 3)
    nw!(dp, w1, w2, p)
end
