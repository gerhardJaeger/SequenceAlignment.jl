


@doc raw"""
`levenshteinAlign` returns the Levenshtein-optimal pairwise alignment
between two sequences (abstract string or vector) as a two-column
matrix
"""
function levenshteinAlign(s1::T, s2::T) where {T<:Union{AbstractString,Vector}}
    n, m = length.([s1, s2])
    dp = Matrix{Int}(undef, n + 1, m + 1)
    pointer = Matrix{Int}(undef, n + 1, m + 1)
    dp[:, 1] .= 0:n
    dp[1, :] .= 0:m
    pointer[:, 1] .= 1
    pointer[1, :] .= 2
    pointer[1, 1] = -1
    for i = 1:m
        for j = 1:n
            insrt = dp[j, i+1] + 1
            dlt = dp[j+1, i] + 1
            mtch = dp[j, i] + (s1[j] != s2[i])
            states = [mtch, insrt, dlt]
            dp[j+1, i+1] = minimum(states)
            pointer[j+1, i+1] = argmin(states) - 1
        end
    end
    a1 = Union{eltype(s1), Missing}[]
    a2 = Union{eltype(s2), Missing}[]
    i, j = m + 1, n + 1
    while pointer[j, i] >= 0
        p = pointer[j, i]
        if p == 0
            pushfirst!(a1, s1[j-1])
            pushfirst!(a2, s2[i-1])
            i -= 1
            j -= 1
        elseif p == 1
            pushfirst!(a1, s1[j-1])
            pushfirst!(a2, missing)
            j -= 1
        else
            pushfirst!(a1, missing)
            pushfirst!(a2, s2[i-1])
            i -= 1
        end
    end
    hcat(a1, a2)
end

##
@doc raw"""
`levenshteinDistance` computes the edit distance between two iterables
(abstract string or vector)
"""
function levenshteinDistance(
    s1::T,
    s2::T,
) where {T<:Union{AbstractString,Vector}}
    n::Int = length(s1)
    m::Int = length(s2)
    dp = Matrix{Int}(undef, n + 1, m + 1)
    dp[1:end, 1] .= 0:n
    dp[1, 1:end] .= 0:m
    for i = 1:m, j = 1:n
        insrt = dp[j, i+1] + 1
        dlt = dp[j+1, i] + 1
        mtch = dp[j, i] + (s1[j] != s2[i])
        states = [mtch, insrt, dlt]
        dp[j+1, i+1]::Int = minimum(states)
    end
    dp[end, end]
end

##

@doc raw"""
`ldn` computes the normalized edit distance between two iterables
(abstract string or vector), i.e.
$L(x,y) /max(|x|, |y|)$
"""
function ldn(s1::T, s2::T) where {T<:Union{AbstractString,Vector}}
    levenshteinDistance(s1, s2) / maximum(length.([s1, s2]))
end
