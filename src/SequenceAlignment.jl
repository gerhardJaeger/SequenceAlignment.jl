__precompile__(true)


module SequenceAlignment

using ArgCheck
using Distributions
using LinearAlgebra
using StatsFuns
using DataStructures


include("levenshtein.jl")

export levenshteinAlign
export levenshteinDistance
export ldn

include("phmm.jl")
include("viterbi.jl")
include("forward.jl")
include("backward.jl")


export Phmm
export viterbi
export forward
export backward


end
