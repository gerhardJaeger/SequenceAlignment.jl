#__precompile__(true)


module SequenceAlignment

using ArgCheck
using Distributions
using LinearAlgebra
using StatsFuns
using DataStructures
using JSON


include("levenshtein.jl")

export levenshteinAlign
export levenshteinDistance
export ldn

include("phmm.jl")
export Phmm

include("viterbi.jl")
export viterbi

include("forward.jl")
export forward

include("backward.jl")
export backward

include("baumWelch.jl")
export baumWelch
export phmmExpectations
export Phmm
export print
export read
export write
export randomPhmm
export uniformPhmm


end
