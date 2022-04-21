module SequenceAlignment


using ArgCheck
using Distributions
using LinearAlgebra
using StatsFuns
import StatsFuns: logsumexp
using DataStructures
using JSON
using Memoize
using Flux


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
export forward0
export conditionalLL

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

include("forwardGradient.jl")

export ∇forward
export transformParameters
export transformBack
export auxArrays
export ∇conditionalLL


include("needlemanWunsch.jl")
export NW
export nw
export nwAlign


end
