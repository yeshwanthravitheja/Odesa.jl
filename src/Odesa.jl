module Odesa
# using MKL
using LinearAlgebra
using BenchmarkTools
using Traceur
using Random
using ProgressBars
using Dates
using JLD
using TimerOutputs
# using Threads

include("fc_odesa_mod.jl")
include("feast_mod.jl")
import .Feast
import .FullyConnected
using Statistics
using Dates
end
