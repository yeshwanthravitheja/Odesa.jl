module Odesa
# using MKL
using LinearAlgebra
using BenchmarkTools
using Traceur
using Random
using ProgressMeter
using Dates
using JLD2
using TimerOutputs
# using Threads

include("fc_odesa_mod.jl")
include("feast_mod.jl")
import .Feast
import .FullyConnected
using Statistics
end
