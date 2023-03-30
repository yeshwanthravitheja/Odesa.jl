using CSV
using Odesa
using CSV, Tables, DataFrames
using Revise
using Plots
import Plots.plot
df = CSV.read("times_for_yesh.csv",DataFrame)
nodes = df.x1
times = df.x2
pop_size = length(unique(df.x1))
feast_layer_nNeurons::Int16 = 20
feast_layer_eta::Float16 = 0.001
feast_layer_threshEta::Float16 = 0.001
feast_layer_thresholdOpen::Float16 = 0.01
feast_layer_tau::Float32 =  1.0/Int(round(sum(unique(times))/(pop_size*2)))
feast_layer_traceTau::Float16 = 0.81
precision::UInt16 = convert(UInt16,0)  

feast_layer = Odesa.Feast.FC(precision,Int16(1),UInt16(pop_size),feast_layer_nNeurons,feast_layer_eta,feast_layer_threshEta,feast_layer_thresholdOpen,feast_layer_tau,feast_layer_traceTau)

perm = sortperm(times)
nodes = nodes[perm]
times = times[perm]
winners = []
p1=plot(feast_layer.thresh)
function collect_distances(feast_layer,nodes,times)
    for i in 1:325
        Odesa.Feast.reset_time(feast_layer)
        for (y,ts) in zip(nodes,times)
            winner = Odesa.Feast.forward(feast_layer, Int16(1), Int16(y), Float16(ts))    
            if i==324
                distances = feast_layer.dot_prod
                append!(winners,winner)
            end
            
        end
        display(plot!(p1,feast_layer.thresh,legend=false))
    end
    distances
end
distances = collect_distances(feast_layer,nodes,times)

#@assert 