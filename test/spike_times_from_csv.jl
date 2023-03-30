using CSV
using Odesa
using CSV, Tables, DataFrames
using Revise
using Plots
import Plots.plot
using StatsBase

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
    distances = feast_layer.dot_prod

    for i in 1:325
        Odesa.Feast.reset_time(feast_layer)
        for (y,ts) in zip(nodes,times)
            winner = Odesa.Feast.forward(feast_layer, Int16(1), Int16(y), Float16(ts))    
            distances = feast_layer.dot_prod
            
        end
        display(plot!(p1,feast_layer.thresh,legend=false))
    end
    distances
end
distances = collect_distances(feast_layer,nodes,times)
@assert length(distances) !=0
@assert mean(distances) !=0
function get_ts(nodes,times)
    #num_spikes = length(nodes)
    # The temporal resolution of the final timesurface
    dt = 10
    num_neurons = Int(length(unique(nodes)))+1#int(df.max(axis=0)['x1'])
    total_time =  Int(maximum(times))
    time_resolution = Int(round(total_time/dt))
    # Final output. 
    final_timesurf = zeros((num_neurons, time_resolution+1))
    # Timestamp and membrane voltage store for generating time surface
    timestamps = zeros((num_neurons)) .- Inf
    mv = zeros((num_neurons))
    tau = 200
    last_t = 0
    for (tt,nn) in zip(times,nodes)
        #Get the current spike
        neuron = Int(nn) 
        time = Int(tt)        
        # If time of the next spikes leaps over, make sure to generate 
        # timesurfaces for all the intermediate dt time intervals and fill the 
        # final_timesurface.
        if time > last_t
            timesurf = similar(final_timesurf[:,1])
            for t in collect(last_t:dt:time)
                @. timesurf = mv*exp((timestamps-t)/tau)
                final_timesurf[:,1+Int(round(t/dt))] = timesurf
            end
            last_t = time
        end
        # Update the membrane voltage of the time surface based on the last value and time elapsed
        mv[neuron] =mv[neuron]*exp((timestamps[neuron]-time)/tau) +1
        timestamps[neuron] = time
        # Update the latest timestamp at the channel. 
    end
    # Generate the time surface for the rest of the time if there exists no other spikes. 
    timesurf = similar(final_timesurf[:,1])
    for t in collect(last_t:dt:total_time)
        @. timesurf = mv*exp((timestamps-t)/tau)
        final_timesurf[:,1+Int(round(t/dt))] = timesurf
    end
    return final_timesurf

end
final_timesurf = get_ts(nodes,times);
@show(sum(final_timesurf))
display(Plots.heatmap(final_timesurf))
@assert mean(final_timesurf) !=0
