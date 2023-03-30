using CSV
using Odesa

feast_layer_nNeurons::Int16 = 20# pop_size*2
feast_layer_eta::Float16 = 0.001
feast_layer_threshEta::Float16 = 0.001
feast_layer_thresholdOpen::Float16 = 0.01
feast_layer_tau::Float16 =  1.0/Int(round(sum(unique(times))/(pop_size*2)))#/2.0)/2.0#0.464
# This doesn't matter, it is used in ODESA but not in FEAST 
feast_layer_traceTau::Float16 = 0.81
precision = 16

convert_precision_float(x,precision) = eval("convert(x,Float{$precision})")
convert_precision_Int(x,precision) = eval("convert(x,Int{$precision})")

# Create a Feast layer with the above parameters
feast_layer = Odesa.Feast.FC(precision,Int16(1),UInt16(pop_size*2),feast_layer_nNeurons,feast_layer_eta,feast_layer_threshEta,feast_layer_thresholdOpen,feast_layer_tau,feast_layer_traceTau)

perm = sortperm(times)
nodes = nodes[perm]
times = times[perm]
winners = []
p1=plot(feast_layer.thresh)
display(SNN.raster([E,I]))
for i in 1:325
    Odesa.Feast.reset_time(feast_layer)
    for (y,ts) in zip(nodes,times)
        winner = Odesa.Feast.forward(feast_layer, Int32(1), Int32(y), ts)    
        if i==325
            distances = feast_layer.dot_prod
            append!(winners,winner)
        end
        
    end
    layer.event_context .= layer.event_context ./ norm(layer.event_context)
    #temp = zeros(feast_layer.dot_prod)
    #mul!(temp, feast_layer.w', view(feast_layer.event_context, :))
    display(plot!(p1,feast_layer.thresh,legend=false))
end
