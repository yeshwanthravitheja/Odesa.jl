
# Equations being implemented:

# https://ieeexplore.ieee.org/document/9864144
module FullyConnected
    using LinearAlgebra
    using JLD
    using Revise
    using SparseArrays
    using Traceur
    using Distributed
    # using Threads
    mutable struct FC
        #TODO:  Discuss with @Russell regd using Float32 or Float64 

        #Layer input dimensions #TODO: This need not be stored. It is useless for FC
        in_rows::Int32
        in_cols::Int32

        # Flattened input dimensions
        context_size::Int32

        #Hyper Parameters for learning

        # Number of neurons in the layer
        nNeurons::Int32
        # Learning rate of the weight update
        eta::Float32
        # Learning rate of the threshold update
        threshEta::Float32
        # Threshold Open constant
        thresholdOpen::Float32
        # Time constant of the timesurface
        tau::Float32
        # TIme constant used for keeping a trace of the layer
        traceTau::Float32

        # Weights of the neurons
        w::Matrix{Float32}
        #Threholds of the neurons
        thresh::Array{Float32,1}

        # Delta stores the latest input context for which the neuron fired. 
        delta::Array{Float32,2}
        # DeltaThresh stores the latest dotproduct when the neuron fired.
        deltaThresh::Array{Float32,1}

        # timestamps and polarity are used to calculate the timesurface. 
        # They keep track of the latest time of the event from a given input channel. 
        timestamps::Array{Float32,2}
        polarity::Array{Float32,2}

        # WinnerTrace is allocated memory to keep track of the firing times of the neurons in the layer. 
        winnerTrace::Array{Float32,1}
        # WinnerMV is also utilized to calculate the trace of the layer
        winnerMV::Array{Float32,1}

        # NoWinnerTrace remembers the last time at which there was no winner
        noWinnerTrace::Float32

        ### Pre-allocated memories for commonly calculated matrices/Vectors along the training

        # tempTrace is pre-allocated memory to populate with calculated trace of the neurons
        tempTrace::Array{Float32,1}
        # Event context is pre-allocated memory to hold the event contexts
        event_context::Array{Float32,2}
        # Dot Product is the pre-allocated memory to hold the dot products
        dot_prod::Array{Float32,1}

        function FC(
            input_rows::Int32,
            input_cols::Int32,
            nNeurons::Int32,
            eta::Float32,
            threshEta::Float32,
            thresholdOpen::Float32,
            tau::Float32,
            traceTau::Float32
        )

            #Flattened dimension size
            context_size::Int32 = input_rows * input_cols 

            event_context =
                zeros(Float32, input_rows, input_cols)::Array{Float32,2}


            w = rand(Float32, context_size, nNeurons)::Matrix{Float32}


            #@inbounds 

            ##
            # slower
            ##
            #@inbounds for row in eachrow(w)                
             #   row = row ./ norm(row)
            #end
            

            ##
            # faster
            ##
            @inbounds for row in range(1, size(w, 2), step = 1)
                w[:, row] = w[:, row] ./ norm(w[:, row])
            end

            thresh = zeros(Float32, nNeurons)::Array{Float32,1}
            dot_prod = zeros(Float32, nNeurons)::Array{Float32,1}

            delta = zeros(Float32, context_size, nNeurons)::Array{Float32,2}
            deltaThresh = zeros(Float32, nNeurons)::Array{Float32,1}

            #TODO: At some point we have to make the timestamp matrix more generic
            #depending on if the timestamps are Integer or Floating point value. 
            timestamps = Array{Float32,2}(undef, input_rows, input_cols)
            polarity  = Array{Float32,2}(undef, input_rows, input_cols)

            winnerTrace = Array{Float32,1}(undef, nNeurons)
            winnerMV = zeros(Float32, nNeurons)::Array{Float32,1}
            tempTrace = zeros(Float32, nNeurons)::Array{Float32,1}

            noWinnerTrace = Float32(0.0)
            new(
                input_rows,
                input_cols,
                context_size,
                nNeurons,
                eta,
                threshEta,
                thresholdOpen,
                tau,
                traceTau,
                w,
                thresh,
                delta,
                deltaThresh,
                timestamps,
                polarity,
                winnerTrace,
                winnerMV,
                noWinnerTrace,
                tempTrace,
                event_context,
                dot_prod,
            )

        end

    end

    function reset_time(layer::FC)
        """
        Reset the timer of the layer. All the time stamps and polarities will be reset. 
        All the traces will be set to 0. reset_time has the same effect as no input events for a really
        long time.
        """

        fill!(layer.delta, zero(eltype(layer.delta))) 
        fill!(layer.deltaThresh, zero(eltype(layer.deltaThresh)))
        
        fill!(layer.timestamps, typemin(eltype(layer.timestamps))) 
        fill!(layer.polarity, zero(eltype(layer.polarity))) 

        fill!(layer.winnerTrace, typemin(eltype(layer.winnerTrace)))
        fill!(layer.winnerMV, zero(eltype(layer.winnerTrace)))

        layer.noWinnerTrace = typemin(Float32)

        return nothing

    end

    function add_event(layer::FC, x::Int32, y::Int32, ts::Float32)
        """
        Adds the input event to the timestamp store. Polarity is set to whatever is the decayed value at that
        channel until time ts, and then added 1 to it. 
        """

        layer.polarity[x, y] =
            layer.polarity[x, y] * exp((layer.timestamps[x, y] - ts) / layer.tau) + 1
        layer.timestamps[x, y] = ts

        return nothing

    end

    function add_trace_event(layer::FC, winner::Int32, ts::Float32)
        """
        Adds a trace event/ adds the trace of a recently fired neuron. This works the same as "add_event" but 
        it is keeping track of the firing of own neurons rather than the input spikes. 
        """
        layer.winnerMV[winner] =
            layer.winnerMV[winner] * exp((layer.winnerTrace[winner] - ts) / layer.traceTau) + 1
        layer.winnerTrace[winner] = ts

        return nothing
    end

    function record_individual_neuron(layer::FC,neuron::Int32,ts::Float32,punishFlag::Bool)
        # neuron_trace gives the decaying trace of the neuron's activity
        neuron_trace::Float32 = layer.winnerMV[neuron] * exp((layer.winnerTrace[neuron] - ts) / layer.traceTau)

        if neuron_trace >= 0.1

            reward(layer, neuron)
            
        else
            # If Neuron hasn't spiked in a while and there was a recent input spike for which there was no winner
            if punishFlag
                # Reduce the threshold
                layer.thresh[neuron] = layer.thresh[neuron] - layer.thresholdOpen
            end
        end

        return nothing
    end

    function record(layer::FC, ts::Float32)
        """
        Record next layer attention signal
        """
        punishFlag = exp((layer.noWinnerTrace - ts) / layer.traceTau) >= 0.1

        # Record the attention signal for each neuron
        # Threads.@threads for n::Int32 = 1:layer.nNeurons
        for n::Int32 = 1:layer.nNeurons
            #TODO: Recording the attention signal can be done to each neuron parallely
            record_individual_neuron(layer,n,ts,punishFlag)
        end

        return nothing
    end

    function reward(layer::FC, neuron::Int32)
        """
        Reward individual neuron 
        """

        layer_w_n = view(layer.w, :, neuron)
        layer_delta_n = view(layer.delta, :, neuron)

        # W = W + η*ΔW
        layer_w_n .=  layer_w_n .+ layer.eta .* layer_delta_n

    
        # Normalize the weights
        layer_w_n .= layer_w_n ./ norm(layer_w_n)

        # Updated threshold value based on Thresh = Thresh + η*ΔThresh
        updatedThresh::Float32 = layer.thresh[neuron] +
            layer.threshEta * layer.deltaThresh[neuron]
            
        # If updated threshold is less than 0, then set it to 0. (There is no point of negative thresholds)
        updatedThresh = updatedThresh < 0.0 ? zero(Float32) : updatedThresh
        layer.thresh[neuron] = updatedThresh

        return nothing
    end

    function compute_context(layer::FC,ts::Float32)
        """
        Compute the normalized event context at time ts based on the timestamp and polarity stores. 
        """
        layer.event_context .=
            layer.polarity .*
            exp.(
                (layer.timestamps .- ts) ./
                layer.tau,
            )
        layer.event_context .= layer.event_context ./ norm(layer.event_context)
        #TODO: Ask Russell if it is a good convention to return the layer 
        #every time or just modify the layer inside the functions.
        # return layer
        return nothing
    end

    function forward(layer::FC, x::Int32, y::Int32, ts::Float32)
        """
        Key forward function of the layer. 
            Find the normalized context
            Perform dotproduct
            Find winner neuron
            Save the delta and deltaThresh
        """
        # Check if it is a valid input spike
        if x < 1 || y < 1 
            # Negative winner value indicates no winner. The same convention is used everywhere
            return -1
        end
        # Add event to the timestamp and polarity stores
        add_event(layer, x, y, ts)

        # Compute the event context
        compute_context(layer, ts)
        
        # Find the dotproduct

        # layer.dot_prod .= layer.w  vec(layer.event_context)
        # a = copy(layer.w')
        # b = zeros(Float32, 15, 1)
        # @btime mul!($b,$layer.w',vec($layer.event_context))
        # @btime mul!($layer.dot_prod,$layer.w',$layer.event_context[:])
        mul!(layer.dot_prod, layer.w', view(layer.event_context, :))
        # @btime $layer.dot_prod .= $layer.w' * vec($layer.event_context)

        # dot_prod = layer.w * reshape(layer.event_context, :, 1)
        # @btime all($layer.dot_prod < $layer.thresh)
        ##TODO For Loop
        winnerNeuron::Int32 = -1
        max_value::Float32 = 0.0

        # Find the neuron with highest dot product among the neurons whose dotproduct has
        # crossed their thresholds.
        # Threads.@threads
        @inbounds for neuron = 1:layer.nNeurons
            if layer.dot_prod[neuron] >= layer.thresh[neuron]
                if layer.dot_prod[neuron] > max_value
                    winnerNeuron = neuron
                    max_value = layer.dot_prod[neuron]
                end
            end
        end


        if winnerNeuron > -1
            # If there is a winner, then save the delta and delta thresh which will be used later to reward 
            layer_delta = view(layer.delta, :, winnerNeuron)
            layer_delta .= view(layer.event_context, :) .-  view(layer.w, :, winnerNeuron)
            layer.deltaThresh[winnerNeuron] = layer.deltaThresh[winnerNeuron] - layer.dot_prod[winnerNeuron]
            # Add this winning to the neuron's firing activity
            add_trace_event(layer, winnerNeuron, ts)
        else
            # If there is no winner save the time when there was no winner
            layer.noWinnerTrace = ts

        end

        
        return winnerNeuron

    end

    function punish(layer::FC)
        """
        Punish all the neurons in the layer
        """
        layer.thresh = layer.thresh .- layer.thresholdOpen

        return nothing
    end

    function saveAll(layer::FC, filename::String)
        """
        Save the weights and thresholds in a file using JLD
        """
        save(filename, "w", layer.w, "thresh", layer.thresh)
    end

    function loadAll(layer::FC, filename::String)
        """
        Load the weights and thresholds in a file using JLD
        """
        file_contents = load(filename)
        layer.w = file_contents["w"]
        layer.thresh = file_contents["thresh"]
    end


    mutable struct Classifier
        """
        Classifier/ output layer
        """
        #Layer input dimensions
        #Layer input dimensions #TODO: This need not be stored. It is useless for FC
        in_rows::Int32
        in_cols::Int32
        context_size::Int32

        #Hyper Parameters for learning
        # Total number of neurons in the layer
        nNeurons::Int32
        # Number of neurons for each class 
        nNeurons_per_class::Int32
        # Number of classes 
        nClasses::Int32


        # Learning rate for the weights update
        eta::Float32
        # Learning rate for the threshold update
        threshEta::Float32
        # Threshold Open for punishment
        thresholdOpen::Float32

        # Time Constant for calculating the timesurface
        tau::Float32
        
        # Weights and Thresholds
        w::Matrix{Float32}
        thresh::Array{Float32,1}

        # Delta and DeltaThresh to save the last winning context and dot product
        delta::Array{Float32,2}
        deltaThresh::Array{Float32,1}

        # Timestamp stores. 
        timestamps::Array{Float32,2}
        polarity::Array{Float32,2}

        event_context::Array{Float32,2}
        dot_prod::Array{Float32,1}

        #=
        function norm_weight(w)
            @inbounds for row in range(1, size(w, 1), step = 1)
                w[row, :] = w[row, :] ./ norm(w[row, :])
            end
            w
        end
        =#

        function Classifier(
            input_rows::Int32,
            input_cols::Int32,
            nNeurons_per_class::Int32,
            nClasses::Int32,
            eta::Float32,
            threshEta::Float32,
            thresholdOpen::Float32,
            tau::Float32,
        )
            context_size::Int32 = input_rows * input_cols 
            nNeurons::Int32 = nClasses * nNeurons_per_class

            event_context =
                zeros(Float32, input_rows, input_cols)::Array{Float32,2}


            w = rand(Float32, context_size, nNeurons)::Matrix{Float32}


            #@inbounds 

            ##
            # slower
            ##
            #@inbounds for row in eachrow(w)                
            #   row = row ./ norm(row)
            #end
            

            ##
            # faster
            ##
            @inbounds for row in range(1, size(w, 2), step = 1)
                w[:, row] = w[:, row] ./ norm(w[:, row])
            end

            thresh = zeros(Float32, nNeurons)::Array{Float32,1}
            dot_prod = zeros(Float32, nNeurons)::Array{Float32,1}

            delta = zeros(Float32, context_size, nNeurons)::Array{Float32,2}
            deltaThresh = zeros(Float32, nNeurons)::Array{Float32,1}

            #TODO: At some point we have to make the timestamp matrix more generic
            #depending on if the timestamps are Integer or Floating point value. 
            # timestamps = zeros(Float32, input_rows, input_cols)::Array{Float32,2}
            timestamps = Array{Float32,2}(undef, input_rows, input_cols)
            # polarity = zeros(Float32, input_rows, input_cols)::Array{Float32,2}
            polarity  = Array{Float32,2}(undef, input_rows, input_cols)

            new(
                input_rows,
                input_cols,
                context_size,
                nNeurons,
                nNeurons_per_class,
                nClasses,
                eta,
                threshEta,
                thresholdOpen,
                tau,
                w,
                thresh,
                delta,
                deltaThresh,
                timestamps,
                polarity,
                event_context,
                dot_prod
            )
        end

    end

    function reset_time(layer::Classifier)
        """
        Reset the timer of the layer. All the time stamps and polarities will be reset. 
        All the traces will be set to 0. reset_time has the same effect as no input events for a really
        long time.
        """

        fill!(layer.delta, zero(eltype(layer.delta))) # = layer.delta .*0
        fill!(layer.deltaThresh, zero(eltype(layer.deltaThresh)))# = layer.deltaThresh .* 0
        
        fill!(layer.timestamps, typemin(eltype(layer.timestamps))) #layer.timestamps = 0 .*layer.timestamps
        fill!(layer.polarity, zero(eltype(layer.polarity))) #layer.polarity = 0 .* layer.polarity

    end


    function add_event(layer::Classifier, x::Int32, y::Int32, ts::Float32)
        """
        Adds the input event to the timestamp store. Polarity is set to whatever is the decayed value at that
        channel until time ts, and then added 1 to it. 
        """
        layer.polarity[x, y] =
            layer.polarity[x, y] * exp((layer.timestamps[x, y] - ts) / layer.tau) + 1
        layer.timestamps[x, y] = ts
    end

    function compute_context(layer::Classifier,ts::Float32)
        """
        Compute the normalized event context at time ts based on the timestamp and polarity stores. 
        """
        layer.event_context .=
            layer.polarity .*
            exp.(
                (layer.timestamps .- ts) ./
                layer.tau,
            )
        layer.event_context .= layer.event_context ./ norm(layer.event_context)
        #TODO: Ask Russell if it is a good convention to return the layer 
        #every time or just modify the layer inside the functions.
        # return layer
    end

    function forward(layer::Classifier, x::Int32, y::Int32, ts::Float32)
        """
        Key forward function of the layer. This is the entry point into the layer when there in an input event
            Add event to timestamp and polarity stores
            Compute the normalized event context
            Find winner and winning class. 
            Save the delta and deltaThresh
        """
        if x < 1 || y < 1 
            return -1, -1
        else
            # Add the input event to the timestamp and polarity stores. 
            add_event(layer, x, y, ts)
        end

        # Compute the normalized event context at time ts. 
        compute_context(layer, ts)
        
        # Calculate the dot products 

        # layer.dot_prod .= layer.w  vec(layer.event_context)
        # a = copy(layer.w')
        # b = zeros(Float32, 15, 1)
        # @btime mul!($b,$layer.w',vec($layer.event_context))
        # @btime mul!($layer.dot_prod,$layer.w',$layer.event_context[:])
        mul!(layer.dot_prod, layer.w', view(layer.event_context, :))
        # @btime $layer.dot_prod .= $layer.w' * vec($layer.event_context)

        # dot_prod = layer.w * reshape(layer.event_context, :, 1)
        # @btime all($layer.dot_prod < $layer.thresh)
        ##TODO For Loop
        winnerNeuron::Int32 = -1
        winnerClass::Int32 = -1
        max_value::Float32 = 0.0
        # Find the winner
        @inbounds for neuron = 1:layer.nNeurons
            if layer.dot_prod[neuron] >= layer.thresh[neuron]
                if layer.dot_prod[neuron] > max_value
                    winnerNeuron = neuron
                    max_value = layer.dot_prod[neuron]
                end
            end
        end

        if winnerNeuron > -1
            # If winner, save the delta and deltaThresh and also find the class of the winner neuron
            layer.delta[:, winnerNeuron] .= view(layer.event_context, :) .- view(layer.w, : , winnerNeuron)
            layer.deltaThresh[winnerNeuron] = layer.dot_prod[winnerNeuron] - layer.deltaThresh[winnerNeuron]
            winnerClass = ceil(winnerNeuron / layer.nNeurons_per_class)
        end



        return winnerNeuron, winnerClass

    end


    function reward(layer::Classifier, neuron::Int32)
        """
        Reward individual neuron 
        """

        layer_w_n = view(layer.w, :, neuron)
        layer_delta_n = view(layer.delta, :, neuron)

        # W = W + η*ΔW
        layer_w_n .=  layer_w_n .+ layer.eta .* layer_delta_n

    
        # Normalize the weights
        layer_w_n .= layer_w_n ./ norm(layer_w_n)

        # Updated threshold value based on Thresh = Thresh + η*ΔThresh
        updatedThresh::Float32 = layer.thresh[neuron] +
            layer.threshEta * layer.deltaThresh[neuron]
            
        # If updated threshold is less than 0, then set it to 0. (There is no point of negative thresholds)
        updatedThresh = updatedThresh < zero(Float32) ? zero(Float32) : updatedThresh
        layer.thresh[neuron] = updatedThresh

        return nothing
    end


    function punish(layer::Classifier, winner::Int32, label::Int32)
        """
        Punishing the layer for wrong classification
        """
        if winner > -1
            # Apply anti-STDP to the wrongly spiked neuron
            layer_w_n = view(layer.w, :, winner)
            layer_delta_n = view(layer.delta, :, winner)

            # W = W - η*ΔW
            layer_w_n .=  layer_w_n .- layer.eta .* layer_delta_n
            # Keep the weights normalized always
            layer_w_n .= layer_w_n ./ norm(layer_w_n)
        end
        # Lower the thresholds of all the correct class neurons

        labelNeurons = (label-1)*layer.nNeurons_per_class+1:label*layer.nNeurons_per_class
        labelNeuronThresholds_view = view(layer.thresh, labelNeurons)
        labelNeuronThresholds_view .= labelNeuronThresholds_view .- layer.thresholdOpen

        # This is considering that layer.dot_prod has not changed
        # after forward is called and before punish is called. 
        #This is generally the case and it depends on the FC Model implementation
        # In the current form of ODESA it is always true.
        closestNeuron::Int32 = -1
        max_value::Float32 = -1.0
        @inbounds for neuron in labelNeurons

            if layer.dot_prod[neuron] > max_value
                closestNeuron = neuron
                max_value = layer.dot_prod[neuron]
            end
        end
        # Update the delta and deltaThresh with the current event context. 
        layer.delta[:, closestNeuron] .= view(layer.event_context, :) .- view(layer.w, : , closestNeuron)
        layer.deltaThresh[closestNeuron] = layer.dot_prod[closestNeuron] - layer.thresh[closestNeuron]
        reward(layer,closestNeuron)

        return nothing

    end

    function saveAll(layer::Classifier, filename::String)
        save(filename, "w", layer.w, "thresh", layer.thresh)
    end

    function loadAll(layer::Classifier, filename::String)
        file_contents = load(filename)
        layer.w = file_contents["w"]
        layer.thresh = file_contents["thresh"]
    end


    mutable struct FCModel
        hidden_layers::Vector{FC}
        output_layer::Classifier

        function FCModel(hidden_layers::Vector{FC}, output_layer::Classifier)
            new(hidden_layers, output_layer)
        end
    end

    function reset(model::FCModel)
        @inbounds for (layer_id, layer) in enumerate(model.hidden_layers)
            reset_time(layer)
        end
        reset_time(model.output_layer)
    end

    function saveModel(model::FCModel, filename::String)
        @inbounds for (layer_id, layer) in enumerate(model.hidden_layers)
            saveAll(layer, filename * "_fc_hidden_$(layer_id).jld")
        end
        saveAll(model.output_layer, filename * "_fc_output.jld")
    end

    function loadModel(model::FCModel, filename::String)
        for (layer_id, layer) in enumerate(model.hidden_layers)
            loadAll(layer, filename * "_fc_hidden_$(layer_id).jld")
        end
        loadAll(model.output_layer, filename * "_fc_output.jld")
    end

    function forward(model::FCModel, x::Int32, y::Int32, ts::Float32, label::Int32)
        #println("inside FCModel::Forward")

        #TODO: Implement single layer FC Models( No Hidden Layers)

        # Place holder winners memory allocation. Maybe this is not necessary
        winners::Array{Int32,1} = [Int32(-1) for i in model.hidden_layers]

        # Go through each layer and call the forward function. And if a layer spikes, then call the record function
        # on the  layer before it. This is local attention mechanism
        @inbounds for (layer_idx, layer) in enumerate(model.hidden_layers)
            if layer_idx == 1
                winners[layer_idx] = forward(layer, x, y, ts)
            else
                winners[layer_idx] = forward(
                    layer,
                    1,
                    winners[layer_idx-1],
                    ts,
                )
                if winners[layer_idx] > -1
                    record(
                        model.hidden_layers[layer_idx-1],
                        ts
                    )
                end
            end
        end

        # After all hidden layers. Go to forward function of output layer. 
        output_winner, output_class = forward(
            model.output_layer,
            Int32(1),
            last(winners),
            ts
        )


        # If output layer spikes, then call record on the last hidden layer. 
        if output_winner > -1
             #@code_warntype
             record(last(model.hidden_layers), ts)
        end

        # If the incoming event has a label, then the model should have predicted the label.
        # TODO: No winner from the layers is represented as -1, and no label is represented as 0. 
        # This has to be unified. 
        if label > 0
            @inbounds for (layer_idx, winner) in enumerate(winners)
                if winner < 0
                    punish(model.hidden_layers[layer_idx])
                    break
                else
                    reward(model.hidden_layers[layer_idx], winner)

                end
            end

            if output_winner > -1
                if label == output_class
                    reward(model.output_layer, output_winner)
                else
                    punish(model.output_layer, output_winner, label)
                end
            else
                if last(winners) > -1
                    punish(model.output_layer, output_winner, label)
                end
            end
        end

        
        return winners, output_winner, output_class

    end

    function infer(model::FCModel, x::Int32, y::Int32, ts::Float32)
        """
        Infer the outputs from the model. This is similar to forward function of the model, but without changing 
        the weights and thresholds. 
        """
        winners = [Int32(-1) for i in model.hidden_layers]::Array{Int32}

        @inbounds for (layer_idx, layer) in enumerate(model.hidden_layers)
            if layer_idx == 1
                winners[layer_idx] = forward(layer, x, y, ts)
            else
                winners[layer_idx] = forward(
                    layer,
                    1,
                    winners[layer_idx-1],
                    ts,
                )
            end
        end

        output_winner, output_class = forward(
            model.output_layer,
            Int32(1),
            last(winners),
            ts,
        )

        return winners, output_winner, output_class
    end


    
end # end of module
