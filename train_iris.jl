using Revise
using Odesa

using BenchmarkTools
using ProfileView

using JLD
using ScikitLearn.CrossValidation
using Printf

## !!!RUN prepare_iris.jl before running this script to generate the dataset!!! ## 

#This should be changed to be picked up configuration from some file
function create_model(input_rows::Int32,input_cols::Int32)
    # First hidden layer parameters
    layer_1_nNeurons::Int32 = 10

    layer_1_eta::Float32 = 0.001
    layer_1_threshEta::Float32 = 0.001

    layer_1_thresholdOpen::Float32 = 0.1

    layer_1_tau::Float32 = 0.464
    layer_1_traceTau::Float32 = 0.81

    # Create a Hidden FullyConnected layer with the above parameters
    layer_1 = Odesa.FullyConnected.FC(
            input_rows,
            input_cols,
            layer_1_nNeurons,
            layer_1_eta,
            layer_1_threshEta,
            layer_1_thresholdOpen,
            layer_1_tau,
            layer_1_traceTau
        )

    output_layer_input_rows::Int32 = 1
    output_layer_input_cols::Int32 = layer_1_nNeurons
    output_layer_nNeurons_per_class::Int32 = 1
    output_layer_nClasses::Int32 = 3
    output_layer_eta::Float32 = 0.01
    output_layer_threshEta::Float32 = 0.01
    output_layer_tau::Float32 = 0.81
    output_layer_thresholdOpen::Float32 = 0.1



    output_layer = Odesa.FullyConnected.Classifier( output_layer_input_rows,
                                                    output_layer_input_cols,
                                                    output_layer_nNeurons_per_class,
                                                    output_layer_nClasses,
                                                    output_layer_eta,
                                                    output_layer_threshEta,
                                                    output_layer_thresholdOpen,
                                                    output_layer_tau
                                                )
    
    hidden_layers = Vector{Odesa.FullyConnected.FC}()
    push!(hidden_layers, layer_1)
    model = Odesa.FullyConnected.FCModel(hidden_layers,output_layer)

    return model

end






function run_training(epochs::Int32,
                        model::Odesa.FullyConnected.FCModel,
                        inputs::Vector{Vector{Tuple{Int32,Float32}}},
                        targets::Array{Int32,1},
                        training_order::Array{Int32,1},
                     )


    # Allotting memory for housekeeping the training. 
    winners::Array{Int32,1} = [-1  for i in model.hidden_layers]
    # temp = [(-1, -1, -1)  for i in model.hidden_layers]::Array{Tuple{Int32,Int32, Int32}}
    # winners::Array{Tuple{Int32,Int32, Int32}} = temp
    output_winner::Int32 = -1
    output_class::Int32 = -1
    best_accuracy::Float32 = 0

    correct_count::Int32 = 0
    no_count::Int32 = 0
    wrong_count::Int32 = 0

    correct_class_percent::Float32 = zero(Float32)
    wrong_class_percent::Float32 = zero(Float32)
    no_class_percent::Float32 = zero(Float32)

    t::Float32 = 0.0 #Global time


    @inbounds for epoch in 1:epochs

        Odesa.FullyConnected.reset(model)

        correct_count = 0
        no_count = 0
        wrong_count = 0

        t = 0.0

        


        @inbounds for example_index in Odesa.ProgressBar(training_order)
            events::Vector{Tuple{Int32,Float32}} = inputs[example_index]
            label::Int32 = targets[example_index]
            total_n_events::Int32 = size(events)[1]

            t = t + 5.0


            @inbounds for event_idx in 1:total_n_events
                event::Tuple{Int32,Float32} = events[event_idx]
                x::Int32 = 1
                y::Int32 = event[1]

                ts::Float32 = t + event[2]

                if event_idx < total_n_events
                    winners, output_winner, output_class = Odesa.FullyConnected.forward(model, x, y, ts, Int32(-1))

                else
                    winners, output_winner, output_class = Odesa.FullyConnected.forward(model, x, y, ts,label)

                    if output_winner > -1
                        if output_class == label
                            correct_count += 1
                        else
                            wrong_count += 1
                        end
                    else
                        no_count += 1
                    end
                end
            end

        end
        correct_class_percent = correct_count/(correct_count+wrong_count+no_count)
        wrong_class_percent = wrong_count/(correct_count+wrong_count+no_count)
        no_class_percent = no_count/(correct_count+wrong_count+no_count)

        if correct_class_percent > best_accuracy
            best_accuracy = correct_class_percent
            Odesa.FullyConnected.saveModel(model,"first_try_iris")
        end

        @printf("Epoch : %d => Correct : %f, Wrong : %f, No Prediction : %f \n",epoch,correct_class_percent, wrong_class_percent, no_class_percent)

        # println(correct_class_percent, wrong_class_percent, no_class_percent)

    end

    return model


end




function run_testing(   model::Odesa.FullyConnected.FCModel,
                        inputs::Vector{Vector{Tuple{Int32,Float32}}},
                        targets::Array{Int32,1},
                        testing_order::Array{Int32,1}
                    )


    # Allotting memory for housekeeping the testing. 
    winners::Array{Int32,1} = [-1  for i in model.hidden_layers]
    # temp = [(-1, -1, -1)  for i in model.hidden_layers]::Array{Tuple{Int32,Int32, Int32}}
    # winners::Array{Tuple{Int32,Int32, Int32}} = temp
    output_winner::Int32 = -1
    output_class::Int32 = -1

    correct_count::Int32 = 0
    no_count::Int32 = 0
    wrong_count::Int32 = 0

    correct_class_percent::Float32 = zero(Float32)
    wrong_class_percent::Float32 = zero(Float32)
    no_class_percent::Float32 = zero(Float32)

    t::Float32 = 0.0 #Global time



    Odesa.FullyConnected.reset(model)

    correct_count = 0
    no_count = 0
    wrong_count = 0

    t = 0.0




    @inbounds for example_index in Odesa.ProgressBar(testing_order)
            
        events::Vector{Tuple{Int32,Float32}} = inputs[example_index]
        label::Int32 = targets[example_index]
        total_n_events::Int32 = size(events)[1]

        t = t + 5.0


        @inbounds for event_idx in 1:total_n_events
                
            event::Tuple{Int32,Float32} = events[event_idx]
            x::Int32 = 1
            y::Int32 = event[1]

            ts::Float32 = t + event[2]

            if event_idx < total_n_events
                winners, output_winner, output_class = Odesa.FullyConnected.infer(model, x, y, ts)

            else
                winners, output_winner, output_class = Odesa.FullyConnected.infer(model, x, y, ts)

                if output_winner > -1
                    if output_class == label
                        correct_count += 1
                    else
                        wrong_count += 1
                    end
                else
                    no_count += 1
                end
            end

        end
        
    end
    correct_class_percent = correct_count/(correct_count+wrong_count+no_count)
    wrong_class_percent = wrong_count/(correct_count+wrong_count+no_count)
    no_class_percent = no_count/(correct_count+wrong_count+no_count)
    @printf("Test Percentages => Correct : %f, Wrong : %f, No Prediction : %f",correct_class_percent, wrong_class_percent, no_class_percent)
    # println(correct_class_percent, wrong_class_percent, no_class_percent)



    return nothing


end

# Code entry point
function main(epochs::Int32)

    # The dataset generated from "prepare_iris.jl" 
    dataset = JLD.load("./iris_data.jld")

    # Inputs and targets of the dataset
    inputs::Vector{Vector{Tuple{Int32,Float32}}} = dataset["inputs"]
    targets::Array{Int32,1} = dataset["targets"]

    # Dataset example ids ranging from 1 to 150
    ids = collect(1:size(inputs)[1])
    # Odesa.shuffle!(ids)


    # Selecting the same training set as the python version of the code. Using odd and even numbered 
    # examples for training and testing. 
    training_order::Vector{Int32} = ids[1 : 2 : size(inputs)[1]]
    testing_order::Vector{Int32} = ids[2 : 2 : size(inputs)[1]]

    # Shuffling to make sure different classes fall together. 
    Odesa.shuffle!(training_order)
    Odesa.shuffle!(testing_order)

    # Input dimensions of the dataset
    input_rows::Int32 = 1
    input_cols::Int32 = 20

    # Make the model with some pre-defined hyper-params
    model = create_model(input_rows,input_cols)

    # Training the model on the training set. 
    model = run_training(epochs,
                        model,
                        inputs,
                        targets,
                        training_order
                        )

    # Evaluating the model on the testing set. 
    run_testing(model,
                inputs,
                targets,
                testing_order
                )
    # Save the model weights and thresholds. ß
    Odesa.FullyConnected.saveModel(model,"Iris_Julia")
	return nothing

end
@time main(Int32(400))

