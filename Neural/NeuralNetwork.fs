namespace ML.Neural

open System
open ML.Linalg

/// Represents a neural network.
/// This is a list (layers) of lists (neurons) of vectors (weights.)
type NeuralNetwork = Vector list list

/// NeuralNetwork module for manipulation of NeuralNetwork types.
module NeuralNetwork =

    /// Smooth (differentiable) step function.
    let sigmoid (x: double) =
        1./(1. + Math.Exp(-x))

    /// Weights include the bias term at the end, neurons have a bias of 1.
    let getNeuronOutput (weights: Vector) (inputs: Vector) =
        Vector.dot weights inputs
        |> sigmoid

    /// Feeds the input vector through the neural network.  Returns the outputs
    /// of all the layers (not just the last one.)
    let rec feedForward (inputVector: Vector) (neuralNetwork: NeuralNetwork) =        
        seq {
            match neuralNetwork with
            | [] -> ()
            | head::tail ->                
                let inputVector' = inputVector @ [1.] // Add bias to input vector.
                let output = [for neuron in head -> getNeuronOutput neuron inputVector']
                yield output
                yield! feedForward output tail
        } |> List.ofSeq

    /// Get final out of the resulting vector.
    let getResult (neuralNetworkResult: Vector list) =
        neuralNetworkResult
        |> List.last
        |> List.head

    /// Given a neural network, an input vector, and a target vector, make a prediction and
    /// compute the gradient of the squared error loss with respect to the neuron weights.
    let sqErrorGrads (neuralNetwork: NeuralNetwork) (inputVector: Vector) (targetVector: Vector) =

        // Forward pass.
        let fp = neuralNetwork |> feedForward inputVector
        let hiddenOutputs, outputs = fp |> List.head, fp |> List.last

        // Gradients with respect to output neuron pre-activation outputs.
        let outputDeltas =
            [for output, target in List.zip outputs targetVector -> output * (1.-output)*(output-target)]

        // Gradients with respect to output neuron weights.
        let outputGrads =            
            [for i in [0..(neuralNetwork |> List.last |> List.length)-1] do
                [for hiddenOutput in hiddenOutputs @ [1.] -> outputDeltas.[i] * hiddenOutput]]

        // Gradients with respect to hidden neuron pre-activation outputs.
        let hiddenDeltas =
            hiddenOutputs
            |> List.mapi (fun i hiddenOutput ->
                let dp = Vector.dot outputDeltas [for n in neuralNetwork |> List.last -> n.[i]]
                hiddenOutput * (1.-hiddenOutput) * dp)

        // Gradients with respect to hidden neuron weights.
        let hiddenGrads =
            [for i in [0..(neuralNetwork |> List.head |> List.length)-1] do
                [for input in inputVector @ [1.] -> hiddenDeltas.[i] * input]]

        [hiddenGrads; outputGrads]

    /// Train the neural network with passed trasing data (xs,ys) for a given number of epochs with a given learning rate.
    let train (xs: Vector list) (ys: Vector list) (noEpochs: int) (learningRate: double) (neuralNetwork: NeuralNetwork) =
        [for _ in [0..noEpochs-1] do for x, y in List.zip xs ys -> x,y]        
        |> List.fold (fun nn (x,y) ->
            let gradients = sqErrorGrads nn x y
            [for layer, layerGrad in List.zip nn gradients do
                [for neuron, grad in List.zip layer layerGrad -> Vector.gradientStep neuron grad -learningRate]]
        ) neuralNetwork        
            