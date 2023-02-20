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