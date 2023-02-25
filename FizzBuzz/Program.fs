﻿open ML.Neural
open ML.Linalg
open System
open System.Diagnostics

[<EntryPoint>]
let main argv =

    // Encode number to a list containing binary representation.
    let binaryEncode (n: int) =        
        let b = Convert.ToString(n, 2) |> List.ofSeq |> List.map (string >> double) in
        List.init (10-b.Length) (fun _ -> 0.) @ b

    // Encode number to a list containing state.
    let fizzBuzzEncode n =
        match n % 15, n % 5, n % 3 with
        | 0, _, _ -> [0.; 0.; 0.; 1.]
        | _, 0, _ -> [0.; 0.; 1.; 0.]
        | _, _, 0 -> [0.; 1.; 0.; 0.]
        | _       -> [1.; 0.; 0.; 0.]

    // Solve fizzbuzz using passed neural network and return score out of 100.
    let solve neuralNetwork =        
        [1..100]
        |> List.map (fun n ->
            let x = binaryEncode n
            let predicted = neuralNetwork |> NeuralNetwork.feedForward x |> List.last |> Vector.argmax
            let actual = n |> fizzBuzzEncode |> Vector.argmax
            //let labels = [(string n).PadRight(8); "Fizz    "; "Buzz    "; "FizzBuzz"]
            //printfn "%s: Actual: %s Predicted: %s" ((string n).PadRight(3)) labels.[actual] labels.[predicted]
            if predicted = actual then 1 else 0)
        |> List.sum

    // Create initial neural network with random weights.
    let createRandomNeuralNetwork noHiddenLayers : NeuralNetwork =                
        let rnd = System.Random()        
        [// Hidden layer - 10 inputs -> numHiddenLayers outputs.
            [for _ in [1..noHiddenLayers] -> List.init (10+1) (fun _ -> rnd.NextDouble())]
            // Output layer - numHiddenLayers inputs -> 4 outputs.
            [for _ in [1..4] -> List.init (noHiddenLayers+1) (fun _ -> rnd.NextDouble())]
        ]

    // Train neural network and use it to solve fizzbuzz. Once we have a perfect network
    // serialise the results and quit.
    let rec train xs ys epochs learningRate noHiddenLayers =        
        let neuralNetwork =
            createRandomNeuralNetwork noHiddenLayers
            |> NeuralNetwork.train xs ys epochs learningRate
        let score = neuralNetwork |> solve
        if score = 100 then
            let fileName = "fizzbuzz_neural_network_"+DateTime.Now.ToString("yyyyMMddhhmmss")+".ser"
            printfn "Perfect result! This has been serialised to %s" fileName
            neuralNetwork |> NeuralNetwork.serialise fileName
        else
            printfn "Accuracy was %d/100, retrying." score
            train xs ys epochs learningRate noHiddenLayers        

    printfn "FizzBuzz solving neural network"
    printfn "-------------------------------"

    // Encode training data.
    let xs = [101..1023] |> List.map (fun x -> binaryEncode x)
    let ys = [101..1023] |> List.map (fun y -> fizzBuzzEncode y)
    let noHiddenLayers = 25

    // Create hyper parameters.
    let noEpochs = 500
    let learningRate = 1.0

    // Train a neural network till we have the perfect weights.
    let sw = Stopwatch()
    sw.Start()
    let neuralNetwork = train xs ys noEpochs learningRate noHiddenLayers
    sw.Stop()
    printfn "Fully trained the neural network in %dms (%dmin %ds):"
        sw.ElapsedMilliseconds (sw.ElapsedMilliseconds/60000L) (sw.ElapsedMilliseconds/1000L % 60L)
    printfn "%A" neuralNetwork
    0