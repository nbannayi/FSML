namespace ML.Neural

open ML.Neural
open ML.Linalg
open System
open System.IO

/// Represents a fizzbuzz neural network.
type FizzBuzzNetwork = NeuralNetwork
    
module FizzBuzzNetwork =

    /// Location of pre-trained fizzbuzz neural networks,
    let private savedDirectory = Path.Combine(__SOURCE_DIRECTORY__, "saved")

    /// Create initial neural network with random weights.
    let private createRandomNeuralNetwork noHiddenLayers : NeuralNetwork =                
        let rnd = System.Random()        
        [
            // Hidden layer - 10 inputs -> numHiddenLayers outputs.
            [for _ in [1..noHiddenLayers] -> List.init (10+1) (fun _ -> rnd.NextDouble())]
            // Output layer - numHiddenLayers inputs -> 4 outputs.
            [for _ in [1..4] -> List.init (noHiddenLayers+1) (fun _  -> rnd.NextDouble())]
        ]

    /// Encode number to a list containing binary representation.
    let binaryEncode (n: int) =        
        let b = Convert.ToString(n, 2) |> List.ofSeq |> List.map (string >> double) in
        List.init (10-b.Length) (fun _ -> 0.) @ b

    /// Encode number to a list containing state.
    let fizzBuzzEncode n =
        match n % 15, n % 5, n % 3 with
        | 0, _, _ -> [0.; 0.; 0.; 1.]
        | _, 0, _ -> [0.; 0.; 1.; 0.]
        | _, _, 0 -> [0.; 1.; 0.; 0.]
        | _       -> [1.; 0.; 0.; 0.]

    /// Solve fizzbuzz using passed neural network and return score out of 100.
    /// If display is set to true will display results.
    let solve display neuralNetwork =
        let score =
            [1..100]
            |> List.map (fun n ->
                let x = binaryEncode n
                let predicted = neuralNetwork |> NeuralNetwork.feedForward x |> List.last |> Vector.argmax
                let actual = n |> fizzBuzzEncode |> Vector.argmax
                if display then
                    let labels = [(string n).PadRight(8); "Fizz    "; "Buzz    "; "FizzBuzz"]
                    printfn "%s: Actual: %s Predicted: %s" ((string n).PadRight(3)) labels.[actual] labels.[predicted]
                if predicted = actual then 1 else 0)
            |> List.sum
        if display then
            printfn "Accuracy is %d/100" score
        score

    /// Train neural network and use it to solve fizzbuzz. Once we have a perfect network
    /// serialise the results and quit.
    let rec train xs ys epochs learningRate =        
        let neuralNetwork =
            createRandomNeuralNetwork 25 // Note: by trial and error 25 hidden layers seems best, more than this leads
                                         // to overfitting, less leads to underfitting 
            |> NeuralNetwork.train xs ys epochs learningRate
        let score = neuralNetwork |> solve false
        // If we get 96% accuracy or over, save it.
        if score >= 96 then
            let fileName = sprintf "fizzbuzz_neural_network_%d_%s.txt" score (DateTime.Now.ToString("yyyyMMddhhmmss"))
            let filePath = Path.Combine(savedDirectory, fileName)
            printfn "Result >= 97! This has been serialised to %s" filePath
            neuralNetwork |> NeuralNetwork.serialise filePath
        if score < 100 then
            printfn "Accuracy was %d/100, retrying." score
            train xs ys epochs learningRate     

    /// Create a fizzbuzz neural network, if one is saved from a previous training session use the best
    /// available, otherwise generate a new random one.  Return a tuple of the neural network and true if
    /// one was found, false otherwise.
    let create () =
        let bestNetwork =
            Directory.EnumerateFiles(savedDirectory)        
            |> Seq.map (fun f -> f.Split('_').[3] |> int, f)
            |> Seq.sortByDescending (fun (s,_) -> s)
            |> Seq.tryHead
        match bestNetwork with
        | None ->
            printfn "Did not find any saved networks, generating random weights."
            createRandomNeuralNetwork 25, false
        | Some bestNetwork ->
            let score, network = bestNetwork
            printfn "Best saved network with score of %d found, this will be loaded." score 
            // Note this code is somewaht specific to the shape of a fizzbuzz neural network but could be adapted
            // to a passed number of hidden and output layers.
            let weights =
                (File.ReadAllText network).Replace(" ","").Replace("\n","").Replace("[","").Replace("]","").Split(';')
                |> Array.map (double)
            let hiddenWeights = weights |> Array.chunkBySize 11
            let hiddenLayer   = hiddenWeights.[0..24] |> Array.map (fun hw -> hw |> List.ofArray) |> List.ofArray
            let outputWeights = weights.[(25*11)..]   |> Array.chunkBySize 26
            let outputLayer   = outputWeights.[0..3]  |> Array.map (fun ow -> ow |> List.ofArray) |> List.ofArray
            [hiddenLayer; outputLayer], true