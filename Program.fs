open ML.Neural
open ML.Linalg

[<EntryPoint>]
let main argv =

    let andPerceptron = {Weights = [2.;2.]; Bias = -3.}

    printfn "AND perceptron"
    printfn "--------------"
    printfn "Perceptron [1;1] gives = %A" (andPerceptron |> Perceptron.getOutput [1.;1.])
    printfn "Perceptron [0;1] gives = %A" (andPerceptron |> Perceptron.getOutput [0.;1.])
    printfn "Perceptron [1;0] gives = %A" (andPerceptron |> Perceptron.getOutput [1.;0.])
    printfn "Perceptron [0;0] gives = %A" (andPerceptron |> Perceptron.getOutput [0.;0.])
    printfn ""

    let orPerceptron = {Weights = [2.;2.]; Bias = -1.}

    printfn "OR perceptron"
    printfn "-------------"
    printfn "Perceptron [1;1] gives = %A" (orPerceptron |> Perceptron.getOutput [1.;1.])
    printfn "Perceptron [0;1] gives = %A" (orPerceptron |> Perceptron.getOutput [0.;1.])
    printfn "Perceptron [1;0] gives = %A" (orPerceptron |> Perceptron.getOutput [1.;0.])
    printfn "Perceptron [0;0] gives = %A" (orPerceptron |> Perceptron.getOutput [0.;0.])
    printfn ""

    let notPerceptron = {Weights = [-2.]; Bias = 1.}

    printfn "NOT perceptron"
    printfn "--------------"
    printfn "Perceptron [0] gives = %A" (notPerceptron |> Perceptron.getOutput [0.])
    printfn "Perceptron [1] gives = %A" (notPerceptron |> Perceptron.getOutput [1.])
    printfn ""

    let (xorNeuralNetwork: NeuralNetwork) =
        [// Hidden layer.
         [[20.;20.;-30.];  // AND neuron. 
          [20.;20.;-10.]]; // OR neuron.
         // Output layer.
         [[-60.;60.;-30.]]]

    printfn "XOR neural network"
    printfn "-------------------"
    printfn "Neural network [0;0] gives = %A" (xorNeuralNetwork |> NeuralNetwork.feedForward [0.;0.] |> NeuralNetwork.getResult)
    printfn "Neural network [1;0] gives = %A" (xorNeuralNetwork |> NeuralNetwork.feedForward [1.;0.] |> NeuralNetwork.getResult)
    printfn "Neural network [0;1] gives = %A" (xorNeuralNetwork |> NeuralNetwork.feedForward [0.;1.] |> NeuralNetwork.getResult)
    printfn "Neural network [1;1] gives = %A" (xorNeuralNetwork |> NeuralNetwork.feedForward [1.;1.] |> NeuralNetwork.getResult)
    printfn ""

    printfn "Train XOR neural network"
    printfn "------------------------"

    // Training data.
    let xs = [[0.;0.];[0.;1.];[1.;0.];[1.;1.]]
    let ys = [[0.];[1.];[1.];[0.]]

    // Start with random weights.
    let rnd = System.Random()
    let xorNeuralNetwork' = 
        [// Hidden layer - 2 inputs -> 2 outputs
         [[for _ in [0..2] -> rnd.NextDouble()];  // 1st hidden neuron.
          [for _ in [0..2] -> rnd.NextDouble()]]; // 2nd hidden neuron.
         // Output layer - 2 inputs -> 1 output
         [[for _ in [0..2] -> rnd.NextDouble()]]
        ]
    printfn "Starting random weights:"
    printfn "%A" xorNeuralNetwork'

    let noEpochs = 20000
    let learningRate = 1.
    let xorNeuralNetwork'' = xorNeuralNetwork' |> NeuralNetwork.train xs ys noEpochs learningRate

    printfn "After training:"
    printfn "%A" xorNeuralNetwork''
    printfn "Check if it learned XOR:"
    printfn "Neural network [0;0] gives = %A" (xorNeuralNetwork'' |> NeuralNetwork.feedForward [0.;0.] |> NeuralNetwork.getResult)
    printfn "Neural network [1;0] gives = %A" (xorNeuralNetwork'' |> NeuralNetwork.feedForward [1.;0.] |> NeuralNetwork.getResult)
    printfn "Neural network [0;1] gives = %A" (xorNeuralNetwork'' |> NeuralNetwork.feedForward [0.;1.] |> NeuralNetwork.getResult)
    printfn "Neural network [1;1] gives = %A" (xorNeuralNetwork'' |> NeuralNetwork.feedForward [1.;1.] |> NeuralNetwork.getResult)    
    0