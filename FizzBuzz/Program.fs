open ML.Neural
open System.Diagnostics

[<EntryPoint>]
let main argv =

    printfn "FizzBuzz solving neural network"
    printfn "-------------------------------"

    // First try to create network.
    let network, found = FizzBuzzNetwork.create ()
    
    let network' = 
        // If no pre-trained network is found, create and train a new one.
        // Note this can take a very long time to get 100% accuracy, possibly 30 mins+ (though this was on an old Macbook.)
        if not found then

            // Hyper parameters.
            let noEpochs     = 500 // Increase this for better accuracy (but will take longer.)    
            let learningRate = 1.0 // I don't think decreasing this helps. 

            // Encode training data, use numbers other than 1 to 100 as that would be cheating!
            let xs = [101..1023] |> List.map (fun x -> FizzBuzzNetwork.binaryEncode x)
            let ys = [101..1023] |> List.map (fun y -> FizzBuzzNetwork.fizzBuzzEncode y)

            // Train it.
            let sw = Stopwatch()
            sw.Start()
            FizzBuzzNetwork.train xs ys noEpochs learningRate 
            sw.Stop()
            printfn "Training complete in %dms (%dmin %ds.)"
                sw.ElapsedMilliseconds (sw.ElapsedMilliseconds/60000L) (sw.ElapsedMilliseconds/1000L % 60L)

            // Get best network trained.
            FizzBuzzNetwork.create () |> fst
        else
            network

    // Finally, solve fizzbuzz!
    network' |> FizzBuzzNetwork.solve true |> ignore  
    0