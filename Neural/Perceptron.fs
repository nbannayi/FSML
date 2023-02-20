namespace ML.Neural

open ML.Linalg

/// Represents a single perceptron.
type Perceptron =
    {
        /// Vector of weights to use in calculation. 
        Weights: Vector
        /// Bias to offset the weights calculation based on passed input.
        Bias:    double
    }

/// Perceptron module for manipulation of Perceptron types.
module Perceptron =

    /// Returns 1.0 if input is greater than 0.0, else 0.0.
    let stepFunction (x: double) =
        match x with
        | x when x >= 0.0 -> 1.0
        | _               -> 0.0

    /// Returns 1.0 if the perceptron 'fires', 0.0 if not. 
    let getOutput (x: Vector) (perceptron: Perceptron) =
        let calc = perceptron.Bias + Vector.dot perceptron.Weights x
        stepFunction calc