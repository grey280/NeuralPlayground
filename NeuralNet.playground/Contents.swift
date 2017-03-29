//: Playground - noun: a place where people can play

import UIKit
import PlaygroundSupport

class Sigmoid{ // We'll be using sigmoid neurons for the network
    private var inputs = [Double]()
    private var weights = [Double]()
    var bias = 0.0
    
    func addInput(_ input: Double, weight: Double){ // Add a single input; makes sure we have the same number of inputs and weights, because otherwise... problems.
        inputs.append(input)
        weights.append(weight)
    }
    
    func addInputs(_ newInputs: [Double], weights newWeights: [Double]){ // Helper function
        for (input, weight) in zip(newInputs, newWeights){
            inputs.append(input)
            weights.append(weight)
        }
    }
    
    private func sum() -> Double{ // Sums everything up. Basically, \exp(-\sum_j w_j x_j-b)
        var out = 0.0
        for (input, weight) in zip(inputs, weights){
            out += input*weight
            out -= bias
        }
        return out
    }
    
    var output: Double{
        return 1/(1+exp(-1.0 * sum()))
    }
}

class Layer{
    var neurons = [Sigmoid]()
}
class Network{
    var layers = [Layer]()
}