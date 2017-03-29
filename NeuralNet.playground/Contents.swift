//: Playground - noun: a place where people can play

import UIKit
import PlaygroundSupport

protocol Neuron{
    var output: Double{ get }
}

class Constant: Neuron{
    var amount: Double = 0.0
    
    var output: Double{
        return amount
    }
}

class Sigmoid: Neuron{ // We'll be using sigmoid neurons for the network
    private var inputs = [Neuron]()
    private var weights = [Double]()
    var bias = 0.0
    
    func addInput(_ input: Neuron, weight: Double){ // Add a single input; makes sure we have the same number of inputs and weights, because otherwise... problems.
        inputs.append(input)
        weights.append(weight)
    }
    
    func addInputs(_ newInputs: [Neuron], weights newWeights: [Double]){ // Helper function
        for (input, weight) in zip(newInputs, newWeights){
            inputs.append(input)
            weights.append(weight)
        }
    }
    
    private func sum() -> Double{ // Sums everything up. Basically, \exp(-\sum_j w_j x_j-b)
        var out = 0.0
        for (input, weight) in zip(inputs, weights){
            out += input.output*weight
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
