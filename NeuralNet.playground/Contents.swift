//: Playground - noun: a place where people can play

import UIKit
import PlaygroundSupport

struct config{
    static let defaultWeight = 0.5
}

protocol Neuron{ // Having this allows constant vs. sigmoid neurons, while also making it possible to gracefully interlink the two.
    var output: Double{ get }
}

class InputNeuron: Neuron, Equatable{ // Constant value, used for feeding inputs to the network
    var amount: Double = 0.0
    
    init(withValue value: Double){
        amount = value
    }
    
    var output: Double{
        return amount
    }
}
func ==(lhs: InputNeuron, rhs: InputNeuron) -> Bool{
    return lhs.amount == rhs.amount
}

class Sigmoid: Neuron, Equatable{ // We'll be using sigmoid neurons for the network
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
    init(fromLayer inputLayer: Layer){ // Feed in an entire layer at once, assigning default weight
        for neuron in inputLayer.neurons{
            inputs.append(neuron)
            weights.append(config.defaultWeight) // TODO: replace with randomizing
        }
    }
    
    fileprivate func sum() -> Double{ // Sums everything up. Basically, \exp(-\sum_j w_j x_j-b)
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
func ==(lhs: Sigmoid, rhs: Sigmoid) -> Bool{
    return lhs.bias == rhs.bias && lhs.output == rhs.output && lhs.sum() == rhs.sum()
}

class Layer{
    var neurons = [Neuron]()
    
    func softmax() -> [Double]{ // Gets softmax info for the entire layer at once
        let sum = softMaxSum()
        var output = [Double]()
        for neuron in neurons{
            output.append(exp(neuron.output)/sum)
        }
        return output
    }
    
    private func softMaxSum() -> Double{
        var output = 0.0
        for neuron in neurons{
            output += exp(neuron.output)
        }
        return output
    }
}
class Network{
    var layers = [Layer]()
    
    func buildDefaultNetwork() -> Network{
        let net = Network()
        let layer1 = Layer()
        for _ in 0..<8{
            layer1.neurons.append(InputNeuron(withValue: 0))
        }
        let layer2 = Layer()
        for _ in 0..<8{
            layer2.neurons.append(Sigmoid(fromLayer: layer1))
        }
        let layer3 = Layer()
        for _ in 0..<4{
            layer3.neurons.append(Sigmoid(fromLayer: layer2))
        }
        
        let layer4 = Layer()
        layer4.neurons.append(Sigmoid(fromLayer: layer3))
        layer4.neurons.append(Sigmoid(fromLayer: layer3))
        
        net.layers.append(layer1)
        net.layers.append(layer2)
        net.layers.append(layer3)
        net.layers.append(layer4)
        
        return net
    }
}