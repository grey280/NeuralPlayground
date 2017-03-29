import UIKit
import PlaygroundSupport

struct config{
    static let defaultWeight = 0.5  // Default weight to use when building Sigmoids
    static let layerInfo = [8, 4, 2, 2] // Default layer structure; [Int], where each value is the number of neurons in the layer. First layer will be InputNeurons, the rest will be Sigmoids
    static let defaultInput = 0.0 // Default input for the InputNeurons
}

enum NeuralNetError: Error{
    case InputMismatch // Given input does not match the shape of the input layer
}

protocol Neuron{ // Having this allows constant vs. sigmoid neurons, while also making it possible to gracefully interlink the two.
    var output: Double{ get } // The main useful value
    func reset() // Clears any caching that the neuron is doing
}

class Layer{
    var neurons = [Neuron]()
    
    func reset(){
        for neuron in neurons{
            neuron.reset()
        }
    }
    
    func softmax() -> [Double]{ // Gets softmax info for the entire layer at once
        let sum = softMaxSum()
        var output = [Double]()
        for neuron in neurons{
            output.append(exp(neuron.output)/sum)
        }
        return output
    }
    
    private func softMaxSum() -> Double{ // Sum up everything in order to get softmax per-neuron
        var output = 0.0
        for neuron in neurons{
            output += exp(neuron.output)
        }
        return output
    }
}

class InputNeuron: Neuron, Equatable{ // Constant value, used for feeding inputs to the network
    var amount: Double = 0.0
    
    init(withValue value: Double){
        amount = value
    }
    
    func reset(){
        return // Do nothing, since we don't need to clear any cache here
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
    
    var cachedOutput: Double?
    
    func reset(){
        for neuron in inputs{ // Invalidate cache, bubble upwards
            neuron.reset()
        }
        cachedOutput = nil
    }
    
    var output: Double{
        if cachedOutput != nil{
            return cachedOutput!
        }
        cachedOutput = 1/(1+exp(-1.0 * sum()))
        return cachedOutput!
    }
}
func ==(lhs: Sigmoid, rhs: Sigmoid) -> Bool{
    return lhs.bias == rhs.bias && lhs.output == rhs.output && lhs.sum() == rhs.sum()
}

class Network: CustomStringConvertible{
    var layers = [Layer]()
    
    func reset(){
        layers[layers.count - 1].reset() // since it bubbles up, don't need to reset each layer, only the last one
    }
    
    var lastLayer: Layer{ // Helper for accessing the last layer; useful for getting outputs, I suspect
        return layers[layers.count - 1]
    }
    
    var firstLayer: Layer{ // Helper for accessing hte first layer; useful for feeding inputs, I suspect
        return layers[0]
    }
    
//    func cost() -> Double{ // C(w,b) \equiv \frac{1}{2n} \sum_x \| y(x) - a\|^2.
//        
//    }
    
    func evaluate(_ input: [Double]) throws -> [Double]{ // Evaluate the network on a single input
        guard input.count == firstLayer.neurons.count else{
            throw NeuralNetError.InputMismatch
        }
        for i in 0..<input.count{
            (firstLayer.neurons[i] as! InputNeuron).amount = input[i]
        }
        var out = [Double]()
        for neuron in lastLayer.neurons{
            out.append(neuron.output)
        }
        return out
    }
    
    func buildDefaultNetwork() -> Network{
        let net = Network()
        var didInputLayer = false
        var previousLayer: Layer = Layer()
        for ly in config.layerInfo{
            let thisLayer = Layer()
            for _ in 0..<ly{
                if !didInputLayer{
                    thisLayer.neurons.append(InputNeuron(withValue: config.defaultInput))
                }else{
                    thisLayer.neurons.append(Sigmoid(fromLayer: previousLayer))
                }
            }
            didInputLayer = true
            net.layers.append(thisLayer)
            previousLayer = thisLayer
        }
        return net
    }
    
    public var description: String{
        var out = "Network with \(layers.count) layers: "
        for layer in layers{
            out += "\(layer.neurons.count) "
        }
        return out
    }
}

func buildInput(_ inp: UInt8) -> (input: [Double], output: [Double]){ // Helper to build a properly-shaped in/out pair
    var input = [Double]()
    let hold = Int(inp)
    var output = [Double]()
    if inp % 2 == 0{ // even number!
        output = [0, 1]
    } else {
        output = [1, 0]
    }
    
    input = [Double(hold/128 % 2), Double(hold/64 % 2), Double(hold/32 % 2), Double(hold/16 % 2), Double(hold/8 % 2), Double(hold/4 % 2), Double(hold/2 % 2), Double(hold % 2)]
    return (input: input, output: output)
}

//    Data structure? (input: [Double], output: [Double])
let net = Network().buildDefaultNetwork()

var trainingData = [(input: [Double], output: [Double])]()
for i in 0..<256{
    trainingData.append(buildInput(UInt8(i)))
}
var testOut = [[Double]]()
for dataPoint in trainingData{
    do{
        try testOut.append(net.evaluate(dataPoint.input))
    } catch NeuralNetError.InputMismatch{
        print("C'mon use the helper function, it's fool-resilient")
    }
    
}

print(testOut)