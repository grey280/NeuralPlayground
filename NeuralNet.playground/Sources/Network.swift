import Foundation

enum NeuralNetError: Error{
    case InputMismatch // Given input does not match the shape of the input layer
    case NoDataSet // Attempted to call an analytic function without having given the network a dataset
    case NeuronMismatch // Attempted to call a function on a neuron that didn't have that type of function
    case InterlinkFailure // Attempted to find properties of a linked neuron when the neuron wasn't actually linked
}

func onionPrime(_ input: Double) -> Double{ // First derivative of the sigmoid function, σ - or, as I refer to it, 'onion'
    let eX = exp(input)
    let bottom = pow(eX+1, 2)
    return eX/bottom
}

open class Network: CustomStringConvertible{
    public var layers = [Layer]() // All the layers in the network
    private var lastEvalSet:[(input: [Double], output: [Double])]? // Used to store the previous evaluation set, which we use for analytic functions
    
    public func reset(){ // Clear every cache in the network.
        layers[layers.count - 1].reset() // since it bubbles up, don't need to reset each layer, only the last one
    }
    
    public var lastLayer: Layer{ // Helper for accessing the last layer; useful for getting outputs
        return layers[layers.count - 1]
    }
    
    public var firstLayer: Layer{ // Helper for accessing the first layer; useful for feeding inputs
        return layers[0]
    }
    
    /**
    Evaluate the network on a single input
    - returns:
    The softmax'd output of the network
    - parameters:
        - input: A single input
    - throws: An error of type `NeuralNetworkError`
    */
    public func evaluate(_ input: [Double]) throws -> [Double]{ // Evaluate the network on a single input
        guard input.count == firstLayer.neurons.count else{
            throw NeuralNetError.InputMismatch
        }
        for i in 0..<input.count{
            (firstLayer.neurons[i] as! InputNeuron).amount = input[i]
        }
        return lastLayer.softmax()
    }
    
    /**
    Calculate the cost of the network on the last data set that was run
    - returns: The cost of the network.
    - throws: An error of type `NeuralNetworkError`
    */
    public func cost() throws -> Double{ // Cost function - figures out how wrong the network is, basically.
        // C_{MST}(W,B,S^r,E^r)=0.5\sum_j(a^L_j-E^r_j)^2 \equiv \frac{1}{2n}\sum||y(x)-a||^2
        guard let dataSet = lastEvalSet else{
            throw NeuralNetError.NoDataSet
        }
        var sum = 0.0
        for dataPoint in dataSet{
            do{
                let thisOut = try evaluate(dataPoint.input)
                let component1 = thisOut[0] - dataPoint.output[0]
                let component2 = thisOut[1] - dataPoint.output[1]
                let localSum = component1*component1 + component2*component2
                sum += localSum
            }
        }
        return sum / Double((2*dataSet.count))
    }
    
    /**
     Evaluate the network on a set of inputs
     - returns:
     A tuple of the output results and the cost across all the inputs
     - parameters:
        - input: An array of input-output tuples
     
     - throws: An error of type `NeuralNetworkError`
     */
    public func evaluate(_ input: [(input: [Double], output: [Double])]) throws -> (output: [[Double]], cost: Double){
        lastEvalSet = input
        var outs = [[Double]]()
        do {
            for (inVal, _) in input{
                try outs.append(evaluate(inVal))
            }
            return (output: outs, cost: try cost())
        }
    }
    
    /**
    Train the network on a set of inputs.
    - parameters:
        - input: An array of input-output tuples
    
    - throws: An error of type `NeuralNetworkError`
    */
    public func train(_ input: [(input: [Double], output: [Double])]) throws{
        // Train on a subset at a time, making it stochastic
        // Gradient descent algorithm
        // Change the biases of nodes, and the weights of their interconnections
        let stepSize = config.defaultStepSize
        
        // Build subsets
        var shuffledInput = input.sorted { (in1, in2) -> Bool in
            return arc4random_uniform(2) == 0
        }
        var subsets = [[(input: [Double], output: [Double])]]()
        for i in 0..<(shuffledInput.count/10){
            var temp = [(input: [Double], output: [Double])]()
            for j in 0..<10{
                temp.append(shuffledInput[j*i])
            }
            subsets.append(temp)
        }
        
        do{
            for subset in subsets{
                // Calculate error
                let _ = lastLayer.errorCalc(withInputs: subset) // Calculate the error on the final layer
                for i in 0..<layers.count-1{ // Calculate the error on the rest of the layers
                    let _ = try layers[(layers.count-2)-i].errorCalc() // -2: -1 so no overflow error, and -1 since we already did the last layer
                }
                
                // remember, the new bias on a node is oldBias - (stepSize) * (error on that node)
                // a new weight is oldWeight - (stepSize) * ((input along that weight, unchanged by the weight) * (error on the node))
                for layer in layers{
                    for neuron in layer.neurons{
                        if let thisNeuron = neuron as? Sigmoid{
                            thisNeuron.bias -= stepSize * thisNeuron.error
                            for i in 0..<thisNeuron.weights.count{
                                let inputAlongWeight = thisNeuron.inputs[i].output * thisNeuron.error
                                thisNeuron.weights[i] -= stepSize * inputAlongWeight
                            }
                        }
                    }
                }
            }
        }
    }
    
    
    /**
    Creates the default network with randomized weights
    - returns: A network with randomized weights
    */
    static func buildDefaultNetwork() -> Network{ // Creates the default network with randomized weights
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
    /**
    Creates the default network with predesigned weights.
    - returns: A network with pre-configured weights
    */
    public static func buildPredesignedNetwork() -> Network{ // Creates the default network, with pre-set weights. Definitely not perfect, but also significantly better than the randomized default.
        let net = Network.buildDefaultNetwork()
        if let sigList = net.lastLayer.neurons as? [Sigmoid]{
            for i in 0..<8{
                sigList[0].weights[i] = 0
                sigList[1].weights[i] = 0
            }
            sigList[0].weights[6] = -1.0
            sigList[0].weights[7] = +1.0
            sigList[1].weights[6] = +1.0
            sigList[1].weights[7] = -1.0
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
    
    public init(){
        // Nothing needs doing
    }
}

