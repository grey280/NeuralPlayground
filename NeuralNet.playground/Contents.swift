import UIKit
import PlaygroundSupport

struct config{
    static let layerInfo: [Int] = [8, 2] // Default layer structure; [Int], where each value is the number of neurons in the layer. First layer will be InputNeurons, the rest will be Sigmoids
    static let defaultInput: Double = 0.0 // Default input for the InputNeurons
    static let defaultStepSize: Double = 0.5 // Default step size for training
    static let stepSizeChange: Double = 0.9 // Multiplier by which to change the step size after every training iteration
    static let trainingIterations: Int = 5 // Number of training iterations to run
}

enum NeuralNetError: Error{
    case InputMismatch // Given input does not match the shape of the input layer
    case NoDataSet // Attempted to call an analytic function without having given the network a dataset
    case NeuronMismatch // Attempted to call a function on a neuron that didn't have that type of function
    case InterlinkFailure // Attempted to find properties of a linked neuron when the neuron wasn't actually linked
}

protocol Neuron{ // Having this allows constant vs. sigmoid neurons, while also making it possible to gracefully interlink the two.
    var output: Double{ get } // The main useful value
    var linkedNeurons: [Neuron] { get set }
    var error: Double { get set }
    func setError(_ input: Double)
    func reset() // Clears any caching that the neuron is doing
    func sum() -> Double
    func addLinkedNeuron(_ input: Neuron)
    func weightFor(_ input: Neuron) throws -> Double
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
    
    func errorCalc() throws -> [Double]{
        var outs = [Double]()
        
        for neuron in neurons{
            var errorSum = 0.0
            for linkedNeuron in neuron.linkedNeurons{
                do{
                    let thisError = try linkedNeuron.weightFor(neuron) * linkedNeuron.error
                    errorSum += thisError
                }
            }
            errorSum *= onionPrime(neuron.sum())
            neuron.setError(errorSum)
            outs.append(errorSum)
        }
        
        return outs
    }
    
    private func errorCalc(withInput input: (input: [Double], output: [Double])) -> [Double]{
        let selfOut = softmax()
        var secondBits = [Double]()
        for neuron in self.neurons{
            secondBits.append(onionPrime(neuron.sum()))
        }
        var outputs = [Double]()
        for secondBit in secondBits{
            let firstBit = vectorDistance(x: selfOut[0] - input.output[0], y: selfOut[1] - input.output[1])
            outputs.append(firstBit * secondBit)
        }
        
        for i in 0..<self.neurons.count{ // store the error for later use
            self.neurons[i].error = outputs[i]
        }
        
        return outputs
    }
    
    func errorCalc(withInputs inputs: [(input: [Double], output: [Double])]) -> [Double]{
        var outs = [[Double]]()
        for input in inputs{
            outs.append(errorCalc(withInput: input))
        }
        var output = [0.0, 0.0]
        var outSums = [0.0, 0.0]
        for out in outs{
            outSums[0] += out[0]
            outSums[1] += out[1]
        }
        output[0] = outSums[0] / Double(outs.count)
        output[1] = outSums[1] / Double(outs.count)
        neurons[0].error = output[0]
        neurons[1].error = output[1]
        return output
    }
}

class InputNeuron: Neuron, Equatable{ // Constant value, used for feeding inputs to the network
    var amount: Double = 0.0
    
    var error: Double{
        get{
            return 0
        }
        set {
            return // ignore attempts to set
        }
    }
    
    func setError(_ input: Double) {
        return // ignore error being set
    }
    
    var linkedNeurons = [Neuron]()
    
    init(withValue value: Double){
        amount = value
    }
    
    func reset(){
        return // Do nothing, since we don't need to clear any cache here
    }
    
    func sum() -> Double{
        return amount
    }
    
    var output: Double{
        return amount
    }
    
    func addLinkedNeuron(_ input: Neuron){
        return // do nothing, since backpropagation doesn't matter to inputs
    }
    
    func weightFor(_ input: Neuron) -> Double {
        return 0.0
    }
}
func ==(lhs: InputNeuron, rhs: InputNeuron) -> Bool{
    return lhs.amount == rhs.amount
}

class Sigmoid: Neuron, Equatable{ // We'll be using sigmoid neurons for the network
    var inputs = [Neuron]()
    var linkedNeurons = [Neuron]()
    var weights = [Double](){
        didSet{
            reset()
        }
    }
    var bias = 0.0{
        didSet{
            reset()
        }
    }
    var error = 0.0
    func setError(_ input: Double) {
        error = input
    }
    init(fromLayer inputLayer: Layer){ // Feed in an entire layer at once, doing all the linkages and randomizing the weights
        for neuron in inputLayer.neurons{
            inputs.append(neuron)
            neuron.addLinkedNeuron(self)
            weights.append(drand48())
        }
    }
    
    func sum() -> Double{ // Sums everything up. Basically, \exp(-\sum_j w_j x_j-b)
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
        return 1/(1+exp(-1.0 * sum()))
//        if cachedOutput != nil{
//            return cachedOutput!
//        }
//        cachedOutput = 1/(1+exp(-1.0 * sum()))
//        return cachedOutput!
    }
    
    func addLinkedNeuron(_ input: Neuron) { // should only ever be used by the initializer of something linking itself to this one
        linkedNeurons.append(input)
    }
    
    func weightFor(_ input: Neuron) throws -> Double { // Swift doesn't have a way to make a protocol equatable that I can figure out, so I'm basically replacing array.index(of:) in here and it is *hell*
        guard let inputSig = input as? Sigmoid else{
            throw NeuralNetError.NeuronMismatch
        }
        var indO: Int?
        for i in 0..<inputs.count{
            if inputs[i] as! Sigmoid == inputSig{
                indO = i
                break
            }
        }
        guard let ind = indO else{
            throw NeuralNetError.InterlinkFailure
        }
        
        return weights[ind]
    }
}
func ==(lhs: Sigmoid, rhs: Sigmoid) -> Bool{
    return lhs.bias == rhs.bias && lhs.output == rhs.output && lhs.sum() == rhs.sum()
}

func vectorDistance(x: Double, y: Double) -> Double{
    return sqrt(x*x + y*y)
}
func onionPrime(_ input: Double) -> Double{ // First derivative of the sigmoid function, σ - or, as I refer to it, 'onion'
    let eX = exp(input)
    let bottom = pow(eX+1, 2)
    return eX/bottom
}

class Network: CustomStringConvertible{
    var layers = [Layer]()
    private var lastEvalSet:[(input: [Double], output: [Double])]?
    
    func reset(){
        layers[layers.count - 1].reset() // since it bubbles up, don't need to reset each layer, only the last one
    }
    
    var lastLayer: Layer{ // Helper for accessing the last layer; useful for getting outputs, I suspect
        return layers[layers.count - 1]
    }
    
    var firstLayer: Layer{ // Helper for accessing the first layer; useful for feeding inputs, I suspect
        return layers[0]
    }

    private func evaluate(_ input: [Double]) throws -> [Double]{ // Evaluate the network on a single input; for internal use only
        guard input.count == firstLayer.neurons.count else{
            throw NeuralNetError.InputMismatch
        }
        for i in 0..<input.count{
            (firstLayer.neurons[i] as! InputNeuron).amount = input[i]
        }
        return lastLayer.softmax()
//        var outs = [Double]()
//        for neuron in lastLayer.neurons{
//            outs.append(neuron.output)
//        }
//        return outs
    }
    
    func cost() throws -> Double{ // C(w,b) \equiv \frac{1}{2n} \sum_x \| y(x) - a\|^2.
        guard let dataSet = lastEvalSet else{
            throw NeuralNetError.NoDataSet
        }
        var sum = 0.0
        for dataPoint in dataSet{
            do{
                let thisOut = try evaluate(dataPoint.input)
                let distance = vectorDistance(x: thisOut[0]-dataPoint.output[0], y: thisOut[1]-dataPoint.output[1])
                sum += distance*distance
            }
        }
        return sum / Double((2*dataSet.count))
    }

    
    func evaluate(_ input: [(input: [Double], output: [Double])]) throws -> (output: [[Double]], cost: Double){
        lastEvalSet = input
        var outs = [[Double]]()
        do {
            for (inVal, _) in input{
                try outs.append(evaluate(inVal))
            }
            return (output: outs, cost: try cost())
        }
    }
    
    func train(_ input: [(input: [Double], output: [Double])]) throws{
        // Train on a subset at a time, making it stochastic
        // Gradient descent algorithm
        // Change the biases of nodes, and the weights of their interconnections
        var stepSize = config.defaultStepSize
        
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
                    try layers[(layers.count-2)-i].errorCalc() // -2: -1 so no overflow error, and -1 since we already did the last layer
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
                // Update the step size; we'll shrink slowly, for now
                stepSize *= config.stepSizeChange
            }
        }
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


// Testing

let net = Network().buildDefaultNetwork()

var trainingData = [(input: [Double], output: [Double])]()
for i in 0..<256{
    trainingData.append(buildInput(UInt8(i)))
}
do{
    let out = try net.evaluate(trainingData)
    out.output
    print(out.cost)
    for i in 0..<config.trainingIterations{
        try net.train(trainingData)
        let out2 = try net.evaluate(trainingData)
        out2.output
        print(out2.cost)
    }
} catch NeuralNetError.InputMismatch{
    print("C'mon use the helper function, it's fool-resilient")
} catch {
    print("Something else went wrong, smart guy")
}

for layer in net.layers{
    print("Layer with \(layer.neurons.count) neurons: ")
    for neuron in layer.neurons{
        if let nSig = neuron as? Sigmoid{
            print("  Sigmoid with \(nSig.bias) bias and \(nSig.weights) weights")
        } else {
            print("  Input neuron")
        }
    }
}