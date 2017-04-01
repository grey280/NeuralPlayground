import Foundation

public class Sigmoid: Neuron, Equatable{ // We'll be using sigmoid neurons for the network
    var inputs = [Neuron]()
    public var linkedNeurons = [Neuron]()
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
    public var error = 0.0
    public func setError(_ input: Double) {
        error = input
    }
    public init(fromLayer inputLayer: Layer){ // Feed in an entire layer at once, doing all the linkages and randomizing the weights
        for neuron in inputLayer.neurons{
            inputs.append(neuron)
            neuron.addLinkedNeuron(self)
            weights.append(drand48())
        }
    }
    
    public func sum() -> Double{ // Sums everything up. Basically, \exp(-\sum_j w_j x_j-b)
        var out = 0.0
        for (input, weight) in zip(inputs, weights){
            out += input.output*weight
            out -= bias
        }
        return out
    }
    
    var cachedOutput: Double?
    
    public func reset(){
        for neuron in inputs{ // Invalidate cache, bubble upwards
            neuron.reset()
        }
        cachedOutput = nil
    }
    
    public var output: Double{
        //        return 1/(1+exp(-1.0 * sum()))
        if cachedOutput != nil{
            return cachedOutput!
        }
        cachedOutput = 1/(1+exp(-1.0 * sum()))
        return cachedOutput!
    }
    
    public func addLinkedNeuron(_ input: Neuron) { // should only ever be used by the initializer of something linking itself to this one
        linkedNeurons.append(input)
    }
    
    public func weightFor(_ input: Neuron) throws -> Double { // Swift doesn't have a way to make a protocol equatable that I can figure out, so I'm basically replacing array.index(of:) in here and it is *hell*
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
public func ==(lhs: Sigmoid, rhs: Sigmoid) -> Bool{
    return lhs.bias == rhs.bias && lhs.output == rhs.output && lhs.sum() == rhs.sum()
}
