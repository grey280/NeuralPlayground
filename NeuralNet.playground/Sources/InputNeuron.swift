import Foundation

public class InputNeuron: Neuron, Equatable{ // Constant value, used for feeding inputs to the network
    var amount: Double = 0.0
    
    public var error: Double{
        get{
            return 0 // input neurons can't have error
        }
        set {
            return // ignore attempts to set
        }
    }
    
    public func setError(_ input: Double) {
        error = input // does nothing, of course
    }
    
    public var linkedNeurons = [Neuron]()
    
    public init(withValue value: Double){
        amount = value
    }
    public init(){
        amount = config.defaultInput
    }
    
    public func reset(){
        return // Do nothing, since we don't need to clear any cache here
    }
    
    public func sum() -> Double{
        return amount // This one probably won't get called, actually, but if it is, the sum is the input since it's an input neuron
    }
    public var output: Double{
        return amount
    }
    
    public func addLinkedNeuron(_ input: Neuron){
        return // do nothing, since backpropagation doesn't matter to inputs
    }
    
    public func weightFor(_ input: Neuron) -> Double {
        return 0.0 // Should never be called, since nothing comes before an input node
    }
}
public func ==(lhs: InputNeuron, rhs: InputNeuron) -> Bool{
    return lhs.amount == rhs.amount
}
