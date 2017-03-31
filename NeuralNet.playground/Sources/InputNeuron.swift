import Foundation

public class InputNeuron: Neuron, Equatable{ // Constant value, used for feeding inputs to the network
    var amount: Double = 0.0
    
    public var error: Double{
        get{
            return 0
        }
        set {
            return // ignore attempts to set
        }
    }
    
    public func setError(_ input: Double) {
        return // ignore error being set
    }
    
    public var linkedNeurons = [Neuron]()
    
    init(withValue value: Double){
        amount = value
    }
    
    public func reset(){
        return // Do nothing, since we don't need to clear any cache here
    }
    
    public func sum() -> Double{
        return amount
    }
    
    public var output: Double{
        return amount
    }
    
    public func addLinkedNeuron(_ input: Neuron){
        return // do nothing, since backpropagation doesn't matter to inputs
    }
    
    public func weightFor(_ input: Neuron) -> Double {
        return 0.0
    }
}
public func ==(lhs: InputNeuron, rhs: InputNeuron) -> Bool{
    return lhs.amount == rhs.amount
}
