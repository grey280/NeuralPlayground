import Foundation

public class Layer{
    
    public var neurons = [Neuron]()
    
    public init(){
        
    }
    
    public func reset(){
        for neuron in neurons{
            neuron.reset()
        }
    }
    
    public func softmax() -> [Double]{ // Gets softmax info for the entire layer at once
        reset() // Since we softmax our output, this is a good place to reset everything so we don't have caching problems
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
    
    public func errorCalc() throws -> [Double]{
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
            let firstBitA = selfOut[0] - input.output[0]
            let firstBitB = selfOut[1] - input.output[1]
            let firstBit = firstBitA * firstBitA + firstBitB * firstBitB
            
            outputs.append(sqrt(firstBit) * secondBit)
        }
        
        for i in 0..<self.neurons.count{ // store the error for later use
            self.neurons[i].error = outputs[i]
        }
        
        return outputs
    }
    
    public func errorCalc(withInputs inputs: [(input: [Double], output: [Double])]) -> [Double]{
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
