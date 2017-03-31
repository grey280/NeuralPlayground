import UIKit
import PlaygroundSupport
/*:
 ## Neural Networks
 It seems like everybody is taking about artificial intelligence these days, and for good reason - it's impressive stuff. Actually understanding how it works, though, can be rather difficult.
 */
let network = Network.buildPredesignedNetwork()
/*:
 ### Structure
 The structure of a neural network is based off
 */



var tests = [(test: [Double], original: Int)]()
tests.append((test: config.buildInput(62).input, original: 62))
tests.append((test: config.buildInput(63).input, original: 63))
tests.append((test: config.buildInput(65).input, original: 65))
do{
    for test in tests{
        print("Testing \(test.original)")
        try print(network.evaluate(test.test))
        network.lastLayer.neurons.forEach({ (neuron) in
            print("  \(neuron.output)")
        })
    }
    var trainingData = [(input: [Double], output: [Double])]()
    for i in 0..<256{
        trainingData.append(config.buildInput(UInt8(i)))
    }
    try network.train(trainingData)
    for test in tests{
        print("Testing \(test.original)")
        try print(network.evaluate(test.test))
        network.lastLayer.neurons.forEach({ (neuron) in
            print("  \(neuron.output)")
        })
    }
} catch {
    print("Something threw an error in the testing function.")
}