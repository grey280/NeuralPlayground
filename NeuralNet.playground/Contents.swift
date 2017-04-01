import UIKit
import PlaygroundSupport
/*:
 ## Neural Networks
 It seems like everybody is taking about artificial intelligence these days, and for good reason - it's impressive stuff. Actually understanding how it works, though, can be rather difficult.
 */
let network2 = Network()
/*:
 ### Structure
 The structure of a neural network is based off that of the human brain; information goes in, gets passed from neuron to neuron, and comes out turned into different information. It's not a direct analogue, of course: the brain has many ways of passing information around, and it can move in any direction, whereas in most neural networks there's just one form of information passing, and it's usually in one direction - sequential.
 */




let network = Network.buildPredesignedNetwork()
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
