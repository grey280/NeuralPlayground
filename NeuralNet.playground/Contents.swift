import UIKit
import PlaygroundSupport
/*:
 ## Neural Networks
 It seems like everybody is taking about artificial intelligence these days, and for good reason - it's impressive stuff. Actually understanding how it works, though, can be rather difficult.
 Let's start by making a **network**.
 */
let network2 = Network()
/*:
 The structure of a neural network is based off that of the human brain; information goes in, gets passed from neuron to neuron, and comes out turned into different information. It's not a direct analogue, of course: the brain has many ways of passing information around, and it can move in any direction, whereas in most neural networks there's just one form of information passing, and it's usually in one direction - sequential.
 For that sequential structure, we make a series of **layers** and fill them with **neurons**. (Don't worry about the term 'hidden layer' - that just means that it's not the input or the output layer.)
 */
let inputLayer = Layer()
let hiddenLayer = Layer()
let outputLayer = Layer()
/*:
 There's a couple different types of neurons that we're using here: Input neurons, and sigmoids. Input neurons do what it sounds like they do - they make it possible to put data into the network. Sigmoids do the heavy lifting of the neural network. Let's fill the layers we made earlier - we want 8 input neurons, and 2 sigmoids in the output layer, but in between can be as many layers with as many neurons as you'd like.
 */
for i in 0..<8{
    let neuron = InputNeuron()
    inputLayer.neurons.append(neuron)
}
for i in 0..<4{
    let neuron = Sigmoid(fromLayer: inputLayer)
    hiddenLayer1.neurons.append(neuron)
}
for i in 0..<2{
    let neuron = Sigmoid(fromLayer: hiddenLayer1)
    outputLayer.neurons.append(neuron)
}
/*:
Note that each neuron is linked to the previous layer. This is how information is passed around: each neuron takes **input** from *every* neuron in the previous layer. As it takes that input, however, it adjusts it by a **weight**. Once it's collected all of those adjusted inputs, it'll combine them all together, adjust by a **bias**, and then output the result, which gets passed to every neuron in the next layer, repeating the whole process.
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
