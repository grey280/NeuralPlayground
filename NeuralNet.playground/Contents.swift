import UIKit
import PlaygroundSupport
/*:
 ## Neural Networks
 It seems like everybody is taking about artificial intelligence these days, and for good reason - it's impressive stuff. Actually understanding how it works, though, can be rather difficult.
 Let's start by making a **network**.
 */
let network = Network()
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
    hiddenLayer.neurons.append(neuron)
}
for i in 0..<2{
    let neuron = Sigmoid(fromLayer: hiddenLayer)
    outputLayer.neurons.append(neuron)
}
/*:
Note that each neuron is linked to the previous layer. This is how information is passed around: each neuron takes **input** from *every* neuron in the previous layer. As it takes that input, however, it adjusts it by a **weight**. Once it's collected all of those adjusted inputs, it'll combine them all together, adjust by a **bias**, and then output the result, which gets passed to every neuron in the next layer, repeating the whole process.
The final step in assembling our network is to put all the layers into it:
 */
network.layers.append(inputLayer)
network.layers.append(hiddenLayer)
network.layers.append(outputLayer)
/**:
Now that we've got our network, let's put together some data to test it.
 */
var tests = [(test: [Double], original: Int)]()
tests.append((test: config.buildInput(62).input, original: 62))
tests.append((test: config.buildInput(63).input, original: 63))
tests.append((test: config.buildInput(65).input, original: 65))
/**:
Take a look at what comes back from `config.buildInput()`: `(input: [Double], output: [Double])`. The input is an array of 8 doubles, either 1 or 0 - the binary representation of the original number, spread out to be easier to feed into the network. (This is why we made 8 input neurons.)
 The output is an array of 2 doubles: the first is the chance that the number is an odd number, and the second is the chance that it's an even number. Why do it like this?
 It's the same reason we had two neurons in the output layer, in fact. The output of a sigmoid neuron is... sorta useless on its own. It's an arbitrary number, and we don't know what it means. When we have multiple outputs, though, we can get more meaning by comparing them. To get the output of the network, we pass the output of the last layer through a **softmax** function: a fun little bit of math that converts the outputs into a percentage.
 */
do{
    for test in tests{
        print("Testing network with input \(test.original)")
        let result = try network.evaluate(test.test)
        print("  Chance it's odd: \(result[0]), from sigmoid output \(network.lastLayer.neurons[0].output)")
        print("  Chance it's even: \(result[1]), from sigmoid output \(network.lastLayer.neurons[1].output)")
    }
} catch {
    // Uh oh, something went wrong!
}
/**:
 Well, *that* wasn't very accurate. I can't say for certain how inaccurate it was, because every time we initialize the sigmoid neurons, their weights are randomized; if you were *very* lucky, maybe it was 100% accurate on all the tests!
 That probably isn't the case, though, so what can we do about that?
 Well, time for the big reason people use neural networks: machine learning. If we build a big data set that has both the inputs *and* the correct outputs, we can use it to **train** the network.
 */
var trainingData = [(input: [Double], output: [Double])]()
for i in 0..<256{
    trainingData.append(config.buildInput(UInt8(i)))
}

//let network = Network.buildPredesignedNetwork()
//do{
//    for test in tests{
//        print("Testing \(test.original)")
//        try print(network.evaluate(test.test))
//        network.lastLayer.neurons.forEach({ (neuron) in
//            print("  \(neuron.output)")
//        })
//    }

//    try network.train(trainingData)
//    for test in tests{
//        print("Testing \(test.original)")
//        try print(network.evaluate(test.test))
//        network.lastLayer.neurons.forEach({ (neuron) in
//            print("  \(neuron.output)")
//        })
//    }
//} catch {
//    print("Something threw an error in the testing function.")
//}
