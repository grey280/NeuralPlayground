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
/*:
Now that we've got our network, let's put together some data to test it.
 */
var tests = [(test: [Double], original: Int)]()
tests.append((test: config.buildInput(62).input, original: 62))
tests.append((test: config.buildInput(63).input, original: 63))
tests.append((test: config.buildInput(65).input, original: 65))
/*:
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
    // Uh oh, something went wrong! Which kind of NeuralNetError was it?
}
/*:
 Well, *that* wasn't very accurate. I can't say for certain how inaccurate it was, because every time we initialize the sigmoid neurons, their weights are randomized; if you were *very* lucky, maybe it was 100% accurate on all the tests!
 
 That probably isn't the case, though, so what can we do about that?
 
 Well, time for the big reason people use neural networks: machine learning. If we build a big data set that has both the inputs *and* the correct outputs, we can use it to **train** the network.
 */
var trainingData = [(input: [Double], output: [Double])]()
for i in 0..<256{
    trainingData.append(config.buildInput(UInt8(i)))
}
do{
    try network.train(trainingData)
}catch{
    // Uh oh, something went wrong! Check the NeuralNetError.
}
/*:
 What just happened there? It probably took a little bit to run - training involves a whole lot of math.
 
 Basically put, training involves feeding in an input and then comparing the result to the known answer. That error, or **cost**, is then used to figure out the error on every neuron in the network through something called **backpropagation**. Again, a whole lot of math, [explained here](http://neuralnetworksanddeeplearning.com/chap2.html) much more clearly than I'd be able to do.
 
 Once we know the error on each neuron, we adjust the weights and the bias. The exact manner we do this is known as **stochastic gradient descent** or, more generally, **gradient descent**. (The "stochastic" there means "there's too much data to do it all at once, so we'll do it with a small chunk of the data set at a time.") We calculate the gradient of the error and then adjust things in the direction that reduces the error - the common example is setting a ball on the side of a valley and seeing where it stops in order to find the deepest point in the valley.
 
 Now, let's see how well it worked:
 */
do{
    for test in tests{
        print("Testing network with input \(test.original)")
        let result = try network.evaluate(test.test)
        print("  Chance it's odd: \(result[0]), from sigmoid output \(network.lastLayer.neurons[0].output)")
        print("  Chance it's even: \(result[1]), from sigmoid output \(network.lastLayer.neurons[1].output)")
    }
}catch{
    // Uh oh, something went wrong! Check the NeuralNetError.
}
/*:
 Hopefully that was a *touch* more accurate. It probably wasn't too much better, though - one iteration of training isn't all that much. Neural networks generally take quite a bit of training before they'll work well for what their intended use is.
 
 Of course, this is a *very* limited example - there's only 2^8 possible inputs, after all, and we started with a single hidden layer. Large scale neural networks can have billions of inputs requiring tens of thousands of input neurons and millions of neurons all working in concert.
 
 The principles are the same, though: layers of neurons, passing values from one layer to the next, and learning from known data sets.
 */