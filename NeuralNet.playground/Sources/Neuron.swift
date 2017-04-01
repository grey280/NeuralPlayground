public protocol Neuron{ // Having this allows constant vs. sigmoid neurons, while also making it possible to gracefully interlink the two. Plus I get to play around with making protocols.
    var output: Double { get } // The output of the neuron, probably the most important part for evaluating the network.
    var error: Double { get set } // The error of the neuron; used during training.
    var linkedNeurons: [Neuron] { get set } // All the neurons that this neuron feeds to; used during backpropagation as part of the training process
    
    func setError(_ input: Double) // Set the error on the neuron
    func reset() // Clears any caching that the neuron is doing, and calls reset() on all the neurons the current one receives input from
    func sum() -> Double // Sum of the inputs to the neuron; intermediary step in calculating the output, but also important for error calculation.
    func addLinkedNeuron(_ input: Neuron) // Add a linked neuron
    func weightFor(_ input: Neuron) throws -> Double // Get the weight for a neuron
}

