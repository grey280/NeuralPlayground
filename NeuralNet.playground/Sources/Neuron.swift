public protocol Neuron{ // Having this allows constant vs. sigmoid neurons, while also making it possible to gracefully interlink the two.
    var output: Double{ get } // The main useful value
    var linkedNeurons: [Neuron] { get set }
    var error: Double { get set }
    func setError(_ input: Double)
    func reset() // Clears any caching that the neuron is doing
    func sum() -> Double
    func addLinkedNeuron(_ input: Neuron)
    func weightFor(_ input: Neuron) throws -> Double
}

