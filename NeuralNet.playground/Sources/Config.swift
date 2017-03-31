public struct config{
    static let layerInfo: [Int] = [8, 2] // Default layer structure; [Int], where each value is the number of neurons in the layer. First layer will be InputNeurons, the rest will be Sigmoids
    static let defaultInput: Double = 0.0 // Default input for the InputNeurons
    static let defaultStepSize: Double = 0.1 // Default step size for training
    static let stepSizeChange: Double = 0.95 // Multiplier by which to change the step size after every training iteration
    static let trainingIterations: Int = 25 // Number of training iterations to run
}
