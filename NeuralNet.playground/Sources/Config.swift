public struct config{
    static let layerInfo: [Int] = [8, 2] // Default layer structure; [Int], where each value is the number of neurons in the layer. First layer will be InputNeurons, the rest will be Sigmoids
    static let defaultInput: Double = 0.0 // Default input for the InputNeurons
    static let defaultStepSize: Double = 0.1 // Default step size for training
    static let stepSizeChange: Double = 0.95 // Multiplier by which to change the step size after every training iteration
    static let trainingIterations: Int = 25 // Number of training iterations to run
    public static func buildInput(_ inp: UInt8) -> (input: [Double], output: [Double]){ // Helper to build a properly-shaped in/out pair
        var input = [Double]()
        let hold = Int(inp)
        var output = [Double]()
        if inp % 2 == 0{ // even number!
            output = [0, 1]
        } else {
            output = [1, 0]
        }
        
        input = [Double(hold/128 % 2), Double(hold/64 % 2), Double(hold/32 % 2), Double(hold/16 % 2), Double(hold/8 % 2), Double(hold/4 % 2), Double(hold/2 % 2), Double(hold % 2)]
        return (input: input, output: output)
    }
}
