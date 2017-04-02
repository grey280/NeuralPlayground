public struct config{
    static let layerInfo: [Int] = [8, 2] // Default layer structure; [Int], where each value is the number of neurons in the layer. First layer will be InputNeurons, the rest will be Sigmoids
    static let defaultInput: Double = 0.0 // Default input for the InputNeurons
    static let defaultStepSize: Double = 0.1 // Default step size for training
    static let trainingIterations: Int = 25 // Number of training iterations to run
    
    /**
     Convert UInt8 to an input-output tuple of the format expected by the networks.
     - returns:
     A tuple of the formatted input-output pair for feeding into Network.evaluate
     
     - parameters:
       - inp: The input number to be converted
     */
    public static func buildInput(_ inp: UInt8) -> (input: [Double], output: [Double]){
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
    
    /**
     Converts the output of the network into a more legible format.
      - returns:
    A tuple of the chance the number fed into the network was odd or even
     
     - parameters:
       - input: An output of a neural network
     */
    public static func parseOutput(_ input: [Double]) -> (chanceOdd: Double, chanceEven: Double){
        guard input.count > 1 else{
            return (chanceOdd: 0, chanceEven: 0)
        }
        return (chanceOdd: input[0]*100, chanceEven: input[1]*100)
    }
}
