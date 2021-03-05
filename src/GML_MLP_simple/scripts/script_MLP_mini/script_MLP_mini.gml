#region /// INFORMATION ABOUT SCRIPT
/*
____________________________________________________________________________________________________

	This is more simplified version of array MLP. 
	This uses "sin(x)" as activation function. 
		-> Activation function needs to be non-linear. 
		-> Derivative of "sin(x)" is "cos(x)". Derivative of activation function is needed for "gradient descent"-learning.
		-> Sin as activation function might not yield best results, but it's built-in function and good enough for example.
	Cost function is Mean squared error, in learning derivative is used. Doesn't calculate MSE itself, or doesn't return total error.
	Stochastic Gradient Descent is learning algorithm, minibatch-size 1.
____________________________________________________________________________________________________
	How to use
		Create:			mlp = new mlp_mini(layers);			Define layer sizes as integer array. Input/output layers are first/last layer. Here input-size is 8, output-size 4.
		Get output		output = mlp.Forward(array);		Input is fed as array.
		Train MLP		mlp.Minimize(input, output);		Updates weights+biases from given input/output pair. Leanrate is .01.
____________________________________________________________________________________________________
*/
#endregion

/// @func	mlp_mini(layerSizes);
/// @desc	Multi-Layer Perceptron, neural network. All layers are fully-connected to next one.
/// @desc	Used as constructor "neural_network = new mlp_mini([2,8,2])".
/// @param	{array}	layerSizes
function mlp_mini(layerSizeArray) constructor {
	
	#region /// INITIALIZE NETWORK - Neurons, weights and biases.
		// Configure layers.
		layerCount	= array_length(layerSizeArray);		// How many layers there are, defined when struct is created
		layerSizes	= array_create(layerCount);			// Cache how large layers are. Looks cleaner too when don't have to do something like: "array_length(output[i])".
		array_copy(layerSizes, 0, layerSizeArray, 0, layerCount);
		
		// Initialize neurons + their deltas.
		activity	= [[undefined]];					// Combined output signals from previous layers times linked weight. Need to save for learning.
		output		= [[undefined]];					// Output signal: activation_function(activity + bias). 
		bias		= [[undefined]];					// Bias for neuron activity
		delta		= [[undefined]];					// Difference between actual output and wanted output. Calculated from current example
		
		var i, j, k;
		for(i = 0;	i < layerCount;	i++) {				// In actuality input-layer doesn't use activity or bias, so they don't need values.
		for(j = 0;	j < layerSizes[i]; j++) {			// But for simplicity they are defined for input too.
			activity[i][j]	= 0;
			output[i][j]	= 0;
			bias[i][j]		= random_range(-.5,.5);
			delta[i][j]		= 0;
		}}
	
		// Initialize weights & gradients
		weights		= [[[undefined]]];					// Connection between neurons
		gradients	= [[[undefined]]];					// Gradient is used in training to update weights. 
		for(i = 1; i < layerCount; i++) {
		for(j = 0; j < layerSizes[i]; j++) {
		for(k = 0; k < layerSizes[i-1]; k++) {
			weights[i][j][k] = random_range(-2, +2);
			gradients[i][j][k] = 0;		
		}}}
		
	#endregion
			
			
	/// @func	Forward(input);
	/// @desc	Updates MLP outputs by given input-array. Sends signal forward.
	/// @param	{array}		input		Give inputs as array.
	static Forward = function(inputArray) {				
		// Set input. Put first layers neurons' output as given input
		array_copy(output[0], 0, inputArray, 0, array_length(inputArray));
		
		// Update values.
		var i, j, k;
		for(i = 1; i < layerCount; i++) {
		for(j = 0; j < layerSizes[i]; j++) {
			activity[i][j] = 0;
			for(k = 0; k < layerSizes[i-1]; k++) {
				activity[i][j] += output[i-1][k] * weights[i][j][k];
			}
			output[i][j] = sin(activity[i][j] + bias[i][j]);		// output = activation_function(activity + bias)
		}}
		
		// Return output. 
		return output[layerCount-1];
	}
			
			
	/// @func	Minimize(exampleInput, exampleOutput);
	/// @desc	Tries to mimize error by updating weights and biases.
	/// @desc	Uses Stochastic Gradient Descent -learning algorithm. Batch-size 1.
	/// @desc	Backpropagates error from last layer to first layer (from output to input).
	/// @param	{array}		exampleInput		Used to update outputs.
	/// @param	{array}		exampleOutput		Desired outcome of given example input
	static Minimize = function(exampleInput, exampleOutput) {
		
		// Get predicted output
		Forward(exampleInput);

		// Compare predicted output to example-output.
		var  i, j, k;
		i = layerCount - 1;
		for(j = 0; j < layerSizes[i]; j++) {		
			delta[i][j] = output[i][j] - exampleOutput[j];				// Assumes derivative of Mean Squared error. (regression problem).
		}
		
		// Backpropagate through layers. From last to first.
		for(i = layerCount - 1; i > 0; i--) {
		for(j = 0; j < layerSizes[i]; j++) {							// Activation function is "sin()", derivative of it is "cos()".
			delta[i][j] = delta[i][j] * cos(activity[i][j] + bias[i][j]);	// delta = delta * activation_derivative(activity+bias)
			for(k = 0; k < layerSizes[i-1]; k++) {
				gradients[i][j][k] += output[i-1][k] * delta[i][j];
				delta[i-1][k] += weights[i][j][k] * delta[i][j];
			}
		}}
		
		// Update weights & biases
		var learnRate = .01;											// For learning negative learn rate is used, to update weights & biases in opposite direction of delta & gradient.
		for(i = 1; i < layerCount; i++) {
		for(j = 0; j < layerSizes[i]; j++) {
			bias[i][j] += -learnRate * delta[i][j];
			delta[i][j] = 0;
			for(var k = 0; k < layerSizes[i-1]; k++) {
				weights[i][j][k] += -learnRate * gradients[i][j][k];
				gradients[i][j][k] = 0;
			}
		}}
	}
}








