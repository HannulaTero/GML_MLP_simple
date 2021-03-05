#region /// INFORMATION ABOUT SCRIPT
/*
____________________________________________________________________________________________________
	How to use
		Create:			mlp = new mlp_array(layers);			Give parameters as instructed in constructor.
		
		Get output		mlp.Forward(array);						Input is fed as array. Array size should be same size as input-layer. (though input is clamped).
		Get output		mlp.Output();							Returns output of latest Forward().
		
		Train MLP		1.	mlp.Forward(input);					First update MLP output with example-input, then use backpropagate with example-output. 
						2.	mlp.Cost(target);					Use cost-function. This updated error delta of output.
						3.	mlp.Backward();						Backpropagates error delta of output-layer through MLP, updating deltas and gradients.
						4.	mlp.Apply(.03);						Stochastic Gradient Descent. This uses gradients which backpropagation has build up, calculates average of them. 
																
		Train MLP with batches:		Repeat steps 1-3 with different input/target -pairs. Batch-size is implicitly how many times you repeat this process.
									Repeating this process accumulates deltas and gradients, and won't update weights & biases yet.
									Repeat count can be arbitrary, average gradients and deltas are automatically calculated when "Apply()" is used.
									Use "Apply()" to use accumulated gradients and deltas to update weights & biases.
									Gradients and deltas are resetted, and ready for accumulating next batch.
									Repeat this until error is at desired level.
																
		Destroy:		As this is struct+arrays, it is grarbage-collected automatically, when not anymore referenced.
____________________________________________________________________________________________________
*/
#endregion

/// @func	mlp_array(layerSizes);
/// @desc	Multi-Layer Perceptron, neural network. All layers are fully-connected to next one.
/// @param	{array}		layerSizes		Give layer-sizes as array of integers.		Like:	[10, 32, 16, 4]
function mlp_array(layerSizeArray) constructor {
	
	#region /// INITIALIZE NETWORK - Neurons, weights and biases.
		// Configure layers.
		session	= 0;															// How many training session has been before Apply(). This is to divides gradients to get average of several examples. This is here to allow user calculate examples over time.
		layerCount	= array_length(layerSizeArray);								// How many layers there are, defined when struct is created
		layerSizes	= array_create(layerCount);									// Cache how large layers are. Looks cleaner too.
		array_copy(layerSizes, 0, layerSizeArray, 0, layerCount);
																	
		// Initialize neurons + their deltas.									
		activity	= [[undefined]];											// Combined output signals from previous layers times linked weight. Need to save for learning.
		output		= [[undefined]];											// Output signal: Activation(activity + bias). 
		bias		= [[undefined]];											// Bias for neuron activity
		delta		= [[undefined]];											// Difference between actual output and wanted output. Calculated from current example
		deltaAdd	= [[undefined]];											// Additivie/Cumulative delta from several examples
																				
		var i, j, k;															
		for(i = 0; i < layerCount;	i++) {										// Initialize starting values.
		for(j = 0; j < layerSizes[i]; j++) {									
			activity[i][j]	= 0;												
			output[i][j]	= 0;												
			bias[i][j]		= random_range(-.2,+.2);							// Give default random bias.
			delta[i][j]		= 0;												
			deltaAdd[i][j]	= 0;												
		}}																		
																				
		// Initialize weights & gradients										
		weights		= [[[undefined]]];											// Connection between neurons
		gradients	= [[[undefined]]];											// Gradient is used in training to update weights. 
		for(i = 1; i < layerCount; i++) {										// Initialize start values 
		for(j = 0; j < layerSizes[i]; j++) {									
		for(k = 0; k < layerSizes[i-1]; k++) {									
			weights[i][j][k] = random_range(-.5,+.5);							// Give default random weight.
			gradients[i][j][k] = 0;		
		}}}
		
	#endregion
		
	
	/// @func	Output();
	/// @desc	Returns output-array from MLP, doesn't update
	/// @return	{array}
	static Output = function() {
		return output[layerCount-1];
	}
	
		
	/// @func	Forward(input);
	/// @desc	Updates MLP outputs by given input-array. for example: mlp.Forward( [1,0,0,1] );
	/// @param	{array}		input		Give inputs as array.
	static Forward = function(inputArray) {

		var i, j, k, minSize;
				
		// Set input. Put first layers neurons' output as given input
		i = 0;
		minSize = min(layerSizes[i], array_length(inputArray));
		array_copy(output[i], 0, inputArray, 0, minSize);
		
		// Update values.
		for(i = 1; i < layerCount; i++) {
		for(j = 0; j < layerSizes[i]; j++) {
			activity[i][j] = 0;													// Reset activity. It is stored between rounds for backpropagation
			for(k = 0; k < layerSizes[i-1]; k++) {								
				activity[i][j] += output[i-1][k] * weights[i][j][k];			// Activity is sum of all directed signals from previous layer. Signal is output * weight, weight is connection between two neurons.
			}																	
			output[i][j] = Tanh(activity[i][j] + bias[i][j]);					// Output is (activity + bias) but also run through Activation function, which should be non-linear.
		}}
		
		// Return output. 
		return Output();
	}
		

	/// @func	Cost(targetOutput);
	/// @desc	Cost function. Uses Mean Squared Error.
	/// @desc	Calculates error, and updates error delta of output -layer.
	/// @param	{array}		targetOutput	example output, desired outcome of given input
	static Cost = function(target) {
		var i = layerCount - 1;
		// Calculate delta, use derivative of MSE.
		for(var j = 0; j < layerSizes[i]; j++) {
			delta[i][j] = (output[i][j] - target[j]);
		}
		// Calculate error, use MSE
		var error = 0;
		for(var j = 0; j < layerSizes[i]; j++) {
			error += sqr(output[i][j] - target[j]);
		}
		// Return average error. (2 is there for technicality, so derivative is correct.)
		return error / (layerSizes[i] * 2);
	}
	
	
	/// @func	Backward();
	/// @desc	Backpropagates output-error towards input 
	static Backward = function() {
		var i, j, k;													
		
		// Backpropagate through layers. From last to first.
		var varDeltaActivity;
		for(i = layerCount - 1; i > 0; i--) {
		for(j = 0; j < layerSizes[i]; j++) {
			varDeltaActivity = delta[i][j] * TanhDerivative(activity[i][j]);		// Cache this. Activation function and its derivative can be expensive calculations.
			for(k = 0; k < layerSizes[i-1]; k++) {
				gradients[i][j][k] += output[i-1][k] * varDeltaActivity;			// Gradients are cumulative, summed up. At Apply() they are divided to get average.
				delta[i-1][k] += weights[i][j][k] * varDeltaActivity;				// Update delta for connected layer. This will be used when we move to previous layer.
			}																		
			deltaAdd[i][j] += varDeltaActivity;										// Cumulative delta
			delta[i][j] = 0;														// Reset for next example.
		}}																			
		session++;																	// This was 1 training, add to count. In "Apply" this is used to divide deltas & gradients to get average.
	}
	
	
	/// @func	Apply(learnRate);
	/// @desc	Stochastic Gradient Descent. 
	/// @desc	Updates weights and biases according to gradients/delta. These are calculated in backpropagation
	/// @param	{real}	learnRate	How quickly MLP learns (usually between 0.1 and 0.001). 
	///								If learnrate is too high, it can jump over local minimum. This again can cause MLP to keep jumping over it from another direction.
	static Apply = function(learnRate) {
		var i, j, k;
		
		// Divide by training session-count to get average of several trainings. Otherwise gradients are just sum of them.
		learnRate = - learnRate / session;
																				
		// Update weights & biases												
		for(i = 1; i < layerCount; i++) {										
		for(j = 0; j < layerSizes[i]; j++) {									
			bias[i][j] += learnRate * deltaAdd[i][j];							// We want to update in opposite direction of deltas & gradients, therefore minus.
			for(var k = 0; k < layerSizes[i-1]; k++) {
				weights[i][j][k] += learnRate * gradients[i][j][k];
			}
		}}
		
		// Reset gradients and deltas for next training session.
		for(i = 1; i < layerCount; i++) {
		for(j = 0; j < layerSizes[i]; j++) {
			deltaAdd[i][j] = 0;
			for(k = 0; k < layerSizes[i-1]; k++) {
				gradients[i][j][k] = 0;
			}
		}}
		
		// Counting starts again.
		session = 0;
	}
	
	
	/// @func	Destroy();
	static Destroy = function() {
		// Destroy datastructures.
		activity = undefined;
		output = undefined;
		bias = undefined;
		delta = undefined;
		deltaAdd = undefined;
		weights = undefined;
		gradients = undefined;
		
		// Storing no layers.
		layerCount = 0;
		layerSizes = [0];	
	}
}








