#region /// INFORMATION ABOUT SCRIPT
/*
____________________________________________________________________________________________________

	MLP with calculating weights with grids.
	Uses Stochastic Gradient Descent is learning algorithm. Can use 1 example or minibatch.
____________________________________________________________________________________________________
	How to use
		Create:			mlp = new mlp_grid(layers);				Give parameters as instructed in constructor.
																
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

		Destroy:		mlp.Destroy();							As this uses grids, need to be explicitly called.
____________________________________________________________________________________________________
*/
#endregion

/// @func	mlp_grid(layerSizes);
/// @desc	Multi-Layer Perceptron, neural network. All layers are fully-connected to next one.
/// @param	{array}		layerSizes		Give layer-sizes as array of integers.				Like:	[10, 32, 16, 4]
function mlp_grid(layerSizeArray) constructor {

	#region /// INITIALIZE NETWORK - Neurons, weights and biases.
		// Configure layers.
		session	= 0;									// How many training session has been before Apply(). This is to divides gradients to get average of several examples. This is here to allow user calculate examples over time.
		layerCount	= array_length(layerSizeArray);		// How many layers there are, defined when struct is created
		layerSizes	= array_create(layerCount);			// Cache how large layers are. Looks cleaner too.
		array_copy(layerSizes, 0, layerSizeArray, 0, layerCount);

		// Initialize neurons + their deltas.
		activity	= [[undefined]];					// Combined output signals from previous layers times linked weight. Need to save for learning.
		output		= [[undefined]];					// Output signal: Activation(activity + bias). 
		bias		= [[undefined]];					// Bias for neuron activity
		delta		= [[undefined]];					// Difference between actual output and wanted output. Calculated from current example
		deltaAdd	= [[undefined]];					// Additivie/Cumulative delta from several examples
		
		var i, j, k;
		for(i = 0; i < layerCount;	i++) {				// Initialize starting values.
		for(j = 0; j < layerSizes[i]; j++) {
			activity[i][j]	= 0;
			output[i][j]	= 0;
			bias[i][j]		= random_range(-.2,+.2);	// Give default random bias.
			delta[i][j]		= 0;
			deltaAdd[i][j]	= 0;
		}}
	
		// Initialize weights & gradients
		weights		= [undefined];						// Connection between neurons
		gradients	= [undefined];						// Gradient is used in training to update weights. 
		for(i = 1; i < layerCount; i++) {
			weights[i] = ds_grid_create(layerSizes[i], layerSizes[i-1]);
			gradients[i] = ds_grid_create(layerSizes[i], layerSizes[i-1]);
			for(j = 0; j < layerSizes[i]; j++) {
			for(k = 0; k < layerSizes[i-1]; k++) {
				weights[i][# j, k] = random_range(-.5,+.5);	// Give default random weight.
			}}
		}
		
	#endregion
			
				
	/// @func	Output();
	/// @desc	Returns output-array from MLP, doesn't update
	/// @return	{array}
	static Output = function() {
		return output[layerCount-1];
	}

			
	/// @func	Forward(input);
	/// @desc	Updates MLP outputs by given input-array. for example: mlp.Forward( [1,0,0,1] );
	/// @param	{array}		input
	static Forward = function(inputArgument) {

		// Set input. Put first layers neurons' output as given input
		var i, j, k, minSize;
			i = 0;
		minSize = min(layerSizes[i], array_length(inputArgument));
		array_copy(output[i], 0, inputArgument, 0, minSize);
		
		// Update values.
		var jEnd, kEnd, J, K;
		var calc = ds_grid_create(0,0);	// Help calculator.
		for(i = 1; i < layerCount; i++) {
			// To make less array-fetching
			jEnd = layerSizes[i];		J = jEnd-1;
			kEnd = layerSizes[i-1];		K = kEnd-1;
			ds_grid_resize(calc, jEnd, kEnd);

			// Get signals from all outputs
			for(k = 0; k < kEnd; k++) {
				ds_grid_set_region(calc, 0, k, J, k, output[i-1][k]);
			}
			// Multiply signals with weights
			ds_grid_multiply_grid_region(calc, weights[i], 0, 0, J, K, 0, 0);

			// Get combined signal for neurons (activity), then calculate output. 
			for(j = 0; j < jEnd; j++) {
				activity[i][j] = ds_grid_get_sum(calc, j, 0, j, K);
				output[i][j] = Tanh(activity[i][j] + bias[i][j]);
			}
		}
		// Destroy temp calc-grid
		ds_grid_destroy(calc);
		
		// Return output. 
		return Output();
	}

	
	/// @func	Cost(target);
	/// @desc	Cost function. Uses Mean Squared Error.
	/// @desc	Calculates error, and updates error delta of output -layer.
	/// @param	{array}	target	example output, desired outcome of given input
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
	/// @desc	Backpropagates error from output-layer to input-layer. 
	static Backward = function() {
		var i, j, k, jEnd, kEnd, J, K;
	
		// Backpropagate through hidden layers.
		var calc = ds_grid_create(0,0);
		for(i = layerCount - 1; i > 0; i--) {
			jEnd = layerSizes[i];		J = jEnd-1;
			kEnd = layerSizes[i-1];		K = kEnd-1;
			ds_grid_resize(calc, jEnd, kEnd);
			
			// Finalize current deltas and calculate gradients
			for(j = 0; j < jEnd; j++) {
				delta[i][j] = delta[i][j] * TanhDerivative(activity[i][j]);
				deltaAdd[i][j] += delta[i][j];
				ds_grid_set_region(calc, j, 0, j, K, delta[i][j]);
			}
			for(k = 0; k < kEnd; k++) {
				ds_grid_multiply_region(calc, 0, k, J, k, output[i-1][k]);
			}
			ds_grid_add_grid_region(gradients[i], calc, 0, 0, J, K, 0, 0);

			// Calculate previous layer deltas
			for(j = 0; j < jEnd; j++) {
				ds_grid_set_region(calc, j, 0, j, K, delta[i][j]);
			}
			ds_grid_multiply_grid_region(calc, weights[i], 0, 0, J, K, 0, 0);
			for(k = 0; k < kEnd; k++) {
				delta[i-1][k] = ds_grid_get_sum(calc, 0, k, J, k);
			}
		}
		// Destroy temp calc-grid
		ds_grid_destroy(calc);
		session++;
	}
	
	
	/// @func	Apply(learnRate);
	/// @desc	Stochastic Gradient Descent. 
	/// @desc	Updates weights and biases according to gradients/delta. These are calculated in backpropagation
	/// @param	{real}	learnRate	How quickly MLP learns (usually between 0.1 and 0.001). 
	///								If learnrate is too high, it can jump over local minimum. This again can cause MLP to keep jumping over it from another direction.
	static Apply = function(learnRate) {		
		learnRate = learnRate / session;	// To get average of several trainings (otherwise gradients are just sum of them).
		session = 0;						// Counting starts again.
		// Update weights and biases.
		var i, j, jEnd, kEnd, J, K;
		for(i = 1; i < layerCount; i++) {
			// Update weights
			jEnd = layerSizes[i];		J = jEnd-1;
			kEnd = layerSizes[i-1];		K = kEnd-1;
			ds_grid_multiply_region(gradients[i], 0, 0, J, K, -learnRate);
			ds_grid_add_grid_region(weights[i], gradients[i], 0, 0, J, K, 0, 0);
			ds_grid_clear(gradients[i], 0);	// Clear for next round
			// Update biases
			for(j = 0; j < jEnd; j++) {
				bias[i][j] += -learnRate * deltaAdd[i][j];
				deltaAdd[i][j] = 0;
			}
		}
	}


	/// @func	Destroy();
	static Destroy = function() {
		// Destroy datastructures.
		activity = undefined;
		output = undefined;
		bias = undefined;
		delta = undefined;
		deltaAdd = undefined;
		for(var i = 1; i < layerCount; i++) {
			ds_grid_destroy(weights[i]);
			ds_grid_destroy(gradients[i]);
		}
		weights = undefined;
		gradients = undefined;
			
		// Storing no layers.
		layerCount = 0;
		layerSizes = [0];	
	}
}








