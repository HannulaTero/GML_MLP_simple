/// @desc TRAIN With 1. Example

// Training, 1. example.
exampleInput = [0,1,0,1];	// Match input layer -> size 4.
exampleOutput = [.5, .9];	// Match output layer -> size 2

// mlp_mini uses only 1 function.
mlpm.Minimize(exampleInput, exampleOutput);
/*---------------------------------------*/
// mlp_array and -grid do following:

// Get prediction
mlpa.Forward(exampleInput);
mlpg.Forward(exampleInput);

// Calculate error delta for output layer
mlpa.Cost(exampleOutput);	// Also: "error = mlp.Cost(...)";
mlpg.Cost(exampleOutput);

// Backpropagate output-error through network
mlpa.Backward();
mlpg.Backward();

// Apply Gradients & Deltas to update Weights & Biases.
mlpa.Apply(.1);		// Learn rate shouldn't be too large to not overshoot.
mlpg.Apply(.1);		// Also too small increases time used for learning.

