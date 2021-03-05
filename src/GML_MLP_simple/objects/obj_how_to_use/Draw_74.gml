/// @desc a) TRAIN WITH BATCH

// (mlp_mini doesn't support batching.)

// Calculate batch.
repeat(batchSize) {
	// Get prediction
	mlpa.Forward( examples[index].input );
	mlpg.Forward( examples[index].input );

	// Calculate error delta for output layer
	mlpa.Cost( examples[index].output );
	mlpg.Cost( examples[index].output );

	// Backpropagate output-error through network
	mlpa.Backward();
	mlpg.Backward();
	
	// Choose next example
	index = (index+1) mod array_length(examples);
}

// After batch is done, apply cumulated values.
// Apply automatically calculates average of these.
// Apply Gradients & Deltas to update Weights & Biases.
mlpa.Apply(.1);		// Learn rate shouldn't be too large to not overshoot.
mlpg.Apply(.1);		// Also too small increases time used for learning.


// This takes minibatch of all examples, loops through them eventually and starts over.
// You don't need to loop during single step, but you can accumulate batch during several steps. Just don't Apply before batch is done.
// Also you don't need to go examples in order either.

