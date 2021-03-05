/// @desc b) TRAIN WITH BATCH 
// This way batch is calculated over several steps
// Weights & Biases are updated only when batch is completed.

// Get prediction
mlpa.Forward( examples[index].input );
mlpg.Forward( examples[index].input );

// Calculate error delta for output layer
mlpa.Cost( examples[index].output );
mlpg.Cost( examples[index].output );

// Backpropagate output-error through network
mlpa.Backward();
mlpg.Backward();
	
// Choose next example. Advance batch.
index = (index+1) mod array_length(examples);
batchPosition++;

// Apply Gradients & Deltas to update Weights & Biases.
if (batchPosition >= batchSize) {
	mlpa.Apply(.1);
	mlpg.Apply(.1);
	batchPosition = 0;	// New batch starts.
}




