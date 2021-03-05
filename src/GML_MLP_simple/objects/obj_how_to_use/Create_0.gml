/// @desc CREATION

// This object isn't for real use. Just shows how scripts are used.
// It isn't meant to actually work if you put this on the room.
// But just showcase how mlp's are used.

// Creation
var layers = [4,32,16,2];
mlpa = new mlp_array(layers);
mlpg = new mlp_grid(layers);
mlpm = new mlp_mini(layers);




// Just Assume "examples" is array, which holds lot of separate training examples.
index = 0;	// Array index for examples.
examples = [];			// You can build however you want your example-list.
examples[0] = {			// Doesn't need to hold structs.
	input : [0,0,0,0], 
	output : [0,0]		// Output should range -1 to +1.
};						// Because activation function "Tanh" can only output these
						// So you need to normalize output if they are outside these ranges.
						
						
						
// Batch
batchSize = 25;
batchPosition = 0;