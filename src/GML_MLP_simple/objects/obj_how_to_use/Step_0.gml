/// @desc UPDATE

// Get input
var input = [];	// 4 inputs as first layer's size is 4
input[0] = 0;
input[1] = 0;
input[2] = 0;
input[3] = 0;

// Update mlp
mlpa.Forward(input);
mlpg.Forward(input);
mlpm.Forward(input);