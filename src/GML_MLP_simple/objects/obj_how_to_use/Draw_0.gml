/// @desc RESULTS
// Different ways of accessing results.
// mlp_mini doesn't have Output-method.
// Returns array of outputs.

result = mlpa.Forward();
result = mlpg.Forward();
result = mlpm.Forward();

result = mlpa.Output();	// Returns latest Forward-result.
result = mlpg.Output();

draw_text(0,0, result[0]);
draw_text(0,0, result[1]);
