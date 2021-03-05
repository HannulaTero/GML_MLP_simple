#region /// INFORMATION ABOUT SCRIPT
/*
____________________________________________________________________________________________________

	Here is genetic algorithms for both array and grid MLP types.
	This function is not a "ground truth", but just one simple way of implementing genetic algorithm. 
	Genetic algorithm to updates given population weight + biases
	
		1) Selection:	choose best for elite
		2) Crossover:	make children from best
		3) Mutation:	tiny changes to children
	
	Hox! Given population should already be arranged with Fitness-function. 
	Fitness-function is depent of use-case, so it's not included here. 
	You can sort array with array_sort(...), and use own sorting function.
____________________________________________________________________________________________________
*/
#endregion

/// @func	mlp_genetic_algorithm(population, elitism, mutationism, mutationRate);
/// @param	{array}		population		Array of mlp's. All need to be same type.
/// @param	{real}		elitism			Which portion of elite population continues to next generation. Rest are childs of elite
/// @param	{real}		mutationism		Which portion of new generation will be mutated
/// @param	{real}		mutationAmount	How much mutate given individual
/// @param	{real}		mutationRate	Maximium amount for random mutation
function mlp_genetic_algorithm(population, elitism, mutationism, mutationAmount, mutationRate) {
	if (instanceof(population[0]) == "mlp_grid") {
		mlp_grid_genetic_algorithm(population, elitism, mutationism, mutationAmount, mutationRate) 
	} else {
		mlp_array_genetic_algorithm(population, elitism, mutationism, mutationAmount, mutationRate) 
	}
}


/// @func	mlp_array_genetic_algorithm(population, elitism, mutationism, mutationRate);
/// @param	{array}		population		Array of mlp's. (mlp_array)	
/// @param	{real}		elitism			Which portion of elite population continues to next generation. Rest are childs of elite
/// @param	{real}		mutationism		Which portion of new generation will be mutated
/// @param	{real}		mutationAmount	How much mutate given individual
/// @param	{real}		mutationRate	Maximium amount for random mutation
function mlp_array_genetic_algorithm(population, elitism, mutationism, mutationAmount, mutationRate) {
	// Genetic algorithm does three things:
	// 1) Selection:	choose best for elite
	// 2) Crossover:	make children from best
	// 3) Mutation:		tiny changes to children
	
	var i, j, k, a, b, c;
	var populationCount, eliteCount;
	var child, parent, parentA, parentB;

	// 1) Selection
	// Take portion of population as Elite for next generation.
	// Assumes array has arranged Best-Worst already. Fitness-function is the how you arrange them, the way varies case-to-case.
	populationCount = array_length(population)
	eliteCount = max(1, ceil(populationCount * elitism));

	// 2) Cross-over
	// Make childs of elite population. Children copy randomly parts from parents.
	// In this example children has two parents, but you could have more.
	for(c = eliteCount; c < populationCount; c++) {
		a = irandom(eliteCount-1);
		b = irandom(eliteCount-1);
		while((a == b) && (eliteCount > 1)) {
			b = irandom(eliteCount-1);
		}
		parentA = population[a];
		parentB = population[b];
		child = population[c];
	
		// Go through all weights & biases
		for(i = 1; i < child.layerCount; i++) {
		for(j = 0; j < child.layerSizes[i]; j++) {
			parent = choose(parentA, parentB);
			child.bias[i][j] = parent.bias[i][j];
			for(k = 0; k < child.layerSizes[i-1]; k++) {
				parent = choose(parentA, parentB);
				child.weights[i][j][k] = parent.weights[i][j][k];
			}
		}}
	}

	// 3) Mutation
	// Mutate some of the childs with given mutation rate
	// To keep it simple we allow mutations happen to same specimen again.
	repeat((populationCount - 1) * mutationism) {
		c = irandom_range(1, populationCount-1);	// Save elite from mutations.
		with(population[c]) {
		for(i = 1; i < layerCount; i++) {
			// Mutate weights
			repeat(max(1, mutationAmount * layerSizes[i] * layerSizes[i-1])) {
				j = irandom(layerSizes[i]-1);
				k = irandom(layerSizes[i-1]-1);
				weights[@i][@j][@k] += random_range(-mutationRate, +mutationRate);
			}
			// Mutate biases
			repeat(max(1, mutationAmount * layerSizes[i])) {
				j = irandom(layerSizes[i]-1);
				bias[@i][@j] += random_range(-mutationRate, +mutationRate);
			}
		}
	}}
}


/// @func	mlp_grid_genetic_algorithm(population, elitism, mutationism, mutationRate);
/// @param	{array}		population		Array of mlp's. (mlp_grid)	
/// @param	{real}		elitism			Which portion of elite population continues to next generation. Rest are childs of elite
/// @param	{real}		mutationism		Which portion of new generation will be mutated
/// @param	{real}		mutationAmount	How much mutate given individual
/// @param	{real}		mutationRate	Maximium amount for random mutation
function mlp_grid_genetic_algorithm(population, elitism, mutationism, mutationAmount, mutationRate) {
	// Genetic algorithm does three things:
	// 1) Selection:	choose best for elite
	// 2) Crossover:	make children from best
	// 3) Mutation:		tiny changes to children
	
	var i, j, k, a, b, c;
	var populationCount, eliteCount;
	var child, parent, parentA, parentB;

	// 1) Selection
	// Take portion of population as Elite for next generation.
	// Assumes array has arranged Best-Worst already. Fitness-function is the how you arrange them, the way varies case-to-case.
	populationCount = array_length(population)
	eliteCount = max(1, ceil(populationCount * elitism));

	// 2) Cross-over
	// Make childs of elite population. Children copy randomly parts from parents.
	// In this example children has two parents, but you could have more.
	for(c = eliteCount; c < populationCount; c++) {
		a = irandom(eliteCount-1);
		b = irandom(eliteCount-1);
		while((a == b) && (eliteCount > 1)) {
			b = irandom(eliteCount-1);
		}
		parentA = population[a];
		parentB = population[b];
		child = population[c];
	
		// Go through all weights & biases
		for(i = 1; i < child.layerCount; i++) {
		for(j = 0; j < child.layerSizes[i]; j++) {
			parent = choose(parentA, parentB);
			child.bias[i][j] = parent.bias[i][j];
			for(k = 0; k < child.layerSizes[i-1]; k++) {
				parent = choose(parentA, parentB);
				child.weights[i][# j, k] = parent.weights[i][# j, k];
			}
		}}
	}

	// 3) Mutation
	// Mutate some of the childs with given mutation rate
	// To keep it simple we allow mutations happen to same specimen again.
	repeat((populationCount - 1) * mutationism) {
		c = irandom_range(1, populationCount-1);	// Save elite from mutations.
		with(population[c]) {
		for(i = 1; i < layerCount; i++) {
			// Mutate weights
			repeat(max(1, mutationAmount * layerSizes[i] * layerSizes[i-1])) {
				j = irandom(layerSizes[i]-1);
				k = irandom(layerSizes[i-1]-1);
				weights[i][# j, k] += random_range(-mutationRate, +mutationRate);
			}
			// Mutate biases
			repeat(max(1, mutationAmount * layerSizes[i])) {
				j = irandom(layerSizes[i]-1);
				bias[@i][@j] += random_range(-mutationRate, +mutationRate);
			}
		}
	}}
}