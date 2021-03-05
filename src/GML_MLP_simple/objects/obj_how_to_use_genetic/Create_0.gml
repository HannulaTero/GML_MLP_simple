/// @desc CREATION
// This too is just example object
// Isn't meant to work as it is, but just showcases how functions are used.


// Create population
populationSize = 100;
population = array_create(populationSize);
for(var i = 0; i < populationSize; i++) {
	population[i] = new mlp_array([10, 32, 16, 4]);
	
								// Create a way to comparing population.
	population[i].points = 0;	// Be careful to not override mlp variables though.
								// This is just simple way of doing this, you can do it anyway you want.
								
}

/// Here fitness function is simply comparing population's points against each other.
/// Part of fitness function is defining how population gains points.
function PopulationSort(mlpA, mlpB) {
	return mlpB.points - mlpA.points;
}


// You can define however you want when algorithm is applied.
// For example every 15 seconds, or when all are dead etc.
timer = room_speed * 15;
