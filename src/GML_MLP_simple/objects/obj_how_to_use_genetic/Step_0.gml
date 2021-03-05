/// @desc APPLY GENETIC ALGORITHM

timer--;

if (timer <= 0) {
	// Sort population by fitness function
	array_sort(population, PopulationSort);
	
	// Apply genetic algorithm
	mlp_genetic_algorithm(population, .2, .7, .2, .2);
	timer = room_speed * 15;
}
