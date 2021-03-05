# GML_MLP_simple
 Simple Multi-Layer Perceptron in GML
 
 There are three different MLP versions:
  - Array
  - Grid
  - Mini

Array and Grid are different how they store and calculate weights. 
Mini is similiar to Array, but it's more simplified.

Uses Mean Squared Error as cost function.
Uses Tanh as activation function.

Included learning methods:
 - Stochastic Gradient Descent. 
 - Genetic Algorithm. 

SGD uses backpropagation, and batch size can be 1 or more. 
Genetic Algorithm is just one simple way of implementing it. It includes 1) Selection, 2) Offsprings, 3) Mutations.
