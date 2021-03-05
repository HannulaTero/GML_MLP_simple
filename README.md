# GML_MLP_simple
 Simple Multi-Layer Perceptron in GML. Uses GMS2.3
 These scripts are simplified versions of other GML MLP scripts I have been writing.  
 
 This included three different MLP versions:
  - Array
  - Grid
  - Mini

Array and Grid are different how they store and calculate weights. 
Mini is similiar to Array, but it's more simplified.
Uses Tanh as activation function. 

Included learning methods:
 - Stochastic Gradient Descent. Uses backpropagation, and batch size can be 1 or more. Uses Mean Squared Error as cost function.
 - Genetic Algorithm. One simple way of implementing it. It includes 1) Selection, 2) Offsprings, 3) Mutations.
