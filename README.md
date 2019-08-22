# Self Driving Car Simulation

This is a simple simulation of a self driving car. I prepared an environment in
which a car can move, register collisions and detect obstacles.

Neural network is used as a decision model of the car. This neural network is trained in 2 ways: 

- The car is taught how to drive from data collected by manual driving. 

- Neural network is trained by genetic algorithm. The car learns how to drive without external data.

## Project structure

- "matlab" folder: Contains matlab code used to train the neural network on data collected by manual driving

- "python" folder: This is the main folder of the project. It contains the code for simulation, graphical elements and neuroevolution.

## Dependencies

[Panda3D](https://www.panda3d.org/) framework is used to display 3D models.