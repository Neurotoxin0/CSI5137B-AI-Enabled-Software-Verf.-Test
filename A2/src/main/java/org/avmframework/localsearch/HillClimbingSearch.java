package org.avmframework.localsearch;

import org.avmframework.TerminationException;
import org.avmframework.Vector;
import org.avmframework.objective.ObjectiveFunction;
import org.avmframework.objective.ObjectiveValue;

public class HillClimbingSearch extends LocalSearch {

    @Override
    protected void performSearch(Vector vector, ObjectiveFunction objFun) throws TerminationException {
        ObjectiveValue currentValue = objFun.evaluate(vector);
        ObjectiveValue bestValue = currentValue;
        int maxIterations = 100; // Set a maximum iteration limit

        for (int i = 0; i < maxIterations; i++) {
            Vector neighbor = generateNeighbor(vector); // Generate a neighboring solution
            ObjectiveValue neighborValue = objFun.evaluate(neighbor);

            if (neighborValue.betterThan(currentValue)) {
                vector = neighbor;
                currentValue = neighborValue;

                if (currentValue.betterThan(bestValue)) {
                    bestValue = currentValue;
                }
            } else {
                // Stop if no improvement is found
                break;
            }
        }
    }

    // Method to generate a neighboring solution by making a small change to the current solution
    private Vector generateNeighbor(Vector vector) {
        Vector neighbor = new Vector(vector); // Create a copy of the current vector
        int indexToModify = (int) (Math.random() * vector.size()); // Randomly pick an index to modify
        neighbor.getVariable(indexToModify).randomlyMutate(); // Apply a small mutation at the chosen index
        return neighbor;
    }
}
