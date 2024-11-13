# Hill Climbing and AVM Framework Extension

## Overview

This project extends the AVM framework by integrating a Hill Climbing algorithm as an alternative to the default Alternating Variable Method (AVM) for search-based test input generation. The framework is designed to support various local search algorithms for optimizing test inputs to achieve specific branch coverage.

## Structure

1. **GenerateInputData.java**: Main entry point, where test object data generation and algorithm selection occur.
2. **HillClimbingSearch.java**: Implements the Hill Climbing search algorithm.
3. **Variable Classes** (e.g., `AtomicVariable`, `VectorVariable`, `StringVariable`, `CharacterVariable`): Extended with a `mutate()` function to support incremental changes required by Hill Climbing.

## How to Run

### Command Line Usage

To run the `GenerateInputData` with the Hill Climbing algorithm or any supported algorithm, use the following command:

```bash
java org.avmframework.examples.GenerateInputData <testobject> <branch> [search]
```
- `<testobject>`: Name of the test object (e.g., `"Calendar"`, `"Line"`, `"Triangle"`).
- `<branch>`: Branch ID in the form X(T|F), where X is the branching node number (e.g., `"5T"`).
- `[search]` (optional): Specify the search algorithm. Options are:
    - `"HillClimbingSearch"` (default)
    - `"IteratedPatternSearch"`
    - `"GeometricSearch"`
    - `"PatternSearch"`
    - `"LatticeSearch"`

### Example
```bash
java org.avmframework.examples.GenerateInputData Calendar 1T HillClimbingSearch
```

## Debug Env

Several debugging environments are configured for IntelliJ IDEA to support direct run of different algorithm.