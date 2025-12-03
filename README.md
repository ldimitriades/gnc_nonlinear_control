#NON-LINEAR CONTROL---SPACECRAFT ATTITUDE CONTROL AND DYNAMICS

## Overview
Spacecraft attitude dynamics are inherently nonlinear, which limits the accuracy of conventional linear control approaches. This project simulates spacecraft attitude motion under several control methods—linear control, nonlinear dynamic inversion (NDI), time-scale separated NDI, and incremental NDI (INDI). The goal is to compare their performance and illustrate why linear controllers struggle in nonlinear regimes while nonlinear and incremental methods provide more accurate and robust control.

## Key Concepts
- Nonlinear spacecraft attitude dynamics
- Linear vs. nonlinear control performance comparison
- Nonlinear Dynamic Inversion (NDI)
- Time-scale separated NDI
- Incremental Nonlinear Dynamic Inversion (INDI)
- Quaternion-based attitude representation and control
- Simulation of real-time control response
- Robustness considerations for nonlinear controllers

## What I Did
- Implemented the full nonlinear attitude dynamics model using quaternions
- Developed linear, NDI, time-scale separated NDI, and INDI control laws
- Built a simulation framework to compare controller performance under identical conditions
- Implemented numerical integration and real-time control loops in Python
- Generated attitude response plots to visualize stability, tracking accuracy, and robustness
- Analyzed limitations of linear control and validated improvements from nonlinear methods

## How to run
python main.py

## Tools used
-Python (NumPy, SciPy, Matplotlib)
-Github
-Sublime text
