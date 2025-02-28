#!/bin/bash
# Create a script to ensure the package structure is correct

# Create the required directory structure if it doesn't exist
mkdir -p /home/rishabh/Desktop/Experiments/VSR-LLM/src/models

# Make sure __init__.py files are in place
touch /home/rishabh/Desktop/Experiments/VSR-LLM/src/__init__.py
touch /home/rishabh/Desktop/Experiments/VSR-LLM/src/models/__init__.py

echo "Package structure set up correctly."
