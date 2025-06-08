#!/bin/bash

# Define the source directory containing the notebook files and the output directory
SOURCE_DIR="../solving_economic_models"
OUTPUT_DIR="./docs/sem"

# Ensure the output directory exists
mkdir -p "$OUTPUT_DIR"

# Loop over each notebook file in the source directory
for NOTEBOOK_FILE in "$SOURCE_DIR"/*.ipynb; 
# for NOTEBOOK_FILE in "$SOURCE_DIR"/4-optimization.ipynb; 
# for NOTEBOOK_FILE in "$SOURCE_DIR"/3-application_stoch_proc.ipynb; 
do
    # Convert the notebook to Markdown and save it in the specified output directory
    jupyter nbconvert --to markdown "$NOTEBOOK_FILE" --output-dir="$OUTPUT_DIR"
done

echo "Conversion complete! The Markdown files are saved in $OUTPUT_DIR"