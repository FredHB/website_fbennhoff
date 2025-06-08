#!/bin/bash

# Define the source directory containing the notebook files and the output directory
SOURCE_DIR="../bayesian_macro"
OUTPUT_DIR="./docs/bmetrics"

# Ensure the output directory exists
mkdir -p "$OUTPUT_DIR"

# Loop over each notebook file in the source directory
# for NOTEBOOK_FILE in "$SOURCE_DIR"/*.ipynb; 
# # for NOTEBOOK_FILE in "$SOURCE_DIR"/4-optimization.ipynb; 
# # for NOTEBOOK_FILE in "$SOURCE_DIR"/3-application_stoch_proc.ipynb; 
# do
#     # Convert the notebook to Markdown and save it in the specified output directory
#     jupyter nbconvert --to markdown "$NOTEBOOK_FILE" --output-dir="$OUTPUT_DIR"
# done

jupyter nbconvert --to markdown "$SOURCE_DIR/BayReg.ipynb" --output-dir="$OUTPUT_DIR"
jupyter nbconvert --to markdown "$SOURCE_DIR/TVP-AR.ipynb" --output-dir="$OUTPUT_DIR"
jupyter nbconvert --to markdown "$SOURCE_DIR/UCSV.ipynb" --output-dir="$OUTPUT_DIR"

echo "Conversion complete! The Markdown files are saved in $OUTPUT_DIR"