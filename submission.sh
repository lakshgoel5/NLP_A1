#!/bin/bash

# Define Entry Number
ENTRY_NO="2023CS10848"

# 1. Create the submission directory structure [cite: 156]
echo "Creating directory structure for $ENTRY_NO..."
mkdir -p "$ENTRY_NO/src"

# 2. Copy the mandatory root-level files [cite: 157-160]
echo "Copying root scripts and writeup..."
cp -p compile.sh "$ENTRY_NO/"
cp -p install_requirements.sh "$ENTRY_NO/"
cp -p run_model.sh "$ENTRY_NO/"
cp -p writeup.txt "$ENTRY_NO/"

# 3. Copy ONLY the specific source files into the src directory
# This maintains your hardcoded relative paths [cite: 161]
echo "Copying specific model and inference files..."
cp -p src/model.py "$ENTRY_NO/src/"
cp -p src/inference_task1.py "$ENTRY_NO/src/"
cp -p src/inference_task2.py "$ENTRY_NO/src/"

# 4. Create the zip file [cite: 156]
echo "Zipping files into ${ENTRY_NO}.zip..."
zip -r "${ENTRY_NO}.zip" "$ENTRY_NO"

# 5. Cleanup
echo "Cleaning up temporary directory..."
rm -rf "$ENTRY_NO"

echo "✅ Submission script finished: ${ENTRY_NO}.zip"