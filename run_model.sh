#!/bin/bash

# Check if arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: ./run_model.sh \"<Player Name>\" <stat> <line>"
    echo "Example: ./run_model.sh \"LeBron James\" pts 25.5"
    exit 1
fi

PLAYER="$1"
STAT="$2"
LINE="$3"

# Ensure dependencies are installed (optional, can be commented out if slow)
# pip install -r requirements.txt > /dev/null 2>&1

# Run the prediction script
python src/predict_player_prop.py --player "$PLAYER" --stat "$STAT" --line "$LINE"
