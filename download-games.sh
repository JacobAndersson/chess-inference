#!/bin/bash

GAMES_DIR="./games"
LIST_FILE="./list.txt"
LIMIT=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

mkdir -p "$GAMES_DIR"

count=0
while IFS= read -r url; do
    # Skip empty lines
    [[ -z "$url" ]] && continue

    filename=$(basename "$url")
    filepath="$GAMES_DIR/$filename"

    if [[ -f "$filepath" ]]; then
        echo "Skipping $filename (already exists)"
    else
        echo "Downloading $filename..."
        curl -L -o "$filepath" "$url"
        ((count++))
    fi

    if [[ $LIMIT -gt 0 && $count -ge $LIMIT ]]; then
        echo "Reached download limit of $LIMIT"
        break
    fi
done < "$LIST_FILE"

echo "Done. Downloaded $count files."
