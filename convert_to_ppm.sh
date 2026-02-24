#!/bin/bash

ROOT_DIR="${1:-.}"

find "$ROOT_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" \) | while read img; do
    out="${img%.*}.pgm"
    echo "Converting $img → $out"

    if convert "$img" "$out"; then
        rm "$img"
        echo "Removed $img"
    else
        echo "Failed: $img"
    fi
done

echo "Done."