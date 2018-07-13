#!/bin/bash
find $1 -name "*.jpg" | while IFS= read -r file; do
#  convert "$file" -resize 448x448 "$file";
    convert "$file" -resize "448x448^" -gravity center -crop 448x448+0+0 +repage "$file"
done

find $1 -name "*.png" | while IFS= read -r file; do
    convert "$file" -resize "448x448^" -gravity center -crop 448x448+0+0 +repage "$file"
done
