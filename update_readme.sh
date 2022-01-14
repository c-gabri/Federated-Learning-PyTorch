#!/bin/sh

usage=$(python src/main.py -h)

usage=$(echo "$usage" | \
sed 's/^usage:\ //' | \
sed -E 's/(^python\ .*)/```\1```/' | \
sed -E 's/(^[a-z])/###\ \1/' | \
sed 's/^\ \ --/\*\ --/' | \
sed -E 's/(--[^[:space:]]+(\ [^[:space:]]+)?(,\ -[^[:space:]]+(\ [^[:space:]]+)?)?)/```\1```:/' | \
sed -E 's/default:\ (.*)\)/default:\ ```\1```\)/')

requirements="* python $(grep 'Python version' src/main.py | cut -d ':' -f2 | tr -d ' ')"
requirements=$(echo "$requirements"; sort requirements.txt | sed 's/==/\ /; s/^/*\ /')

export usage requirements
perl -0 -i.bak -pe 's/(## Usage\n)(.*?)(\n## )/$1$ENV{usage}\n$3/s; s/(## Requirements\n)(.*?)(\n## )/$1$ENV{requirements}\n$3/s' README.md

cat README.md

exit 0
