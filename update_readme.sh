#!/bin/sh

usage=$(python src/main.py -h)

usage=$(echo "$usage" | \
sed 's/^\ \ --/\*\ --/' | \
sed -E 's/\ ((-){1,2}[a-zA-Z_]+(\ [A-Z_]+)?)/\ ```\1```/g' | \
sed -E 's/```(\ |$)/```:\1/' | \
sed -E 's/(main\.py.*)/```\1```/' | \
sed -E 's/default:\ (.*)\)/default:\ ```\1```\)/' | \
sed -E 's/(^[a-z])/###\ \1/')

requirements="* python $(grep 'Python version' src/main.py | cut -d ':' -f2 | tr -d ' ')"
requirements=$(echo "$requirements"; sort requirements.txt | sed 's/==/\ /; s/^/*\ /')

export usage requirements
perl -i.bak -0 -pe 's/(## Usage\n)(.*?)(\n## )/$1$ENV{usage}\n$3/s; s/(## Requirements\n)(.*?)(\n## )/$1$ENV{requirements}\n$3/s' README.md

cat README.md

exit 0
