#!/bin/sh

help=$(python src/main.py -h)

help=$(echo "$help" | \
sed 's/^\ \ --/\*\ --/' | \
sed -E 's/((-){1,2}[a-zA-Z_]+(\ [A-Z_]+)?)/```\1```/g' | \
sed -E 's/(main\.py.*)/```\1```/' | \
sed -E 's/default:\ (.*)\)/default:\ ```\1```\)/' | \
sed -E 's/(^[a-z])/####\ \1/')
export help

perl -i.bak -0 -pe 's/(## Usage\n)(.*?)(\n## )/$1$ENV{help}\n$3/s' README.md

cat README.md

exit 0
