#!/bin/bash

help=$(python main.py -h)
help=$(echo -e '```\n'"$help"'\n```')

pipreqs --force
requirements=$(cat requirements.txt)
requirements=$(echo -e '```\n'"$requirements"'\n```')

export help requirements
perl -0 -i.bak -pe 's/(## Help\n)(.*?)(\n## )/$1$ENV{help}\n$3/s; s/(## Requirements\n)(.*?)(\n## )/$1$ENV{requirements}\n$3/s' README.md

cat README.md

exit 0
