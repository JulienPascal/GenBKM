#!/bin/bash
echo "CONVERTING TO MARKDOW"
jupyter nbconvert --to markdown $PWD/GenBKM.ipynb
echo "DONE"

