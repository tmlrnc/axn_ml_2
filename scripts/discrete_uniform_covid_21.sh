#!/usr/bin/env bash
for I in 3 4; do
    for J in 3 4; do
    echo "Running $I $J"
    FIX="$I$J"
    echo "$FIX"
    done
done