#!/usr/bin/env bash


python -m luigi --module pipeline MyTask \
--x 100 \
--local-scheduler