#!/bin/sh
xargs -P 10 -n 2 make < file_list
find . -empty | xargs rm
