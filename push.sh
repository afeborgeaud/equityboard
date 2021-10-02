#!/bin/sh
git add .
git commit -m "updates price data"
git lfs push --all origin main
git push --no-verify
