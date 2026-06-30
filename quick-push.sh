#!/bin/bash
# Usage: ./quick-push.sh "Your GitHub repo URL"

git init
git branch -m master main
git add .
git commit -m "Initial commit"
git remote add origin $1
git push -u origin main --force