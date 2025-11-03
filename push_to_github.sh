#!/bin/bash
if ! command -v gh &> /dev/null; then
  echo "gh CLI not found. Install from https://cli.github.com/"
  exit 1
fi
read -p "GitHub repo name (user/repo): " REPO
git init
git add .
git commit -m "Initial commit - log-ids-ml"
gh repo create "$REPO" --public --source=. --remote=origin --push
