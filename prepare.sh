#!/bin/bash
# re-validate login information
mkdir -p ./.auth
uv run browser_env/auto_login.py
