#!/usr/bin/env bash
# Exit on error
set -o errexit

# Start the application
gunicorn app:app