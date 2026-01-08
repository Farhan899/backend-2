#!/bin/bash
# Deployment script for Railway

# Run database migrations
alembic upgrade head

# Start the application
exec "$@"