"""Pytest configuration for headless test execution."""

import matplotlib

# Set non-interactive backend for headless test runs
matplotlib.use("Agg")
