"""Installed application entry point."""
import sys


def main():
    if len(sys.argv) > 1:
        from .main import main as run
    else:
        from .desktop import main as run
    run()
