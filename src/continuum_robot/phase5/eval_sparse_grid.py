"""Evaluate saved sparse-grid models, without training."""
import sys
from .pipeline import main
if __name__ == "__main__":
    arguments = sys.argv[1:]
    for step in (2, 5, 10):
        sys.argv = [sys.argv[0], "evaluate", "--experiment", f"sparse_{step}"] + arguments
        main()
