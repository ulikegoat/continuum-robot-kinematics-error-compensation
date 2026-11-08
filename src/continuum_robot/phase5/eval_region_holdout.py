"""Evaluate saved contiguous-region models, without training."""
import sys
from .pipeline import main
if __name__ == "__main__":
    arguments = sys.argv[1:]
    for zero in (1, 2, 3):
        sys.argv = [sys.argv[0], "evaluate", "--experiment", f"region_{zero}"] + arguments
        main()
