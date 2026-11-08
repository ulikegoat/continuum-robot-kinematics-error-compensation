"""Prepare residual targets and all seven reproducible split files."""
import sys
from .pipeline import main
if __name__ == "__main__":
    sys.argv.insert(1, "prepare")
    main()
