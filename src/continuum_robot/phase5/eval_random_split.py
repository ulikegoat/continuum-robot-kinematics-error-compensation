"""Evaluate saved random-split models, without training."""
import sys
from .pipeline import main
if __name__ == "__main__":
    sys.argv[1:1] = ["evaluate", "--experiment", "random"]
    main()
