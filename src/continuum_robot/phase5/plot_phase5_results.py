"""Render saved results and research answers without training."""
import sys
from .pipeline import main
if __name__ == "__main__":
    sys.argv.insert(1, "plot")
    main()
