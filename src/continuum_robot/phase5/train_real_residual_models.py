"""Train the four residual models on saved training/validation indices."""
import sys
from .pipeline import main
if __name__ == "__main__":
    sys.argv.insert(1, "train")
    main()
