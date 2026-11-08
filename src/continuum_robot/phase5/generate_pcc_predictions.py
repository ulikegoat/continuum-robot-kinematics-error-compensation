"""Generate predictions using an externally verified coordinate calibration."""
import sys
from .pipeline import main
if __name__ == "__main__":
    sys.argv.insert(1, "predict")
    main()
