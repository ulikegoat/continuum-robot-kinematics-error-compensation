"""Phase 6 analysis entry point; phase5 is the legacy technical module name."""
import sys
from .pipeline import main
if __name__ == "__main__":
    sys.argv.insert(1, "analyze")
    main()
