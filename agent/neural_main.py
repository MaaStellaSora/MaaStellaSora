"""Windows x64 Agent: register the optional tower alongside the standard tasks."""
from pathlib import Path
import sys

# The portable Python distribution does not add the script directory to sys.path.
agent_dir = str(Path(__file__).resolve().parent)
if agent_dir not in sys.path:
    sys.path.insert(0, agent_dir)

from main import main
from custom.action.climb_tower_neural import register_if_supported

register_if_supported()

if __name__ == "__main__":
    main()
