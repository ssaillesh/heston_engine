import os
import sys

# The backend modules use bare imports (e.g. "from backend.core.parameters import ...")
# that assume heston_engine/ itself is on sys.path, not the repo root. run.py gets this
# for free because Python auto-adds a directly-executed script's own directory to
# sys.path[0]. pytest does not do that, so replicate it here for test collection.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'heston_engine'))
