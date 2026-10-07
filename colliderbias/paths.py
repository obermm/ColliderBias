import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.environ.get("COLLIDERBIAS_RESULTS", os.path.join(ROOT, "results"))
RUNS = os.path.join(RESULTS, "runs")
