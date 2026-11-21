# Project 05 — FP-Growth support–confidence sensitivity

**Research question:** How do mining thresholds change the number of association rules and masked-item recommendation accuracy?

The experiment uses genuine [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail) UK transactions. It evaluates **all 27** combinations of three chronological training origins (50/60/70%), three supports (0.5/1/2%), and three directional confidences (10/20/40%) on their following 10% held-out invoice windows.

Outputs: HR@5, HR@10, itemset counts, rule counts, support–confidence heatmap, complexity plot and evidence-generated manuscript. No test-selected winner or statistically independent fold claim.

```bash
pip install -r requirements.txt
python -m pytest -q tests/test_project05_fpgrowth_sensitivity.py
python - <<'PY'
from src.data import download_dataset,extract_dataset
print(extract_dataset(download_dataset()))
PY
python -m projects.project05_fpgrowth_sensitivity.analysis "data/raw/Online Retail.xlsx"
python -m projects.project05_fpgrowth_sensitivity.paper reports/generated/project05_sensitivity.json
```
