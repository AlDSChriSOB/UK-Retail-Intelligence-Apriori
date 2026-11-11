# Run the research notebook on Windows without VS Code

The notebook alone is not a self-contained project: it imports `src.data`, `src.chronological_holdout`, `src.recommendation_benchmark` and `src.paired_bootstrap`. A Windows drive search reporting “Not found on any drive” means the project source directory was not found; it does not indicate a notebook calculation failure.

1. Open https://github.com/AlDSChriSOB/UK-Retail-Intelligence-Apriori and choose **Code → Download ZIP**.
2. Extract the **whole** archive to a folder such as `Documents/UK-Retail-Intelligence-Apriori-main`. Do not move the notebook out of the extracted project.
3. In File Explorer, open the extracted project folder, click its address bar, type `powershell` and press Enter.
4. Run:

```powershell
py -m pip install -r requirements.txt
py -m jupyter notebook
```

5. In the Jupyter browser, open `notebooks/02_recommendation_research.ipynb` and choose **Run All**.

The first notebook code cell searches upward for `src/data.py`, then downloads the original UCI workbook if absent. Download requires internet access and may take several minutes. The research benchmark can be computationally intensive.

If Python is not installed, install it first from https://www.python.org/downloads/ and reopen PowerShell.

The notebook writes `reports/generated/per_invoice_hits.csv` and `results/figures/hit_rate_by_method.png`. Do not interpret a blank output or a kernel restart as successful execution.

## Validate after running

- Confirm that the cleaning summary, matrix shape, benchmark table, chart and bootstrap table all display.
- Confirm the plot file exists and the bootstrap interval is computed from the generated invoice-level CSV.
- Compare computed values with the repository's historical executed reports, investigate discrepancies rather than replacing them with expected values.

## Cloud alternatives

A standalone notebook uploaded to Google Colab also lacks the `src` modules. Clone or upload the **whole repository** into the Colab runtime before running it; note that hosted runtime storage is ephemeral.
