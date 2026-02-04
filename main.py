"""Runner script: quick EDA, create DB, train model.

This script orchestrates the helper scripts: db_setup.py and train_model.py
and prints brief summaries. It's a convenience entrypoint.
"""

import subprocess
import sys
import pandas as pd

EXCEL_PATH = "Supply chain logistics problem.xlsx"

def quick_eda():
	print('Running quick EDA...')
	df = pd.read_excel(EXCEL_PATH)
	print('Shape:', df.shape)
	print('Columns:', df.columns.tolist())
	print('Head:')
	print(df.head().to_string())

def run_script(name):
	print(f'Running {name}...')
	res = subprocess.run([sys.executable, name], capture_output=True, text=True)
	print(res.stdout)
	if res.returncode != 0:
		print('Error:', res.stderr)
		raise SystemExit(res.returncode)

if __name__ == '__main__':
	quick_eda()
	run_script('db_setup.py')
	run_script('train_model.py')
	print('All done. You can run: streamlit run app.py')
