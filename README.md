# Supply Chain Project

This project provides EDA, a small ML pipeline, a SQLite database, and a Streamlit app to explore a supply chain Excel dataset.

Quick steps (Windows PowerShell):

1. Create a virtual environment and activate it:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Create the database from Excel:

```powershell
python db_setup.py
```

4. Train a model (if numeric data exists):

```powershell
python train_model.py
```

5. Run the Streamlit app:

```powershell
streamlit run app.py
```

Notes:
- The scripts try to be resilient to different column names, but you may need to adapt `db_setup.py` and `train_model.py` for your dataset.
- If pandas complains about reading Excel, ensure `openpyxl` is installed (it's in requirements.txt).
