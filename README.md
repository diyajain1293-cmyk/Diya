# Employee Attrition Intelligence (Streamlit)

## Run locally
pip install -r requirements.txt
streamlit run app.py

## Deploy (Streamlit Cloud)
- Create a GitHub repo.
- Upload app.py, requirements.txt (and your dataset if desired).
- Point Streamlit Cloud to app.py on main branch.

## Notes
- Dataset must contain an 'Attrition' column.
- Sidebar filters: JobRole (multi-select) and any '*Satisfaction' column threshold.
- Tabs:
  1) Insights (5 charts)
  2) More Analytics (quick models, combined ROC, feature importances)
  3) Train & Evaluate (full 5-fold CV & plots)
  4) Upload & Predict (score new files + download results)
