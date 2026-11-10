# ICED Topic Exploration

Interactive exploration of the fixed ICED relational SQLite database.

The application provides topic, source, year, estimated TRL, document, evidence,
relational graph, semantic graph, historical, and personalized visualizations.
The SQLite database is versioned in this repository so the Streamlit deployment
uses a fixed reproducible dataset.

## Run locally

```powershell
python -m pip install -r requirements.txt
streamlit run iced_relational_database/app.py
```

## Deploy on Streamlit Community Cloud

1. Select the `melklr/Topic-exploration-` repository.
2. Select the `main` branch.
3. Set the main file to `iced_relational_database/app.py`.
4. Deploy.

The database is read-only at runtime and is located at
`iced_relational_database/iced_relational.sqlite`.
