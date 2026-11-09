# data/

Everything in this folder except this file is ignored by Git. The starter
notebook fills it by calling the loaders in `src/data.py`, which download
from Kaggle or UCI on first use.

Never commit a dataset. The files are large, some have licence terms that
don't allow redistribution, and a 500 MB CSV in Git history is there forever.
If you want to skip the download on your next session, cache a cleaned copy
in your Google Drive instead (the starter notebook has a cell for that).
