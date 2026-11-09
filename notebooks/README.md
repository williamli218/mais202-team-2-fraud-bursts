# notebooks/

**One notebook per person.** Colab notebooks don't merge in Git: when two
people save the same `.ipynb`, the second push is rejected and untangling it
by hand costs an afternoon. Each teammate works in their own file, named
`track<NN>_<yourname>_<topic>.ipynb`, for example
`track03_amira_features.ipynb`. Anything a teammate needs to reuse goes in
`src/` as a function, where Git can merge it line by line.

What's here:

- `00_getting_started.ipynb`: run this once, before anything else. Git from
  Colab, tokens in Secrets, loading big files, splits, leakage, and writing
  a metric against tests.
- `starter_track<NN>_*.ipynb`: your track's starter. Make a copy with your
  name in it and work there; leave the original as it is so a teammate can
  start from a clean one. The starters for the other tracks can be deleted.

Save a notebook to the repository with **File → Save a copy in GitHub**,
and `git pull` before you start a session so you're editing the latest
`src/`.
