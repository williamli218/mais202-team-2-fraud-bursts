# tests/fixtures/

Tiny synthetic files with the same names and columns as the real datasets.
A few dozen rows each, generated with a fixed seed. None of it is real data,
and none of it is big enough to train on; it exists so the tests can run in
under a minute without downloading anything.

| Folder | Mimics | Used by tracks |
|---|---|---|
| `sparkov/` | `fraudTrain.csv`, `fraudTest.csv` | 1, 2 |
| `paysim/` | the PaySim log CSV | 3 |
| `baf/` | `Base.csv` | 4, 10 |
| `phiusiil/` | `PhiUSIIL_Phishing_URL_Dataset.csv` (raw labels: 1 = legitimate) | 6 |
| `ulb/` | `creditcard.csv` | 7 |
| `elliptic/` | the three Elliptic CSVs | 9 |
| `sms/` | the SMS Spam Collection file (getting-started notebook only) | none |

Don't edit these. If a test needs a different fixture, tell your TPM.
