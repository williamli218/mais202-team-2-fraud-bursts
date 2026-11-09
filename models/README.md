# models/

Saved weights and thresholds that the demo needs, and nothing bigger than a
few MB. `.gitignore` blocks weight files everywhere except in this folder,
and here it allows `.json`, `.npy` and `.pt` only, so a checkpoint saved in
the wrong place can't be pushed by accident. GitHub refuses files over
100 MB, so keep what goes here small.

If your model is small, save it here (`numpy.save` for arrays,
`torch.save(model.state_dict(), "models/weights.pt")` for a PyTorch model)
and load it in `src/predict.py`. If it's large, keep it in Google Drive and
have `load_model()` download it once; the demo can take a few seconds to
start.

Thresholds and the cost table go here too, as `thresholds.json`, once D3.1
is done.
