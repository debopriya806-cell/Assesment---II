# Rumor_RvNN — Antigravity prompt pack

**What's wrong with the repo:** it's Python 2 + Theano 0.8.2. It won't run on a current Python (`reload(sys)`, `print x`, `has_key`, `theano.compat.python2x`), and Theano can't be installed any more.
**The fix:** I already ported everything to Python 3 + PyTorch (`Rumor_RvNN_fixed.zip`). The data loading was tested on all folds; the training loop needs your first run (see Step 3).

## Step 0 — before you open Antigravity
1. Fork/clone the GitHub repo you were assigned, so you have your own copy: `git clone <your-repo-url>`.
2. Unzip `Rumor_RvNN_fixed.zip` and copy its contents over your repo: `model/*.py`, `README.md`, `requirements.txt`. Keep your `nfold/` and `resource/` folders.
3. Open the repo folder in Antigravity.

---

## Prompt 1 — environment
```
Open this repo. Create a Python virtual environment in .venv, activate it, and install requirements.txt (numpy and PyTorch, CPU build is fine). Then run `python -c "import torch, numpy; print(torch.__version__, numpy.__version__)"` and show me the output. If the install fails on this Python version, tell me which Python version I should use instead of guessing.
```
📸 Screenshot 1 → `screenshots/01_setup.png`

## Prompt 2 — quick smoke test (about a minute)
```
cd into model/ and run:
python Main_BU_RvNN.py --obj Twitter15 --fold 3 --epochs 1 --max_samples 30
python Main_TD_RvNN.py --obj Twitter15 --fold 2 --epochs 1 --max_samples 30
Both must finish without errors and print an epoch loss and a "results:" line. If anything fails, read the full traceback, fix the root cause in the code (do not just catch the exception), and re-run until both pass. Explain each fix in one line.
```

## Prompt 3 — real runs (these are the screenshots)
Start with 20 epochs. Each epoch trains on about 1,200 trees one at a time on CPU, so it is slow. Check the time of the first epoch, then pick the epoch count. Ten or more is enough to show it working.
```
From model/, run the bottom-up model and let it finish:
python Main_BU_RvNN.py --obj Twitter15 --fold 3 --epochs 20 --eval_every 5
Then run the top-down model:
python Main_TD_RvNN.py --obj Twitter15 --fold 2 --epochs 20 --eval_every 5
Keep the terminal output visible. When each run ends, tell me the final accuracy (the number after 'acc:') and Favg from the last "results:" line. If a run is too slow (first epoch over ~3 minutes), tell me and suggest an epoch count instead of silently changing it.
```
📸 Screenshot 2 (BU run: loading lines, epoch losses, `results:`) → `screenshots/02_bu_run.png`
📸 Screenshot 3 (TD run, same) → `screenshots/03_td_run.png`

## Prompt 4 — if something breaks (paste the traceback)
```
This is the full error from running <command>:
<paste traceback>
Find the root cause, fix it in the source, and re-run the same command. Don't change the model architecture or the data files. Summarise the change in one line.
```

## Prompt 5 — README and push
```
1. Create a `screenshots/` folder in the repo root. I will drop 01_setup.png, 02_bu_run.png and 03_td_run.png into it.
2. In README.md, fill in the "Final metrics" table with the accuracy and Favg from my two runs (epochs: <N>). Do not invent numbers; use only what the terminal printed.
3. Add a .gitignore entry for .venv/ and __pycache__/.
4. Show me `git status` and a proposed commit message, then commit and push to origin main.
```
Alternative: do the git part yourself.
```bash
git add .
git commit -m "Port Rumor_RvNN to Python 3 + PyTorch, add run instructions and screenshots"
git push origin main
```
Before you push, open the README preview and check that the three images render. Image paths are case-sensitive on GitHub.

---

### Things to know if your teacher asks
- **Why PyTorch?** The original needs Theano 0.8.2 and numpy 1.11, which can't be installed on a modern Python. The architecture and maths are kept the same: tree GRU, hard-sigmoid gates, squared-error loss on softmax, gradient clipping at 5, SGD, and a learning rate that halves when the loss goes up.
- **Why does BU use only ~680 training trees on Twitter15?** The provided `data.BU_RvNN` file only holds 1,793 of the 3,098 trees. That is the original behaviour and I didn't touch it. TD uses all of them.
- **A quirk in the original:** its "momentum" never took effect. The velocity was a Python `0.` fixed when the graph was built. So the original is effectively plain SGD with clipping, and the port keeps that.
- Accuracy after 20 epochs will be well below the paper's (about 0.7–0.8 at 500+ epochs). That's expected.
