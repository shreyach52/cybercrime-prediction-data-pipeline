# PS 26184 — Member A: Data & Synthetic Dataset Engineer

## What's in this folder
- `00_primer.py` — tiny practice script (4 core techniques), run this first
- `01_schema.py` — the shared data schema (interface contract for the team)
- `02_generate_complaints.py` — generates synthetic complaint records
- `data/` — where generated CSV/JSON files will be saved (empty for now)
- `requirements.txt` — list of Python libraries this project needs

## One-time setup (do this once)

1. **Install VS Code** if you don't have it: https://code.visualstudio.com/
2. **Install the Python extension** inside VS Code: click the Extensions icon
   on the left sidebar (looks like 4 squares), search "Python", install the
   Microsoft one (it's the top result, has a blue icon).
3. **Install Python itself** if you don't have it: https://www.python.org/downloads/
   During install on Windows, tick "Add Python to PATH" — easy to miss, matters a lot.
4. **Download this folder** to your laptop (I'll give you a zip) and put it
   somewhere sensible, e.g. `Documents/ps26184-member-a`.
5. **Open the folder in VS Code**: File → Open Folder → select `ps26184-member-a`.

## Every time you start working (do this each session)

1. Open VS Code, open this folder if it's not already open.
2. Open a terminal inside VS Code: Terminal menu → New Terminal (or `` Ctrl+` ``).
3. Create a virtual environment — this keeps this project's Python packages
   separate from everything else on your machine, so nothing conflicts:
   ```
   python -m venv venv
   ```
   (only needed once — skip this step after the first time)
4. Activate it:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
   You'll know it worked because your terminal prompt will show `(venv)` at the start.
5. Install the required packages (only needed once, or whenever requirements.txt changes):
   ```
   pip install -r requirements.txt
   ```
6. Run any script:
   ```
   python 00_primer.py
   python 02_generate_complaints.py
   ```
   You should see output printed in the terminal — tables, numbers, etc.

## VS Code tips for a beginner
- Click on any `.py` file in the left sidebar (Explorer panel) to open it.
- The ▶ "Run" button top-right of the editor does the same thing as typing
  `python filename.py` in the terminal — either works.
- If VS Code asks you to "select a Python interpreter", pick the one that
  says `venv` in its path — that's the one with your installed packages.
- Errors will show up in red in the terminal — read the LAST line of the
  error first, it usually tells you what actually went wrong.

## Where we are in the overall workflow
- [x] Step 1: Research & calibration (real NCRP/I4C stats)
- [x] Step 2: Schema design (`01_schema.py`)
- [x] Step 3: Generate Complaints (`02_generate_complaints.py`)
- [ ] Step 4: Generate Accounts + ATMs (with reuse pool)
- [ ] Step 5: Generate Transactions + Withdrawals
- [ ] Step 6: Add noise/incompleteness
- [ ] Step 7: Validate distributions
- [ ] Step 8: Export CSV/JSON
- [ ] Step 9: Build ingestion API (FastAPI)
