# Loopy Hint

Upload a screenshot of an SGT Puzzles **Loopy** game (pentagonal / Cairo grid)
and get:

- a mistake check against the puzzle's unique solution
- one next move with a short explanation — never the whole answer

## How it works

OpenCV reads the dots, the yellow/black/gray edge states, and the number clues
(template matching). The grid's faces are reconstructed from the planar graph,
a SAT solver (CaDiCaL via python-sat) finds the loop, and a rule engine finds
the simplest forced move from your current position.

No LLM calls, no data stored. Scope: pentagonal-grid Loopy phone screenshots
(~900–1100 px wide).

## Run locally

    pip install -r requirements.txt
    streamlit run app.py
