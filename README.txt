School Selection Advisor
========================

A tool that recommends optimal 3-school selection strategies based on
your exam score, class rank, and 5 years of historical admission data.


HOW TO SET UP (one time only)
-----------------------------

1. Make sure Python 3.9+ is installed
   - Mac: Install from https://www.python.org/downloads/
   - Or via Homebrew: brew install python3

2. Open Terminal, navigate to this folder:
   cd path/to/School_selection

3. Run the setup script:
   bash setup.sh

   This creates a virtual environment and installs all dependencies.


HOW TO RUN
----------

1. Open Terminal, navigate to this folder:
   cd path/to/School_selection

2. Run:
   bash run.sh

3. A browser window will open automatically at http://localhost:8501

4. In the sidebar:
   - Enter your latest exam score
   - Enter your class rank (1 = best)
   - Adjust total students if needed
   - Click "Analyze & Recommend"


WHAT YOU'LL SEE
---------------

1. School-by-School Analysis
   - Predicted cutoff for each school (based on 5-year trends)
   - Your score margin (positive = you're above the cutoff)
   - Admission probability at 1st and 2nd choice level
   - Category: Reach / Target / Safety

2. Recommended Strategies (shown for both optimistic and pessimistic scenarios)
   - Aggressive: 1st choice is a Reach school (high risk, high reward)
   - Balanced:   1st choice is a Target school (realistic stretch)
   - Conservative: 1st choice is a Safety school (maximize placement)

3. Strategic Summary
   - Which schools are Reach / Target / Safety for you
   - Key principles for choosing

4. Historical Cutoff Trends
   - Line chart showing how each school's cutoff changed over 5 years


FILES IN THIS PACKAGE
---------------------

School_selection/
  README.txt          - This file
  setup.sh            - One-time setup script
  run.sh              - Launch the app
  requirements.txt    - Python dependencies
  school.xlsx         - Historical school admission data (2021-2025)
  student.xlsx        - Sample student exam history
  tools/
    school_advisor.py - The main application


NOTES
-----

- The tool treats each student independently. Enter YOUR score and rank.
- Probabilities are estimates based on statistical models, not guarantees.
- The model assumes ~200 students competing for 12 schools with 1-2 quota each.
- Adjust "Score Variability" in Advanced Settings if your scores fluctuate
  more or less than +/-20 points between exams.
