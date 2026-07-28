================================================================
  LAB RAT AI  -  Data Analytics Lab (Python)
  Tuned against a real Class Test 1 paper.  Offline.  No admin.
================================================================


HOW TO RUN
----------

  1. COPY THIS WHOLE FOLDER TO THE LAB PC FIRST (Desktop is fine).

     Do not run it from the pendrive. The model file is 4.7 GB and gets
     read into RAM at startup - over USB that takes minutes, from the
     lab PC's SSD it takes seconds. Copying a folder needs no admin.

  2. Double-click  launch-da.bat

  3. Wait for the browser to open at  http://localhost:8080

  4. Press Ctrl+F5 once on first load.

  5. Type "hi" and send it, BEFORE the paper is handed out.

     The first question is slow because the system prompt has to be
     processed once. After that it is cached and everything is faster.
     Do not let question 1 of the exam be the first thing you ask.

  To stop: close the console window.


HOW TO USE IT IN THE EXAM
-------------------------

  PASTE THE WHOLE PAPER IN ONE MESSAGE. Include the instructions block
  and the dataset-loading code exactly as printed, then every task.

  It answers the entire paper in one go - all tasks, one code block.
  Expect several minutes. Start it, then read the paper properly while
  it works. Do not retype the questions one at a time; you lose the
  connections between tasks (the split, X_train, y_train) that way.

  IMPORTANT - where the CSV must live:
  The exam gives you diamonds.csv on the lab drive. The generated code
  says pd.read_csv('diamonds.csv'), which looks in the SAME FOLDER as
  your notebook. So save your .ipynb next to diamonds.csv, or copy the
  CSV into your notebook's folder. If you get FileNotFoundError, that
  is all this is.


CHECK THESE FIVE THINGS BEFORE YOU SUBMIT
-----------------------------------------

  The model cannot run code and never sees output. It is tuned hard
  against these five, but check anyway - they are where marks go:

  1. Is the target OUT of X?
       y = df['price']
       X = df.drop(columns=['price'])        <- correct
     If you see a long feature_cols list instead, read it carefully.

  2. Is the scaler fitted on TRAINING data only?
       scaler.fit_transform(X_train[...])    <- fit here
       scaler.transform(X_test[...])         <- transform only
     Never fit_transform on X_test or on the whole frame.

  3. "Highest correlation with price" must NOT answer "price".
     The correct answer is carat (about 0.92). If it prints price at
     1.000, the target leaked into X - see point 1.

  4. Did every task actually print something? A step that computes
     silently earns no marks. Each task should print a shape, a count,
     a head() or a computed value.

  5. Look at the FIRST column name in df.info().
     If it is called "Unnamed: 0" (or "index"), the CSV was saved with
     its row numbers. That is not a feature. Add this one line right
     after the read_csv line and re-run:

         df = df.drop(columns=['Unnamed: 0'])

     This is the one thing the AI does not do reliably on its own -
     it treats it as extra code you did not ask for. Harmless if you
     forget (nothing crashes, and the answers stay correct), but it
     leaves a meaningless extra column in X. Ten seconds to fix.

  6. RUN THE CODE. Every cell. Before you submit. The answer is only
     worth what actually executes.


IF THE .BAT FLASHES AND DISAPPEARS
----------------------------------

  Python was not found. Open PowerShell and type:  python --version
  If that prints a version, the launcher has a bug.
  If it says "not recognized", look for Anaconda or Jupyter in the
  Start menu - if the lab has Jupyter, Python IS installed, and the
  launcher prints instructions for pointing at it.


BEFORE EXAM DAY - AT HOME, WITH INTERNET
----------------------------------------

      pip install numpy pandas matplotlib seaborn scikit-learn

  That is all this exam needs. The paper supplies diamonds.csv, so you
  do not depend on seaborn downloading anything.

  If you want the seaborn built-in datasets to work offline too (in
  case a question uses titanic or tips instead), also run:

      python -c "import seaborn as sns; [sns.load_dataset(n) for n in ['titanic','diamonds','tips','iris','penguins','mpg','flights']]; print('cache warmed')"

  The data\ folder here holds the same CSVs as a backup.

  Full checklist: SETUP_FOR_EXAM.md


IF IT FEELS SLOW ON THE LAB PC
------------------------------

  12th-gen Intel chips mix fast P-cores and slow E-cores. The launcher
  guesses a thread count and the guess is not always best. A LOWER
  number is often faster. Open cmd in this folder and try:

      set LABRAT_THREADS=6
      launch-da.bat

  Try 4, 6 and 8; keep whichever feels quickest.


A NOTE ON TRUST
---------------

  The model never executes anything and never sees any output. It is
  instructed never to state a result it has not printed - so when you
  want a number, the code prints it and you read it off your screen.

  If the Notes ever name a winner ("carat is strongest", "Saturday is
  highest") without the code printing it, treat that sentence as a
  guess and check it yourself.
