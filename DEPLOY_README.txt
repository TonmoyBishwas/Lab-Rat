================================================================
  LAB RAT AI  -  Data Analytics Lab (Python)
  Offline. No admin. No internet.
  Tuned against TWO real Class Test papers (diamonds + penguins).
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

  To stop: close the console window.


DO THIS BEFORE THE PAPER IS HANDED OUT
--------------------------------------

  STEP 1 - POINT IT AT THE DATA FILE.   <-- the most important step

     There is a "Dataset" box just above where you type. Put the CSV in
     it and press Enter (or click Load):

         penguins.csv                        <- a bare name works
         C:\Users\student\Desktop\data.csv   <- a full path works too

     It should turn green and show something like "344 rows x 7 cols".

     WHY THIS MATTERS MORE THAN ANYTHING ELSE:
     the AI cannot see the file. Without this step it GUESSES what is
     inside - and in the last exam it guessed the sex column held
     "Male"/"Female" when the file actually held "MALE"/"FEMALE". The
     guess produced code that ran perfectly and silently turned that
     whole column into blanks. With the box set it reads the real column
     names and the real category values, and cannot get them wrong.

     If the exam hands you a file you have not seen before, set this box
     before you type a single question.

  STEP 2 - WARM IT UP.  Type "hi" and send it.

     The first message of a session is slow because the instructions are
     processed once. After that they are cached and everything is much
     faster. Do not let question 1 of the exam be the first thing you ask.


HOW TO ASK
----------

  DO IT ONE TASK AT A TIME. This is now the recommended way, and it is
  what scored best when measured against a real paper:

      task by task   30/31 checks, a usable answer every ~2 minutes
      whole paper    23/31 checks, nothing at all for ~8 minutes

  Type Task 1, paste the answer into your notebook and run it, then type
  Task 2, and so on. Each answer is short, carries its own imports, and
  continues from the variables the previous ones created instead of
  rebuilding everything.

  Mention the dataset only in your first message - the Dataset box keeps
  the real column names available for every later question.

  YOU DO NOT NEED TO SPELL COLUMN NAMES CORRECTLY, or at all. Once the
  Dataset box is set, "impute the numeric columns using the median of
  their species group" works fine. It reads the exact names from the file
  and corrects your typing. Less typing, fewer mistakes.

  (Pasting the whole paper in one message still works if you prefer it,
  but it is slower to first answer and scored lower.)

  If a reply takes too long or goes wrong, press STOP - the Send button
  becomes a red Stop button while it is writing. The "+" new-chat button
  also works at any time now; it cancels whatever is in flight.

  WHERE THE CSV MUST LIVE FOR YOUR NOTEBOOK:
  the generated code says pd.read_csv('<name>.csv'), which looks in the
  SAME FOLDER as your notebook. Save your .ipynb next to the CSV, or copy
  the CSV into your notebook's folder. FileNotFoundError is only ever this.


CHECK THESE BEFORE YOU SUBMIT
-----------------------------

  The AI never runs anything and never sees any output. It is tuned hard
  against all of these, but check anyway - this is where marks go:

  1. RUN EVERY CELL, in order. The answer is worth only what executes.

  2. DID ANY COLUMN GO BLANK? After a label-encoding step, look at that
     column. If it is all NaN, the mapping did not match the real values.
     The code now prints "unmapped: 0" after each mapping - if that
     number is not 0, that column is broken and every later step that
     uses it is wrong.

  3. IS THE TARGET OUT OF X?
         y = df['species']
         X = df.drop(columns=['species'])     <- correct
     If you see a long hand-written column list instead, read it closely.

  4. IS THE SCALER FITTED ON TRAINING DATA ONLY?
         scaler.fit_transform(X_train[...])   <- fit here
         scaler.transform(X_test[...])        <- transform only
     Never fit_transform on X_test or on the whole frame.

  5. DID EVERY TASK PRINT SOMETHING? A step that computes silently earns
     no marks. Each task should print a shape, a count, a head() or a
     computed value.

  6. LOOK AT THE FIRST COLUMN NAME IN df.info(). If it is "Unnamed: 0"
     (or "index"), the CSV was saved with its row numbers. Add this line
     after read_csv and re-run:

         df = df.drop(columns=['Unnamed: 0'])

     Harmless if you forget - nothing crashes and the answers stay
     correct - but it leaves a meaningless extra column in X.


IF IT FEELS SLOW ON THE LAB PC
------------------------------

  12th-gen Intel chips mix fast P-cores and slow E-cores, and the
  launcher's thread guess is not always best. A LOWER number is often
  faster. Open cmd in this folder and try:

      set LABRAT_THREADS=6
      launch-da.bat

  Try 4, 6 and 8; keep whichever feels quickest.

  The FIRST message of a session is always the slow one. Every message
  after it is much faster - which is exactly why you warm it up with
  "hi" before the paper arrives.


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

  That is all this exam needs. The paper supplies its own CSV, so you do
  not depend on seaborn downloading anything.

  If you want seaborn's built-in datasets to work offline too (in case a
  question uses titanic or tips instead), also run:

      python -c "import seaborn as sns; [sns.load_dataset(n) for n in ['titanic','diamonds','tips','iris','penguins','mpg','flights']]; print('cache warmed')"

  The data\ folder here holds the same CSVs as a backup - and they are
  what the Dataset box offers in its dropdown.

  Full checklist: SETUP_FOR_EXAM.md


A NOTE ON TRUST
---------------

  The model never executes anything and never sees any output. It is
  instructed never to state a result it has not printed - so when you
  want a number, the code prints it and you read it off your screen.

  If a Notes line ever names a winner ("carat is strongest", "Gentoo has
  the largest flippers") without the code printing it, treat that
  sentence as a guess and check it against your own output.
