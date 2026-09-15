================================================================
  LAB RAT AI  -  Data Analytics Lab (Python)
  Offline. No admin. No internet.
  Tuned against THREE real Class Test papers:
    CT-1  diamonds, penguins   (Parts 1-2: preprocessing + EDA)
    CT-2  bank.csv             (Parts 3-4: machine learning + deep learning)
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

     It should turn green and show something like "4521 rows x 17 cols".

     IT NOW REMEMBERS. Once set, the box survives the "+" new-chat button AND
     a brand-new browser tab. Set it once at the start of the exam and forget
     it. (If you ever reopen the page and the box looks empty, retype the
     name - it takes two seconds and it is worth checking.)

     IT ALSO DETECTS THE SEPARATOR. bank.csv is semicolon-separated, so the
     correct load is pd.read_csv('bank.csv', sep=';'). Without sep=';' pandas
     returns ONE column and every single task fails. The box reads the real
     separator off the file and tells the AI, so you cannot get this wrong.

     WHY THIS MATTERS MORE THAN ANYTHING ELSE:
     the AI cannot see the file. Without this step it GUESSES what is
     inside - and in the last exam it guessed the sex column held
     "Male"/"Female" when the file actually held "MALE"/"FEMALE". The
     guess produced code that ran perfectly and silently turned that
     whole column into blanks. With the box set it reads the real column
     names and the real category values, and cannot get them wrong.

     If the exam hands you a file you have not seen before, set this box
     before you type a single question.

  STEP 2 - LET IT WARM UP.  It now does this by itself.

     The instructions are long and get processed ONCE per session. That
     takes a few minutes, and it used to happen in front of your first real
     question. It no longer does: the moment the page loads (and again
     whenever you change the Dataset box) it quietly warms itself up in the
     background, and the Dataset box shows "... ready" when it is done.

     So: OPEN IT EARLY. Launch it and set the Dataset box while the paper is
     still being handed out, and by the time you type Q1 the slow part is
     already paid. Everything after that is fast, including every new tab -
     the processed instructions stay cached, so a new chat costs almost
     nothing.

     You can still type "hi" first if you want to see it respond. You no
     longer have to.


HOW TO ASK
----------

  ONE QUESTION BLOCK PER CHAT. Paste ALL the parts of one question together
  - Q1 (a), (b) and (c) in a single message - then press "+" for a brand-new
  chat and paste Q2, then "+" again for Q3.

  Why a new chat each time: the answer stays short and fast because the AI
  is not re-reading the whole conversation. It is tuned to understand that an
  empty chat does NOT mean an empty notebook - a block numbered Q2 or Q3, or
  one that says "your trained model", tells it to CONTINUE from what the
  earlier cells built rather than start over. That "start over" behaviour is
  what wrecked the last Class Test.

  IF THE PAPER GIVES STARTER CODE, RUN IT YOURSELF FIRST. Copy the starter
  block straight from the paper into your notebook and run it. Do not ask
  the AI to write it - it already knows that code exists and will start from
  the first thing the starter did NOT do.

  Give the dataset description only with Q1. From Q2 on, just paste the
  question - the Dataset box keeps the real column names and the real
  separator available in every new chat.

  IT ALL HAS TO LAND IN ONE NOTEBOOK, IN ORDER. Paste Q1's answer, run it,
  then Q2's answer below it, run it, then Q3's. The later blocks deliberately
  do not rebuild the earlier ones - they use the variables those cells left
  behind. If you skip a block, the next one will not run.

  EACH CONTINUATION BLOCK STARTS WITH A LINE LIKE:
      # continues the notebook - uses: mlp, X_test_s, y_test
  Read it. If one of those names is not in your notebook, you skipped
  something - fix that before running the cell.

  YOU DO NOT NEED TO SPELL COLUMN NAMES CORRECTLY, or at all. Once the
  Dataset box is set, "impute the numeric columns using the median of
  their species group" works fine. It reads the exact names from the file
  and corrects your typing. Less typing, fewer mistakes.

  (Pasting the whole paper in one message still works if you prefer it,
  but it is slower to first answer and has always scored lower.)

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

     "NameError: name 'X' is not defined" IS A FIVE-SECOND FIX, NOT A DISASTER.
     It means one import line was left out, or one variable came from a cell
     you have not run yet. Read the name it complains about:
        - a class name (OneHotEncoder, Pipeline, StandardScaler) -> add the
          import at the top of that cell and re-run it;
        - a data name (X_train_s, mlp, cm) -> you skipped an earlier block, or
          ran them out of order. Run them top to bottom.
     This is the single most likely thing to go wrong, and it costs you nothing
     if you notice it. Which is why you run every cell as you paste it, rather
     than pasting all three answers and running at the end.

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

  FOR A MACHINE-LEARNING / DEEP-LEARNING PAPER (CT-2 shape), also check:

  A. DID A PIPELINE GET PRE-SCALED DATA? A Pipeline that contains a
     StandardScaler must be handed the RAW frame:
         pipe.fit(X_train, y_train)          <- correct
         pipe.fit(X_train_s, y_train)        <- WRONG, scales twice
     Same for cross_val_score(pipe, X, y) - raw X, never X_train_s. It runs
     either way and quietly gives a worse model, so nothing warns you.

  B. ARE THE HYPER-PARAMETERS THE ONES THE PAPER NAMED? Read them back
     one by one: hidden_layer_sizes=(32, 16), activation='relu',
     solver='adam', max_iter=500, random_state=42, n_estimators=100.
     Markers check these individually.

  C. .loss_ OR .loss_curve_ ? .loss_ is ONE final number. .loss_curve_ is
     the list you plot. Asking for one and printing the other loses the mark.

  D. "Stochastic Optimizer: Maximum iterations reached" IS NOT AN ERROR.
     The cell ran, the results count. Do not change max_iter to silence it.

  E. DID THE THRESHOLD QUESTION PRINT BOTH COUNTS? Raising the threshold
     above 0.5 must make False Positives go DOWN and False Negatives go UP.
     If your printed numbers show the opposite, something is wrong with the
     matrix you compared against.

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

  The slow part is the ONE-TIME warm-up, not your questions. It now runs by
  itself as soon as the page opens, so the fix for "it feels slow" is simply
  to open the app earlier - not to wait until the paper is in your hand.


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

  That is all this exam needs - the CT-2 syllabus (machine learning and
  deep learning) is entirely scikit-learn. There is NO tensorflow, keras or
  pytorch anywhere in it: the course's neural network is
  sklearn.neural_network.MLPClassifier. The paper supplies its own CSV, so
  you do not depend on seaborn downloading anything.

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
