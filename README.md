This is a manual to reproduce the results of nsm with a clean WikiTableQuestions download, which was not preprocessed by the authors of nsm.
For the preprocessed version follow the instructions on the project's repository on GitHub.

The manual also includes necessary adjustments for a custom dataset to run.

## Prerequisites:

- Setup nsm as instructed in the [GitHub](https://github.com/crazydonkey200/neural-symbolic-machines) repository (adjust to your machine, like paths and venv stuff).
- Download [WikiTableQuestions](https://nlp.stanford.edu/software/sempre/wikitable/)
- Download pretrained [GloVe:](https://nlp.stanford.edu/projects/glove/)[Wikipedia 2014 + Gigaword 5](https://nlp.stanford.edu/projects/glove/)
- **(Optional)**  download the wikitable processed for nsm as described on the git project page.
- **(Optional) For custom datasets:** download and install [SEMPRE](https://github.com/percyliang/sempre).
After that, follow the instructions to **install tables SEMPRE** - [link](https://github.com/percyliang/sempre/tree/master/tables)

## Preprocessing:

- Make a base folder for your data with 3 folders in it: raw_input, processed_input and output
- Create a GloVe matrix (npy) and list (json).
  - Use can use create_glove_matrix.py from the scripts folder to do so. Make sure you change the hardcoded paths in it.
  - Implementing your own (in python):
    - Load the glove file, find all words in the training set that appear in glove and in the tables column names.
    - Create a dictionary of these words and the matching vectors.
    - Save the keys list in a json file and save the matrix (from stacking the vectors) to a npy file.\
      Note that the order of the matrix should match the order of the words in the json list.
  - Examples can be found on the processed wikitable that can be downloaded from the project's page.
- Move the npy and json file created in the previous section to raw_input.
- Add the downloaded WikiTableQuestions folder to raw_input.
- Add stop_words.json and trigger_word_all.json to raw_input, they can be found either in the scripts folder or on the processed dataset wikitable version in it's raw_input folder.
- Add the file 'preprocess_fixed.py' from the script folder to (...)/neural-symbolic-machines/table/wtq in your nsm dir.
- In the same folder, find 'preprocess.sh' (in neural-symbolic-machines/table/wtq), change it to run preprocess_fixed.py and change the paths to your base folder.
- **Run the edited preprocess.sh.** It should go without any errors and finish in a few minutes.
- In the same folder, find 'explore.sh'. **It's run is rather heavy and takes a long time!\
  I personally had to split it into iterations on my local, so chose one of the following based of your hardware:**
  - **(a)** Change the paths in 'explore.sh' to your base folder and run it. (simple)
  - **(b)** Use 'explore_iterations.sh' from the scripts folder, it iterates over the exploration.
      - Change random_explore.py to write in "a" mode and not "w" mode (line 231).
      - You can alter "inc" to whatever value you would like, I used 8 because my local have 8 cores.
- Copy output/random_explore/saved_program.json to processed_input rename to all_train_saved_program.json.
- **If explore_iterations.sh was chosen (option b):** Inside all_train_saved_program.json replace "}{" with ", ", since the append simply concat different jsons (this should make the json format legal and correct).

## Training:

Edit 'run.sh' in neural-symbolic-machines/table/wtq to have the correct paths.
Other information can be found on the project's GitHub page.\
Notes:
1. n_actors is heavy, on local I recommand changing it to 3.
2. train_use_gpu/eval_use_gpu should be changed to 0 if you don't have CUDA or gpu tensorflow.
3. User train_gpu_id/eval_gpu_id for the id of the gpu to use (if you have multiple)
4. Add debug flag to run if something went wrong.

## Adjusting a custom dataset:

- Make sure your dataset is in the WikiTableQuestions format and folder structure (expanded in details on subsections). I recommend getting to know the wtq folder structure so that the following instruction would make sense.
Note that not all the files in wtq are necessary.
  - **Base folder:** base folder for your dataset.
  - **Tables:** should be in csv format the same as in wtq with the name "%d.csv" where %d is the number of the table.\
    The tables should be under a folder named "csv" and subfolder 200-csv (or some other number).
  - **Questions:**\
    You should have a training dataset, testing and also some dev splits (in wtq there are 5).\
    All files should be under a folder named "data" and in 2 formats: tsv (regular tsv) and examples.\
    I recommand keeping the same names for your datasets as the ones in wtq since some of them are hard coded in nsm preprocessing.\
    *.examples is in lisp format, it's for the SEMPRE tagging. Here is a breakdown of a line (question) in the examples format:
    ```
    (example (id nt-0) (utterance "what was the last year where this team was a part of the usl a-league?") (context (graph tables.TableKnowledgeGraph csv/204-csv/590.csv)) (targetValue (list (description "2004"))))
    ```
    __Notes about *.examples format:__
    - The format should be kept the same.
    - Id is the question id and should be in format "nt-%d" where %d is the question number.
    - Utterance is the question text
    - Context should be kept with the same format, make changes only to the path. In the example above the path is BASEFOLDER/csv/204-csv/590.csv.
    - targetValue is a list of answers.
  - **Tagged data:** Create an empty "tagged" folder under the base folder.
- You need to tag your data, so download and install SEMPRE and SEMPRE tables (as mentioned in the downloads section).
- Follow the instructions on how to tag questions and table [here](https://github.com/percyliang/sempre/tree/master/tables#generating-tagged-files) (Courtesy of wtq's author, Panupong Pasupat).\
  Tag your tables, train set and test set.\
  NOTE: the path to "table-base-directory" you provided will be copied to the tag file. You should remove anything before "csv/%d1-csv/%d2.csv".
- Move the tagged data to the "tagged" folder under you base folder.\
  Rename the train data to training.tagged and test data to pristine-unseen-tables.tagged.
- In preproccess_fixed.py, many of the file names are hard coded.\
  Only concern yourself with the main function, as it is the one handling the file paths.\
  Change file names/logic according to your dataset.
  - Common change: the script assumes you have 5 splits for train and dev out of your training dataset, change it according to the number of splits your dataset have (line 629).
- Now you can preprocess your data as mentioned in the Preprocessing section.
