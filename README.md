# import-brain-as-speech
## Team name
import brain as speech

## Team members
Görcs András        - PKDICK

Józsa Richárd       - XF4R8G

Salamon Ádám László - ASRR8L


## Project introduction

In this project we are trying to synthetize words based on EEG signals. We approach this with a classification problem similarly to the [https://arxiv.org/abs/1612.05369] paper. The project's goal is to achive better results and to extend the solution to multiple speaker.

We use the KaraOne dataset to achive the mentioned goals. This contains two types of EEG signal: Imagined EEG signal -EEG signals that generated by imagining speaking the word- and Speaking EEG signal -EEG singals that generated by speaking the word-.

## File functions and running options.
### Files
train.ipynb: Think of it as our project's main function, it calls the other components. Trains the neural net, runs predictions and evaluates them using the most well-known metrics.

hyperparameter tuner.ipynb: Jupyter notebook for training using hyperparameter optimization. It selects the best model it can find in the given hyperparameter space, then does some further training on it. Runs predictions and evaluations after the retraining just like train.ipynb. Logs the results of the optimization in the tuner folder. A copy of our logs: https://drive.google.com/file/d/10ziuIT791IK5tvYg92FrUb6SDojdMmPU/view?usp=sharing

example.py: Demonstrates the implemented functions, how we get a ready dataset with a few lines of code.

database.py: Implements the database via the Database class.

Its most important functions are:

    download( participant ): Downloads the given participant's files in a compressed .tar.bz2 file.

    extract( participant ): Extracts the given participant's downloaded .tar.bz2 files.

    preprocess_eeg( participant ): Preprocesses the participant's EEG signal, splits it into imagined and spoken trials, then finally saves them.
    
    load_data( participant, eeg_type, train_size, test_size ): Loads the EEG trials and labels, already split into training, validation and test sections.

utils.py: Implements utility functions that come in handy.

Its most importatnt functions are:

    train_valid_test_split(trials, labels, train_size, test_size): Splits the dataset into train-validation-test sets.

### Running options
First of all, download the repository.
If you want to do hyperparameter optimization (that is to try out a range of parameters), run the hyperparameter tuner.ipynb Jupyter Notebook.
If you only want to train a net the regular way, open the train.ipynb Jupyter Notebook.

This applies to both notebooks:
Before the first run: Specify which participant's data to download in block 2. Be careful, as one participant can take around 10 minutes to download and process, downloading all can take a long time. After the first run, set 'download' to False, unless you want to add more participants.
You can switch between EEG types in block 3.

After setting these parameters, just run every block. You're going to see feedback about the training process, the best performing model's going to be saved. After the training process ends, the script will load the best model and show its evaluation and metrics.
The hyperparameter optimizer runs more trainings, and further trains the best model, but essentially works the same way.
