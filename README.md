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
example.py: Demonstrates the implemented functions, how we get a ready dataset with a few lines of code.

database.py: Implements the database via the Database class.

Its most important functions are:

    download( participant ): Downloads the given participant's files in a compressed .tar.bz2 file.

    extract( participant ): Extracts the given participant's downloaded .tar.bz2 files.

    preprocess_eeg( participant ): Preprocesses the participant's EEG signal, splits it into imagined and spoken trials, then finally saves them.

    load_labels( participant ): Loads the trial labels.

    load_eeg_trials( participant ): Loads the EEG trials.

utils.py: Implements utility functions that comes in handy.

Its most importatnt functions are:

    train_valid_test_split(trials, labels, train_size, test_size): Splits the dataset into train-validation-test sets.

### Running options
The current running options are to use the implemented functions from a python terminal or python script.