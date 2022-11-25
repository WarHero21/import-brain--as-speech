# Imports
from database import Database
from utils import train_valid_test_split
import os

# Create an empty directory for the database.
if not os.path.exists("Database"):
    os.mkdir("Database")

# Initialize the Database class with the empty directory.
db = Database(os.path.join(os.curdir,"Database"))

# Downloading the "MM05" participants files. The downloaded file will be in Database/TAR
db.download("MM05")

# Extracting the downloaded files. The extracted files will be in Database/PARTICIPANTS/MM05
db.extract("MM05")

# Get the EEG files for each trial. These trials will be in Database/PARTICIPANTS/MM05/DATA/{SPEAKING or THINKING}.
# Also creates figures for the EEG signals which are in Database/PARTICIPANTS/MM05/FIGURE
db.preprocess_eeg("MM05")

# Download, extract and preprocess in one single step.
db.initialize("MM08")

# Prepare data for training.
X_train, X_valid, X_test, Y_train, Y_valid, Y_test = db.load_data("MM05","thinking")

#print(X_train[0])
#print(Y_train[0])
#print(X_valid[0])
#print(Y_valid[0])
#print(X_test[0])
#print(Y_test[0])
