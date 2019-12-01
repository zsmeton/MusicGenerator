from glob import glob
import random
import shutil
from sklearn.model_selection import train_test_split

testing_ratio = 0.2
training_ration = 1-(testing_ratio)

random_gen = random.Random()
X_train_files, X_test_files = train_test_split(glob('*.mid'),test_size=testing_ratio)
for i in X_test_files:
    shutil.move(f'{i}',f'testing/{i}')
for i in X_train_files:
    shutil.move(f'{i}', f'training/{i}')

