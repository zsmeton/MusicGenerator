from glob import glob
import random
import shutil

testing_ratio = 0.2
validation_ratio = 0.15
training_ration = 1-(validation_ratio+testing_ratio)

random_gen = random.Random()
for i in glob('*.mid'):
    random_num = random_gen.random()
    if random_num < testing_ratio:
        shutil.move(f'{i}',f'testing/{i}')
    elif random_num < testing_ratio+validation_ratio:
        shutil.move(f'{i}', f'validation/{i}')
    else:
        shutil.move(f'{i}', f'training/{i}')

