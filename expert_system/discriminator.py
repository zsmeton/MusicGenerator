from keras import Sequential
from keras.layers import Dense, Flatten

from training_utils import *

from expert_system.mt_generator import Generator

major_scale = [0, 2, 4, 5, 7, 9, 11]
minor_scale = [0, 2, 3, 5, 7, 8, 10]

def create_music(n=50, length=50):
    maj_generator = Generator()
    min_generator = Generator()

    songs = []

    for i in range(int(n/2)):
        songs.append(maj_generator.generate_music(length))
        maj_generator.new_offset(60 + random.randint(0, 11))

    for i in range(int(n/2)):
        songs.append(min_generator.generate_music(length))
        min_generator.new_offset(60 + random.randint(0, 11))

    return songs

def create_discriminator():
    discriminator_layers = [Flatten(input_shape=(129,50)), Dense(1024), Dense(512), Dense(1, activation='sigmoid')]
    discriminator = Sequential(discriminator_layers)
    return discriminator

def train_discriminator():
    X_train, X_val = getX_train_val()
    generated_music = create_music()

    real_songs = len(X_train)

    y_train = np.ones(real_songs)

    generated_songs = len(generated_music)

    y_train = np.append(y_train, np.zeros(generated_songs))

    X_train = np.append(X_train, generated_music)

    discriminator = create_discriminator()

    discriminator.compile(loss='binary_crossentropy',optimizer="adam", metrics=['accuracy'])

    discriminator.fit(X_train, y_train, batch_size=10, epochs=5)

if __name__ == '__main__':
    train_discriminator()