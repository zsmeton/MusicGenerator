import train_notes
import numpy as np

if __name__ == "__main__":
    model = train_notes.create_model_notes(train_notes.read_size_of_data())
    model.load_weights('models/notes/note-model-03-0.0675.hdf5')
    model.summary()
    print(model.input_shape)
    train_x = np.load('batch_data/notes/train/x2.npy', allow_pickle=True)
    train_y = np.load('batch_data/notes/train/y2.npy', allow_pickle=True)
    for x,y in zip(train_x, train_y):
        y_pred = model.predict(np.array([x]))
        print(y_pred,y)
