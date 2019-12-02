from tkinter import *
from tkinter import filedialog, messagebox
from tkinter import simpledialog
from tkinter.ttk import Progressbar

from src.generate_music import generate_music
from src.prep_batch_loading import read_pitchnames, read_size_of_data
from src.train_notes import create_model
from multiprocessing import freeze_support, Value, Lock, Process
import time
n = 2


def thread_generate(input_file, output_file, length, temperature):
    n_vocab = len(read_pitchnames())  # get amount of pitch names
    model = create_model(read_size_of_data(), n_vocab)
    model.summary()
    model.load_weights("../files/models/notes/model-13-4.7501.hdf5")

    # Generate music
    generate_music(model, input_file, output_file, length, temperature=temperature)


class Application(Frame):
    def __init__(self, master=None):
        super().__init__(master)
        # Create buttons in window
        self.master = master
        self.pack()
        self.create_widgets()
        # Page Title
        self.master.title("Apollo Music Generator")
        # Window Size
        # Gets the requested values of the height and widht.
        self.master.geometry("400x350")
        windowWidth = root.winfo_reqwidth()
        windowHeight = root.winfo_reqheight()
        print("Width", windowWidth, "Height", windowHeight)
        # Gets both half the screen width/height and window width/height
        positionRight = int(root.winfo_screenwidth()/2 - windowWidth / 2)
        positionDown = int(root.winfo_screenheight()/3 - windowHeight/2)
        # Positions the window in the center of the page.
        self.master.geometry("400x350+{}+{}".format(positionRight, positionDown))

        # load model
        n_vocab = len(read_pitchnames())  # get amount of pitch names
        self.last_save_location = None
        self.file = None
        self.time = None

    def create_widgets(self):
        # Spacing formatting
        self.grid_columnconfigure(n, minsize=25)
        self.grid_rowconfigure(n, minsize=25)

        # Title
        self.title = Label(self, text="Apollo Music Generator", font=("Verdana", 14))
        self.title.grid(row=n - 2, column=0, columnspan=2)

        self.team = Label(self, text="By: Snoopy Squad", font=("Verdana", 11))
        self.team.grid(row=n - 1, column=0, columnspan=2)

        # Add text for instructions
        guide = "Instructions"
        self.add_guide = Label(self, text=guide, font=("Verdana", 10))
        self.add_guide.grid(row=n, column=0, sticky=W+E)

        guide = "1) Upload your MIDI File "
        self.add_ins1 = Label(self, text=guide, font=("Verdana", 8))
        self.add_ins1.grid(row=n + 1, column=0, sticky=W)

        guide = "2) Pick song length"
        self.add_ins4 = Label(self, text=guide, font=("Verdana", 8))
        self.add_ins4.grid(row=n + 2, column=0, sticky=W)

        guide = "3) Generate Music"
        self.add_ins2 = Label(self, text=guide, font=("Verdana", 8))
        self.add_ins2.grid(row=n + 3, column=0, sticky=W)

        guide = "4) Listen to music \n in your MIDI Player"
        self.add_ins3 = Label(self, text=guide, font=("Verdana", 8))
        self.add_ins3.grid(row=n + 4, column=0, sticky=W)

        # Get file button
        self.get_file = Button(self, text="MIDI File Input", command=self.file_input)
        self.get_file.grid(row=n + 1, column=1, sticky=W+E)

        # Get time to run button
        self.get_time = Button(self, text="Enter Song Duration", command=self.time_to_run)
        self.get_time.grid(row=n + 2, column=1, sticky=W+E)

        # Randomness slider
        self.get_randomness = Scale(self, from_=0.1, to=1.5, label='Randomness', orient=HORIZONTAL, resolution=0.01, )
        self.get_randomness.grid(row=n+3, column=1)
        self.get_randomness.set(0.3)

        # Generate music button
        self.generate_music = Button(self, text="Generate Music", command=self.generate_new_music)
        self.generate_music.grid(row=n + 4, column=1, sticky=W+E)

        # Quit buttom
        self.quit = Button(self, text="Quit", fg="red", command=self.master.destroy)
        self.quit.grid(row=n + 5, column=0, columnspan=2, padx=5, pady=10)

        # Note about model training
        guide = "The model used to generate this music was trained on MIDI files\n which are licensed under cc-by-sa Germany License to\nName: Bernd Krueger \nSource: http://www.piano-midi.de"
        self.add_ins1 = Label(self, text=guide, font=("Verdana", 6))
        self.add_ins1.grid(row=n + 7, column=0, columnspan=2)

    def file_input(self):
        # Get file action when button pressed
        self.file = filedialog.askopenfilename(initialdir="../", title="Select input file",
                                                 filetypes=(("midi files", "*.mid"), ("all files", "*.*")))
        if not self.file:
            self.file = None
        print(self.file)

    def time_to_run(self):
        # Get time to run the music generation
        self.time = simpledialog.askinteger("Generate Time", "Length of song extension in seconds")
        if not self.time or self.time <= 0:
            self.time = None

    def generate_new_music(self):
        # Generate music
        if self.file is None:
            messagebox.showerror("Error", "Must enter input file")
            return
        elif self.time is None:
            messagebox.showerror("Error", "Must enter desired song duration")
            return

        # Saves the last place we chose to save for faster usage when saving to the same location
        if self.last_save_location:
            # If the last_saved_location points to a file set it to point to that files path
            if "." in self.last_save_location:
                self.last_save_location = self.last_save_location[:str(self.last_save_location).rfind("/")]
            save_location = filedialog.asksaveasfilename(initialdir=self.last_save_location, title="Save as",
                                                 filetypes=(("midi files", "*.mid"), ("all files", "*.*")))
        else:
            save_location = filedialog.asksaveasfilename(initialdir="../", title="Save as",
                                         filetypes=(("midi files", "*.mid"), ("all files", "*.*")))
        # If the user presses cancel during file selection cancel the transaction
        if not save_location:
            return
        else:
            self.last_save_location = save_location

        print(self.last_save_location)

        # generate music on a thread
        p = Process(target=thread_generate, args=(self.file, self.last_save_location, self.time, self.get_randomness.get()))
        p.start()
        # Start loading bar
        progress = Progressbar(self, orient=HORIZONTAL,
                               length=300, mode='indeterminate')
        progress.grid(row=n+6, column=0, columnspan=2)

        # animate loading bar going left to right while music is being generated
        last_time = time.time()
        sign = 1
        t = 0
        while p.is_alive():
            self.update_idletasks()
            t += sign*(time.time()-last_time)
            last_time = time.time()
            if t > 1.0 or t < 0.0:
                t = 1.0 if t > 1.0 else 0.0
                sign = -sign
            progress['value'] = 100 * t
        progress.destroy()
        p.join()


root = Tk()
app = Application(master=root)

app.mainloop()

###########KEY###############
# app.time = time to make music
# app.file = .mid file
