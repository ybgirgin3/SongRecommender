from rich.console import Console
import matplotlib.pyplot as plt
from pprint import pprint
import numpy as np
import librosa
import librosa.display
import sys
import os

# some init
console = Console()
print = console.log
#log = console.log

# output folder
def to_output(fn):
    # uzantıyı np kendisi ekliyor zaten

    fn, ext = os.path.splitext(fn)
    fn = fn.split("/")[1]
    print("filename:", fn)
    output_folder = "output"
    ret = f"{os.path.join(output_folder, fn)}.npy"
    print(ret)
    return ret


# plotting
def plot_song(data: str):
    """
    data is path
    """
    data = np.load(data, allow_pickle=True)
    #print(data.item())
    plt.plot(data[0])
    plt.show()


# read sound
song = sys.argv[1]

print("parsing song...")
#x, sr = librosa.load(librosa.util.example_audio_file(), duration=5.0)
x, sr = librosa.load(song)

# display spectrum
print("Fourier Transform...")
# data to Fourier Transform
x  = librosa.stft(x)
# amp to db
xdb = librosa.amplitude_to_db(x)
print("xdb:")
print("type:", type(xdb))
print("shape:", xdb.shape)
print("dimension:", xdb.ndim)
print("itself:", xdb)

print("writing to np file")
# save xdb to file
np.save(to_output(song), xdb)
#np.savetxt(os.path.splitext(song)[0], xdb)

print("plotting..")
plot_song(to_output(song))


