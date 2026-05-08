import math
import sys
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.image as image

from PIL import Image

def line_plot(file_path, labels=None):
    data = np.loadtxt(file_path)
    t = data[:,0]

    _, ax = plt.subplots()

    for i, col in enumerate(data[:,1:].T):
        if labels is None:
            ax.plot(t,col)
        else:
            ax.plot(t,col, label=labels[i])

    if labels is not None:
        ax.legend()

    plt.show()


def ground_tracking(file_path: str, img_path: str):
    """
    Plots ground tracking data from file.

    The file is expected to contain the following data:
        1. Image of ground
        2. Longitude and Latitude of satellite

    :param file_path: File path to ground tracking data
    :param img_path: Image path of ground tracking image
    """

    # Load data
    data = np.loadtxt(file_path)
    t, lons, lats = data.T

    # If data uses the same time at the beginning then
    # remove one
    if t[0] == t[1]:
        t = np.delete(t, 0)
        lons = np.delete(lons, 0)
        lats = np.delete(lats, 0)

    # Load ground track image
    ground = image.imread(img_path)

    # Setup plot
    _, ax = plt.subplots()
    ax.imshow(ground, extent=(-180.0, 180.0, -90.0, 90.0), zorder=0)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)

    # Find jumps larger than pi
    jumps = np.abs(np.diff(lons)) > np.pi
    split_indices = np.where(jumps)[0] + 1

    # Convert the longitude and latitudes to degrees to
    # better fit plot
    lons *= 180.0 / np.pi
    lats *= 180.0 / np.pi

    # Create segments from each split segment
    lon_segments = np.split(lons, split_indices)
    lat_segments = np.split(lats, split_indices)

    # Draw ground track to plot
    for lo, la in zip(lon_segments, lat_segments):
        ax.plot(lo, la, color='r', zorder=1)

    dt = t[1]-t[0] # Assumes constant delta time
    N = 60 * 60 / dt # Mark every 60min

    if N % 1.0 != 0:
        print(f"The time interval is not a full number and therefor will contain deviations!!!.")

    N = int(N)
    ax.scatter(lons[::N], lats[::N], color='red', s=10)

    plt.show()

def main(argv):
    if len(argv) == 3:
        plot_type = argv[1]
        file_path = argv[2]
        if plot_type == 'lineplot':
            line_plot(file_path)
        else:
            print("Plot type not supported yet.")
    else:
        print("Wrong number of arguments. Expected 2 (plot_type, file_path) got {}".format(len(argv)-1))

if __name__ == "__main__":
    main(sys.argv)

###################################
# Assignment 6 | Algorithms       #
###################################

def plot_ground_track(longitude, latitude, degrees=True, earth_image="earth_grid.jpg", title="Ground Track"):
    """
    Plot satellite ground track.

    Parameters
    ----------
    longitude : array-like
        Longitudes.

    latitude : array-like
        Latitudes.

    degrees : bool
        If True, inputs are already in degrees.
        Otherwise radians are assumed.

    earth_image : str or None
        Optional path to Earth texture image.

    title : str
        Plot title.
    """

    longitude = np.asarray(longitude)
    latitude = np.asarray(latitude)

    if not degrees:
        longitude = np.degrees(longitude)
        latitude = np.degrees(latitude)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Optional Earth texture
    if earth_image is not None:

        try:
            with Image.open(earth_image) as im:

                ax.imshow(
                    im,
                    extent=(-180, 180, -90, 90),
                    aspect='auto'
                )

        except FileNotFoundError:

            print(f"Could not find image: {earth_image}")

    # Detect longitude jumps
    jumps = np.where(
        np.abs(np.diff(longitude)) > 180
    )[0]

    start = 0

    for idx in jumps:

        ax.plot(
            longitude[start:idx+1],
            latitude[start:idx+1],
            linewidth=2
        )

        start = idx + 1

    # Final segment
    ax.plot(
        longitude[start:],
        latitude[start:],
        linewidth=2
    )

    # Formatting
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)

    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")

    ax.set_title(title)

    ax.set_xticks(np.arange(-180, 181, 30))
    ax.set_yticks(np.arange(-90, 91, 15))

    ax.grid(True)

    plt.tight_layout()
    plt.show()