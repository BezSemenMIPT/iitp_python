import click
import matplotlib.pyplot as plt

from . import __version__
from .draw import detecting_lines
from .image_processing import read_image


@click.command()
@click.argument("pic_path")
@click.version_option(version=__version__)
def main(pic_path) -> None:
    """Main file.

    Choosing file trhow terminal, apply discrete Radon trnsformation and
    after that displaying picture detecting lines on sinogram
    """
    FOLDER = "src/Radon_transform/images/"
    img = read_image(FOLDER, pic_path)

    fig, ax = plt.subplots()
    fig.set_figwidth(12)
    fig.set_figheight(12)
    fig.suptitle("Lines detection")

    max_radon, img_look = detecting_lines(img)

    fig, ax = plt.subplots()
    ax.imshow(img_look)
    fig.set_figwidth(12)
    fig.set_figheight(12)
    fig.suptitle("Radon image")
    plt.show()
