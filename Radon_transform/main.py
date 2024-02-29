import matplotlib.pyplot as plt
from image_processing import read_image, find_max_value
from draw import draw_line, detecting_lines


if __name__ == "__main__":
    FOLDER = "images/"
    print("Input picture file name from /images folder:")
    pic_path = input()
    img = read_image(FOLDER, pic_path)

    fig, ax = plt.subplots()
    fig.set_figwidth(12)
    fig.set_figheight(12)
    fig.suptitle("Lines detection")
    max_radon, img_look = detecting_lines(img, draw=1)
    fig.savefig("lines detection_" + pic_path)

    fig, ax = plt.subplots()
    ax.imshow(img_look)
    fig.set_figwidth(12)
    fig.set_figheight(12)
    fig.suptitle("Radon image")
    fig.savefig("transformed_picture_" + pic_path)
    plt.show()
