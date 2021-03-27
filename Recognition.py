import numpy as np


def recognize():
    # DA MODIFICARE
    gallery_data = np.load("npy_db/gallery_data.npy")
    return gallery_data[:, 0]


def main():
    pass


if __name__ == '__main__':
    main()
