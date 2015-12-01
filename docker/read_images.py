import tensorflow as tf


def read_image(filename):
    tf.read_file(filename)


def main():
    print read_image(tf.constant('/pairs/1/90457d13.nef'))

if __name__ == '__main__':
    main()