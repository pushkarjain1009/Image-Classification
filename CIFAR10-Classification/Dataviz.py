import numpy as np
import matplotlib.pyplot as plt
from funcs import load_batch


class_names = ["Aeroplain", "Automobiles", "Bird", "Cat", "Deer", 
               "Dog", "Frog", "Horse", "Ship", "Truck"]

def display(dataset_path, batch_no, sample_no):
    features, labels = load_batch(dataset_path, batch_no)

    if not (0<sample_no<len(features)):
        print('{} Samples in batch {}. {} is out of range.'.format(len(features), batch_no, sample_no))
        return None

    print('\n Stats of batch #{}'.format(batch_no))
    print("# of Samples: {}\n".format(len(features)))

    label_counts = dict(zip(*np.unique(labels, return_counts=True)))

    for key, value in label_counts.items():
        print("Label Counts of [{}]({}) : {}".format(key, class_names[key], value))

    sample_image= features[sample_no]
    sample_labels= labels[sample_no]

    print("\n Eg of image: {}".format(sample_no))
    print("Image Shape: {}".format(sample_image.shape))
    print("Label: {}".format(class_names[sample_labels]))

    plt.imshow(sample_image)
    plt.show()


display("cifar-10-batches-py", 3, 7000)

