#!/usr/bin/env python3
import os
from subprocess import run

if __name__ == "__main__":
    directory_pictures = "./Data/dataset2/pictures"
    directory_texts = "./Data/dataset2/texts"
    size_total = len(os.listdir(directory_pictures))
    size_valid = int(0.2*size_total)
    size_train = int(0.8*size_total)
    name_list = list(os.listdir(directory_pictures))

    with open("train_my.txt","w") as train_file:
        for pic in name_list[:size_train]:
            name = pic[:-4]
            train_file.write(name+"\n")
            run(['mkdir', f"./Data/ourData/{name}"])
            run(["mv", f"{directory_pictures}/{name}.jpg", f"./Data/ourData/{name}"])
            run(["mv", f"{directory_texts}/{name}.txt", f"./Data/ourData/{name}"])

    with open("valid_my.txt","w") as valid_file:
        for pic in name_list[size_train:]:
            name = pic[:-4]
            valid_file.write(name+"\n")
            run(['mkdir', f"./Data/ourData/{name}"])
            run(["mv", f"{directory_pictures}/{name}.jpg", f"./Data/ourData/{name}"])
            run(["mv", f"{directory_texts}/{name}.txt", f"./Data/ourData/{name}"])
