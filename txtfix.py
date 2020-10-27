#!/usr/bin/env python3
from os import listdir

if __name__ == "__main__":
    files_txt = ["test.txt", "train.txt"]


    list_of_files = listdir("./Data/primusCalvoRizoAppliedSciences2018/")

    for txt in files_txt:
        with open("./Data/"+txt) as oldfile:
            with open("./Data/new_"+txt,"w") as newfile:
                for line in oldfile.readlines():
                    # print(line, line.strip() in list_of_files)
                    if line.strip() in list_of_files:
                        newfile.write(line)
