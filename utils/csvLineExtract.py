'''
author: Jiajun Mao

description: this script is used to cut all csvs in a directory  to only its first n rows
usage: python csvLineExtract.py --path <dir_of_csvs> --length <n>    *n needs to be a string instead of int
'''

from subprocess import call
import os, sys, getopt


def process(base_path, length):
    file_list = os.listdir(base_path)

    for file in file_list:
        print ("processing: " + file)
        index = file.find("-")
        number = int(file[index-1:index])+len(file_list)
        #print (number)
        result_file_name = file[0:index-1]+str(number)+"-"+file[index:]
        print(result_file_name)
        os.system("head -n " + str(length) + " " + base_path + file + " > " + base_path+result_file_name)

        os.system("mv " + base_path + result_file_name + " " + base_path + file)


def main(argv):
    filepath = ""
    length = ""
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["path=","length="])
    except getopt.GetoptError:
        print ('csvLineExtract.py -path <dir_of_csvs> -length <length>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-help':
            print ('csvLineExtract.py -path <dir_of_csvs> -length <length>')
            sys.exit()
        elif opt in ("-path", "--path"):
            filepath = arg
        elif opt in ("-length", "--length"):
            length = arg
    process(filepath, length)


if __name__ == "__main__":
   main(sys.argv[1:])