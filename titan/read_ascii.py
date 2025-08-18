import pandas as pd
import numpy as np

def read_ascii(file_path):

    matrix = np.zeros((118, 151))
    df = pd.DataFrame()
    row = 0
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # Each 'line' variable will contain one line from the file
                # including the newline character '\n' at the end.
                # You can process or print the line here.
                # print(line.strip())  # .strip() removes leading/trailing whitespace, including '\n'
                # skip commented lines
                if (line[0] == '#'):
                    tokens = line.split(" ")
                    # print(tokens[0])
                    if ("labels" in tokens[0]):
                        # parse the header separately
                        headings = tokens[1].split(",")
                        print("headings: ", headings)
                        df.columns = headings
                else:
                    tokens = line.split(" ")
                    #  NSimpleTracks,ComplexNum,SimpleNum,
                    # parse a row of data
                    for i in range(0, len(tokens)):
                        matrix[row][i] = tokens[i].float()
                        #   list comprehension convert to floats          
                    # the last columns is a polygon with 79 points; make it an array
                    # place in a dataframe? or xarray?
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    print("finally!")

    # return data_frame, polygons
    return df
