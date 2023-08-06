import pandas as pd

def readColumn(file:str, column:int, clean:bool):
    data = pd.read_csv(file)
    column_data = data[data.columns[column]].tolist()
    if clean:
        for i, column in enumerate(column_data):
            column_data[i] = 'pre post start ' + column
    return column_data
