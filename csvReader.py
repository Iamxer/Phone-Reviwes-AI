 import pandas as pd

def readColumn(file, column, clean):
    data = pd.read_csv(file)
    column_data = data[data.columns[column]]
    if clean:
        column_data = column_data.str.lower().replace('[\.,1234567890!?()\-\$%"\']', '', regex=True)
    return column_data.tolist()
