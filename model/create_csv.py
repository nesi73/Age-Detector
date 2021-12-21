import os
from pandas import DataFrame

class LoadData:

    def __init__(self):
        self.folder_database = "../../databaseAgeMioNoPreprocess/"
        folder = os.listdir(self.folder_database)
        self.cont = 0
        self.lenFolder = len(folder)
        df_list = map(self.write_csv, folder)
        df = DataFrame(df_list, columns=['Path', 'Age', 'Categorical'])
        df.to_csv('database_ageMio.csv')

    # [0 - 29] adolescentes, [30 - 49] adultez, [51 - > ] mayor
    def write_csv(self, file):
        age = 'mayores'
        categorical = '2'
        
        if int(file.split("_")[2]) < 30:
            age = 'adolescente'
            categorical = '0'
        elif int(file.split("_")[2]) < 50:
            age = 'adultez'
            categorical = '1'
        self.cont += 1

        return ['databaseAgeMioNoPreprocess/' + file, age, categorical]


l = LoadData()