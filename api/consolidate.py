from importlib.metadata import files
import pandas as pd
import os

PATH_DATA = 'data'
PATH_DATA_USERS = 'data_users'
PATH_DATA_CONSOLIDATED = f'{PATH_DATA}_consolidated'

if not os.path.exists(PATH_DATA_CONSOLIDATED):
    os.mkdir(PATH_DATA_CONSOLIDATED)

def save_consolidated(path, start='dataTW_GP_', end='_start'):

    files_data = os.listdir(path)

    roots = list(set([x.split(start,1)[1].split(end,1)[0] for x in files_data]))

    for root in roots:
        root_files = [x for x in files_data if root in x]
        df_collector = pd.DataFrame()
        for file in root_files:
            df_i = pd.read_csv(os.path.join(path, file))
            df_collector = df_collector.append(df_i).reset_index(drop=True)

        df_collector = df_collector.drop_duplicates()

        file_consolidated = f'{start}_{root}.csv'
        df_collector.to_csv(os.path.join(PATH_DATA_CONSOLIDATED, file_consolidated), index=False)

save_consolidated(PATH_DATA, start='dataTW_GP_', end='_start')
save_consolidated(PATH_DATA_USERS, start='usersTW_GP_', end='_start')