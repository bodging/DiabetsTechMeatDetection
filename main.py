import csv
import glob, os
import pandas as pd
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
#PATH_TO_DMMSR_DATASET=".\DMMSR_Dataset\DMMSR_Dataset"
PROJECT_HOMEFOLDER=os.getcwd()
PATH_TO_DMMSR_DATASET=PROJECT_HOMEFOLDER+".\DMMSR_Dataset\DMMSR_Dataset"

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #merge all excel into a dataframe, if it doesnt exist yet
    isFile = os.path.isfile(PATH_TO_DMMSR_DATASET+"/merged.csv")
    if not isFile:
        df_tot=pd.DataFrame()
        os.chdir(PATH_TO_DMMSR_DATASET+"/adolecents")
        for file in glob.glob("*.csv"):
            df = pd.read_csv(file)
            df_tot=df_tot.append(df,ignore_index=True)

        os.chdir(PATH_TO_DMMSR_DATASET+"/adults")
        for file in glob.glob("*.csv"):
            df = pd.read_csv(file)
            df_tot=df_tot.append(df,ignore_index=True)

        os.chdir(PATH_TO_DMMSR_DATASET + "/children")
        for file in glob.glob("*.csv"):
            df = pd.read_csv(file)
            df_tot=df_tot.append(df,ignore_index=True)
        df_tot.to_csv(PATH_TO_DMMSR_DATASET+'/merged.csv',index=False)
        print("merged all files together")
    else:
        df_tot=pd.read_csv(PATH_TO_DMMSR_DATASET+'/merged.csv')

    #filter out #average
    df_tot = df_tot[
        (df_tot['name'] != 'adolescent#average') & (df_tot['name'] != 'adult#average') & (df_tot['name'] != 'child#average')]
    pass
    #binar<y labeling

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
