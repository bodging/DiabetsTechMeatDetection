import glob, os

import numpy as np
import pandas as pd

PROJECT_HOMEFOLDER=os.getcwd()
PATH_TO_DMMSR_DATASET=PROJECT_HOMEFOLDER+".\DMMSR_Dataset\DMMSR_Dataset"

def loadcsv():

    # merge all excel into a dataframe, if it doesnt exist yet
    isFile = os.path.isfile(PATH_TO_DMMSR_DATASET + "/merged.csv")
    if not isFile:
        df_tot = pd.DataFrame()
        os.chdir(PATH_TO_DMMSR_DATASET + "/adolecents")
        for file in glob.glob("*.csv"):
            df = pd.read_csv(file)
            df_tot = df_tot.append(df, ignore_index=True)

        os.chdir(PATH_TO_DMMSR_DATASET + "/adults")
        for file in glob.glob("*.csv"):
            df = pd.read_csv(file)
            df_tot = df_tot.append(df, ignore_index=True)

        os.chdir(PATH_TO_DMMSR_DATASET + "/children")
        for file in glob.glob("*.csv"):
            df = pd.read_csv(file)
            df_tot = df_tot.append(df, ignore_index=True)
        df_tot.to_csv(PATH_TO_DMMSR_DATASET + '/merged.csv', index=False)
        print("merged all files together")
    else:
        df_tot = pd.read_csv(PATH_TO_DMMSR_DATASET + '/merged.csv')



    return df_tot

def writetimewindows(df_tot,timewindow):

    # split into time windows, if not done already
    isFile = os.path.isfile(PATH_TO_DMMSR_DATASET + "/timewindowed.csv")
    if not isFile:
        df_tot['timewindow_nr']=0
        df_tot_grouped = df_tot.groupby('name')
        df_result=pd.DataFrame()
        for group_name, df_group in df_tot_grouped:
            df_group['timewindow_nr'] =np.arange(len(df_group)) // timewindow
            # last timewindow might be unequal it total/timewindow =/= nat. number!
            if len(df_group)%timewindow !=0:
                df_group=df_group.drop([df_group['timewindow_nr'].idxmax()])
            df_result = pd.concat([df_result, df_group], axis=0)
        df_tot = df_result
        df_tot.to_csv(PATH_TO_DMMSR_DATASET + '/timewindowed.csv', index=False)
        print("timewindowed merged file")
    else:
        df_tot = pd.read_csv(PATH_TO_DMMSR_DATASET + '/timewindowed.csv')
    return df_tot



if __name__ == '__main__':
    df_tot=loadcsv()
    df_tot.drop(df_tot.columns.difference(['sqInsulinNormalBolus','minutesPastSimStart','cgm','name']), 1, inplace=True)
    # filter out #average
    df_tot = df_tot[
        (df_tot['name'] != 'adolescent#average') & (df_tot['name'] != 'adult#average') & (
                df_tot['name'] != 'child#average')]
    timewindow=15
    #each file has rows 1 to 44640
    df_tot=writetimewindows(df_tot,timewindow)
    #df_tot=binarize()
    # df_tot=splitintosets()

    pass


