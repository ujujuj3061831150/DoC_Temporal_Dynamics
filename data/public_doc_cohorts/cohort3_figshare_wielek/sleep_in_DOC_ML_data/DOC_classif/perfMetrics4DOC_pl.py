'''
Author: Tomasz Wielek wielekto@stud.sbg.ac.at
Scripts used to generate Fig4: Classification performance for (single) DOC.
Additionaly bar plot with F1 scores for each subjects and correlation with CRS score.
DATA:
=> csv file for each patient with epoch-wise: classifier predictions, eyes state and
    day/night period information
=> diag_bothSL.csv file with CRS-R diagnosis for each patient

'''
import os
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#plt.style.use('ggplot')
#change path
path = "C:\\Users\\b1016533\\Desktop\\doc_paper_10.17\\manuscript\\submit\\plos\\data\\results\\DOC_classif\\"
file_names = sorted(filter(lambda x: x.endswith('pred.csv'), os.listdir(path)))
legend = True

def perf_compute(path2preds):
    weighted_precission = []
    f1_score_store = []
    sbj_name = []; accuracy_score_store = []
    for index in range(len(file_names) ):
        print index
        df = pd.DataFrame.from_csv(path = path+file_names[index])
        df["trueEyes_CorO"] = df["trueEyes"].replace("O/C", "O") #O/C considered as O
        df = df[ df["trueEyes_CorO"] != "nv"]#remove "nv" class from eyes scoring
        df.loc[:,"predEyes_Cor"] = df["myPredict"] #first copy column then recode
        #wake as open, N1, N2, N3, REM as closed
        df.replace({"predEyes_Cor": {"wake":"O", "N1": "C","N2": "C", "N3": "C","REM":"C"}}, inplace = True)
        new_df =df[["trueEyes_CorO", "predEyes_Cor"]].copy().dropna() #drop nan
        weighted_precission.append(precision_score(new_df["trueEyes_CorO"], new_df["predEyes_Cor"], average="weighted"))#
        f1_score_store.append(f1_score(new_df["trueEyes_CorO"], new_df["predEyes_Cor"], average="weighted", labels = ["O", "C"]))#
        accuracy_score_store.append(accuracy_score(new_df["trueEyes_CorO"], new_df["predEyes_Cor"]))
        sbj_name.append(file_names[index][0:4])
    return zip(sbj_name, weighted_precission, f1_score_store, accuracy_score_store)

diagnosis = pd.read_csv( path +"diag_bothSL.csv", header = None, names = ["name", "diag","CRS"])
#in paper 11 subjects with 'high' F1 > 0.7, and 4 with low F1 < 0.3
performance = pd.DataFrame(perf_compute(path2preds = path), columns = ["name", "w_prec", "f1", "accuracy"])
#find diagnosis for performance names
performance["diag"] = "nan"
for i in  range(len(performance)):
    found_diag_pd = diagnosis["diag"][performance["name"].iloc[i] == diagnosis["name"].str.slice(0,4,1)]
    performance["diag"].iloc[i] = found_diag_pd.iloc[0]
    print i
    print performance["name"].iloc[i]
#plot perf matric as a bar plot
ax = performance["f1"].plot(kind='bar', title ="Classificaion performance for DOC",
            figsize=(15, 10), legend=True, fontsize=12)
ax.set_xlabel("Subjects", fontsize=12)
ax.set_ylabel("F1 score", fontsize=12)
ax.set_xticklabels(performance["name"])
#plt.show()

'''
Main visualisations: plot predictions against eyes open / closed
'''
#recode prediction into numeric labels
def recode_labeling(df):
    df['myPredict_numeric'] = df.replace({"myPredict": {"wake": 5, "N1": 3,"N2": 2, "N3": 1,"REM":4}}, \
                            inplace = False)['myPredict']
    return df

#colors-stages mapping and colors-day/night mapping
from collections import OrderedDict
colors = {'Open':'gold', 'Open/Closed':'red', 'Closed':'blue', 'N/A':'dimgrey', 'nan': 'white'}
#sort: O,C, O/C, N/A
c_sort = sorted(zip([2,4,0,3,1], colors.items()))
c_sort = [j for i,j in c_sort]
c_sort = OrderedDict(c_sort)
colors_dn = {'day': 'sandybrown', 'night': 'deepskyblue'}
#create custom legend based on colors to stages mapping
def built_legend():
    handle = []
    for label, color in c_sort.items():
        if label == 'nan': continue
        handle.append(matplotlib.lines.Line2D([], [], color=color, \
                linewidth = 4, alpha=1, label=label))
    return handle
handles = built_legend()
labels = [h.get_label() for h in handles]
#
plt.style.use('seaborn-white')
from matplotlib.gridspec import GridSpec

def combine_day1_day2(df):
    '''Function for combining day with day2'''
    pred_d1 = df['myPredict_numeric'][df['time_record'] == 'day1']
    pred_d2 = df['myPredict_numeric'][df['time_record'] == 'day2']
    eyes_d1 = df['trueEyes_renamed'][df['time_record'] == 'day1']
    eyes_d2 = df['trueEyes_renamed'][df['time_record'] == 'day2']
    day = pd.concat([pred_d1, pred_d2], 0).reset_index()
    eyes = pd.concat([eyes_d1, eyes_d2], 0).reset_index()
    return day, eyes

def rename_eyes(df):
    renaming = {'O': 'Open', 'O/C' : 'Open/Closed', 'C':'Closed', 'nv':'N/A'}
    for k,i in renaming.items():
        df.loc[df['trueEyes'] == k, 'trueEyes_renamed'] = i
    return df

#def plot_pred_eyes(path2preds):
for index in [4]:#range(len(file_names) ):
    df = pd.DataFrame.from_csv(path = path+file_names[index])
    df = recode_labeling(df)
    df = rename_eyes(df)
    time_classes = np.unique(df['time_record'])
    #order day1, night, day2
    #time_classes = ['day1', 'night', 'day2'] if len(time_classes) == 3 else time_classes
    time_classes = ['day', 'night']
    #if len(time_classes) > 3: continue #skip KG
    fig, ax = plt.subplots(nrows=len(time_classes), ncols=2,sharex=True,sharey=False,
            gridspec_kw={'width_ratios': [11, 1]}, figsize=(8,4))
    record_len = 0
    for i, tc in enumerate(time_classes):
        if tc in ['day']:
            datpred, dateyes = combine_day1_day2(df)
            dateyes = dateyes.fillna('nan')
            df4plot = pd.DataFrame({'datpred': datpred['myPredict_numeric'], \
                    'dateyes': dateyes['trueEyes_renamed'], 'markpos': 0.3})
        else:
            datpred = df['myPredict_numeric'][df['time_record'] == tc]
            dateyes = df['trueEyes_renamed'][df['time_record'] == tc]
            dateyes = dateyes.fillna('nan')
            df4plot = pd.DataFrame({'datpred': datpred,'dateyes': dateyes, 'markpos': 0.3})

        ax[i,0].plot(df4plot.index, df4plot['datpred'], c = 'black', linewidth = 1)
        ax[i,0].scatter(df4plot.index, df4plot['markpos'], s= 30, alpha=0.4, marker='|',\
            c=df4plot['dateyes'].apply(lambda x: colors[x]) )
        ax[i,0].set_ylim(0,6)
        ax[i,0].set_ylabel('Classif. prediction')
        tic_locs_y = range(0,6)
        ax[i,0].set_yticks(tic_locs_y); ax[i,0].set_xticks(tic_locs_y)
        ax[i,0].set_yticklabels(['Eyes', 'N3', 'N2', 'N1', 'REM', 'wake'] )
        #fill right columns (DAY, NIGHT)
        #ax[i,1].set_axis_bgcolor(colors_dn[tc])
        if tc == 'day':
            ax[i,1].text(8,0.53,tc.upper(), rotation=-90, size = 20)
        elif tc == 'night':
            ax[i,1].text(8,0.63,tc.upper(), rotation=-90, size = 20)
        ax[i,1].set_yticks([])
        ax[i,1].tick_params(labelbottom='off')
        print(len(df4plot))
        #find max length across recording, use then for x formating
        record_len = len(df4plot) if len(df4plot) > record_len else record_len
        tic_locs_x = range(0,record_len, 120) #1h = 120 epochs [epoch is 30s]
        ax[-1,0].set_xticks(tic_locs_x)
        ax[-1,0].set_xticklabels(np.array(tic_locs_x) / 120)
        ax[-1,0].set_xlabel('Time [hr]')
        ax[-1,0].xaxis.set_label_coords(1.04, -0.03)
    #fig.tight_layout()
    anchor_y = -0.1 if len(time_classes) == 2 else -0.2
    anchor_y = -.06
    if legend == True: # singel legend for several subjects
        ax[-1,0].legend(handles=handles,labels=labels,loc='upper center', \
            bbox_to_anchor=(0.5,anchor_y), ncol = len(handles), prop={'size': 15}, frameon=False)
    save_path = 'C:\\Users\\b1016533\Desktop\\'
    save_name = file_names[index].split('pred')[0]
    #fig.savefig(save_path+save_name, dpi = 600)#, bbox_extra_artists=(ax[-1,0],), \
        #bbox_inches='tight')
plt.show()

#plot_pred_eyes(path)
'''
#correalte f1 score with crs and visualisations
diagnosis = pd.read_csv( path +"diag_bothSL.csv", header = None, names = ["name", "diag","CRS"])
performance = pd.DataFrame(perf_compute(path2preds = path), columns = ["name", "w_prec", "f1", "accuracy"])
diagnosis['name_abrev'] = [diagnosis['name'][i][0:2] for i in range(len(diagnosis))]
diagnosis['f1'] = performance['f1']
from scipy.stats import pearsonr, spearmanr
diagnosis_clean = diagnosis[diagnosis['f1'].notnull()]
spearmanr(diagnosis_clean['f1'], diagnosis_clean['CRS'].rank())
diagnosis['f1'].corr(diagnosis['CRS'])
diagnosis_clean['diag_str'] = diagnosis_clean['diag'].replace({0: 'UWS', 1: 'MCS'})
import seaborn as sns
sns.set_style('ticks')

g = sns.lmplot('CRS', 'f1', hue="diag_str", data=diagnosis_clean, fit_reg=False, \
     markers=["o", "x"], palette="Set1" )
sns.regplot('CRS', 'f1', data=diagnosis_clean, scatter=False, ax=g.axes[0, 0])


plt.xlabel('CRS_R total')
plt.ylabel('F1 score')
plt.show()
'''
