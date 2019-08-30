import scipy.io
import TopoPB
import numpy as np
import FeaturesExtraction
import pandas as pd


    #load .mat
mat_LA0 = scipy.io.loadmat('data_LA0.mat')
mat_LA1 = scipy.io.loadmat('data_LA1.mat')
mat_LA2 = scipy.io.loadmat('data_LA2.mat')
mat_LA3 = scipy.io.loadmat('data_LA3.mat')
mat_LA4 = scipy.io.loadmat('data_LA4.mat')
mat_LA5 = scipy.io.loadmat('data_LA5.mat')
mat_LA6 = scipy.io.loadmat('data_LA6.mat')

mat_LAw0 = scipy.io.loadmat('data_LAw0.mat')
mat_LAw1 = scipy.io.loadmat('data_LAw1.mat')
mat_LAw2 = scipy.io.loadmat('data_LAw2.mat')
mat_LAw3 = scipy.io.loadmat('data_LAw3.mat')
mat_LAw4 = scipy.io.loadmat('data_LAw4.mat')
mat_LAw5 = scipy.io.loadmat('data_LAw5.mat')
mat_LAw6 = scipy.io.loadmat('data_LAw6.mat')

    #get data --> array
data_LA00= mat_LA0['S2_unconscious_RawData']
data_LA0=np.delete(data_LA00,[39,44],0)
data_LA11= mat_LA1['S3_unconscious_RawData']
data_LA1=np.delete(data_LA11,[39,44],0)
data_LA2= mat_LA2['S4_unconscious_RawData']
data_LA33= mat_LA3['S5_unconscious_RawData']
data_LA3=np.delete(data_LA33,[39,44],0)
data_LA44= mat_LA4['S6_unconscious_RawData']
data_LA4=np.delete(data_LA44,[39,44],0)
data_LA55= mat_LA5['S8_unconscious_RawData']
data_LA5=np.delete(data_LA55,[39,44],0)
data_LA66= mat_LA6['S10_unconscious_RawData']
data_LA6=np.delete(data_LA66,[39,44],0)

data_LAw00= mat_LAw0['S2_conscious_RawData']
data_LAw0=np.delete(data_LAw00,[39,44],0)
data_LAw11= mat_LAw1['S3_conscious_RawData']
data_LAw1=np.delete(data_LAw11,[39,44],0)
data_LAw2= mat_LAw2['S4_conscious_RawData']
data_LAw33= mat_LAw3['S5_conscious_RawData']
data_LAw3=np.delete(data_LAw33,[39,44],0)
data_LAw44= mat_LAw4['S6_conscious_RawData']
data_LAw4=np.delete(data_LAw44,[39,44],0)
data_LAw55= mat_LAw5['S8_conscious_RawData']
data_LAw5=np.delete(data_LAw55,[39,44],0)
data_LAw66= mat_LAw6['S10_conscious_RawData']
data_LAw6=np.delete(data_LAw66,[39,44],0)

    #append data --> list
list_LA=[]
list_LA.append(data_LA0)
list_LA.append(data_LA1)
list_LA.append(data_LA2)
list_LA.append(data_LA3)
list_LA.append(data_LA4)
list_LA.append(data_LA5)
list_LA.append(data_LA6)

list_LAw=[]
list_LAw.append(data_LAw0)
list_LAw.append(data_LAw1)
list_LAw.append(data_LAw2)
list_LAw.append(data_LAw3)
list_LAw.append(data_LAw4)
list_LAw.append(data_LAw5)
list_LAw.append(data_LAw6)

            #compute avPSD 

powerBlist_LA=[pd.DataFrame(columns=['delta','theta','alpha','sigma','beta','low gamma'], index=range(len(list_LA[0]))) for x in range(7)]
powerBlist_LAw=[pd.DataFrame(columns=['delta','theta','alpha','sigma','beta','low gamma'], index=range(len(list_LAw[0]))) for x in range(7)]
            #filled with zeros

ave_LA=np.zeros((len(list_LA[0]),6))
divider_LA=np.zeros((len(list_LA[0]),6))

ave_LAw=np.zeros((len(list_LAw[0]),6))
divider_LAw=np.zeros((len(list_LAw[0]),6))

for i in range(7):
    data_LA=list_LA[i]
    data_LAw=list_LAw[i]

    for j in range(len(data_LA)):
        data_row_LA=data_LA[j]    #extract data for each channel
        data_row_LAw=data_LAw[j]

        PSD_LA=FeaturesExtraction.computePSD(data_row_LA,300)     #compute power spectral denstity for each channel
        PSD_LAw=FeaturesExtraction.computePSD(data_row_LAw,300) 

        f_LA=PSD_LA[0]                      #assign frequency and power spectral density to 2 different vectors
        p_LA=PSD_LA[1]

        f_LAw=PSD_LAw[0]
        p_LAw=PSD_LAw[1]

        powerBrow_LA=FeaturesExtraction.computePowerBands(f_LA, p_LA)   #compute absolute power bands for each channel
        powerBlist_LA[i].iloc[j]=powerBrow_LA                    #core.frame.DataFrame containing absolute power bands for each channel

        powerBrow_LAw=FeaturesExtraction.computePowerBands(f_LAw, p_LAw)   #compute absolute power bands for each channel
        powerBlist_LAw[i].iloc[j]=powerBrow_LAw 
    
    for k in range(len(data_LA)):            #each channel
        for y in range(6):           #each band
            if not np.isnan(powerBlist_LA[i].iloc[k,y]):       
                ave_LA[k,y] += powerBlist_LA[i].iloc[k,y]     #sum dataframes element by element
                divider_LA[k,y] += 1     

    for k in range(len(data_LAw)):            #each channel
        for y in range(6):           #each band
            if not np.isnan(powerBlist_LAw[i].iloc[k,y]):       
                ave_LAw[k,y] += powerBlist_LAw[i].iloc[k,y]     #sum dataframes element by element
                divider_LAw[k,y] += 1 

ave_LA/=divider_LA                 #average power band per electrode for all subjects
ave_LAw/=divider_LAw

   
av_delta_LA=ave_LA[:,0]           #extract average for each band
av_theta_LA=ave_LA[:,1]
av_alpha_LA=ave_LA[:,2]
av_beta_LA=ave_LA[:,4]
av_lowgamma_LA=ave_LA[:,5]

list_av_LA=[]
list_av_LA.append(av_delta_LA)
list_av_LA.append(av_theta_LA)
list_av_LA.append(av_alpha_LA)
list_av_LA.append(av_beta_LA)
list_av_LA.append(av_lowgamma_LA)


av_delta_LAw=ave_LAw[:,0]          
av_theta_LAw=ave_LAw[:,1]
av_alpha_LAw=ave_LAw[:,2]
av_beta_LAw=ave_LAw[:,4]
av_lowgamma_LAw=ave_LAw[:,5]

list_av_LAw=[]
list_av_LAw.append(av_delta_LAw)
list_av_LAw.append(av_theta_LAw)
list_av_LAw.append(av_alpha_LAw)
list_av_LAw.append(av_beta_LAw)
list_av_LAw.append(av_lowgamma_LAw)



'''
#script 1 subjects


def Topo_1_sub(file_name, data_name):

    mat= scipy.io.loadmat(file_name)
    data_LA00= mat[data_name]

    if file_name == 'data_LA2.mat' or file_name == 'data_LAw2.mat':
        data_LA0=data_LA00
    else: 
        data_LA0=np.delete(data_LA00,[39,44],0)

    powerBlist_LA=pd.DataFrame(columns=['delta','theta','alpha','sigma','beta','low gamma'], index=range(len(data_LA0)))

    for j in range(len(data_LA0)):

        data_row_LA=data_LA0[j]    #extract data for each channel
        PSD_LA=FeaturesExtraction.computePSD(data_row_LA,300)     #compute power spectral denstity for each channel

        f_LA=PSD_LA[0]                      #assign frequency and power spectral density to 2 different vectors
        p_LA=PSD_LA[1]

        powerBrow_LA=FeaturesExtraction.computePowerBands(f_LA, p_LA)   #compute absolute power bands for each channel
        powerBlist_LA.iloc[j]=powerBrow_LA                    #core.frame.DataFrame containing absolute power bands for each channel


    av_delta_LA=powerBlist_LA['delta'].to_numpy()           #extract average for each band
    av_theta_LA=powerBlist_LA['theta'].to_numpy() 
    av_alpha_LA=powerBlist_LA['alpha'].to_numpy() 
    av_beta_LA=powerBlist_LA['beta'].to_numpy() 
    av_lowgamma_LA=powerBlist_LA['low gamma'].to_numpy() 

    list_av_LA=[]
    list_av_LA.append(av_delta_LA)
    list_av_LA.append(av_theta_LA)
    list_av_LA.append(av_alpha_LA)
    list_av_LA.append(av_beta_LA)
    list_av_LA.append(av_lowgamma_LA)

    return list_av_LA










