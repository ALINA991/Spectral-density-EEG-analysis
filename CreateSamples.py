import mne
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import FeaturesExtraction
import Examine


def rawsamples(file_name, s_len, nb_subjects):

   

    '''
    !!! bad channels have to be removed beforehand!!!

    data : has to be string with file name without .edf & named after 'data'n'.edf' with n = file number corresponding to subject number
    s_len : desired sample lenth (sec)
    aq_time : duration of aquisition (sec)
    nb_subjects : nb of files to load 

    returns list of nb_electrodes*nb_subject items --> each containing nb_samples rows 

    '''

    
    sample_sets=[] 
    samples_idx=[] 

    subjects=range(nb_subjects)

    files=[]                #create a list of file names as found in work directory  --> change as needed      
    for i in subjects:
        files.append(file_name +str(i)+ '.edf')

    for i in subjects:                  #iterate over each subject file
        raw=mne.io.read_raw_edf(files[i])       #importing raw data
        data=raw.get_data()                     #extracting data

        #data=np.loadtxt(files[i])                  #if data .txt  !!!! sfreq has to be specified !!!

        sfreq=raw.info['sfreq']                #extracting / calculating relevant info
        points_per_sample = sfreq * s_len
        points_tot = np.size(data,1)
        nb_samples_1_el = points_tot / points_per_sample


        if int(points_tot/ points_per_sample) != points_tot/ points_per_sample:         #if impossible to create equal samples
            diff = points_tot/ points_per_sample - int(points_tot/ points_per_sample)
            points_to_remove = diff * points_per_sample
            end = int(points_tot-points_to_remove)                  #remove last points so that (points_tot/ points_per_sample) has no digits
            data_del = data[:, : end]                       
            points_tot -= points_to_remove
        else:
            data_del=data          #if nb of points ok

        
        points_per_sample = int(sfreq * s_len)                  #transforming float to int for later use
        nb_samples_1_el = int(points_tot / points_per_sample)
        points_per_el= int(nb_samples_1_el * points_per_sample)

                  #empty
        samples_1_el=np.zeros((nb_samples_1_el,points_per_sample))    #create an array of shape(nb_samples_1_el,points_per_sample)  
                                                                        #for each row(=electrode) of data (1 subj)

        n=0
        for j in range(len(data_del)):  
        
            for k in range(nb_samples_1_el): 
                
                while n < points_per_el: 
                    samples_1_el[k,:]=data_del[j,n:n+points_per_sample]   #fill sample array with one row of data array 
                    n+=points_per_sample 
                n=0       
                                                      
            sample_sets.append(samples_1_el)      #list of samples for all subject (1 sample array = 1 electrode of 1 subject)
            samples_idx.append(i)                   #create list of labels (subject number corresponding to sample array)
            samples_1_el=np.zeros((nb_samples_1_el,points_per_sample))

    return sample_sets, samples_idx


def PSDsamples(file_name, s_len, sfreq, nb_subjects, ext):

    '''
    data : has to be string with file name without .txt & named after 'data'n'.txt' with n = file number corresponding to subject number
    s_len : desired sample lenth (sec)
    aq_time : duration of aquisition (sec)
    nb_subjects : nb of files to load +1 

    returns one 3D array per power band, containing mean spectral density per sample per electrode per subject 
    order: delta / theta / alpha / sigma / beta / low gamma

    '''

        #empty
    samples_PSD_delta=[]
    samples_PSD_theta=[]
    samples_PSD_alpha=[]
    #samples_PSD_sigma=[]
    samples_PSD_beta=[]
    samples_PSD_lowgamma=[]

    files=[]                         #create a list of file names as found in work directory  --> change as needed     
    subjects=range(nb_subjects) 
    
    for i in subjects:
        if ext == '.txt':
            files.append(file_name +str(i)+ '.txt')
        elif ext == '.npy':
            files.append(file_name +str(i)+'.npy')


    for i in subjects:     
        
        if ext == '.txt':                 #load data from text file
            data=np.loadtxt(files[i])
        elif ext == '.npy':
            data=np.load(files[i])

        #raw=mne.io.read_raw_edf(files[i])       #importing raw data if data .edf
        #data=raw.get_data()          

                                 #extracting / calculating relevant info
        points_per_sample = sfreq * s_len
        points_tot = np.size(data,1)
        nb_samples_1_el = points_tot / points_per_sample
        nb_electrodes=len(data)
        


        if int(points_tot/ points_per_sample) != points_tot/ points_per_sample:         #if impossible to create equal samples
            diff = points_tot/ points_per_sample - int(points_tot/ points_per_sample)
            points_to_remove = diff * points_per_sample
            endd = int(points_tot-points_to_remove)                  #remove last points so that (points_tot/ points_per_sample) has no digits
            data_del = data[:, : endd]                       
            points_tot -= points_to_remove
        else:
            data_del=data                               #if nb of points ok

        points_per_sample = int(sfreq * s_len)                  #transforming float to int for later use
        nb_samples_1_el = int(points_tot / points_per_sample)
        nb_electrodes=int(nb_electrodes)
        points_tot=int(points_tot)

        samples_PSD_delta1=np.zeros((nb_samples_1_el,nb_electrodes))        #empty for 1 subject 
        samples_PSD_theta1=np.zeros((nb_samples_1_el,nb_electrodes)) 
        samples_PSD_alpha1=np.zeros((nb_samples_1_el,nb_electrodes)) 
        #samples_PSD_sigma1=np.zeros((nb_samples_1_el,nb_electrodes))
        samples_PSD_beta1=np.zeros((nb_samples_1_el,nb_electrodes)) 
        samples_PSD_lowgamma1=np.zeros((nb_samples_1_el,nb_electrodes)) 

        n=0

        for j in range(nb_electrodes):  

            for k, n in zip(range(nb_samples_1_el),range(0,points_tot,points_per_sample)):

                sample=data_del[j,n:n+points_per_sample]                #consider n points only (=sample)
                f,p=FeaturesExtraction.computePSD(sample, s_len)              #extract power bands
                powerBrow=FeaturesExtraction.computePowerBands(f, p)  #delta / theta / alpha / sigma / beta / low gamma
               
                samples_PSD_delta1[k][j]=powerBrow[0]             #assign mean power band of n points to correspondig list 
                samples_PSD_theta1[k][j]=powerBrow[1]                 #with i: subject, j:electrode, k:sample
                samples_PSD_alpha1[k][j]=powerBrow[2]
                #samples_PSD_sigma1[k][j]=powerBrow[3]
                samples_PSD_beta1[k][j]=powerBrow[4]
                samples_PSD_lowgamma1[k][j]=powerBrow[5]

                   
        samples_PSD_delta.append(samples_PSD_delta1)
        samples_PSD_theta.append(samples_PSD_theta1)
        samples_PSD_alpha.append(samples_PSD_alpha1)
        #samples_PSD_sigma.append(samples_PSD_sigma1)
        samples_PSD_beta.append(samples_PSD_beta1)
        samples_PSD_lowgamma.append(samples_PSD_lowgamma1)


    return samples_PSD_delta, samples_PSD_theta, samples_PSD_alpha, samples_PSD_beta, samples_PSD_lowgamma


def avPower(samples_PSD_delta, samples_PSD_theta, samples_PSD_alpha, samples_PSD_beta, samples_PSD_lowgamma):

    av_delta=np.zeros(samples_PSD_delta[0].shape)
    av_theta=np.zeros(samples_PSD_theta[0].shape)
    av_alpha=np.zeros(samples_PSD_alpha[0].shape)
    av_beta=np.zeros(samples_PSD_beta[0].shape)
    av_lowgamma=np.zeros(samples_PSD_lowgamma[0].shape)

    divider= np.full((samples_PSD_delta[0].shape),len(samples_PSD_delta))

    for i in range(len(samples_PSD_delta)):
        for j in range(samples_PSD_delta[0].shape[0]):
            for k in range(samples_PSD_delta[0].shape[1]):
                av_delta[j,k] += samples_PSD_delta[i][j][k]

        for j in range(samples_PSD_theta[0].shape[0]):
            for k in range(samples_PSD_theta[0].shape[1]):
                av_theta[j,k] += samples_PSD_theta[i][j][k]

        for j in range(samples_PSD_alpha[0].shape[0]):
            for k in range(samples_PSD_theta[0].shape[1]):
                av_alpha[j,k] += samples_PSD_alpha[i][j][k]

        for j in range(samples_PSD_beta[0].shape[0]):
            for k in range(samples_PSD_beta[0].shape[1]):
                av_beta[j,k] += samples_PSD_beta[i][j][k]

        for j in range(samples_PSD_lowgamma[0].shape[0]):
            for k in range(samples_PSD_lowgamma[0].shape[1]):
                av_lowgamma[j,k] += samples_PSD_lowgamma[i][j][k]

    liste=Examine.appendd(av_delta, av_theta, av_alpha, av_beta, av_lowgamma)

    for i in range(len(liste)): 
        liste[i] /= divider

    avD=np.empty((liste[0].shape[1],1))
    avT=np.empty((liste[1].shape[1],1))
    avA=np.empty((liste[2].shape[1],1))
    avB=np.empty((liste[3].shape[1],1))
    avLG=np.empty((liste[4].shape[1],1))

    
    for k in range(liste[0].shape[1]):
        for j in range(liste[0].shape[0]):
            avD[k] += liste[0][j,k]
            avT[k] += liste[1][j,k]
            avA[k] += liste[2][j,k]
            avB[k] += liste[3][j,k]
            avLG[k] += liste[4][j,k]

    divider2=np.full((avD.shape), liste[0].shape[1])

    listee= Examine.appendd(avD, avT, avA, avB, avLG)

    for i in range(len(listee)): 
        listee[i] /= divider2

    return listee



def printch_nb(file_name, nb_subjects):

    subjects=range(nb_subjects)

    files=[]                #create a list of file names as found in work directory  --> change as needed      
    for i in subjects:
        files.append(file_name +str(i)+ '.edf')

    nb_ch=np.zeros((nb_subjects,2))

    for i in subjects:   
        raw=mne.io.read_raw_edf(files[i])       #importing raw data
        data=raw.get_data()

        nb_ch[i][0]=i    
        nb_ch[i][1]=len(raw.info['chs'])                        #iterate over each subject file

    print(nb_ch)

def printch_names(file_name, nb_subjects):

    subjects=range(nb_subjects)

    files=[]                #create a list of file names as found in work directory  --> change as needed      
    for i in subjects:
        files.append(file_name +str(i)+ '.edf')

    names_ch={}

    for i in subjects:   
        raw=mne.io.read_raw_edf(files[i])       #importing raw data
        names_ch[i]=raw.info['ch_names']                #iterate over each subject file

    good_ch_names=[]
    for a,b,c,d,e,f,g,h,i in zip(names_ch[0],names_ch[1],names_ch[2],names_ch[3],names_ch[4],names_ch[5],names_ch[6],names_ch[7],names_ch[8]):
        if a in {b,c,d,e,f,g,h,i}:
            good_ch_names.append(a)
        
    
        #return [[x for x in a if x not in {b,c,d,e,f,g,h,i}]]
        

def delete_rows(file_name, nb_subjects):

    subjects=range(nb_subjects)

        #channel names --> to remove
    ch_to_remove=['E3','E8','E10', 'E13', 'E23', 'E36', 'E45', 'E46', 'E47','E56', 'E57', 'E63', 'E70', 'E71', 'E100', 'E102', 'E107', 'E115','STI 014']
    ch_2_remove=['C3','T7','LM','01','RM']

    idx_to_remove=[]

    files=[]                #create a list of file names as found in work directory      
    for i in subjects:
        files.append(file_name +str(i)+ '.edf')

    for i in subjects:   
        raw=mne.io.read_raw_edf(files[i])       #importing raw data
        data=raw.get_data()

        for k in range(len(ch_to_remove)):              #get indices of channels from 1st list to remove
            if ch_to_remove[k] in raw.info['ch_names']:
                idx_to_remove.append(raw.info['ch_names'].index(ch_to_remove[k]))
        for k in range(len(ch_2_remove)):               #get indices of channels from 2nd list to remove
            print(k)
            if ch_2_remove[k] in raw.info['ch_names']:
                idx_to_remove.append(raw.info['ch_names'].index(ch_2_remove[k]))

        
        clean_data=np.delete(data,idx_to_remove,0)              #delete channels
        np.savetxt('cl_data_DAw'+str(i)+'.edf', clean_data)      #save clean data



def find_missingch(file_name1, file_name2, nb_subjects):

    subjects=range(nb_subjects)
    ch_2_remove=[]

    for i in subjects:                                   #iterate over all subject channel names (1 condition ex: anesthesia)
        raw0=mne.io.read_raw_edf(file_name1 +str(i) +'.edf')
        channels0= raw0.info['ch_names']                     #get ch names

        for i in subjects:                                   #compare with every other subject
            raw1=mne.io.read_raw_edf(file_name1 +str(i) +'.edf')

            for k in range(len(channels0)):
                if (not channels0[k] in raw1.info['ch_names']) and (not channels0[k] in ch_2_remove):
                                                                         #if ch not present in one of the subjects & not already in the list 
                    ch_2_remove.append(channels0[k])

    for i in subjects:                                   #iterate over all subject channel names (other condition ex: wakefulness)
        raw0=mne.io.read_raw_edf(file_name2 +str(i) +'.edf')
        channels0= raw0.info['ch_names']                     #get ch names

        for i in subjects:                                   #compare with every other subject
            raw1=mne.io.read_raw_edf(file_name2 +str(i) +'.edf')

            for k in range(len(channels0)):
                if (not channels0[k] in raw1.info['ch_names']) and (not channels0[k] in ch_2_remove):
                                                                         #if ch not present in one of the subjects & not already in the list 
                    ch_2_remove.append(channels0[k])

    ch_2_remove.append('STI 014')         #remove stim channel

    return ch_2_remove

def delete_missingch(file_name, ch_to_remove, nb_subjects):

    ''' change file name in np.savetxt as needed
        file_name: as found in work directory without subject n* nor .edf 
            ex= 'data2.edf' ---> file_name= 'data'
    '''

    subjects=range(nb_subjects)
   

    for i in subjects: 

        raw=mne.io.read_raw_edf(file_name +str(i) +'.edf')
        channels=raw.info['ch_names']
        data=raw.get_data()

        idx2rem=[]

        points_to_remove = 500    #remove 500 last points (all identical)
        points_tot = np.size(data,1)
        endd= int(points_tot-points_to_remove)

        for k in range(len(ch_to_remove)):
            if ch_to_remove[k] in channels:
                idx2rem.append(channels.index(ch_to_remove[k]))

        idx2rem.extend((118,120,126))

        clean_data=np.delete(data,idx2rem,0) 

        endd = int(minim-points_to_remove)
        clean_data_del=clean_data[:,:endd]     
        np.savetxt('cl_data_DAw'+str(i)+'.txt', clean_data_del)

        
def delete_missingch_1file(file_name, ch_to_remove, nb_subjects):

        raw=mne.io.read_raw_edf(file_name)
        channels=raw.info['ch_names']
        data=raw.get_data()

        idx2rem=[]
    
        points_to_remove = 500
        points_tot = np.size(data,1)
        endd= int(points_tot-points_to_remove)

        for k in range(len(ch_to_remove)):
            if ch_to_remove[k] in channels:
                idx2rem.append(channels.index(ch_to_remove[k]))

        idx2rem.extend((118,120,126))

        clean_data=np.delete(data,idx2rem,0) 

        endd = int(points_tot-points_to_remove)
        clean_data_del=clean_data[:,:endd]    
        
        np.savetxt('cl_test_DA.txt', clean_data_del)


def del_missingch_LA(data):
    
    idx2rem=[39,44]
    clean_data=[]

    for i in range(2):
        clean=np.delete(data[i],idx2rem,0)
        clean_data.append(clean)
    
    clean_data.append(data[2])

    for i in range(3,7):
        clean=np.delete(data[i],idx2rem,0)
        clean_data.append(clean)

    return clean_data

def diff_per_subject(liste1, liste2):

    liste_diff=[]
    array_diff=np.empty(liste1[0].shape)
    row_diff=[]
    new_list1=[]
    new_list2=[]

    if liste1.shape[0] > liste2.shape[0]:     #if different numbers of subjects
        lenght_list= liste2.shape[0]
        for i in range(lenght_list):
            new_list1.append(liste1[i])
            liste1=new_list1
            new_list1=[]
    elif liste1.shape[0] < liste2.shape[0]:
        lenght_list= liste1.shape[0]
        for i in range(lenght_list):
            new_list2.append(liste2[i])
            liste2=new_list2
            new_list2=[]

    for i in range(len(liste1)):
        new_list1.append(liste1[i].T)
        new_list2.append(liste2[i].T)

    if new_list1[0].shape[1] > new_list2[0].shape[1]:    #get arrays of same shape (delete samples of bigger array)
        shape_row= new_list2[0].shape[1]
        for i in range(len(new_list1)):
            for j in range(new_list1[0].shape[0]):
                new_list1[i][j]= new_list1[i][:shape_row]
    elif new_list1[0].shape[0] < new_list2[0].shape[0]:
        shape_row= new_list1[0].shape[0]
        for i in range(len(new_list2)):
            new_list2[i]= new_list2[i][:shape_row]

    for i in range(len(liste1)):
        for j in range(liste1[i].shape[0]):
            row1=liste1[i][j]
            row2=liste2[i][j]
            for k in range(len(row1)):               #get difference for each value of array
                row_diff.append((row1[k]-row2[k]))
            array_diff[j] = row_diff
            row_diff=[]
        liste_diff.append(array_diff)
        array_diff=[]

    return liste_diff

def diff_per_subject1(liste1, liste2):

    for i in range(len(liste1)):
        for j in range(liste1[i].shape[0]):
            row1=liste1[i][j]
            row2=liste2[i][j]
            for k in range(len(row1)):               #get difference for each value of array
                row_diff.append((row1[k]-row2[k]))
            array_diff[j] = row_diff
            row_diff=[]
        liste_diff.append(array_diff)
        array_diff=[]

    return liste_diff

def diff_per_subject2(list1, list2):

    list1T=[]
    list2T=[]
    list1_mean=[]
    list2_mean=[]

    list_diff=[]
    list_diffT=[]

    array_diff=np.empty((len(list1), list1[0].shape[1]))

    for i in range(len(list1)):
        list1T.append(list1[i].T)
        list2T.append(list2[i].T)

        array_mean1=np.empty((list1T[0].shape[0]))
        array_mean2=np.empty((list1T[0].shape[0]))

        for j in range(list1T[0].shape[0]):
            array_mean1[j]= np.mean(list1T[i][j], axis=0)
            array_mean2[j]= np.mean(list2T[i][j], axis=0)

        list1_mean.append(array_mean1)
        list2_mean.append(array_mean2)

        list_diff.append(list1_mean[i]-list2_mean[i])
        list_diffT.append(list_diff[i].T)

        array_diff[i]=np.hstack(list_diffT[i])

    return array_diff

def rel_diff(list1,list2):

    rel_diff=[]
    for i in range(len(list1)):
        rel_diff.append((list1[i]-list2[i])/list2[i])

    return rel_diff










        

