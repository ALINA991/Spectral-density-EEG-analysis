import mne
import numpy as np
import FeaturesExtraction 
from mne.event import define_target_events
from mne.channels import make_1020_channel_selections
import pandas as pd
import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
import CreateSamples
from scipy import spatial
import scipy
import TopoPB 


'''author of all function found in this file (except if specified otherwise): Alina Weinberger'''


def avPowerband(file_name, nb_subjects, ext):

    #dictionnary to acces data files (1-8) corresponding to subject n*:
    #subjects={0:3, 1:5, 2:6, 3:7, 4:10, 5:11, 6:12, 7:15, 8:17}

    subjects= range(nb_subjects)
    files=[]
    #bands={0:'delta',1:'theta',2:'alpha',3:'sigma',4:'beta',5:'low gamma'}

                #create a list of file names as found in work directory --> change as needed 
    if ext == '.txt':
        for i in subjects:
            files.append(file_name +str(i)+ '.txt')
            test = np.loadtxt(files[0])   # to have right size of ave / divider array 
    elif ext == '.npy':
        for i in subjects:
            files.append(file_name + str(i) + '.npy')
            test = np.load(files[0])    # to have right size of ave / divider array 

    #empty (NaN)
    powerBlist=[pd.DataFrame(columns=['delta','theta','alpha','sigma','beta','low gamma'], index=range(len(test))) for x in subjects]


    #filled with zeros
    ave=np.zeros((len(test),6))
    divider=np.zeros((len(test),6))



    for i in subjects:
    
        #raw=mne.io.read_raw_edf(files[i])       #importing raw data 
        #events=mne.events_from_annotations(raw)      #extracting events 

        if ext == '.txt':
            data= np.loadtxt(files[i])
        elif ext == '.npy':
            data= np.load(files[i])

        for j in range(len(data)):
        
            data_row=data[j]    #extract data for each channel
           
            PSD=FeaturesExtraction.computePSD(data_row,300)     #compute power spectral denstity for each channel 
            f=PSD[0]                      #assign frequency and power spectral density to 2 different vectors
            p=PSD[1]

            powerBrow=FeaturesExtraction.computePowerBands(f, p)   #compute absolute power bands for each channel
            powerBlist[i].iloc[j]=powerBrow                    #core.frame.DataFrame containing absolute power bands for each channel
    
    
        for k in range(len(data)):            #each channel
            for y in range(6):           #each band
                if not np.isnan(powerBlist[i].iloc[k,y]):       
                    ave[k,y] += powerBlist[i].iloc[k,y]     #sum dataframes element by element
                    divider[k,y] += 1     

    ave/=divider                 #average power band per electrode for all subjects
    ave= np.delete(ave, [4,33,40,43,45], axis=0)

    #np.save('average_PB.npy',ave)        #save file 

    #ave=np.delete(ave, (118,120,126),axis=0)        #delete electrode with too high values
   
    av_delta=ave[:,0]           #extract average for each band
    av_theta=ave[:,1]
    av_alpha=ave[:,2]
    #av_sigma=ave[:,3]
    av_beta=ave[:,4]
    av_lowgamma=ave[:,5]

    #dictio={'delta':av_delta, 'theta':av_theta, 'alpha': av_alpha, 'beta':av_beta, 'lowgamma':av_lowgamma}

    return av_delta, av_theta, av_alpha,  av_beta, av_lowgamma


def avPowerband_sub(data):


    #empty (NaN)
    powerBlist=pd.DataFrame(columns=['delta','theta','alpha','sigma','beta','low gamma'], index=range(len(data)))


    #filled with zeros
    ave=np.zeros((len(data),6))
    divider=np.zeros((len(data),6))


    for j in range(len(data)):
        
        data_row=data[j]    #extract data for each channel
           
        PSD=FeaturesExtraction.computePSD(data_row,1180)     #compute power spectral denstity for each channel 
        f=PSD[0]                      #assign frequency and power spectral density to 2 different vectors
        p=PSD[1]

        powerBrow=FeaturesExtraction.computePowerBands(f, p)   #compute absolute power bands for each channel
        powerBlist.iloc[j]=powerBrow                    #core.frame.DataFrame containing absolute power bands for each channel
    
    
    for k in range(len(data)):            #each channel
        for y in range(6):           #each band
            if not np.isnan(powerBlist.iloc[k,y]):       
                ave[k,y] += powerBlist.iloc[k,y]     #sum dataframes element by element
                divider[k,y] += 1     

    ave/=divider                 #average power band per electrode for all subjects
    ave= np.delete(ave, [4,33,40,43,45], axis=0)

    #np.save('average_PB.npy',ave)        #save file 

    #ave=np.delete(ave, (118,120,126),axis=0)        #delete electrode with too high values
   
    av_delta=ave[:,0]           #extract average for each band
    av_theta=ave[:,1]
    av_alpha=ave[:,2]
    #av_sigma=ave[:,3]
    av_beta=ave[:,4]
    av_lowgamma=ave[:,5]

    #dictio={'delta':av_delta, 'theta':av_theta, 'alpha': av_alpha, 'beta':av_beta, 'lowgamma':av_lowgamma}

    return av_delta, av_theta, av_alpha,  av_beta, av_lowgamma



def interp_elect_pos(elect_file_small, elect_file_big):

    elect_small = np.load(elect_file_small + '.npy')     #load reference electrode positions
    elect_big = np.load(elect_file_big + '.npy')        #load electrode positions to interpolate 

    interp_idx_big=[]
    interp_elect_big=[]

    spat_dist= spatial.distance.cdist(elect_small, elect_big)     #caluclate spatial distance between all the coordinates 

    for i in range(len(spat_dist)):
        spat_dist_list= spat_dist[i].tolist()
        interp_idx_big.append(spat_dist_list.index(min(spat_dist_list)))     #get index of minimum distance = index of nearest electrode

    for i in range(len(interp_idx_big)):
        interp_elect_big.append(elect_big[interp_idx_big[i]])     #get coordinates of nearest electrodes

    return interp_elect_big

def interp_data(list_big, elect_big, elect_small, interp_elect):

    elect_big = np.load(elect_big + '.npy')
    elect_small= np.load(elect_small + '.npy')
    interp_elect_big= np.load(interp_elect + '.npy')

    list_interp_data_big=[]

    for i in range(len(list_big)):
        list_interp_data_big.append(scipy.interpolate.griddata(elect_big, list_big[i], interp_elect_big))

    return list_interp_data_big

def interp_data2(elect_big, elect_small, list_big):

    second_min=100
    idx_min=[]
    idx_2min=[]
    mean_elect=[]
    interp_elect_min=[]
    interp_elect_2min=[]
    list_interp_data_big=[]
    divider=2

    spat_dist= spatial.distance.cdist(elect_small, elect_big)

    for i in range(spat_dist.shape[0]):
        
        spat_dist_list= spat_dist[i].tolist()
        minim=min(spat_dist_list)
        idx_min.append(spat_dist_list.index(minim)) 

        for y in range(len(spat_dist_list)):
            if (spat_dist_list[y] < second_min) and (spat_dist_list[y] >  minim) :
                second_min = spat_dist_list[y]
            idx_2min.append(spat_dist_list.index(second_min))

    for i in range(len(idx_min)):
        interp_elect_min.append(elect_big[idx_min[i]])
        interp_elect_2min.append(elect_big[idx_2min[i]])

    for i in range(len(interp_elect_min)):
        mean_elect.append((interp_elect_min[i]+interp_elect_2min))
        mean_elect[i]=mean_elect[i]/divider

    for i in range(len(list_big)):
        list_interp_data_big.append(scipy.interpolate.griddata(elect_big, list_big[i], mean_elect))

    return list_interp_data_big
  

# mindist = np.min(spatial.distance.cdist(elect_LA, elect_DA), axis=1) 


def array_topoplot(toplot, ch_xy, cmap='coolwarm', showtitle=False, titles=None, savefig=False, figpath=None, vmin=-3, vmax=3):

    '''author: Yann Harel'''

    #create fig
    fig, ax = plt.subplots(len(toplot),1, figsize=(20,10))
    #create a topomap for each data array
    for i, data in enumerate(toplot):
        image,_ = mne.viz.plot_topomap(data=data, pos=ch_xy, cmap=cmap, vmin=vmin, vmax=vmax, axes=ax[i], show=False)
        #option for title
        if showtitle == True:
            ax[i].set_title(titles[i], fontdict={'fontsize': 20, 'fontweight': 'heavy'})
    #add a colorbar at the end of the line (weird trick from https://www.martinos.org/mne/stable/auto_tutorials/stats-sensor-space/plot_stats_spatio_temporal_cluster_sensors.html#sphx-glr-auto-tutorials-stats-sensor-space-plot-stats-spatio-temporal-cluster-sensors-py)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax[0])
    ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(image, cax=ax_colorbar)
    ax_colorbar.tick_params(labelsize=8)
    #save plot if specified
    if savefig == True:
        plt.savefig(figpath, dpi=300)
    plt.show()
    return fig, ax

def plot_dec_accu(DA=[],sensors_pos=[],mask=False,DA_thr=None,save_file=None, vmin=-3, vmax=3):

    '''author : Tarek Lajnef'''

    if mask:
        mask_default = np.full((len(DA)), False, dtype=bool)
        mask = np.array(mask_default)
        mask[DA >= DA_thr] = True
        mask_params = dict(marker='*', markerfacecolor='w', markersize=18) # significant sensors appearence
        fig = plt.figure(figsize = (10,5))
        ax,_ = mne.viz.plot_topomap(DA,sensors_pos,
                           cmap='coolwarm',
                           show=False,
                           vmin=vmin,vmax=vmax,
                           contours=True,
                           mask = mask,
                           mask_params = mask_params,
                           extrapolate='local')
        fig.colorbar(ax, shrink=0.25)
        if save_file:
            plt.savefig(save_file, dpi = 300)
    else:
        fig = plt.figure(figsize = (10,5))
        ax,_ = mne.viz.plot_topomap(DA, sensors_pos,cmap='coolwarm',show=False,
        vmin=vmin,vmax=vmax,contours=True)
       #fig.colorbar(ax, shrink=0.25)
        if save_file:
            plt.savefig(save_file, dpi = 300)
    return ax


def array_plot_deccaccu(DA, ch_xy, maskk=True, DA_thr=None, marker='*', markersize=10, cmap='coolwarm', showtitle=False, titles=None, savefig=False, figpath=None, vmin=-3, vmax=3):

    #create fig
    fig, ax = plt.subplots(len(DA),1, figsize=(20,10))
    #create a topomap for each data array
    if maskk: 
        for i, data in enumerate(DA):
            mask_default = np.full((len(DA[i])), False, dtype=bool)
            mask = np.array(mask_default)
            for j in range(len(data)):
                if data[j]> DA_thr:  
                    mask[j] = True
            mask_params = dict(marker=marker, markerfacecolor='w', markersize=markersize) # significant sensors appearence
            image,_ = mne.viz.plot_topomap(data=data, pos=ch_xy, cmap=cmap, vmin=vmin, vmax=vmax, axes=ax[i], contours=True, mask=mask, mask_params=mask_params, show=False)
        #option for title
        if showtitle == True:
            ax[i].set_title(titles[i], fontdict={'fontsize': 20, 'fontweight': 'heavy'})

    #add a colorbar at the end of the line (weird trick from https://www.martinos.org/mne/stable/auto_tutorials/stats-sensor-space/plot_stats_spatio_temporal_cluster_sensors.html#sphx-glr-auto-tutorials-stats-sensor-space-plot-stats-spatio-temporal-cluster-sensors-py)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax[0])
    ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(image, cax=ax_colorbar)
    ax_colorbar.tick_params(labelsize=8)

    #save plot if specified
    if savefig == True:
        plt.savefig(figpath, dpi=300)

    plt.show()
    return fig, ax

def mean_per_subject(liste1, liste2):

    summ=0
    summ2=0
    list_sum=[]
    list_sum2=[]
    list_list_sum=[]
    list_list_sum2=[]
    diff=[]

    divider = liste1[0].shape[0]

    for i in range(len(liste1)):   #nb_subjects
        array= liste1[0]
        for j in range(array.shape[1]):    # nb electrodes
            samples=array[:,j]
            for k in range(samples.shape[0]):   #nb samples
                summ+=samples[k]
            list_sum.append(summ)    #list of sum of samples --> size nb electrodes 
        list_list_sum.append(list_sum)
        list_sum=[]
    
    for i in range(len(list_list_sum)):
        for j in range(len(list_list_sum[i])):
            list_list_sum[i][j]/=divider    #get mean for each electrode

    for i in range(len(liste2)):   #nb_subjects
        array2= liste2[0]
        for j in range(array2.shape[1]):    # nb electrodes
            samples2=array2[:,j]
            for k in range(samples2.shape[0]):   #nb samples
                summ2+=samples2[k]
            list_sum2.append(summ2)    #list of sum of samples --> size nb electrodes 
        list_list_sum2.append(list_sum2)
        list_sum2=[]
    
    for i in range(len(list_list_sum2)):
        for j in range(len(list_list_sum2[i])):
            list_list_sum2[i][j]/=divider     #get mean for each electrode

    for i in range(len(list_list_sum)):
        for j in range(len(list_list_sum[i])):
            diff.append((list_list_sum[i][j] - list_list_sum2[i][j]))    #difference between 2 conditions 

    return diff









