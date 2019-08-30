import TopoPB 
import numpy as np 
import CreateSamples
import scipy.interpolate as interp
import Class2
import Examine 
import preprocessing


#topoplot average Power bands : Deep Anesthesia
'''
avDDA, avTDA, avADA, avBDA, avLGDA= TopoPB.avPowerband('cl_data_DA',9)
avDDAw, avTDAw, avADAw, avBDAw, avLGDAw= TopoPB.avPowerband('cl_data_DAw',9)
'''


sDDA, sTDA, sADA, sBDA, sLGDA= CreateSamples.PSDsamples('cl_data_DA', 2, 500, 9, '.txt')   

avDDA=np.loadtxt('avDelta_DA.txt')
avTDA=np.loadtxt('avTheta_DA.txt')
avADA=np.loadtxt('avAlpha_DA.txt')
avBDA=np.loadtxt('avBeta_DA.txt')
avLGDA=np.loadtxt('avLowgamma_DA.txt')

avDDAw=np.loadtxt('avDelta_DAw.txt')
avTDAw=np.loadtxt('avTheta_DAw.txt')
avADAw=np.loadtxt('avAlpha_DAw.txt')
avBDAw=np.loadtxt('avBeta_DAw.txt')
avLGDAw=np.loadtxt('avLowgamma_DAw.txt')



avDLA=np.load('avLAdelta.npy')
avTLA=np.load('avLAtheta.npy')
avALA=np.load('avLAalpha.npy')
avBLA=np.load('avLAbeta.npy')
avLGLA=np.load('avLAlowgamma.npy')

avDLAw=np.load('avLAwdelta.npy')
avTLAw=np.load('avLAwtheta.npy')
avALAw=np.load('avLAwalpha.npy')
avBLAw=np.load('avLAwbeta.npy')
avLGLAw=np.load('avLAwlowgamma.npy')



#saved 
avDLA=np.load('avLAdelta.npy')
avTLA=np.load('avLAtheta.npy')
avALA=np.load('avLAalpha.npy')
avBLA=np.load('avLAbeta.npy')
avLGLA=np.load('avLAlowgamma.npy')

avDLAw=np.load('avLAwdelta.npy')
avTLAw=np.load('avLAwtheta.npy')
avALAw=np.load('avLAwalpha.npy')
avBLAw=np.load('avLAwbeta.npy')
avLGLAw=np.load('avLAwlowgamma.npy')


interp_avDDA= np.load('interp_avDDA.npy')
interp_avTDA= np.load('interp_avTDA.npy')
interp_avADA= np.load('interp_avADA.npy')
interp_avBDA= np.load('interp_avBDA.npy')
interp_avLGDA= np.load('interp_avLGDA.npy')

list_DA=Examine.appendd(avDDA, avTDA, avADA, avBDA, avLGDA)
list_DAw= Examine.appendd(avDDAw, avTDAw, avADAw, avBDAw, avLGDAw)
list_DA_interp= Examine.appendd(interp_avDDA, interp_avTDA, interp_avADA, interp_avBDA, interp_avLGDA)

list_LA=Examine.appendd(avALA, avTLA, avALA, avBLA, avLGLA)
list_LAw= Examine.appendd(avDLAw, avTLAw, avALAw, avBLAw, avLGLAw)
list_clf= np.load('list_clf_LAT.npy')

diffD = avDDA - avDDAw
diffT = avTDA - avTDAw
diffA = avADA - avADAw
diffB = avBDA - avBDAw
diffLG = avLGDA - avLGDAw

diffDLA = avDLA - avDLAw   #L2 norm euclidische 
diffTLA = avTLA - avTLAw
diffALA = avALA - avALAw
diffBLA = avBLA - avBLAw
diffLGLA = avLGLA - avLGLAw



TopoPB.plotTopo(avDDA, avTDAw, avADAw, avBDAw, avLGDAw, avDDA, avTDA, avADA, avBDA, avLGDA, diffD, diffT, diffA, diffB, diffLG)

idx2rem=[] 
ch2rem= CreateSamples.find_missingch('data_DA','data_DAw', 9 ) 

montage=mne.channels.read_montage('GSN-HydroCel-128') 

for k in range(len(ch2rem)): 
    if ch2rem[k] in montage.ch_names: 
        idx2rem.append(montage.ch_names.index(ch2rem[k]))

idx2rem.extend((118,120,126))    #too  high values

montage.pos=np.delete(montage.pos, idx2rem, axis=0)
montage.ch_names=np.delete(montage.ch_names, idx2rem, axis=0)

sensors_pos=montage.get_pos2d()[3:]
ch_names_info=montage.ch_names[3:]


        #create Topoplots - light anesthesia
avLAdelta, avLAtheta, avLAalpha, avLAbeta, avLAlowgamma= TopoPB.avPowerband('cl_data_LA',7, '.npy')
avLAwdelta, avLAwtheta, avLAwalpha, avLAwbeta, avLAwlowgamma= TopoPB.avPowerband('cl_data_LA',7, '.npy')





        #plot topomaps
TopoPB.plotTopo(avDLA, avTLAw, avALAw, avBLAw, avLGLAw, avDLA, avTLA, avALA, avBLA, avLGLA, diffDLA, diffTLA, diffALA, diffBLA, diffLGLA, 'LA')


        #interpolate new electrode positions LA - saved 
elect_DA_new= TopoPB.interp_elect_pos('elect_LA','elect_DA')  #saved



interp_data_DA= TopoPB.interp_data(list_DA, 'elect_DA', 'elect_LA', 'elect_DA_new')

TopoPB.plotTopo(list_LA, interp_data_DA, 'light', 'deep', 'mixed')



'''
np.savetxt('avDelta_DAw.txt', avDDAw)
np.savetxt('avTheta_DAw.txt', avTDAw)
np.savetxt('avAlpha_DAw.txt', avADAw)
np.savetxt('avBeta_DAw.txt', avBDAw)
np.savetxt('avLowgamma_DAw.txt', avLGDAw)
'''


fig= plt.figure()

    ax1=plt.subplot(3,5,1)
    ax1.mne.viz.plot_topomap(avDeltaDAw,sensors_pos,show=False, cmap='coolwarm')
    plt.title('Delta')  #2-4 Hz

    ax2=plt.subplot(3,5,2)
    ax2.mne.viz.plot_topomap(avThetaDAw,sensors_pos,show=False,cmap='coolwarm')
    plt.title('Theta')  #5-7 Hz

    ax3=plt.subplot(3,5,3)
    ax3.mne.viz.plot_topomap(avAlphaDAw,sensors_pos,show=False,cmap='coolwarm')
    plt.title('Alpha')  #8-13 Hz




    ax4=plt.subplot(3,5,4)
    ax4.mne.viz.plot_topomap(avBetaDAw,sensors_pos,show=False,cmap='coolwarm')
    plt.title('Beta')   #13-30

    ax5=plt.subplot(3,5,5)
    ax5.mne.viz.plot_topomap(avLowgammaDAw,sensors_pos,show=False,cmap='coolwarm')
    plt.title('Low Gamma')  #30-50


    ax6=plt.subplot(3,5,6)
    ax6.mne.viz.plot_topomap(avDeltaDA,sensors_pos,show=False, cmap='coolwarm')
    plt.title('Delta')  #2-4 Hz

    ax7= plt.subplot(3,5,7)
    ax7.mne.viz.plot_topomap(avThetaDA,sensors_pos,show=False,cmap='coolwarm')
    plt.title('Theta')  #5-7 Hz

    ax8=plt.subplot(3,5,8)
    ax8.mne.viz.plot_topomap(avAlphaDA,sensors_pos,show=False,cmap='coolwarm')
    plt.title('Alpha')  #8-13 Hz

    ax9=plt.subplot(3,5,9)
    ax9.mne.viz.plot_topomap(avBetaDA,sensors_pos,show=False,cmap='coolwarm')
    plt.title('Beta')   #13-30

    ax10=plt.subplot(3,5,10)
    ax10.mne.viz.plot_topomap(avLowgammaDA,sensors_pos,show=False,cmap='coolwarm')
    plt.title('Low Gamma')  #30-50


    ax11=plt.subplot(3,5,11)
    ax11.mne.viz.plot_topomap(DiffD,sensors_pos,show=False, cmap='coolwarm')
    plt.title('Delta')  #2-4 Hz

    ax12=plt.subplot(3,5,12)
    ax12.mne.viz.plot_topomap(DiffT,sensors_pos,show=False,cmap='coolwarm')
    plt.title('Theta')  #5-7 Hz

    ax13=plt.subplot(3,5,13)
    ax13.mne.viz.plot_topomap(DiffA,sensors_pos,show=False,cmap='coolwarm')
    plt.title('Alpha')  #8-13 Hz

    ax14=plt.subplot(3,5,14)
    ax14.mne.viz.plot_topomap(DiffB,sensors_pos,show=False,cmap='coolwarm')
    plt.title('Beta')   #13-30

    ax15=plt.subplot(3,5,15)
    ax15.mne.viz.plot_topomap(DiffLG,sensors_pos,show=False,cmap='coolwarm')
    plt.title('Low Gamma')  #30-50

    ax1.text(0.1,0.9, 'wakefulness')
    ax6.text(0.4,0.9, 'deep anesthesia')
    ax11.text(0.8,0.9, 'wake - deep')



#classif 

accu_clf=np.load('accu_clf_DA.npy')
accu_forest=np.load('accu_forest_DA.npy')                                                                                                                        
accu_knn=np.load('accu_knn_DA.npy')                                                                                                                              
accu_lda=np.load('accu_lda_DA.npy')                                                                                                                              
accu_qda=np.load('accu_qda_DA.npy')                                                                                                                              
accu_mlp=np.load('accu_mlp_DA.npy')     

list_clf, list_forest, list_knn, list_lda, list_qda, list_mlp =Class2.split_class(accu_clf, accu_forest, accu_knn, accu_lda, accu_qda, accu_mlp, 85)

list_arr_clf= Examine.list2array(list_clf)                                                                                                                        
list_arr_forest= Examine.list2array(list_forest)                                                                                                                  
list_arr_knn= Examine.list2array(list_knn)                                                                                                                        
list_arr_lda= Examine.list2array(list_lda)                                                                                                                        
list_arr_qda= Examine.list2array(list_qda)                                                                                                                       
list_arr_mlp= Examine.list2array(list_mlp)

list_clfT= np.load('list_clf_LAT.npy')
list_forestT= np.load('list_forest_LAT.npy')
list_knnT= np.load('list_knn_LAT.npy')
list_ldaT = np.load('list_lda_LAT.npy')
list_qdaT = np.load('list_qda_LAT.npy')
list_mlpT = np.load('list_mlp_LAT.npy')

mean_sc_LA= Class2.mean_score(list_clfT, list_forestT, list_knnT, list_ldaT, list_qdaT, list_mlpT)  

TopoPB.plotTopo(list_LA, list_LAw, mean_sc_LA, 'light', 'deep', 'mean accuracy score', 'LA')



scaler = preprocessing.StandardScaler() 
scaled_listLA=[]

 for i in range(len(list_LA)): 
    list_LA_reshape.append(list_LA[i].reshape(-1,1)) 
    
for i in range(len(list_LA)):  
    scaled_listLA.append(scaler.fit_transform(list_LA_reshape[i]))

#run CodeFig2



#plot DA-LA
interp_DDA= np.load('interp_avDDA.npy')
interp_TDA= np.load('interp_avTDA.npy')
interp_ADA= np.load('interp_avADA.npy')
interp_BDA= np.load('interp_avBDA.npy')
interp_LGDA= np.load('interp_avLGDA.npy')

list_interp_DA= Examine.appendd(interp_DDA, interp_TDA,interp_ADA,interp_BDA,interp_LGDA)
sc_interp_DA= Examine.normalize_list(list_interp_DA)


TopoPB.array_topoplot(scaled_interp_DA, sensors_pos_DA_interp)
TopoPB.array_topoplot(scaled_diff_interp, sensors_pos_LA)   #demander a karim quel meilleur 
TopoPB.array_topoplot(scaled_diff_interp, sensors_pos_LA)
TopoPB.array_topoplot(scaled_interp_DA, sensors_pos_LA) 


diff_DDA=CreateSamples.diff_per_subject2(sDDA, sDDAw)
diff_TDA=CreateSamples.diff_per_subject2(sTDA, sTDAw)
diff_ADA=CreateSamples.diff_per_subject2(sADA, sADAw)
diff_BDA=CreateSamples.diff_per_subject2(sBDA, sBDAw)
diff_LGDA=CreateSamples.diff_per_subject2(sLGDA, sLGDAw)

diff_DLA=CreateSamples.diff_per_subject2(sDLA, sDLAw)
diff_TLA=CreateSamples.diff_per_subject2(sTLA, sTLAw)
diff_ALA=CreateSamples.diff_per_subject2(sALA, sALAw)
diff_BLA=CreateSamples.diff_per_subject2(sBLA, sBLAw)
diff_LGLA=CreateSamples.diff_per_subject2(sLGLA, sLGLAw)

np.save('diff_DDA.npy', diff_DDA)
np.save('diff_TDA.npy', diff_TDA)
np.save('diff_ADA.npy', diff_ADA)
np.save('diff_BDA.npy', diff_BDA)
np.save('diff_LGDA.npy', diff_LGDA)

np.save('diff_DLA.npy', diff_DLA)
np.save('diff_TLA.npy', diff_TLA)
np.save('diff_ALA.npy', diff_ALA)
np.save('diff_BLA.npy', diff_BLA)
np.save('diff_LGLA.npy', diff_LGLA)



diff_DA=np.concatenate((diff_DDA, diff_TDA, diff_ADA, diff_BDA, diff_LGDA), axis=1)
T_obs_DA, pval_DA, H0_DA= mne.stats.permutation_t_test(diff_DA)

reldiffDDA= CreateSamples.rel_diff(avDDA, avDDAw)
reldiffTDA= CreateSamples.rel_diff(avTDA, avTDAw)
reldiffADA= CreateSamples.rel_diff(avADA, avADAw)
reldiffBDA= CreateSamples.rel_diff(avBDA, avBDAw)
reldiffLGDA= CreateSamples.rel_diff(avLGDA, avLGDAw)

reldiffDLA= CreateSamples.rel_diff(avDLA, avDLAw)
reldiffTLA= CreateSamples.rel_diff(avTLA, avTLAw)
reldiffALA= CreateSamples.rel_diff(avALA, avALAw)
reldiffBLA= CreateSamples.rel_diff(avBLA, avBLAw)
reldiffLGLA= CreateSamples.rel_diff(avLGLA, avLGLAw)

list_reldiff_DA= Examine.appendd(reldiffDDA, reldiffTDA,reldiffADA,reldiffBDA,reldiffLGDA)
list_reldiff_LA= Examine.appendd(reldiffDLA, reldiffTLA,reldiffALA,reldiffBLA,reldiffLGLA)

T0_DDA, pval_DDA, H0_DDA= mne.stats.permutation_t_test(diff_DDA) 
T0_TDA, pval_TDA, H0_TDA= mne.stats.permutation_t_test(diff_TDA) 
T0_ADA, pval_ADA, H0_ADA= mne.stats.permutation_t_test(diff_ADA) 
T0_BDA, pval_BDA, H0_BDA= mne.stats.permutation_t_test(diff_BDA) 
T0_LGDA, pval_LGDA, H0_LGDA= mne.stats.permutation_t_test(diff_LGDA) 


T0_DLA, pval_DLA, H0_DLA= mne.stats.permutation_t_test(diff_DLA) 
T0_TLA, pval_TLA, H0_TLA= mne.stats.permutation_t_test(diff_TLA) 
T0_ALA, pval_ALA, H0_ALA= mne.stats.permutation_t_test(diff_ALA) 
T0_BLA, pval_BLA, H0_BLA= mne.stats.permutation_t_test(diff_BLA) 
T0_LGLA, pval_LGLA, H0_LGLA= mne.stats.permutation_t_test(diff_LGLA)


T0_DA= Examine.appendd(T0_DDA, T0_TDA, T0_ADA, T0_BDA, T0_LGDA)
T0_LA= Examine.appendd(T0_DLA, T0_TLA, T0_ALA, T0_BLA, T0_LGLA)

H0_DA= Examine.appendd(H0_DDA, H0_TDA, H0_ADA, H0_BDA, H0_LGDA)
H0_LA= Examine.appendd(H0_DLA, H0_TLA, H0_ALA, H0_BLA, H0_LGLA)

flat_H0DA, quant_H0DA= Examine.maxi_distribution(H0_DA) 
flat_H0LA, quant_H0LA= Examine.maxi_distribution(H0_LA) 


TopoPB.array_plot_deccaccu(T0_DA,sensors_pos_DA, maskk=True, DA_thr=quant_H0DA[0], marker='x',markersize=3, cmap='YlOrBr', vmin=0, vmax=7)
TopoPB.array_plot_deccaccu(T0_DA,sensors_pos_DA, maskk=True, DA_thr=quant_H0DA[1], marker='x',markersize=3, cmap='YlOrBr', vmin=0, vmax=7)
TopoPB.array_plot_deccaccu(T0_DA,sensors_pos_DA, maskk=True, DA_thr=quant_H0DA[2], marker='x',markersize=3, cmap='YlOrBr', vmin=0, vmax=7)

TopoPB.array_plot_deccaccu(T0_LA,sensors_pos_LA, maskk=True, DA_thr=quant_H0LA[0], marker='x',markersize=3, cmap='YlOrBr', vmin=0, vmax=7)
TopoPB.array_plot_deccaccu(T0_LA,sensors_pos_LA, maskk=True, DA_thr=quant_H0LA[1], marker='x',markersize=3, cmap='YlOrBr', vmin=0, vmax=7)
TopoPB.array_plot_deccaccu(T0_LA,sensors_pos_LA, maskk=True, DA_thr=quant_H0LA[2], marker='x',markersize=3, cmap='YlOrBr', vmin=0, vmax=7)
 
 

LAdelta, LAtheta, LAalpha, LAbeta, LAlowgamma= TopoPB.avPowerband('cl_data_LAw', 7 , '.npy')
LAwdelta, LAwtheta, LAwalpha, LAwbeta, LAwlowgamma= TopoPB.avPowerband('cl_data_LAw', 7 , '.npy')
list_LA= Examine.appendd(LAdelta, LAtheta, LAalpha, LAbeta, LAlowgamma)
list_LAw= Examine.appendd(LAwdelta, LAwtheta, LAwalpha, LAwbeta, LAwlowgamma)

list_LA_mili=[]
list_LAw_mili=[]

for i in range(len(list_LAw)): 
    list_LAw_mili.append(list_LAw[i]*10e-6) 
for i in range(len(list_LA)): 
    list_LA_mili.append(list_LA[i]*10e-6)

TopoPB.array_topoplot(list_LAw_mili, sensors_pos_LA, vmin=10e-15, vmax=10e-4)
TopoPB.array_topoplot(list_LA_mili, sensors_pos_LA, vmin=10e-9, vmax=10e-4)

diff=[]
for i in range(len(list_LA_mili)):
        diff.append(list_LA_mili[i]-list_LAw_mili[i])



avDLA, avTLA, avALA, avBLA, avLGLA= TopoPB.avPowerband('clean_data_LA', 7, '.npy')
avDLAw, avTLAw, avALAw, avBLAw, avLGLAw= TopoPB.avPowerband('clean_data_LAw', 7, '.npy')

sensors_pos_LA = np.load('elect_pos_LA.npy') 
sensors_pos_LA= np.delete(sensors_pos_LA, [4,33,40,43,44], axis=0) 

list_av_LA=Examine.appendd(avDLA, avTLA, avALA, avBLA, avLGLA)
list_av_LAw=Examine.appendd(avDLAw, avTLAw, avALAw, avBLAw, avLGLAw)

rel_diff=[]
for i in range(len(list_av_LA)):
    rel_diff.append((list_av_LA[i]-list_av_LAw[i])/list_av_LAw[i]*100)

scaled_LA=Examine.normalize_list(list_av_LA)
scaled_LAw=Examine.normalize_list(list_av_LAw)


TopoPB.array_topoplot(list_av_LA, sensors_pos_LA, vmin=0, vmax=100, cmap='coolwarm')
TopoPB.array_topoplot(list_av_LAw, sensors_pos_LA, vmin=0, vmax=100, cmap='coolwarm')
TopoPB.array_topoplot(rel_diff, sensors_pos_LA, vmin=-150, vmax=150, cmap= 'coolwarm')



TopoPB.array_topoplot(scaled_LA, sensors_pos_LA, vmin=-3, vmax=3, cmap='coolwarm')
TopoPB.array_topoplot(scaled_LAw, sensors_pos_LA, vmin=-3, vmax=3, cmap='coolwarm')


s1_LA=np.load('data_LA0.npy')
s2_LA=np.load('data_LA1.npy')
s3_LA=np.load('data_LA2.npy')
s4_LA=np.load('data_LA3.npy')
s5_LA=np.load('data_LA4.npy')
s6_LA=np.load('data_LA5.npy')
s7_LA=np.load('data_LA6.npy')

s1_LAw=np.load('data_LA0.npy')
s2_LAw=np.load('data_LA1.npy')
s3_LAw=np.load('data_LA2.npy')
s4_LAw=np.load('data_LA3.npy')
s5_LAw=np.load('data_LA4.npy')
s6_LAw=np.load('data_LA5.npy')
s7_LAw=np.load('data_LA6.npy')



diff_delta= CreateSamples.diff_per_subject2(sDLA,sDLAw) 
diff_theta= CreateSamples.diff_per_subject2(sTLA,sTLAw)
diff_alpha= CreateSamples.diff_per_subject2(sALA,sALAw)
diff_beta= CreateSamples.diff_per_subject2(sBLA,sBLAw)
diff_lowgamma= CreateSamples.diff_per_subject2(sLGLA,sLGLAw)

Tobs_delta, pval_delta, H0_delta= permutation_t_test(diff_delta)
Tobs_theta, pval_theta, H0_theta= permutation_t_test(diff_theta)
Tobs_alpha, pval_alpha, H0_alpha= permutation_t_test(diff_alpha)
Tobs_beta, pval_beta, H0_beta= permutation_t_test(diff_beta)
Tobs_lowgamma, pval_lowgamma, H0_lowgamma= permutation_t_test(diff_lowgamma)











