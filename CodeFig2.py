import TopoPB 
import numpy as np 
import matplotlib.pyplot as plt  
import CreateSamples
import scipy.interpolate as interp
import Class2
import mne
import Examine 


sensors_pos_DA= np.load('elect_pos_DA_clean.npy')

sensors_pos_LA = np.load('elect_pos_LA.npy')
sensors_pos_LA= np.delete(sensors_pos_LA, [4,33,40,44], axis=0)
#sensors_pos_DA_interp= np.load('elect_DA_new.npy')

scaled_DA= np.load('scaled_list_DA.npy')    #shape (powerbands, electrodes) for topoplots 
scaled_DAw= np.load('scaled_list_DAw.npy')
scaled_LA= np.load('scaled_list_LA.npy')
scaled_LAw= np.load('scaled_list_LAw.npy')
scaled_interp_DA= np.load('list_scaled_interp_DA.npy')

scaled_diff_DA=[]
scaled_diff_LA=[]
scaled_diff_interp=[]

scaled_rel_diff_DA=[]
scaled_rel_diff_LA=[]


for i in range(len(scaled_DA)):
    scaled_diff_DA.append(scaled_DA[i]-scaled_DAw[i])
    scaled_diff_LA.append(scaled_LA[i]-scaled_LAw[i])

    scaled_rel_diff_DA.append((scaled_DA[i]-scaled_DAw[i])/scaled_DAw[i])
    
    scaled_rel_diff_LA.append((scaled_LA[i]-scaled_LAw[i])/scaled_LAw[i])


sDDA=np.load('sDAdelta.npy')    #shape (subjects, samples,  electrodes) for classification 
sTDA=np.load('sDAtheta.npy')
sADA=np.load('sDAalpha.npy')
sBDA=np.load('sDAbeta.npy')
sLGDA=np.load('sDALowgamma.npy')

sDDAw=np.load('sDAwdelta.npy')
sTDAw=np.load('sDAwtheta.npy')
sADAw=np.load('sDAwalpha.npy')
sBDAw=np.load('sDAwbeta.npy')
sLGDAw=np.load('sDAwLowgamma.npy')


sDLA=np.load('sLAdelta.npy') 
sTLA=np.load('sLAtheta.npy')
sALA=np.load('sLAalpha.npy')
sBLA=np.load('sLAbeta.npy')
sLGLA=np.load('sLALowgamma.npy')

sDLAw=np.load('sLAwdelta.npy')    
sTLAw=np.load('sLAwtheta.npy')
sALAw=np.load('sLAwalpha.npy')
sBLAw=np.load('sLAwbeta.npy')
sLGLAw=np.load('sLAwLowgamma.npy')


score_ldaDA=np.load('score_lda_DA_1000.npy')
perm_score_ldaDA=np.load('perm_score_lda_DA_1000.npy')
pval_ldaDA=np.load('pval_lda_DA_1000.npy')

score_knnDA=np.load('score_knn_DA1000.npy')
perm_score_knnDA=np.load('perm_score_knn_DA1000.npy')
pval_knnDA=np.load('pval_knn_DA1000.npy')

#maxidist_knnDA, quant_knnDA= Examine.maxi_distribution(perm_score_knnDA)
#splitlist_scoreknnDA= Examine.split_appendd(score_knnDA, 85)


maxidist_ldaDA=np.load('maxidist_ldaDA.npy')
quant_ldaDA= np.load('quant_ldaDA.npy')
splitlist_scoreldaDA=np.load('splitlist_scoreldaDA.npy')

score_ldaLA=np.load('accu_lda_LA.npy')
perm_score_ldaLA=np.load('permscore_ldaLA.npy')
pval_ldaLA=np.load('pval_ldaLA.npy')

maxidist_ldaLA, quant_ldaLA= Examine.maxi_distribution(perm_score_ldaLA)
splitlist_scoreldaLA= Examine.split_appendd(score_ldaLA, 63)



list_LA= np.load('list_avPB_LA.npy')
list_LAw= np.load('list_avPB_LAw.npy')

reldiffDA=np.load('rel_diff_DA.npy')
reldiffLA=np.load('rel_diff_LA.npy')

diff_DDA=np.load('diff_DDA.npy')    #shape (subjects,electrodes) for t test
diff_TDA=np.load('diff_TDA.npy')
diff_ADA=np.load('diff_ADA.npy')
diff_BDA=np.load('diff_BDA.npy')
diff_LGDA=np.load('diff_LGDA.npy')

diff_DLA=np.load('diff_DLA.npy')
diff_TLA=np.load('diff_TLA.npy')
diff_ALA=np.load('diff_ALA.npy')
diff_BLA=np.load('diff_BLA.npy')
diff_LGLA=np.load('diff_LGLA.npy')


diff_LA0=[]
diff_LA1=[]
diff_LA2=[]
diff_LA3=[]
diff_LA4=[]
diff_LA5=[]
diff_LA6=[]



list_av_LA0=Script_Topo.Topo_1_sub('data_LA0.mat', 'S2_unconscious_RawData') 
list_av_LA1=Script_Topo.Topo_1_sub('data_LA1.mat', 'S3_unconscious_RawData') 
list_av_LA2=Script_Topo.Topo_1_sub('data_LA2.mat', 'S4_unconscious_RawData') 
list_av_LA3=Script_Topo.Topo_1_sub('data_LA3.mat', 'S5_unconscious_RawData') 
list_av_LA4=Script_Topo.Topo_1_sub('data_LA4.mat', 'S6_unconscious_RawData') 
list_av_LA5=Script_Topo.Topo_1_sub('data_LA5.mat', 'S8_unconscious_RawData') 
list_av_LA6=Script_Topo.Topo_1_sub('data_LA6.mat', 'S10_unconscious_RawData') 

list_av_LAw0=Script_Topo.Topo_1_sub('data_LAw0.mat', 'S2_conscious_RawData') 
list_av_LAw1=Script_Topo.Topo_1_sub('data_LAw1.mat', 'S3_conscious_RawData') 
list_av_LAw2=Script_Topo.Topo_1_sub('data_LAw2.mat', 'S4_conscious_RawData') 
list_av_LAw3=Script_Topo.Topo_1_sub('data_LAw3.mat', 'S5_conscious_RawData') 
list_av_LAw4=Script_Topo.Topo_1_sub('data_LAw4.mat', 'S6_conscious_RawData') 
list_av_LAw5=Script_Topo.Topo_1_sub('data_LAw5.mat', 'S8_conscious_RawData') 
list_av_LAw6=Script_Topo.Topo_1_sub('data_LAw6.mat', 'S10_conscious_RawData') 


for i in range(len(list_av_LA0)):
    diff_LA0.append(((list_av_LA0[i]-list_av_LAw0[i])/list_av_LAw0[i])*100)
    diff_LA1.append(((list_av_LA1[i]-list_av_LAw1[i])/list_av_LAw1[i])*100)
    diff_LA2.append(((list_av_LA2[i]-list_av_LAw2[i])/list_av_LAw2[i])*100)
    diff_LA3.append(((list_av_LA3[i]-list_av_LAw3[i])/list_av_LAw3[i])*100)
    diff_LA4.append(((list_av_LA4[i]-list_av_LAw4[i])/list_av_LAw4[i])*100)
    diff_LA5.append(((list_av_LA5[i]-list_av_LAw5[i])/list_av_LAw5[i])*100)
    diff_LA6.append(((list_av_LA6[i]-list_av_LAw6[i])/list_av_LAw6[i])*100)




TopoPB.array_topoplot(list_av_LA0, sensors_pos_LA, cmap='coolwarm', vmin=0, vmax=500)
TopoPB.array_topoplot(list_av_LA1, sensors_pos_LA, cmap='coolwarm', vmin=0, vmax=500)
TopoPB.array_topoplot(list_av_LA2, sensors_pos_LA, cmap='coolwarm', vmin=0, vmax=500)
TopoPB.array_topoplot(list_av_LA3, sensors_pos_LA, cmap='coolwarm', vmin=0, vmax=500)
TopoPB.array_topoplot(list_av_LA4, sensors_pos_LA, cmap='coolwarm', vmin=0, vmax=500)
TopoPB.array_topoplot(list_av_LA5, sensors_pos_LA, cmap='coolwarm', vmin=0, vmax=500)
TopoPB.array_topoplot(list_av_LA6, sensors_pos_LA, cmap='coolwarm', vmin=0, vmax=500)

TopoPB.array_topoplot(list_av_LAw0, sensors_pos_LA, cmap='coolwarm', vmin=0, vmax=500)
TopoPB.array_topoplot(list_av_LAw1, sensors_pos_LA, cmap='coolwarm', vmin=0, vmax=500)
TopoPB.array_topoplot(list_av_LAw2, sensors_pos_LA, cmap='coolwarm', vmin=0, vmax=500)
TopoPB.array_topoplot(list_av_LAw3, sensors_pos_LA, cmap='coolwarm', vmin=0, vmax=500)
TopoPB.array_topoplot(list_av_LAw4, sensors_pos_LA, cmap='coolwarm', vmin=0, vmax=500)
TopoPB.array_topoplot(list_av_LAw5, sensors_pos_LA, cmap='coolwarm', vmin=0, vmax=500)
TopoPB.array_topoplot(list_av_LAw6, sensors_pos_LA, cmap='coolwarm', vmin=0, vmax=500)

TopoPB.array_topoplot(diff_LA0, sensors_pos_LA, cmap='coolwarm', vmin=-100, vmax=100)
TopoPB.array_topoplot(diff_LA1, sensors_pos_LA, cmap='coolwarm', vmin=-100, vmax=100)
TopoPB.array_topoplot(diff_LA2, sensors_pos_LA, cmap='coolwarm', vmin=-100, vmax=100)
TopoPB.array_topoplot(diff_LA3, sensors_pos_LA, cmap='coolwarm', vmin=-100, vmax=100)
TopoPB.array_topoplot(diff_LA4, sensors_pos_LA, cmap='coolwarm', vmin=-100, vmax=100)
TopoPB.array_topoplot(diff_LA5, sensors_pos_LA, cmap='coolwarm', vmin=-100, vmax=100)
TopoPB.array_topoplot(diff_LA6, sensors_pos_LA, cmap='coolwarm', vmin=-100, vmax=100)

TopoPB.array_topoplot(list_av_LA0, sensors_pos_LA, cmap='coolwarm', vmin=-500, vmax=500)
TopoPB.array_topoplot(list_av_LA1, sensors_pos_LA, cmap='coolwarm', vmin=-500, vmax=500)
TopoPB.array_topoplot(list_av_LA2, sensors_pos_LA, cmap='coolwarm', vmin=-500, vmax=500)
TopoPB.array_topoplot(list_av_LA3, sensors_pos_LA, cmap='coolwarm', vmin=-500, vmax=500)
TopoPB.array_topoplot(list_av_LA4, sensors_pos_LA, cmap='coolwarm', vmin=-500, vmax=500)
TopoPB.array_topoplot(list_av_LA5, sensors_pos_LA, cmap='coolwarm', vmin=-500, vmax=500)
TopoPB.array_topoplot(list_av_LA6, sensors_pos_LA, cmap='coolwarm', vmin=-500, vmax=500)

TopoPB.array_topoplot(list_av_LAw0, sensors_pos_LA, cmap='coolwarm', vmin=-500, vmax=500)
TopoPB.array_topoplot(list_av_LAw1, sensors_pos_LA, cmap='coolwarm', vmin=-500, vmax=500)
TopoPB.array_topoplot(list_av_LAw2, sensors_pos_LA, cmap='coolwarm', vmin=-500, vmax=500)
TopoPB.array_topoplot(list_av_LAw3, sensors_pos_LA, cmap='coolwarm', vmin=-500, vmax=500)
TopoPB.array_topoplot(list_av_LAw4, sensors_pos_LA, cmap='coolwarm', vmin=-500, vmax=500)
TopoPB.array_topoplot(list_av_LAw5, sensors_pos_LA, cmap='coolwarm', vmin=-500, vmax=500)
TopoPB.array_topoplot(list_av_LAw6, sensors_pos_LA, cmap='coolwarm', vmin=-500, vmax=500)



rel_diff=[]
for i in range(len(list_LA)):
    rel_diff.append((list_LA[i]-list_LAw[i])/list_LAw[i])

diff_delta=[]
for i in range(len(av_delta)):
    diff_delta.append(av_delta[i]-av_deltaw[i])


    av_delta, av_theta, av_alpha,  av_beta, av_lowgamma


for i in range(len(list_av_LA)): 
    print(np.where(list_av_LA[i] == np.amax(list_av_LA[i])), max(list_av_LA[i])) 