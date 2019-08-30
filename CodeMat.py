import Examine 
import CreateSamples
import TopoPB

        #create PSD samples - deep anesthesia
sDAdelta, sDAtheta, sDAalpha, sDAbeta, sDAlowgamma = CreateSamples.PSDsamples('cl_data_DA', 2, 500, 7, '.txt')


        #create PSD samples - light anesthesia

#.mat files to .npy files & save
files= Examine.loaddmat('data_LA',7)  #load mat files
Examine.saveNpyfromMat_LA(files,'data_LA')  #save as .npy files
files= Examine.loaddmat('data_LAw',7) 
Examine.saveNpyfromMat_LAw(files,'data_LA')
files=Examine.loaddmat('Ldata',7)
Examine.saveNpyfromMat_L(files, 'Ldata') 

LA= Examine.loaddnpy('data_LA',7)     #load all .npy files 
clean_LA= CreateSamples.del_missingch_LA(LA)   #delete channels 33 and 44 for all subjects but nb2
Examine.saveclean(clean_LA, 7, 'clean_data_LA')

LAw= Examine.loaddnpy('data_LAw',7) 
clean_LAw= CreateSamples.del_missingch_LA(LAw)
Examine.saveclean(clean_LAw, 7, 'clean_data_LAw')

LAdelta, LAtheta, LAalpha, LAbeta, LAlowgamma= TopoPB.avPowerband('clean_data_LA', 7, '.npy')  #average powerbands
LAwdelta, LAwtheta, LAwalpha, LAwbeta, LAwlowgamma= TopoPB.avPowerband('clean_data_LA', 7, '.npy')

list_LA= Examine.appendd(LAdelta, LAtheta, LAalpha, LAbeta, LAlowgamma)    #create one list for topoplots
list_LAw= Examine.appendd(LAwdelta, LAwtheta, LAwalpha, LAwbeta, LAwlowgamma)

TopoPB.array_topoplot(list_LA, sensors_pos_LA, vmin=0, vmax=500)

sLAdelta, sLAtheta, sLAalpha, sLAbeta, sLAlowgamma = CreateSamples.PSDsamples('clean_data_LA', 2, 500, 7, '.npy') #create samples per power band
sLAwdelta, sLAwtheta, sLAwalpha, sLAwbeta, sLAwlowgamma = CreateSamples.PSDsamples('clean_data_LAw', 2, 500, 7, '.npy')

Examine.save3Darray('sLAdelta.npy', sLAdelta)   #saving sample arrays                                                                                                                  
Examine.save3Darray('sLAtheta.npy', sLAtheta)                                                                                                                     
Examine.save3Darray('sLAalpha.npy', sLAalpha)                                                                                                                     
Examine.save3Darray('sLAbeta.npy', sLAbeta)                                                                                                                      
Examine.save3Darray('sLAlowgamma.npy', sLAlowgamma)  

Examine.save3Darray('sLAwdelta.npy', sLAwdelta)   #saving sample arrays                                                                                                                  
Examine.save3Darray('sLAwtheta.npy', sLAwtheta)                                                                                                                     
Examine.save3Darray('sLAwalpha.npy', sLAwalpha)                                                                                                                     
Examine.save3Darray('sLAwbeta.npy', sLAwbeta)                                                                                                                      
Examine.save3Darray('sLAwlowgamma.npy', sLAwlowgamma) 




        #create Topoplots - light anesthesia
avLAdelta, avLAtheta, avLAalpha, avLAbeta, avLAlowgamma= TopoPB.avPowerband('cl_data_LA',7, '.npy')
avLAwdelta, avLAwtheta, avLAwalpha, avLAwbeta, avLAwlowgamma= TopoPB.avPowerband('cl_data_LA',7, '.npy')
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



        #interpolate new electrode positions LA - saved 
elect_DA_new= TopoPB.interp_elect_pos('elect_LA','elect_DA')





'''
#code Thomas pour pas mm nombre d'éléctrodes

assert con.shape[0]==uncon.shape[0],    "attention, con et uncon pas le même nb d'elec"
    if con.shape[0] == 65:
        elec = np.ones(shape = (con.shape[0]),dtype = bool)
        elec[[39,44]] = False
        con = con[elec,:]
        uncon = uncon[elec,:]
    con = np.mean(con, axis =1)[...,np.newaxis]

'''


accu_clf=np.load('accu_clf.npy')
accu_forest=np.load('accu_forest.npy')
accu_knn=np.load('accu_knn.npy')
accu_lda=np.load('accu_lda.npy')
accu_qda=np.load('accu_qda.npy')
accu_mlp=np.load('accu_mlp.npy')

list_clf, list_forest, list_knn, list_lda, list_qda, list_mlp= Class2.split_class(accu_clf, accu_forest, accu_knn, accu_lda, accu_qda, accu_mlp,63)               

np.save('list_clf.npy',list_clf)
np.save('list_forest.npy',list_forest)
np.save('list_knn.npy',list_knn)
np.save('list_lda.npy',list_lda)
np.save('list_qda.npy',list_qda)
np.save('list_mlp.npy',list_mlp)

list_clf= np.load('list_clf_LA.npy')
list_forest= np.load('list_forest_LA.npy') 
list_knn= np.load('list_knn_LA.npy')
list_lda = np.load('list_lda_LA.npy')
list_qda = np.load('list_qda_LA.npy')
list_mlp = np.load('list_mlp_LA.npy')

list_arr_clf= Examine.list2array(list_clf)                                                                                                                        
list_arr_forest= Examine.list2array(list_forest)                                                                                                                  
list_arr_knn= Examine.list2array(list_knn)                                                                                                                        
list_arr_lda= Examine.list2array(list_lda)                                                                                                                        
list_arr_qda= Examine.list2array(list_qda)                                                                                                                       
list_arr_mlp= Examine.list2array(list_mlp)

arrTclf= Examine.list_of_arraysT(list_arr_clf)
arrTforest= Examine.list_of_arraysT(list_arr_forest)
arrTknn= Examine.list_of_arraysT(list_arr_knn)
arrTlda= Examine.list_of_arraysT(list_arr_lda)
arrTqda= Examine.list_of_arraysT(list_arr_qda)
arrTmlp= Examine.list_of_arraysT(list_arr_mlp)

np.save('list_clf_LAT.npy',arrTclf)                                                                                                                               

np.save('list_forest_LAT.npy',arrTforest)                                                                                                                         

np.save('list_knn_LAT.npy',arrTknn)                                                                                                                              

np.save('list_lda_LAT.npy',arrTlda)                                                                                                                              

np.save('list_qda_LAT.npy',arrTqda)                                                                                                                              

np.save('list_mlp_LAT.npy',arrTmlp)   

list_clfT= np.load('list_clf_LAT.npy')
list_forestT= np.load('list_forest_LAT.npy') 
list_knnT= np.load('list_knn_LAT.npy')
list_ldaT = np.load('list_lda_LAT.npy')
list_qdaT = np.load('list_qda_LAT.npy')
list_mlpT = np.load('list_mlp_LAT.npy')







