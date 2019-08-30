import numpy as np 
from sklearn.model_selection import train_test_split, permutation_test_score, LeavePGroupsOut
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import pandas as pd

def class_clf(files):
    score_clf=[]
    perm_score_clf=[]
    pval_clf=[]

    file_namesDA=['sDAdelta.npy', 'sDAtheta.npy', 'sDAalpha.npy','sDAbeta.npy','sDAlowgamma.npy']
    file_namesDAw=['sDAwdelta.npy', 'sDAwtheta.npy', 'sDAwalpha.npy','sDAwbeta.npy','sDAwlowgamma.npy']

    file_namesLA=['sLAdelta.npy', 'sLAtheta.npy', 'sLAalpha.npy','sLAbeta.npy','sLAlowgamma.npy']
    file_namesLAw=['sLAwdelta.npy', 'sLAwtheta.npy', 'sLAwalpha.npy','sLAwbeta.npy','sLAwlowgamma.npy']

    listAnest=[]
    listWake=[]

    if files =='DA':
        file_names=file_namesDA
        file_names2=file_namesDAw

    elif files =='LA':
        file_names=file_namesLA
        file_names2=file_namesLAw

    for i, j in zip(file_names, file_names2):

        listAnest.append(np.load(i, allow_pickle=True))
        listWake.append(np.load(j, allow_pickle=True))

    listAnest=np.concatenate(listAnest, axis=2)     
    listWake=np.concatenate(listWake, axis=2)

    listAnest=listAnest.reshape((-1,listAnest.shape[2]))   
    listWake=listWake.reshape((-1,listWake.shape[2]))

    X=np.concatenate((listAnest,listWake), axis=0)
    y=np.concatenate((np.zeros(listAnest.shape[0]),np.ones(listWake.shape[0])))

    if files == 'DA':
        groups=np.concatenate((np.full(150,0),np.full(150,1),np.full(150,2),np.full(150,3),np.full(150,4),np.full(150,5),np.full(150,6),np.full(150,7),np.full(150,8), np.full(112,0),np.full(112,1),np.full(112,2),np.full(112,3),np.full(112,4),np.full(112,5),np.full(112,6),np.full(112,7),np.full(112,8)))
    
    elif files == 'LA':
        groups=np.concatenate((np.full(590,0),np.full(590,1),np.full(590,2),np.full(590,3),np.full(590,4),np.full(590,5), np.full(590,6), np.full(690,0),np.full(690,1),np.full(690,2),np.full(690,3),np.full(690,4), np.full(690,5), np.full(690,6)))

    lpgo= LeavePGroupsOut(2)
    X=X.T
    clf= svm.SVC(gamma='auto')

    for i in range(X.shape[0]):

        x = X[i,:]
        x = x.reshape((-1,1))

        score_clf_val, perm_score_clf_val, pval_clf_val= permutation_test_score(clf, x, y, groups, lpgo, n_permutations=1000, n_jobs=2)

        score_clf.append(score_clf_val)
        perm_score_clf.append(perm_score_clf_val)
        pval_clf.append(pval_clf_val)

        print(i)

    return score_clf, perm_score_clf, pval_clf

def class_knn(files, n_neighbors=20):

    score_knn=[]
    perm_score_knn=[]
    pval_knn=[]

    file_namesDA=['sDAdelta.npy', 'sDAtheta.npy', 'sDAalpha.npy','sDAbeta.npy','sDAlowgamma.npy']
    file_namesDAw=['sDAwdelta.npy', 'sDAwtheta.npy', 'sDAwalpha.npy','sDAwbeta.npy','sDAwlowgamma.npy']

    file_namesLA=['sLAdelta.npy', 'sLAtheta.npy', 'sLAalpha.npy','sLAbeta.npy','sLAlowgamma.npy']
    file_namesLAw=['sLAwdelta.npy', 'sLAwtheta.npy', 'sLAwalpha.npy','sLAwbeta.npy','sLAwlowgamma.npy']

    listAnest=[]
    listWake=[]

    if files =='DA':
        file_names=file_namesDA
        file_names2=file_namesDAw

    elif files =='LA':
        file_names=file_namesLA
        file_names2=file_namesLAw

    for i, j in zip(file_names, file_names2):

        listAnest.append(np.load(i, allow_pickle=True))
        listWake.append(np.load(j, allow_pickle=True))

    listAnest=np.concatenate(listAnest, axis=2)     
    listWake=np.concatenate(listWake, axis=2)

    listAnest=listAnest.reshape((-1,listAnest.shape[2]))   
    listWake=listWake.reshape((-1,listWake.shape[2]))

    X=np.concatenate((listAnest,listWake), axis=0)
    y=np.concatenate((np.zeros(listAnest.shape[0]),np.ones(listWake.shape[0])))

    if files == 'DA':
        groups=np.concatenate((np.full(150,0),np.full(150,1),np.full(150,2),np.full(150,3),np.full(150,4),np.full(150,5),np.full(150,6),np.full(150,7),np.full(150,8), np.full(112,0),np.full(112,1),np.full(112,2),np.full(112,3),np.full(112,4),np.full(112,5),np.full(112,6),np.full(112,7),np.full(112,8)))
    
    elif files == 'LA':
        groups=np.concatenate((np.full(590,0),np.full(590,1),np.full(590,2),np.full(590,3),np.full(590,4),np.full(590,5), np.full(590,6), np.full(690,0),np.full(690,1),np.full(690,2),np.full(690,3),np.full(690,4), np.full(690,5), np.full(690,6)))


    lpgo= LeavePGroupsOut(2)
    X=X.T
    knn=KNeighborsClassifier(n_neighbors=n_neighbors)

    for i in range(X.shape[0]):

        x = X[i,:]
        x = x.reshape((-1,1))

        score_knn_val, perm_score_knn_val, pval_knn_val= permutation_test_score(knn, x, y, groups, lpgo, n_permutations=1000, n_jobs=2)

        score_knn.append(score_knn_val)
        perm_score_knn.append(perm_score_knn_val)
        pval_knn.append(pval_knn_val)

        print(i)

    return score_knn, perm_score_knn, pval_knn

def class_lda(files):

    score_lda=[]
    perm_score_lda=[]
    pval_lda=[]

    file_namesDA=['sDAdelta.npy', 'sDAtheta.npy', 'sDAalpha.npy','sDAbeta.npy','sDAlowgamma.npy']
    file_namesDAw=['sDAwdelta.npy', 'sDAwtheta.npy', 'sDAwalpha.npy','sDAwbeta.npy','sDAwlowgamma.npy']

    file_namesLA=['sLAdelta.npy', 'sLAtheta.npy', 'sLAalpha.npy','sLAbeta.npy','sLAlowgamma.npy']
    file_namesLAw=['sLAwdelta.npy', 'sLAwtheta.npy', 'sLAwalpha.npy','sLAwbeta.npy','sLAwlowgamma.npy']

    listAnest=[]
    listWake=[]

    if files =='DA':
        file_names=file_namesDA
        file_names2=file_namesDAw

    elif files =='LA':
        file_names=file_namesLA
        file_names2=file_namesLAw

    for i, j in zip(file_names, file_names2):

        listAnest.append(np.load(i, allow_pickle=True))
        listWake.append(np.load(j, allow_pickle=True))

    listAnest=np.concatenate(listAnest, axis=2)     
    listWake=np.concatenate(listWake, axis=2)

    listAnest=listAnest.reshape((-1,listAnest.shape[2]))   
    listWake=listWake.reshape((-1,listWake.shape[2]))

    X=np.concatenate((listAnest,listWake), axis=0)
    y=np.concatenate((np.zeros(listAnest.shape[0]),np.ones(listWake.shape[0])))

    if files == 'DA':
        groups=np.concatenate((np.full(150,0),np.full(150,1),np.full(150,2),np.full(150,3),np.full(150,4),np.full(150,5),np.full(150,6),np.full(150,7),np.full(150,8), np.full(112,0),np.full(112,1),np.full(112,2),np.full(112,3),np.full(112,4),np.full(112,5),np.full(112,6),np.full(112,7),np.full(112,8)))
    
    elif files == 'LA':
        groups=np.concatenate((np.full(590,0),np.full(590,1),np.full(590,2),np.full(590,3),np.full(590,4),np.full(590,5), np.full(590,6), np.full(690,0),np.full(690,1),np.full(690,2),np.full(690,3),np.full(690,4), np.full(690,5), np.full(690,6)))


    lpgo= LeavePGroupsOut(2)
    X=X.T
    lda=LinearDiscriminantAnalysis()

    for i in range(X.shape[0]):

        x = X[i,:]
        x = x.reshape((-1,1))

        score_lda_val, perm_score_lda_val, pval_lda_val= permutation_test_score(lda, x, y, groups, lpgo, n_permutations=1000, n_jobs=2)

        score_lda.append(score_lda_val)
        perm_score_lda.append(perm_score_lda_val)
        pval_lda.append(pval_lda_val)

        print(i)

    return score_lda, perm_score_lda, pval_lda

def class_qda(files):

    score_qda=[]
    perm_score_qda=[]
    pval_qda=[]

    file_namesDA=['sDAdelta.npy', 'sDAtheta.npy', 'sDAalpha.npy','sDAbeta.npy','sDAlowgamma.npy']
    file_namesDAw=['sDAwdelta.npy', 'sDAwtheta.npy', 'sDAwalpha.npy','sDAwbeta.npy','sDAwlowgamma.npy']

    file_namesLA=['sLAdelta.npy', 'sLAtheta.npy', 'sLAalpha.npy','sLAbeta.npy','sLAlowgamma.npy']
    file_namesLAw=['sLAwdelta.npy', 'sLAwtheta.npy', 'sLAwalpha.npy','sLAwbeta.npy','sLAwlowgamma.npy']

    listAnest=[]
    listWake=[]

    if files =='DA':
        file_names=file_namesDA
        file_names2=file_namesDAw

    elif files =='LA':
        file_names=file_namesLA
        file_names2=file_namesLAw

    for i, j in zip(file_names, file_names2):

        listAnest.append(np.load(i, allow_pickle=True))
        listWake.append(np.load(j, allow_pickle=True))

    listAnest=np.concatenate(listAnest, axis=2)     
    listWake=np.concatenate(listWake, axis=2)

    listAnest=listAnest.reshape((-1,listAnest.shape[2]))   
    listWake=listWake.reshape((-1,listWake.shape[2]))

    X=np.concatenate((listAnest,listWake), axis=0)
    y=np.concatenate((np.zeros(listAnest.shape[0]),np.ones(listWake.shape[0])))

    if files == 'DA':
        groups=np.concatenate((np.full(150,0),np.full(150,1),np.full(150,2),np.full(150,3),np.full(150,4),np.full(150,5),np.full(150,6),np.full(150,7),np.full(150,8), np.full(112,0),np.full(112,1),np.full(112,2),np.full(112,3),np.full(112,4),np.full(112,5),np.full(112,6),np.full(112,7),np.full(112,8)))
    
    elif files == 'LA':
        groups=np.concatenate((np.full(590,0),np.full(590,1),np.full(590,2),np.full(590,3),np.full(590,4),np.full(590,5), np.full(590,6), np.full(690,0),np.full(690,1),np.full(690,2),np.full(690,3),np.full(690,4), np.full(690,5), np.full(690,6)))

    lpgo= LeavePGroupsOut(2)
    X=X.T
    qda=QuadraticDiscriminantAnalysis()

    for i in range(X.shape[0]):

        x = X[i,:]
        x = x.reshape((-1,1))
   
        score_qda_val, perm_score_qda_val, pval_qda_val= permutation_test_score(qda, x, y, groups, lpgo, n_permutations=1000, n_jobs=2)

        score_qda.append(score_qda_val)
        perm_score_qda.append(perm_score_qda_val)
        pval_qda.append(pval_qda_val)
    
    return score_qda, perm_score_qda, pval_qda

def class_mlp(files):

    score_mlp=[]
    perm_score_mlp=[]
    pval_mlp=[]

    file_namesDA=['sDAdelta.npy', 'sDAtheta.npy', 'sDAalpha.npy','sDAbeta.npy','sDAlowgamma.npy']
    file_namesDAw=['sDAwdelta.npy', 'sDAwtheta.npy', 'sDAwalpha.npy','sDAwbeta.npy','sDAwlowgamma.npy']

    file_namesLA=['sLAdelta.npy', 'sLAtheta.npy', 'sLAalpha.npy','sLAbeta.npy','sLAlowgamma.npy']
    file_namesLAw=['sLAwdelta.npy', 'sLAwtheta.npy', 'sLAwalpha.npy','sLAwbeta.npy','sLAwlowgamma.npy']

    listAnest=[]
    listWake=[]

    if files =='DA':
        file_names=file_namesDA
        file_names2=file_namesDAw

    elif files =='LA':
        file_names=file_namesLA
        file_names2=file_namesLAw

    for i, j in zip(file_names, file_names2):

        listAnest.append(np.load(i, allow_pickle=True))
        listWake.append(np.load(j, allow_pickle=True))

    listAnest=np.concatenate(listAnest, axis=2)     
    listWake=np.concatenate(listWake, axis=2)

    listAnest=listAnest.reshape((-1,listAnest.shape[2]))   
    listWake=listWake.reshape((-1,listWake.shape[2]))

    X=np.concatenate((listAnest,listWake), axis=0)
    y=np.concatenate((np.zeros(listAnest.shape[0]),np.ones(listWake.shape[0])))

    if files == 'DA':
        groups=np.concatenate((np.full(150,0),np.full(150,1),np.full(150,2),np.full(150,3),np.full(150,4),np.full(150,5),np.full(150,6),np.full(150,7),np.full(150,8), np.full(112,0),np.full(112,1),np.full(112,2),np.full(112,3),np.full(112,4),np.full(112,5),np.full(112,6),np.full(112,7),np.full(112,8)))
    
    elif files == 'LA':
        groups=np.concatenate((np.full(590,0),np.full(590,1),np.full(590,2),np.full(590,3),np.full(590,4),np.full(590,5), np.full(590,6), np.full(690,0),np.full(690,1),np.full(690,2),np.full(690,3),np.full(690,4), np.full(690,5), np.full(690,6)))


    lpgo= LeavePGroupsOut(2)
    X=X.T
    mlp=MLPClassifier()

    for i in range(X.shape[0]):

        x = X[i,:]
        x = x.reshape((-1,1))
                
        score_mlp_val, perm_score_mlp_val, pval_mlp_val= permutation_test_score(mlp, x, y, groups, lpgo, n_permutations=1000, n_jobs=2)

        score_mlp.append(score_mlp_val)
        perm_score_mlp.append(perm_score_mlp_val)
        pval_mlp.append(pval_mlp_val)

    return score_mlp, score_mlp_val, pval_mlp



def class_sf_perm(files):

    score_clf=[]
    score_knn=[]
    score_lda=[]
    score_qda=[]
    score_mlp=[]

    perm_score_clf=[]
    perm_score_knn=[]
    perm_score_lda=[]
    perm_score_qda=[]
    perm_score_mlp=[]

    pval_clf=[]
    pval_knn=[]
    pval_lda=[]
    pval_qda=[]
    pval_mlp=[]

    file_namesDA=['sDAdelta.npy', 'sDAtheta.npy', 'sDAalpha.npy','sDAbeta.npy','sDAlowgamma.npy']
    file_namesDAw=['sDAwdelta.npy', 'sDAwtheta.npy', 'sDAwalpha.npy','sDAwbeta.npy','sDAwlowgamma.npy']

    file_namesLA=['sLAdelta.npy', 'sLAtheta.npy', 'sLAalpha.npy','sLAbeta.npy','sLAlowgamma.npy']
    file_namesLAw=['sLAwdelta.npy', 'sLAwtheta.npy', 'sLAwalpha.npy','sLAwbeta.npy','sLAwlowgamma.npy']

    listAnest=[]
    listWake=[]

    if files =='DA':
        file_names=file_namesDA
        file_names2=file_namesDAw

    elif files =='LA':
        file_names=file_namesLA
        file_names2=file_namesLAw

    for i, j in zip(file_names, file_names2):

        listAnest.append(np.load(i, allow_pickle=True))
        listWake.append(np.load(j, allow_pickle=True))

    listAnest=np.concatenate(listAnest, axis=2)     
    listWake=np.concatenate(listWake, axis=2)

    listAnest=listAnest.reshape((-1,listAnest.shape[2]))   
    listWake=listWake.reshape((-1,listWake.shape[2]))

    X=np.concatenate((listAnest,listWake), axis=0)
    y=np.concatenate((np.zeros(listAnest.shape[0]),np.ones(listWake.shape[0])))

    if files == 'DA':
        groups=np.concatenate((np.full(150,0),np.full(150,1),np.full(150,2),np.full(150,3),np.full(150,4),np.full(150,5),np.full(150,6),np.full(150,7),np.full(150,8), np.full(112,0),np.full(112,1),np.full(112,2),np.full(112,3),np.full(112,4),np.full(112,5),np.full(112,6),np.full(112,7),np.full(112,8)))
    
    elif files == 'LA':
        groups=np.concatenate((np.full(590,0),np.full(590,1),np.full(590,2),np.full(590,3),np.full(590,4),np.full(590,5), np.full(590,6), np.full(690,0),np.full(690,1),np.full(690,2),np.full(690,3),np.full(690,4), np.full(690,5), np.full(690,6)))


    lpgo= LeavePGroupsOut(2)

    X=X.T

    for i in range(X.shape[0]):

        x = X[i,:]
        x = x.reshape((-1,1))
        
            #SVM
        clf= svm.SVC(gamma='auto')
        score_clf_val, perm_score_clf_val, pval_clf_val= permutation_test_score(clf, x, y, groups, lpgo, n_permutations=1000, n_jobs=2)

        score_clf.append(score_clf_val)
        perm_score_clf.append(perm_score_clf_val)
        pval_clf.append(pval_clf_val)

            #KNN
        knn=KNeighborsClassifier()
        score_knn_val, perm_score_knn_val, pval_knn_val= permutation_test_score(knn, x, y, groups, lpgo, n_permutations=1000, n_jobs=2)

        score_knn.append(score_knn_val)
        perm_score_knn.append(perm_score_knn_val)
        pval_knn.append(pval_knn_val)

            #LDA
        lda=LinearDiscriminantAnalysis()
        score_lda_val, perm_score_lda_val, pval_lda_val= permutation_test_score(lda, x, y, groups, lpgo, n_permutations=1000, n_jobs=2)

        score_lda.append(score_lda_val)
        perm_score_lda.append(perm_score_lda_val)
        pval_lda.append(pval_lda_val)

            #QDA
        qda=QuadraticDiscriminantAnalysis()
        score_qda_val, perm_score_qda_val, pval_qda_val= permutation_test_score(qda, x, y, groups, lpgo, n_permutations=1000, n_jobs=2)

        score_qda.append(score_qda_val)
        perm_score_qda.append(perm_score_qda_val)
        pval_qda.append(pval_qda_val)

            #MLP
        mlp=MLPClassifier()
        score_mlp_val, perm_score_mlp_val, pval_mlp_val= permutation_test_score(mlp, x, y, groups, lpgo, n_permutations=1000, n_jobs=2)

        score_mlp.append(score_mlp_val)
        perm_score_mlp.append(perm_score_mlp_val)
        pval_mlp.append(pval_mlp_val)


    return score_clf, perm_score_clf, pval_clf, score_knn, perm_score_knn, pval_knn, score_lda, perm_score_lda, pval_lda, score_qda, perm_score_qda, pval_qda, score_mlp, perm_score_mlp, pval_mlp