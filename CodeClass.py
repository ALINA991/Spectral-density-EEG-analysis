import Class_sf_perm
import Examine
import TopoPB 

#score_clf_DA, perm_score_clf_DA, pval_clf_DA, score_knn_DA, perm_score_knn_DA, pval_knn_DA, score_lda_DA, perm_score_lda_DA, pval_lda_DA, score_qda_DA, perm_score_qda_DA, pval_qda_DA, score_mlp_DA, perm_score_mlp_DA, pval_mlp_DA= Class2.class_sf_perm('DA') 

#score_clf_LA, perm_score_clf_LA, pval_clf_LA, score_knn_LA, perm_score_knn_LA, pval_knn_LA, score_lda_LA, perm_score_lda_LA, pval_lda_LA, score_qda_LA, perm_score_qda_LA, pval_qda_LA, score_mlp_LA, perm_score_mlp_LA, pval_mlp_LA= Class2.class_sf_perm('LA') 

score_clf_DA, perm_score_clf_DA, pval_clf_DA= Class_sf_perm.class_clf('DA')
score_knn_DA, perm_score_knn_DA, pval_knn_DA= Class_sf_perm.class_knn('DA')
score_lda_DA, perm_score_lda_DA, pval_lda_DA= Class_sf_perm.class_lda('DA')
score_qda_DA, perm_score_qda_DA, pval_qda_DA= Class_sf_perm.class_qda('DA')
score_mlp_DA, perm_score_mlp_DA, pval_mlp_DA= Class_sf_perm.class_mlp('DA')

score_lda_LA, perm_score_lda_LA, pval_lda_LA= Class_sf_perm.class_lda('LA')
score_knn_LA3, perm_score_knn_LA3, pval_knn_LA3= Class_sf_perm.class_knn('LA', n_neighbors=3)
score_knn_LA5, perm_score_knn_LA5, pval_knn_LA5= Class_sf_perm.class_knn('LA', n_neighbors=5)
score_knn_LA10, perm_score_knn_LA10, pval_knn_LA10= Class_sf_perm.class_knn('LA', n_neighbors=10)
score_knn_LA20, perm_score_knn_LA20, pval_knn_LA20= Class_sf_perm.class_knn('LA', n_neighbors=20)

maxidist_permscore_ldaDA, quant_ldaDA =Examine.maxi_distribution(perm_score_ldaDA)
delta, theta, alpha, beta, lowgamma= Examine.split_PB(score_ldaDA, 85) 
splitlist_scoreldaDA= Examine.appendd(delta, theta, alpha, beta, lowgamma)

delta, theta, alpha, beta, lowgamma= Examine.split_PB(score_knn_DA20, 85) 
split_score_knnDA20= Examine.appendd(delta, theta, alpha, beta, lowgamma)
TopoPB.array_topoplot(split_score_knnDA20, sensors_pos_DA, vmin=0, vmax=1)

delta, theta, alpha, beta, lowgamma= Examine.split_PB(score_knn_DA3, 85) 
split_score_knnDA3= Examine.appendd(delta, theta, alpha, beta, lowgamma)
TopoPB.array_topoplot(split_score_knnDA3, sensors_pos_DA, vmin=0, vmax=1)

delta, theta, alpha, beta, lowgamma= Examine.split_PB(score_knn_DA10, 85) 
split_score_knnDA10= Examine.appendd(delta, theta, alpha, beta, lowgamma)
TopoPB.array_topoplot(split_score_knnDA10, sensors_pos_DA, vmin=0, vmax=1)

delta, theta, alpha, beta, lowgamma= Examine.split_PB(score_knn_DA5, 85) 
split_score_knnDA5= Examine.appendd(delta, theta, alpha, beta, lowgamma)
TopoPB.array_topoplot(split_score_knnDA5, sensors_pos_DA, vmin=0, vmax=1)


split_score_knnLA3= Examine.split_appendd(score_knn_LA3, 63)
split_score_knnLA5= Examine.split_appendd(score_knn_LA5, 63)
split_score_knnLA10= Examine.split_appendd(score_knn_LA10, 63)
split_score_knnLA20= Examine.split_appendd(score_knn_LA20, 63)

TopoPB.array_topoplot(split_score_knnLA3, sensors_pos_LA, vmin=0, vmax=1)
TopoPB.array_topoplot(split_score_knnLA5, sensors_pos_LA, vmin=0, vmax=1)
TopoPB.array_topoplot(split_score_knnLA10, sensors_pos_LA, vmin=0, vmax=1)
TopoPB.array_topoplot(split_score_knnLA20, sensors_pos_LA, vmin=0, vmax=1)


TopoPB.array_plot_deccaccu(splitlist_scoreldaDA,sensors_pos_DA, maskk=True, DA_thr=quant_ldaDA[0], marker='x',markersize=3, cmap='YlOrRd', vmin=0.5, vmax=1)
TopoPB.array_plot_deccaccu(splitlist_scoreldaDA,sensors_pos_DA, maskk=True, DA_thr=quant_ldaDA[1], marker='x',markersize=3, cmap='YlOrRd', vmin=0.5, vmax=1)
TopoPB.array_plot_deccaccu(splitlist_scoreldaDA,sensors_pos_DA, maskk=True, DA_thr=quant_ldaDA[2], marker='x',markersize=3, cmap='YlOrRd', vmin=0.5, vmax=1)


maxidist_knnDA, quant_knnDA= Examine.maxi_distribution(perm_score_knnDA)
splitlist_scoreknnDA= Examine.split_appendd(score_knnDA, 85)

TopoPB.array_plot_deccaccu(splitlist_scoreknnDA,sensors_pos_DA, maskk=True, DA_thr=quant_knnDA[0], marker='x',markersize=3, cmap='YlOrRd', vmin=0.5, vmax=1)
TopoPB.array_plot_deccaccu(splitlist_scoreknnDA,sensors_pos_DA, maskk=True, DA_thr=quant_knnDA[1], marker='x',markersize=3, cmap='YlOrRd', vmin=0.5, vmax=1)
TopoPB.array_plot_deccaccu(splitlist_scoreknnDA,sensors_pos_DA, maskk=True, DA_thr=quant_knnDA[2], marker='x',markersize=3, cmap='YlOrRd', vmin=0.5, vmax=1)

TopoPB.array_plot_deccaccu(splitlist_scoreldaLA,sensors_pos_LA, maskk=True, DA_thr=quant_ldaLA[0], marker='x',markersize=3, cmap='YlOrRd', vmin=0.5, vmax=1)
TopoPB.array_plot_deccaccu(splitlist_scoreldaLA,sensors_pos_LA, maskk=True, DA_thr=quant_ldaLA[1], marker='x',markersize=3, cmap='YlOrRd', vmin=0.5, vmax=1)
TopoPB.array_plot_deccaccu(splitlist_scoreldaLA,sensors_pos_LA, maskk=True, DA_thr=quant_ldaLA[2], marker='x',markersize=3, cmap='YlOrRd', vmin=0.5, vmax=1)


maxi_dist_knn_LA, quant_knn_LA= Examine.maxi_distribution(perm_score_knn_LA)
splitlist_scoreknnLA= Examine.split_appendd(score_knn_LA, 63)









