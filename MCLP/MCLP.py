#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 12:10:32 2019

@author: Yijun Ran
"""

import pickle  
import networkx as nx
import random
import numpy as np
import similarity_nodeijs as sn
import copy
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

path_length = 4 # path length between two nodes

def uplow_bound(pos, neg):  # p1 of LP set and p2 of LN set
    p = 0
    for i in pos:
        if i > 0:
            p += 1
    n = 0
    for i in neg:
        if i > 0:
            n += 1
    return float(p)/len(pos),float(n)/len(neg)

def low_bound(p1, p2): # the AUC lower in unsupervised prediction
    return 0.5+(p1-p2-p1*p2)*0.5
    
def up_bound(p1, p2):  # the AUC upper in unsupervised prediction
    return 0.5+(p1-p2+p1*p2)*0.5

def up_bound_ml(p1, p2): # the AUC upper in supervised prediction
    return 0.5+(p1+p2-p1*p2)*0.5

# load the data 
infile = open('OLP_updated.pickle','rb')  
df = pickle.load(infile)
infile.close()  

# read edge lists for all networks
df_edgelists = df['edges_id']                               
all_name = list(df['network_name'])

data_name = ['network_name','missing_num','no_num','cn_auc','cn_auc_rf','cn_auc_gb','cn_auc_ab',
             'ra_auc','ra_auc_rf','ra_auc_gb','ra_auc_ab','aa_auc','aa_auc_rf','aa_auc_gb','aa_auc_ab',
             'salton_auc','salton_auc_rf','salton_auc_gb','salton_auc_ab','si_auc','si_auc_rf',
             'si_auc_gb','si_auc_ab','hpi_auc','hpi_auc_rf','hpi_auc_gb','hpi_auc_ab',
             'hdi_auc','hdi_auc_rf','hdi_auc_gb','hdi_auc_ab','lhni_auc','lhni_auc_rf',
             'lhni_auc_gb','lhni_auc_ab','jaccard_auc','jaccard_auc_rf','jaccard_auc_gb','jaccard_auc_ab',
             'cn_lower','cn_upper','cn_p1','cn_p2','cn_delta_auc',
             'hei_auc','hei_auc_rf','hei_auc_gb','hei_auc_ab','hoi_auc','hoi_auc_rf','hoi_auc_gb',
             'hoi_auc_ab','hei_lower','hei_upper','hei_p1','hei_p2','hei_delta_auc',
             'lp_auc','lp_auc_rf','lp_auc_gb','lp_auc_ab','katz_auc','katz_auc_rf','katz_auc_gb',
             'katz_auc_ab','fl_auc','fl_auc_rf','fl_auc_gb','fl_auc_ab','rss_auc','rss_auc_rf',
             'rss_auc_gb','rss_auc_ab','spl_auc','spl_auc_rf','spl_auc_gb','spl_auc_ab',
             'lp_lower','lp_upper','lp_p1','lp_p2','lp_delta_auc','l3_auc','l3_auc_rf',
             'l3_auc_gb','l3_auc_ab','ch2l3_auc','ch2l3_auc_rf','ch2l3_auc_gb','ch2l3_auc_ab',
             'l3_lower','l3_upper','l3_p1','l3_p2','l3_delta_auc']  
                          
for ii in range(len(df)):
    edges_orig = df_edgelists.iloc[ii] # a numpy array of edge list for original graph
    file = df['network_name'][ii]
    num_edges = df['number_edges'][ii]
    num_nodes = df['number_nodes'][ii]
    ave_degree = df['ave_degree'][ii]
    domain = df['networkDomain'][ii] 
    print (file)
    g1 = nx.Graph()
    g1.add_edges_from(edges_orig)
    AP = nx.average_shortest_path_length(g1)
    ave_clustering = nx.average_clustering(g1)    
    itel = 0
    edges = list(g1.edges())
    nodes = list(g1.nodes())
    node_num = len(list(g1.nodes()))
    edge_num = int(len(edges)*0.2) 
    edge_all = list(g1.edges())
    all_edges = list(g1.edges())
    for edge in list(g1.edges()):
        all_edges.append((edge[1],edge[0]))
########################  build all nonexistent links ####################          
#    sort_nodes = sorted(nodes)
#    all_pro_edges = list(itertools.combinations(sort_nodes, 2))
#    dif_edge = list(set(all_pro_edges).difference(set(edges)))
#    for edge in edges:
#        if edge in dif_edge:
#            dif_edge.remove(edge)
#        elif (edge[1],edge[0]) in dif_edge:
#            dif_edge.remove((edge[1],edge[0]))    
########################  end ############################################                  
    while itel < 200: # randomly generate 200 pairs of $L^P$ and $L^N$ sets in each network
        all_result = []
        cn_auc = []
        cn_auc_rf = []
        cn_auc_gb = []
        cn_auc_ab = []
        ra_auc = []
        ra_auc_rf = []
        ra_auc_gb = []
        ra_auc_ab = []
        aa_auc = []
        aa_auc_rf = []
        aa_auc_gb = []
        aa_auc_ab = []
        salton_auc = []
        salton_auc_rf = []
        salton_auc_gb = []
        salton_auc_ab = []
        si_auc = []
        si_auc_rf = []
        si_auc_gb = []
        si_auc_ab = []
        hpi_auc = []
        hpi_auc_rf = []
        hpi_auc_gb = []
        hpi_auc_ab = []
        hdi_auc = []
        hdi_auc_rf = []
        hdi_auc_gb = []
        hdi_auc_ab = []
        lhni_auc = []
        lhni_auc_rf = []
        lhni_auc_gb = []
        lhni_auc_ab = []
        jaccard_auc = []
        jaccard_auc_rf = []
        jaccard_auc_gb = []
        jaccard_auc_ab = []
        hei_auc = []
        hei_auc_rf = []
        hei_auc_gb = []
        hei_auc_ab = []
        hoi_auc = []
        hoi_auc_rf = []
        hoi_auc_gb = []
        hoi_auc_ab = []
        lp_auc = []
        lp_auc_rf = []
        lp_auc_gb = []
        lp_auc_ab = []
        katz_auc = []
        katz_auc_rf = []
        katz_auc_gb = []
        katz_auc_ab = []
        fl_auc = []
        fl_auc_rf = []
        fl_auc_gb = []
        fl_auc_ab = []
        rss_auc = []
        rss_auc_rf = []
        rss_auc_gb = []
        rss_auc_ab = []
        spl_auc = []
        spl_auc_rf = []
        spl_auc_gb = []
        spl_auc_ab = []
        l3_auc = []
        l3_auc_rf = []
        l3_auc_gb = []
        l3_auc_ab = []
        ch2l3_auc = []
        ch2l3_auc_rf = []
        ch2l3_auc_gb = []
        ch2l3_auc_ab = []
        
        cn_up_bound = []
        cn_low_bound = []
        hei_up_bound = []
        hei_low_bound = []
        lp_up_bound = []
        lp_low_bound = []
        l3_up_bound = []
        l3_low_bound = []
        
        cn_p1 = []
        cn_p2 = []
        hei_p1 = []
        hei_p2 = []
        lp_p1 = []
        lp_p2 = []
        l3_p1 = []
        l3_p2 = []
        print (itel)
        itel += 1        
        #######  build missing samples  ################################ 
        g = copy.deepcopy(g1)
        ebunch_edges = random.sample(edge_all, edge_num)
        g.remove_edges_from(ebunch_edges)
        ###################### end ################################
        
        ########################  build nonexistent samples ####################    
        dif_edge = []                             
        while len(dif_edge) < len(ebunch_edges):
            u = random.choice(nodes)# random a node
            v = random.choice(nodes)
            if u != v and (u,v) not in all_edges and (v,u) not in all_edges and (u,v) not in dif_edge and (v,u) not in dif_edge:
                dif_edge.append((u,v)) 
        ###################### end ################################
        
        snum = int(len(ebunch_edges)*0.5) # 0.5 or 0.8
        gnum = int(len(dif_edge)*0.5) # 0.5 or 0.8
        cn_pos = []
        cn_neg = [] 
        ra_pos = []
        ra_neg = [] 
        aa_pos = []
        aa_neg = []
        salton_pos = []
        salton_neg = [] 
        si_pos = []
        si_neg = [] 
        hpi_pos = []
        hpi_neg = []
        hdi_pos = []
        hdi_neg = []
        lhni_pos = []
        lhni_neg = [] 
        jaccard_pos = []
        jaccard_neg = []   
        hei_pos = []
        hei_neg = []
        hoi_pos = []
        hoi_neg = []
        lp_pos = []
        lp_neg = []
        katz_pos = []
        katz_neg = []
        fl_pos = []
        fl_neg = []
        rss_pos = []
        rss_neg = []
        spl_pos = []
        spl_neg = []
        l3_pos = []
        l3_neg = []
        ch2l3_pos = []
        ch2l3_neg = []
        
        for i in ebunch_edges:
            cn, ra, aa, salton, si, hpi, hdi, lhni, jaccard, hei, hoi = sn.CNI(g, i, 0.02)
            lp, fl, katz, rss, spl, l3, ch2l3 = sn.Localpath(g, i, 0.02, path_length, node_num)
            cn_pos.append((i, cn))
            ra_pos.append((i, ra))
            aa_pos.append((i, aa))
            salton_pos.append((i, salton))
            si_pos.append((i, si))
            hpi_pos.append((i, hpi))
            hdi_pos.append((i, hdi))
            lhni_pos.append((i, lhni))
            jaccard_pos.append((i, jaccard))
            hei_pos.append((i, hei))
            hoi_pos.append((i, hoi))
            lp_pos.append((i, lp))
            katz_pos.append((i, katz))
            fl_pos.append((i, fl))
            rss_pos.append((i, rss))
            spl_pos.append((i, spl))
            l3_pos.append((i, l3))
            ch2l3_pos.append((i, ch2l3))
                
        for j in dif_edge:
            cn, ra, aa, salton, si, hpi, hdi, lhni, jaccard, hei, hoi = sn.CNI(g, j, 0.02)
            lp, fl, katz, rss, spl, l3, ch2l3 = sn.Localpath(g, j, 0.02, path_length, node_num)
            cn_neg.append((j, cn))
            ra_neg.append((j, ra))
            aa_neg.append((j, aa))
            salton_neg.append((j, salton))
            si_neg.append((j, si))
            hpi_neg.append((j, hpi))
            hdi_neg.append((j, hdi))
            lhni_neg.append((j, lhni))
            jaccard_neg.append((j, jaccard))
            hei_neg.append((j, hei))
            hoi_neg.append((j, hoi))
            lp_neg.append((j, lp))
            katz_neg.append((j, katz))
            fl_neg.append((j, fl))
            rss_neg.append((j, rss))
            spl_neg.append((j, spl))
            l3_neg.append((j, l3))
            ch2l3_neg.append((j, ch2l3))
        
        cn_score_pos = []
        for score in cn_pos[snum:]:
            cn_score_pos.append(score[1])
        cn_score_neg = []
        for score in cn_neg[gnum:]:
            cn_score_neg.append(score[1])
        
        lp_score_pos = []
        for score in lp_pos[snum:]:
            lp_score_pos.append(score[1])
        lp_score_neg = []
        for score in lp_neg[gnum:]:
            lp_score_neg.append(score[1])
            
        l3_score_pos = []
        for score in l3_pos[snum:]:
            l3_score_pos.append(score[1])
        l3_score_neg = []
        for score in l3_neg[gnum:]:
            l3_score_neg.append(score[1])    
            
        hei_score_pos = []
        for score in hei_pos[snum:]:
            hei_score_pos.append(score[1])
        hei_score_neg = []
        for score in hei_neg[gnum:]:
            hei_score_neg.append(score[1])
            
        cn_auc.append(sn.AUC(cn_pos[snum:], cn_neg[gnum:]))
        ra_auc.append(sn.AUC(ra_pos[snum:], ra_neg[gnum:]))
        aa_auc.append(sn.AUC(aa_pos[snum:], aa_neg[gnum:]))
        salton_auc.append(sn.AUC(salton_pos[snum:], salton_neg[gnum:]))
        si_auc.append(sn.AUC(si_pos[snum:], si_neg[gnum:]))
        hpi_auc.append(sn.AUC(hpi_pos[snum:], hpi_neg[gnum:]))
        hdi_auc.append(sn.AUC(hdi_pos[snum:], hdi_neg[gnum:]))
        lhni_auc.append(sn.AUC(lhni_pos[snum:], lhni_neg[gnum:]))
        jaccard_auc.append(sn.AUC(jaccard_pos[snum:], jaccard_neg[gnum:]))
        hei_auc.append(sn.AUC(hei_pos[snum:], hei_neg[gnum:]))
        hoi_auc.append(sn.AUC(hoi_pos[snum:], hoi_neg[gnum:]))
        lp_auc.append(sn.AUC(lp_pos[snum:], lp_neg[gnum:]))
        katz_auc.append(sn.AUC(katz_pos[snum:], katz_neg[gnum:]))
        fl_auc.append(sn.AUC(fl_pos[snum:], fl_neg[gnum:]))
        rss_auc.append(sn.AUC(rss_pos[snum:], rss_neg[gnum:]))
        spl_auc.append(sn.AUC(spl_pos[snum:], spl_neg[gnum:]))
        l3_auc.append(sn.AUC(l3_pos[snum:], l3_neg[gnum:]))
        ch2l3_auc.append(sn.AUC(ch2l3_pos[snum:], ch2l3_neg[gnum:]))
        
        cn_p1_p2 = uplow_bound(cn_score_pos, cn_score_neg)
        cn_up_bound.append(up_bound(cn_p1_p2[0], cn_p1_p2[1]))
        cn_low_bound.append(low_bound(cn_p1_p2[0], cn_p1_p2[1]))
        cn_p1.append(cn_p1_p2[0])  
        cn_p2.append(cn_p1_p2[1])
        
        hei_p1_p2 = uplow_bound(hei_score_pos, hei_score_neg)
        hei_up_bound.append(up_bound(hei_p1_p2[0], hei_p1_p2[1]))
        hei_low_bound.append(low_bound(hei_p1_p2[0], hei_p1_p2[1]))
        hei_p1.append(hei_p1_p2[0])  
        hei_p2.append(hei_p1_p2[1])
        
        lp_p1_p2 = uplow_bound(lp_score_pos, lp_score_neg)
        lp_up_bound.append(up_bound(lp_p1_p2[0], lp_p1_p2[1]))
        lp_low_bound.append(low_bound(lp_p1_p2[0], lp_p1_p2[1]))
        lp_p1.append(lp_p1_p2[0])
        lp_p2.append(lp_p1_p2[1])
        
        l3_p1_p2 = uplow_bound(l3_score_pos, l3_score_neg)
        l3_up_bound.append(up_bound(l3_p1_p2[0], l3_p1_p2[1]))
        l3_low_bound.append(low_bound(l3_p1_p2[0], l3_p1_p2[1]))
        l3_p1.append(l3_p1_p2[0])
        l3_p2.append(l3_p1_p2[1])
        
        cn_list = [[soc[1] for soc in cn_pos]+[soc[1] for soc in cn_neg],[soc[1] for soc in ra_pos]+[soc[1] for soc in ra_neg],
                      [soc[1] for soc in aa_pos]+[soc[1] for soc in aa_neg],[soc[1] for soc in salton_pos]+[soc[1] for soc in salton_neg],
                      [soc[1] for soc in si_pos]+[soc[1] for soc in si_neg],[soc[1] for soc in hpi_pos]+[soc[1] for soc in hpi_neg],
                      [soc[1] for soc in hdi_pos]+[soc[1] for soc in hdi_neg],[soc[1] for soc in lhni_pos]+[soc[1] for soc in lhni_neg],
                      [soc[1] for soc in jaccard_pos]+[soc[1] for soc in jaccard_neg]]
        
        hei_list = [[soc[1] for soc in hei_pos]+[soc[1] for soc in hei_neg], [soc[1] for soc in hoi_pos]+[soc[1] for soc in hoi_neg]]
        
        lp_list = [[soc[1] for soc in lp_pos]+[soc[1] for soc in lp_neg],[soc[1] for soc in katz_pos]+[soc[1] for soc in katz_neg],
                      [soc[1] for soc in fl_pos]+[soc[1] for soc in fl_neg],[soc[1] for soc in rss_pos]+[soc[1] for soc in rss_neg],
                      [soc[1] for soc in spl_pos]+[soc[1] for soc in spl_neg]]
        
        l3_list = [[soc[1] for soc in l3_pos]+[soc[1] for soc in l3_neg], [soc[1] for soc in ch2l3_pos]+[soc[1] for soc in ch2l3_neg]]
        feature_list = cn_list + lp_list + hei_list + l3_list

        pn = len(lp_pos[snum:])
        nn = len(lp_neg[gnum:])
        train_y = np.array([1 for i in range(snum)]+ [0 for i in range(gnum)])
        test_y = np.array([1 for i in range(pn)]+ [0 for i in range(nn)])        
        
        n_est = [50, 100, 150]
        m_depth = [5, 15, 25, 45]
        result = []
        for feature in feature_list:
            train_x = np.array(feature[:snum] + feature[snum+pn:snum+pn+gnum]).reshape(-1,1)
            test_x = np.array(feature[snum:snum+pn] + feature[snum+pn+gnum:snum+pn+gnum+nn]).reshape(-1,1)
            auc_max = []
            for ii in n_est:
                for jj in m_depth:                    
                    clf_rf = RandomForestClassifier(n_estimators = ii, max_depth = jj)
                    clf_rf.fit(train_x, train_y)
                    clf_proba = clf_rf.predict_proba(test_x)
                    auc = roc_auc_score(test_y, clf_proba[:,1])
                    auc_max.append(auc)
            result.append(max(auc_max))
        cn_auc_rf.append(result[0])
        ra_auc_rf.append(result[1])
        aa_auc_rf.append(result[2])
        salton_auc_rf.append(result[3])
        si_auc_rf.append(result[4])
        hpi_auc_rf.append(result[5])
        hdi_auc_rf.append(result[6])
        lhni_auc_rf.append(result[7])
        jaccard_auc_rf.append(result[8])
        lp_auc_rf.append(result[9])
        katz_auc_rf.append(result[10])
        fl_auc_rf.append(result[11])
        rss_auc_rf.append(result[12])
        spl_auc_rf.append(result[13])
        hei_auc_rf.append(result[14])
        hoi_auc_rf.append(result[15])
        l3_auc_rf.append(result[16])
        ch2l3_auc_rf.append(result[17])
        
        n_est = [50, 100, 150]
        m_lr = [0.2, 0.4, 0.6, 0.8]
        result = []
        for feature in feature_list:
            train_x = np.array(feature[:snum] + feature[snum+pn:snum+pn+gnum]).reshape(-1,1)
            test_x = np.array(feature[snum:snum+pn] + feature[snum+pn+gnum:snum+pn+gnum+nn]).reshape(-1,1)
            auc_max = []
            for ii in n_est:
                for jj in m_lr:                    
                    clf_rf = GradientBoostingClassifier(n_estimators = ii, learning_rate = jj)
                    clf_rf.fit(train_x, train_y)
                    clf_proba = clf_rf.predict_proba(test_x)
                    auc = roc_auc_score(test_y, clf_proba[:,1])
                    auc_max.append(auc)
            result.append(max(auc_max))
        cn_auc_gb.append(result[0])
        ra_auc_gb.append(result[1])
        aa_auc_gb.append(result[2])
        salton_auc_gb.append(result[3])
        si_auc_gb.append(result[4])
        hpi_auc_gb.append(result[5])
        hdi_auc_gb.append(result[6])
        lhni_auc_gb.append(result[7])
        jaccard_auc_gb.append(result[8])
        lp_auc_gb.append(result[9])
        katz_auc_gb.append(result[10])
        fl_auc_gb.append(result[11])
        rss_auc_gb.append(result[12])
        spl_auc_gb.append(result[13])
        hei_auc_gb.append(result[14])
        hoi_auc_gb.append(result[15])
        l3_auc_gb.append(result[16])
        ch2l3_auc_gb.append(result[17])
        
        n_est = [50, 100, 150]
        m_lr = [0.2, 0.4, 0.6, 0.8]
        result = []
        for feature in feature_list:
            train_x = np.array(feature[:snum] + feature[snum+pn:snum+pn+gnum]).reshape(-1,1)
            test_x = np.array(feature[snum:snum+pn] + feature[snum+pn+gnum:snum+pn+gnum+nn]).reshape(-1,1)
            auc_max = []
            for ii in n_est:
                for jj in m_lr:                    
                    clf_rf = AdaBoostClassifier(n_estimators = ii, learning_rate = jj)
                    clf_rf.fit(train_x, train_y)
                    clf_proba = clf_rf.predict_proba(test_x)
                    auc = roc_auc_score(test_y, clf_proba[:,1])
                    auc_max.append(auc)
            result.append(max(auc_max))
        cn_auc_ab.append(result[0])
        ra_auc_ab.append(result[1])
        aa_auc_ab.append(result[2])
        salton_auc_ab.append(result[3])
        si_auc_ab.append(result[4])
        hpi_auc_ab.append(result[5])
        hdi_auc_ab.append(result[6])
        lhni_auc_ab.append(result[7])
        jaccard_auc_ab.append(result[8])
        lp_auc_ab.append(result[9])
        katz_auc_ab.append(result[10])
        fl_auc_ab.append(result[11])
        rss_auc_ab.append(result[12])
        spl_auc_ab.append(result[13])
        hei_auc_ab.append(result[14])
        hoi_auc_ab.append(result[15])
        l3_auc_ab.append(result[16])
        ch2l3_auc_ab.append(result[17])       
    
        result_auc = [file,pn,nn,cn_auc[0],cn_auc_rf[0],cn_auc_gb[0],cn_auc_ab[0],
                  ra_auc[0],ra_auc_rf[0],ra_auc_gb[0],ra_auc_ab[0],aa_auc[0],aa_auc_rf[0],aa_auc_gb[0],
                  aa_auc_ab[0],salton_auc[0],salton_auc_rf[0],salton_auc_gb[0],salton_auc_ab[0],si_auc[0],
                  si_auc_rf[0],si_auc_gb[0],si_auc_ab[0],hpi_auc[0],hpi_auc_rf[0],hpi_auc_gb[0],hpi_auc_ab[0],
                  hdi_auc[0],hdi_auc_rf[0],hdi_auc_gb[0],hdi_auc_ab[0],lhni_auc[0],lhni_auc_rf[0],lhni_auc_gb[0],
                  lhni_auc_ab[0],jaccard_auc[0],jaccard_auc_rf[0],jaccard_auc_gb[0],jaccard_auc_ab[0],cn_low_bound[0],
                  cn_up_bound[0],cn_p1[0],cn_p2[0],cn_p1[0]*cn_p2[0],
                  hei_auc[0],hei_auc_rf[0],hei_auc_gb[0],
                  hei_auc_ab[0],hoi_auc[0],hoi_auc_rf[0],hoi_auc_gb[0],hoi_auc_ab[0],hei_low_bound[0],
                  hei_up_bound[0],hei_p1[0],hei_p2[0],hei_p1[0]*hei_p2[0],lp_auc[0],lp_auc_rf[0],lp_auc_gb[0],
                  lp_auc_ab[0],katz_auc[0],katz_auc_rf[0],katz_auc_gb[0],katz_auc_ab[0],
                  fl_auc[0],fl_auc_rf[0],fl_auc_gb[0],fl_auc_ab[0],rss_auc[0],rss_auc_rf[0],
                  rss_auc_gb[0],rss_auc_ab[0],spl_auc[0],spl_auc_rf[0],spl_auc_gb[0],spl_auc_ab[0],
                  lp_low_bound[0],lp_up_bound[0],lp_p1[0],lp_p2[0],lp_p1[0]*lp_p2[0],
                  l3_auc[0],l3_auc_rf[0],l3_auc_gb[0],l3_auc_ab[0],
                  ch2l3_auc[0],ch2l3_auc_rf[0],ch2l3_auc_gb[0],ch2l3_auc_ab[0],
                  l3_low_bound[0],l3_up_bound[0],l3_p1[0],l3_p2[0],l3_p1[0]*l3_p2[0]] 
        all_result.append(result_auc)
        data = pd.DataFrame(all_result)
        data.columns = data_name
        data.to_csv('network_mls0.5_550_kfold.csv', index = None, mode = 'a', header = False)
        