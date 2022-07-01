#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 17:48:34 2019

@author: Yijun Ran
"""
############################link prediction project#############################

#########################import library#####################################
import networkx as nx
import numpy


####################### CN-based and Heterogeneity-based function ######################################
def CNI(G, nodeij, beta):
    """
    CNI(G, nodeij, beta)   
    
    calculate the number of common neighors between two nodes and more 
    
    Parameters
    ----------  
    G             - networkx graph 
    nodeij        - pairs of nodes such as (1, 2)
    beta          - a free heterogeneity exponent
    """
    node_i = nodeij[0]
    node_j = nodeij[1]
    degree_i = G.degree(node_i)
    degree_j = G.degree(node_j)
    neigh_i = set(G.neighbors(node_i))
    neigh_j = set(G.neighbors(node_j))
    neigh_ij = neigh_i.intersection(neigh_j) # common neighbors
    num_cn = len(neigh_ij)
    salton = 0.0
    si = 0.0
    hpi = 0.0
    hdi = 0.0
    lhni = 0.0
    jaccard = 0.0
    aa = 0.0
    ra = 0.0
    hei = abs(degree_i - degree_j)**beta
    hoi = 0.0
    if hei != 0:
        hoi = 1/(hei)
    salton_degree = numpy.math.sqrt(degree_i*degree_j)
    if salton_degree != 0:
        salton = float(num_cn)/salton_degree    
    si_degree = degree_i + degree_j   
    if si_degree != 0:
        si = float(2*num_cn)/si_degree
    hpi_degree = min(degree_i, degree_j)   
    if hpi_degree != 0:
        hpi = float(num_cn)/hpi_degree 
    hdi_degree = max(degree_i, degree_j)  
    if hdi_degree != 0:
        hdi = float(num_cn)/hdi_degree
    lhni_degree = degree_i*degree_j  
    if lhni_degree != 0:
        lhni = float(num_cn)/lhni_degree
    jaccard_num = len(neigh_i.union(neigh_j))   
    if jaccard_num != 0:
        jaccard = float(num_cn)/jaccard_num 
    if len(neigh_ij) > 0:
        for k in neigh_ij:
            degree_k = G.degree(k)
            if degree_k > 1:
                aa = aa+1/(numpy.math.log10(degree_k))    #log10 or log
            if degree_k > 0:
                ra = ra+1.0/degree_k 
    return num_cn, ra, aa, salton, si, hpi, hdi, lhni, jaccard, hei, hoi   
  
########################### end function #########################################

####################### Path-based and L3-based function ######################################
def Localpath(G, nodeij, beta, l, n):
    """
    Localpath(G, nodeij, beta, l, n)    
    
    calculate the number of path between two nodes and more 
    
    Parameters
    ----------  
    G             - networkx graph 
    nodeij        - pairs of nodes such as (1, 2)    
    beta          - controls the weight of paths with different lengths
    l             - controls path length between two nodes
    n             - the number of nodes in a network
    """
    
    node_i = nodeij[0]
    node_j = nodeij[1]
    lp = 0.0
    fl = 0.0
    katz = 0.0
    rss = 0.0
    spl = 0.0
    l3 = 0.0
    ch2l3 = 0.0
    try:
        short_path_length = nx.shortest_path_length(G, source = node_i, target = node_j)
        if short_path_length <= l:
            spl = 1/(short_path_length - 1)
            paths = list(nx.all_simple_paths(G, source = node_i, target = node_j, cutoff = l))
            for i in list(range(2, l+1)):
                num = 1
                path_fl = 0
                path_katz = 0
                path_lp = 0
                for path in paths:
                    if (len(path)-1) == i:
                        path_fl += 1
                        path_katz += 1
                        path_lp += 1
                        nn = 1
                        for node in path:
                            nei_num = G.degree(node)
                            nn *= beta/(nei_num*beta)
                        rss += nn                        
                for j in list(range(2, i+1)):
                    num *= (n - j)
                fl += (1/(i-1))*(path_fl/num)
                katz += (beta**i)*path_katz
                lp += (beta**(i-2))*path_lp
            all_path_nodes = []
            for path in paths:
                all_path_nodes.extend(path)
            uniall_path_nodes = list(set(all_path_nodes))
            uniall_path_nodes.remove(node_i)
            uniall_path_nodes.remove(node_j)
            for path in paths:
                if (len(path)-1) == 3:
                    ku = 1
                    iuv = []
                    euv = []
                    for node in path:
                        if node != node_i and node != node_j:
                            ku *= G.degree(node)
                            neis = list(G.neighbors(node))
                            for nij in [node_i, node_j]:
                                if nij in neis:
                                    neis.remove(nij)
                            iuvnei = set(neis).intersection(set(uniall_path_nodes)) # internal nodes
                            euvnei = set(neis).difference(iuvnei) # external nodes
                            iuv.append(len(iuvnei))
                            euv.append(len(euvnei))
                    l3 += (1/numpy.sqrt(ku))
                    ch2l3 += (numpy.sqrt((1+iuv[0])*(1+iuv[1]))/numpy.sqrt((1+euv[0])*(1+euv[1])))
        else:
            pass
    except:
        pass
    return lp, fl, katz, rss, spl, l3, ch2l3 
########################### end function #####################################################
    
####################### AUC function ######################################
def AUC(real_edges, false_edges):
    """
    AUC(real_edges, false_edges)
    
    calculate AUC   
    
    Parameters
    ----------  
    real_edges:        - the score of node pair in missing links
    false_edges:       - the score of node pair in nonexistent links    
    """
    AUC_real = 0
    AUC_false = 0
    n = len(real_edges)*len(false_edges)
    for score_p in real_edges:
        for score_n in false_edges:
            if score_p[1] > score_n[1]:
                AUC_real += 1
            elif score_p[1] == score_n[1]:
                AUC_false += 1
            
    return float(AUC_real + 0.5*AUC_false)/n 

####################### end function ######################################
