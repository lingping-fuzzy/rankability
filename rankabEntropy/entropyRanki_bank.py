# CFB-Rank-EloCorr: CFB Rankability and Elo Correlation
#
# Author: Thomas R. Cameron
# Date: 11/1/2019
from rankability import specR
from copy import deepcopy
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.preprocessing import normalize

# Elo constants
K = 32.
H = 2.
X = 1000.
loc = 'D:\\workspace\\python\\temp\\specR-master\\'
###########################################################
#                       cfbData                           #
###########################################################
#   Reads CFB data from a certain year and conference.
#   Returns Elo Correlation and rankability for that year
#   and conference.
#   The opt variable determines how the Elo Correlation
#   is measured with either kendalltau (KT), spearmanr (SR),
#   or pearsonr (PR).
###########################################################
def cfbData(conf,year,opt):
    # open file
    f = open((loc+'/DataFiles/CFB/'+str(conf)+'/'+str(year)+'games.txt'))
    # read all lines
    lineList = f.readlines()
    # date, team, and score info
    numGames = len(lineList)
    date=[]; teami = []; scorei = []; teamj = []; scorej = []
    for k in range(numGames):
        row = lineList.pop(0)
        row = row.split(",")
        date.append(eval(row[0]))
        teami.append(eval(row[2]))
        scorei.append(eval(row[4]))
        teamj.append(eval(row[5]))
        scorej.append(eval(row[7]))
    # populate matches, adj, elo_rating, elo_corr, and rankability
    numTeams = max(max(teami),max(teamj))
    matches = [np.zeros((numTeams,numTeams))]
    elo_rating = [np.zeros(numTeams)]
    adj = [np.zeros((numTeams,numTeams))]
    rankability = []
    entropybility =[]
    elo_corr = []
    for k in range(numGames):
        i = teami[k] - 1
        j = teamj[k] - 1
        if(scorei[k]>scorej[k]):
            # team i beat team j
            matches[-1][i,j] = matches[-1][i,j] + 1
            # update team i Elo rating
            d = elo_rating[-1][i] - elo_rating[-1][j]
            u = 1./(1.+10.**(-d/X))
            elo_rating[-1][i] = elo_rating[-1][i] + K*(1.-u)
            # update player j Elo rating
            elo_rating[-1][j] = elo_rating[-1][j] + K*(u-1.)
        elif(scorei[k]<scorej[k]):
            # team j beat team i
            matches[-1][j,i] = matches[-1][j,i] + 1
            # update team j Elo rating
            d = elo_rating[-1][j] - elo_rating[-1][i]
            u = 1./(1.+10.**(-d/X))
            elo_rating[-1][j] = elo_rating[-1][j] + K*(1.-u)
            # update team i Elo rating
            elo_rating[-1][i] = elo_rating[-1][i] + K*(u-1.)
        else:
            # team i and team j tied
            matches[-1][i,j] = matches[-1][i,j] + 0.5
            matches[-1][j,i] = matches[-1][j,i] + 0.5
            # update team i Elo rating
            d = elo_rating[-1][i] - elo_rating[-1][j]
            u = 1./(1.+10.**(-d/X))
            elo_rating[-1][i] = elo_rating[-1][i] + K*(0.5-u)
            # update team j Elo rating
            elo_rating[-1][j] = elo_rating[-1][j] + K*(u-0.5)
        # next round
        if(k<(numGames-1) and (date[k+1]-date[k]+1)>4):
            # update adjacency matrix
            for i in range(numTeams):
                for j in range(i+1,numTeams):
                    total = matches[-1][i,j] + matches[-1][j,i]
                    if(total>0):
                        adj[-1][i,j] = matches[-1][i,j]/total
                        adj[-1][j,i] = matches[-1][j,i]/total
            # Rankability
            rankability.append(specR(adj[-1]))
            entropybility.append(feature_selection_sim(adj[-1]))
            # Elo Correlation
            if(len(elo_rating)>1 and opt=="SR"):
                corr,pval = spearmanr(elo_rating[-1],elo_rating[-2])
                elo_corr.append(corr)
            elif(len(elo_rating)>1 and opt=="KT"):
                corr,pval = kendalltau(elo_rating[-1],elo_rating[-2])
                elo_corr.append(corr)
            elif(len(elo_rating)>1 and opt=="PR"):
                corr,pval = pearsonr(elo_rating[-1],elo_rating[-2])
                elo_corr.append(corr)
            # add next rounds matches, elo_rating, and adjacency storage
            matches.append(deepcopy(matches[-1]))
            elo_rating.append(deepcopy(elo_rating[-1]))
            adj.append(np.zeros((numTeams,numTeams)))
        # last round
        if(k==(numGames-1)):
            # update adjacency matrix
            for i in range(numTeams):
                for j in range(i+1,numTeams):
                    total = matches[-1][i,j] + matches[-1][j,i]
                    if(total>0):
                        adj[-1][i,j] = matches[-1][i,j]/total
                        adj[-1][j,i] = matches[-1][j,i]/total
            # Rankability
            rankability.append(specR(adj[-1]))
            entropybility.append(feature_selection_sim(adj[-1], drawfig=False, year=year))
            # print(adj[-1].shape, '---------------')
            # with open(('tempdata/'+str(year)+'Atlcost.npy'), 'wb') as f:
            #     np.save(f, adj[-1])
            # Elo Correlation
            if(len(elo_rating)>1 and opt=="SR"):
                corr,pval = spearmanr(elo_rating[-1],elo_rating[-2])
                elo_corr.append(corr)
            elif(len(elo_rating)>1 and opt=="KT"):
                corr,pval = kendalltau(elo_rating[-1],elo_rating[-2])
                elo_corr.append(corr)
            elif(len(elo_rating)>1 and opt=="PR"):
                corr,pval = pearsonr(elo_rating[-1],elo_rating[-2])
                elo_corr.append(corr)
    # return
    return elo_corr, rankability, elo_rating[-1], entropybility

def drawgraph(specRdata, id = 0, name = 'bank'):
    import networkx as nx
    import matplotlib.pyplot as plt
    # %matplotlib inline
    G = nx.DiGraph(specRdata)
    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, cmap=plt.cm.get_cmap('jet'), node_size=0)
    labels = {i: i + 1 for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=30)
    nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color="tab:blue", arrows=True, arrowsize=15)
    plt.savefig(('tempdata/'+str(name)+'graph.png'), dpi=300, bbox_inches='tight')


def feature_selection_sim(matrix, measure='luca', p=1, drawfig = False, year = None):
    # Feature selection method using similarity measure and fuzzy entroropy
    # measures based on the article:

    # P. Luukka, (2011) Feature Selection Using Fuzzy Entropy Measures with
    # Similarity Classifier, Expert Systems with Applications, 38, pp. 4600-4607
    import pandas as pd
    data = pd.DataFrame(matrix)
    m = data.shape[0]  # -samples
    t = data.shape[1]   # -features

    if drawfig == True:
        drawgraph(data, name=year)

    sim = data
    sim = normalize(sim, axis=1, norm='l1')
    if measure == 'luca':
        # moodifying zero and one values of the similarity values to work with
        # De Luca's entropy measure
        delta = 1e-10
        sim[sim == 0] = delta
        sim[sim == 1] = 1 - delta

        H = (-sim * np.log(sim) - (1 - sim) * np.log(1 - sim)).sum(axis=1)  #better
        # H = (np.sin(np.pi / 2 * sim) + np.sin(np.pi / 2 * (1 - sim)) - 1).sum(axis=1)
    elif measure == 'park':
        H = (np.sin(np.pi / 2 * sim) + np.sin(np.pi / 2 * (1 - sim)) - 1).sum(axis=1)

    return (H/t).var(axis=0)/m * 1000
    # return (H).var(axis=0)

def feature_selection_sim_old(matrix, measure='luca', p=1):
    # Feature selection method using similarity measure and fuzzy entroropy
    # measures based on the article:

    # P. Luukka, (2011) Feature Selection Using Fuzzy Entropy Measures with
    # Similarity Classifier, Expert Systems with Applications, 38, pp. 4600-4607
    import pandas as pd
    data = pd.DataFrame(matrix)
    m = data.shape[0]  # -samples
    t = data.shape[1]   # -features
    l = 1
    idealvec_s = np.zeros((l, t))
    idealvec_s[:] = data.iloc[:, :].mean(axis=0)
    # scaling data between [0,1]
    # data_v = data.iloc[:, :]
    data_v = data.values
    mins_v = data_v.min(axis=0)
    Ones = np.ones((data_v.shape))
    data_v = data_v + np.dot(Ones, np.diag(abs(mins_v)))

    tmp = []
    tmp.append(abs(mins_v))

    idealvec_s = idealvec_s + tmp
    maxs_v = data_v.max(axis=0)
    # data_v = np.dot(data_v, np.diag(maxs_v ** (-1)))
    tmp2 = [];
    tmp2.append(abs(maxs_v))
    xs = np.ones_like(tmp2)*1e-5
    idealvec_s = idealvec_s / (tmp2+xs)

    # sample data
    datalearn_s = data

    # similarities
    sim = np.zeros((t, m))

    for j in range(m):#sample
        for i in range(t): #feature
             sim[i, j] = (1 - abs(idealvec_s[0,i] ** p - datalearn_s.iloc[j, i]) ** p) ** (1 / p)

    # simtest = np.zeros((t, m)) # the same results as the above 
    # idlcs = np.repeat(idealvec_s, m, axis=0)
    # simtest = (1 - (idlcs - datalearn_s))
    # simtest.transpose()

    # sim = sim.reshape(t, m)
    # possibility for two different entropy measures
    drawfig = False
    if drawfig == True:
        drawgraph(data)

    # sim = data
    sim = normalize(sim, axis=1, norm='l1')
    # normed_matrix = normalize(sim, axis=1, norm='l1')
    # H = (-normed_matrix * np.log(normed_matrix)).sum(axis=1)
    # norm_H = H/np.sum(H)
    # h = (-norm_H * np.log(norm_H)).sum()
    # return 1-h/t
    # return np.std(H)
    # return 1 - (H / t).sum(axis=0)
    if measure == 'luca':
        # moodifying zero and one values of the similarity values to work with
        # De Luca's entropy measure
        delta = 1e-10
        sim[sim == 0] = delta
        sim[sim == 1] = 1 - delta
        # H = (-sim * np.log(sim) - 0.9*(1 - sim) * np.log(1 - sim)).sum(axis=1)

        H = (-sim * np.log(sim) - (1 - sim) * np.log(1 - sim)).sum(axis=1)
        # H = (np.sin(np.pi / 2 * sim) + np.sin(np.pi / 2 * (1 - sim)) - 1).sum(axis=1)
    elif measure == 'park':
        H = (np.sin(np.pi / 2 * sim) + np.sin(np.pi / 2 * (1 - sim)) - 1).sum(axis=1)

        # find maximum feature
    # max_idx = np.argmax(H)  # notice that index is starting from 0
    #
    # # removing feature from the data
    # data_mod = dataold.drop(dataold.columns[max_idx], axis=1)
    #
    # return max_idx, data_mod
    # print('res ', 1-(H/t).sum(axis=0)/m)
    # h = (np.sin(np.pi / 2 * (H/t)) + np.sin(np.pi / 2 * (1 - (H/t))) - 1).sum(axis=0)
    return  (H/t).var(axis=0)/m
    # return 1-(H/t).sum(axis=0)/m #(H/H.sum(axis=0)).sum(axis=0)/m
# 1)- xlog(x)->methemetical meaning, and first use axis=0 up, this is for one player to all others, if use std in second
# sum (last item)
###########################################################
#                    Elo Predictability                   #
###########################################################
#   Use final Elo rating to predict entire season. Return
#   the ratio of the number of correct predictions over the
#   total number of games.
###########################################################
def eloPred(conf,year,elo_rating):
    # open file
    f = open((loc+'/DataFiles/CFB/'+str(conf)+'/'+str(year)+'games.txt'))
    # read all lines
    lineList = f.readlines()
    # compute back_pred
    numGames = len(lineList)
    back_pred = 0
    for k in range(numGames):
        row = lineList.pop(0)
        row = row.split(",")
        teami = eval(row[2]) - 1
        homei = eval(row[3])
        scorei = eval(row[4])
        teamj = eval(row[5]) - 1
        homej = eval(row[6])
        scorej = eval(row[7])
        if(scorei>scorej):
            # team i won at home
            if(homei==1 and elo_rating[teami]>(elo_rating[teamj]-H)):
                back_pred = back_pred + 1
            # team i won on the road
            elif(homei==-1 and elo_rating[teami]>(elo_rating[teamj]+H)):
                back_pred = back_pred + 1
        else:
            # team j won at home
            if(homej==1 and elo_rating[teamj]>(elo_rating[teami]-H)):
                back_pred = back_pred + 1
            # team j won on the road
            elif(homej==-1 and elo_rating[teamj]>(elo_rating[teami]+H)):
                back_pred = back_pred + 1
    # close file
    f.close()
    # return
    return float(back_pred)/float(numGames)

def typeprint(x, y, z, ep):
    corr,pval = spearmanr(x,y);  kcorr,kpval = kendalltau(x,y);
    print('\t specR and EloCorr corr = {:.4f}, kendalltau = {:.4f}'.format( corr, kcorr))
    print('\t specR and EloCorr pval = {:.4f}, kpval = {:.4f}'.format(pval, kpval))

    corr,pval = spearmanr(x,z);  kcorr,kpval = kendalltau(x,z);
    print('\t specR and EloPred  corr = {:.4f}, kendalltau = {:.4f}'.format( corr, kcorr))
    print('\t specR and EloPred pval = {:.4f}, kpval = {:.4f}'.format(pval, kpval))


    corr, pval = spearmanr(ep, x);  kcorr,kpval = kendalltau(ep, x);
    print('\t specR and entrobility corr = {:.4f}, kendalltau = {:.4f}'.format( corr, kcorr))
    print('\t specR and entrobility pval = {:.4f}, kpval = {:.4f}'.format(pval, kpval))
    corr, pval = spearmanr(ep, y); kcorr,kpval = kendalltau(ep, y);
    print('\t entrobility and EloCorr corr = {:.4f}, kendalltau = {:.4f}'.format( corr, kcorr))
    print('\t entrobility and EloCorr pval = {:.4f}, kpval = {:.4f}'.format(pval, kpval))
    corr, pval = spearmanr(ep, z); kcorr,kpval = kendalltau(ep, z);
    print('\t entrobility and EloCorr corr = {:.4f}, kendalltau = {:.4f}'.format( corr, kcorr))
    print('\t entrobility and EloCorr pval = {:.4f}, kpval = {:.4f}'.format(pval, kpval))
###########################################################
#                       main                              #
###########################################################
def main():
    # open files
    f1 = open((loc+'/DataFiles/PythonResults/CFB-Rank-EloCorr-Rounds.csv'),'w+')
    f2 = open((loc+'/DataFiles/PythonResults/CFB-Rank-EloCorr-Summary.csv'),'w+')
    # Atlantic Coast round by round analysis and summary
    f1.write('Atlantic Coast, Year, Round, entropyR, Rankability, EloCorr\n')
    f2.write('Atlantic Coast, Year, entropyR, Rankability, EloCorr, EloPred\n')
    x = []; y = []; z = []; ep =[];
    for year in range(1995,2004):
        elo_corr,rankability,elo_rating, entropybility = cfbData('Atlantic Coast',year,"SR")
        f1.write(',%d,,,\n' % year)
        f2.write(',%d,' % year)
        for k in range(len(rankability)):
            if(k>=1):
                f1.write(',,%d,%.4f,%.4f, %.4f\n' % (k+1,entropybility[k], rankability[k],elo_corr[k-1]))
            else:
                f1.write(',,%d,%.4f,%.4f, %.4f\n' % (k+1,entropybility[k], rankability[k],0))
        x.append(rankability[-1])
        ep.append(entropybility[-1])
        y.append(np.average(elo_corr,weights=[k for k in range(len(elo_corr))]))
        z.append(eloPred('Atlantic Coast',year,elo_rating))
        f2.write(',,%.4f,%.4f,%.4f,%.4f\n' % (ep[-1], x[-1],y[-1],z[-1]))
    # correlation between year summary data
    print('Atlantic Coast: ')
    typeprint(x, y, z, ep)


    # Big East round by round analysis and summary
    f1.write('Big East, Year, Round, entropyR, Rankability, EloCorr\n')
    f2.write('Big East, Year, entropyR, Rankability, EloCorr, EloPred\n')
    x = []; y = []; z = []; ep=[];
    for year in range(1995,2013):
        elo_corr,rankability,elo_rating, entropybility = cfbData('Big East',year,"SR")
        f1.write(',%d,,,\n' % year)
        f2.write(',%d,,\n' % year)
        for k in range(len(rankability)):
            if(k>=1):
                f1.write(',,%d,%.4f,%.4f,%.4f\n' % (k+1,entropybility[k],rankability[k], elo_corr[k-1]))
            else:
                f1.write(',,%d,%.4f,%.4f,%.4f\n' % (k+1,entropybility[k], rankability[k],0))
        x.append(rankability[-1])
        ep.append(entropybility[-1])
        y.append(np.average(elo_corr,weights=[k for k in range(len(elo_corr))]))
        z.append(eloPred('Big East',year,elo_rating))
        f2.write(',,%.4f,%.4f,%.4f,%.4f\n' % (ep[-1], x[-1],y[-1],z[-1]))
    # correlation between year summary data
    print('Big East: ')
    typeprint(x, y, z, ep)

    # Mountain West round by round analysis and summary
    f1.write('Mountain West, Year, Round, entropyR, Rankability, EloCorr\n')
    f2.write('Mountain West, Year, entropyR, Rankability, EloCorr, EloPred\n')
    x = []; y = []; z = []; ep=[]
    for year in range(1999,2012):
        elo_corr,rankability,elo_rating, entropybility = cfbData('Mountain West',year,"SR")
        f1.write(',%d,,,\n' % year)
        f2.write(',%d,,' % year)
        for k in range(len(rankability)):
            if(k>=1):
                f1.write(',,%d,%.4f,%.4f,%.4f\n' % (k+1,entropybility[k],rankability[k],elo_corr[k-1]))
            else:
                f1.write(',,%d,%.4f,%.4f,%.4f\n' % (k+1,entropybility[k], rankability[k],0))
        x.append(rankability[-1])
        ep.append(entropybility[-1])
        y.append(np.average(elo_corr,weights=[k for k in range(len(elo_corr))]))
        z.append(eloPred('Mountain West',year,elo_rating))
        f2.write(',,%.4f,%.4f,%.4f,%.4f\n' % (ep[-1], x[-1],y[-1],z[-1]))
    # correlation between year summary data
    print('Mountain West: ')
    typeprint(x, y, z, ep)
    # close files
    f1.close()
    f2.close()

def get_tuple(a):
    a = a - a.transpose()
    b =  np.where(a<0, 0, a)
    ld = []
    for i in range(9):
        ld.append((i + 1, i + 1))
        for j in range(9):
            if b[i][j] != 0:
                ld.append((i+1,j+1))
    print(ld)

def get_eigen():
    a = np.ones([9, 9])
    b = np.triu(a)
    c = b / (np.sum(b, axis=1))
    ei, ev = np.linalg.eig(c)
    np.argsort(ev[0, :])
    np.argsort(ev[6, :])

if __name__ == '__main__':
    # main()
    cmat = np.ones([9,9]) #--ERRO, USE collectTeams.py
    for year in range(1995, 2004):
        with open(('tempdata/'+str(year)+'Atlcost.npy'), 'rb') as f:
            a = np.load(f)
            # print(a)
            cmat= np.add(cmat, a)
            get_tuple(a)
            print('--------------')
            # print(cmat)
            # Get the unique values and their counts
            # unique_values, counts = np.unique(a, return_counts=True)
            #
            # # Print the results
            # for value, count in zip(unique_values, counts):
            #     print(f"{value} occurs {count} times")
    print(cmat - np.ones([9,9]))
    cmat = cmat - cmat.transpose()
    cmat = np.where(cmat<0, 0, cmat)
    print(cmat)

    ld = []
    for i in range(9):
        ld.append((i + 1, i + 1))
        for j in range(9):
            if cmat[i][j] != 0:
                ld.append((i+1,j+1))
    print(ld)