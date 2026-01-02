import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from copy import deepcopy
from rankability import specR
from entropyRanki_bank import feature_selection_sim
K = 40.
X = 400.

###########################################################
#                       sqfieldData                       #
###########################################################
#   Reads SinquefieldCup data from a certain year. Returns
#   Elo Correlation and rankability for that year. The opt
#   variable determines how the Elo Correlation is measured
#   with either kendalltau (KT), spearmanr (SR), or pearsonr (PR).
###########################################################
loc = 'D:\\workspace\\python\\temp\\specR-master\\'

def sqfieldData(year,opt):
    # open file
    f = open(loc+'/DataFiles/SinquefieldCup/SinquefieldCup'+str(year)+'.csv')
    # read all lines
    lineList = f.readlines()
    # grab first line and split
    row = lineList.pop(0)
    row = row.split(",")
    # store numPlayers and numRounds
    numPlayers = eval(row[0])
    numRounds = eval(row[1])
    # create matches, adj, and r lists
    matches = [np.zeros((numPlayers,numPlayers)) for k in range(numRounds)]
    adj = [np.zeros((numPlayers,numPlayers)) for k in range(numRounds)]
    elo_rating = [np.zeros(numPlayers) for k in range(numRounds)]
    elo_corr = []
    rankability = []
    entropybility =[]
    # populate matches, adj, elo_rating, elo_corr, and rankability
    for k in range(numRounds):
        # deepcopy previous rounds matches and elo_rating
        matches[k] = deepcopy(matches[k-1])
        elo_rating[k] = deepcopy(elo_rating[k-1])
        # play out number of matches in kth round
        for l in range(numPlayers//2):
            row = lineList.pop(0)
            row = row.split(",")
            row = [eval(row[0]),eval(row[1]),eval(row[2])]
            i = row[0]-1
            j = row[2]-1
            if(row[1]>0):
                # player i beat player j
                matches[k][i,j] = matches[k][i,j] + 1
                # update player i Elo rating
                d = elo_rating[k][i] - elo_rating[k][j]
                u = 1./(1.+10.**(-d/X))
                elo_rating[k][i] = elo_rating[k][i] + K*(1.-u)
                # update player j Elo rating
                elo_rating[k][j] = elo_rating[k][j] + K*(u-1.)
            elif(row[1]<0):
                # player j beat player i
                matches[k][j,i] = matches[k][j,i] + 1
                # update player j Elo rating
                d = elo_rating[k][j] - elo_rating[k][i]
                u = 1./(1.+10.**(-d/X))
                elo_rating[k][j] = elo_rating[k][j] + K*(1.-u)
                # update player i Elo rating
                elo_rating[k][i] = elo_rating[k][i] + K*(u-1.)
            else:
                # draw
                matches[k][i,j] = matches[k][i,j] + 0.5
                matches[k][j,i] = matches[k][j,i] + 0.5
                # update player i Elo rating
                d = elo_rating[k][i] - elo_rating[k][j]
                u = 1./(1.+10.**(-d/X))
                elo_rating[k][i] = elo_rating[k][i] + K*(0.5-u)
                # update player j Elo rating
                elo_rating[k][j] = elo_rating[k][j] + K*(u-0.5)
        # update adjacency matrix
        for i in range(numPlayers):
            for j in range(i+1,numPlayers):
                total = matches[k][i,j] + matches[k][j,i]
                if(total!=0):
                    adj[k][i,j] = matches[k][i,j]/total
                    adj[k][j,i] = matches[k][j,i]/total
        # Rankability
        rankability.append(specR(adj[k]))
        entropybility.append(feature_selection_sim(adj[k]))

        # Elo Correlation
        if(k>=1 and opt=="SR"):
            corr,pval = spearmanr(elo_rating[k],elo_rating[k-1])
            elo_corr.append(corr)
        elif(k>=1 and opt=="KT"):
            corr,pval = kendalltau(elo_rating[k],elo_rating[k-1])
            elo_corr.append(corr)
        elif(k>=1 and opt=="PR"):
            corr,pval = pearsonr(elo_rating[k],elo_rating[k-1])
            elo_corr.append(corr)
    # return variables
    return elo_corr, rankability, entropybility

def entropyRank(matrix):
    n = len(matrix)


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
    data_v = data.iloc[:, :]

    mins_v = data_v.min(axis=0)
    Ones = np.ones((data_v.shape))
    data_v = data_v + np.dot(Ones, np.diag(abs(mins_v)))

    tmp = []
    tmp.append(abs(mins_v))

    idealvec_s = idealvec_s + tmp
    maxs_v = data_v.max(axis=0)
    data_v = np.dot(data_v, np.diag(maxs_v ** (-1)))
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

    sim = sim.reshape(t, m)
    # possibility for two different entropy measures
    if measure == 'luca':
        # moodifying zero and one values of the similarity values to work with
        # De Luca's entropy measure
        delta = 1e-10
        sim[sim == 1] = delta
        sim[sim == 0] = 1 - delta
        H = (-sim * np.log(sim) - (1 - sim) * np.log(1 - sim)).sum(axis=1)
    elif measure == 'park':
        H = (np.sin(np.pi / 2 * sim) + np.sin(np.pi / 2 * (1 - sim)) - 1).sum(axis=1)

        # find maximum feature
    # max_idx = np.argmax(H)  # notice that index is starting from 0
    #
    # # removing feature from the data
    # data_mod = dataold.drop(dataold.columns[max_idx], axis=1)
    #
    # return max_idx, data_mod
    return (H/m).sum(axis=0)/m#(H/H.sum(axis=0)).sum(axis=0)/m

def main():
    # open files
    f1 = open((loc+'//DataFiles//PythonResults//SQField-Rank-EloCorr-Rounds.csv'),'w+')
    f2 = open((loc+'/DataFiles/PythonResults/SQField-Rank-EloCorr-Summary.csv'),'w+')
    # round by round analysis and summary
    f1.write('Year, Round, entropyR, Rankability, EloCorr \n')
    f2.write('Year, entropyR, Rankability, EloCorr \n')
    x = []; y = []; z =[]; ep=[]
    for year in range(2013,2020):
        elo_corr,rankability, entropybility = sqfieldData(year,"SR")
        f1.write('%d,,,\n' % year)
        f2.write('%d' % year)
        for k in range(len(rankability)):
            if(k>=1):
                f1.write(',%d,%.4f,%.4f,%.4f\n' % (k+1,entropybility[k], rankability[k], elo_corr[k-1]))
            else:
                f1.write(',%d,%.4f,%.4f,%.4f\n' % (k+1,entropybility[k], rankability[k],0))
        x.append(rankability[-1])
        ep.append(entropybility[-1])
        y.append(np.average(elo_corr,weights=[k for k in range(len(elo_corr))]))
        f2.write(',%.4f,%.4f,%.4f\n' % (ep[-1], x[-1],y[-1]))

    # correlation between year summary data
    print([ep, y, x])
    typeprint_two(ep, y, x)

    # close files
    f1.close()
    f2.close()

def typeprint(ep, y, x):
    corr,pval = spearmanr(ep,y)
    corr1, pval1 = spearmanr(x, ep)
    corr2, pval2 = spearmanr(x, y)
    print('\entrobility and EloCorr corr = %.4f' % corr)
    print('\entrobility and EloCorr pval = %.4f' % pval)
    print('\tspecR and entrobility corr = %.4f' % corr1)
    print('\tspecR and entrobility pval = %.4f' % pval1)

    print('\tspecR and EloCorr corr = %.4f' % corr2)
    print('\tspecR and EloCorr pval = %.4f' % pval2)

def typeprint_two(ep, y, x):

    corr, pval = spearmanr(x, y);  kcorr,kpval = kendalltau(x,y);
    print('\tspecR and EloCorr corr corr = {:.4f}, kendalltau = {:.4f}'.format( corr, kcorr))
    print('\tspecR and EloCorr pval = {:.4f}, kpval = {:.4f}'.format(pval, kpval))

    corr, pval = spearmanr(x, ep);  kcorr,kpval = kendalltau(x,ep);
    print('\tspecR and entrobility corr = {:.4f}, kendalltau = {:.4f}'.format( corr, kcorr))
    print('\tspecR and entrobility pval = {:.4f}, kpval = {:.4f}'.format(pval, kpval))

    corr,pval = spearmanr(ep,y);  kcorr,kpval = kendalltau(ep,y);
    print('\tentrobility and EloCorr corr = {:.4f}, kendalltau = {:.4f}'.format( corr, kcorr))
    print('\tentrobility and EloCorr pval = {:.4f}, kpval = {:.4f}'.format(pval, kpval))

if __name__ == '__main__':
    main()


# t = [0, 0.0909, 0;
# 0.0843, 0.1591, 0.774;
# 0.0121, 0.222, 0.8172;
# 0.0089, 0.2826, 0.1938;
# 0.0085, 0.3336, 0.4556;
# 0.0054, 0.3847, 0.986;
# 0.0037, 0.4252, 0.993;
# 0.0051, 0.4668, 0.6014;
# 0.0035, 0.5219, 0.9231;
# 0.005, 0.5572, 0.8392;
# 0.0074, 0.6021, 0.7972;
# ];

#  t= [0, 0.1111, 0;
# 0.2913, 0.1944, 0.8412;
# 0.3123, 0.2662, 0.8794;
# 0.1037, 0.3319, 0.845;
# 0.0478, 0.3876, 0.9758;
# 0.0591, 0.4882, 0.9515;
# 0.0363, 0.5391, 0.9758;
# 0.0247, 0.5857, 1;
# 0.0246, 0.6389, 0.8182;];
# >>  [ ~, I1] = sort(t(:, 1));
# [~, I2] = sort(t(:, 2));
# [~, I3] = sort(t(:, 3));
# I = [I1, I2, I3];

# t= [1.6176, 0.8750, 0.9639, 0.9167;
# 1.5600, 0.8750, 0.9374, 0.8056;
# 1.5621, 0.8750, 0.9478, 0.9444;
# 1.6640, 0.8750, 0.9646, 0.8889;
# 0.3729, 0.7791, 0.8914, 0.7222;
# 1.5600, 0.8750, 0.9603, 0.9167;
# 0.8630, 0.8313, 0.9275, 0.7500;
# 1.5332, 0.8346, 0.9308, 0.8611;
# 0.9259, 0.8438, 0.8835, 0.8056;];