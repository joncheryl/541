import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols

prob1_data = pd.DataFrame({'level' : np.repeat(['1520', '1600', '1680', '1760', '1840'], 6), 
                           'fail_time' : [1253, 1435, 1771, 4027, 5434, 5614,
                                          1190, 1286, 1550, 2125, 2557, 2845, 
                                          751, 837, 848, 1038, 1361, 1443,
                                          611, 691, 751, 772, 808, 859, 
                                          513, 546, 517, 420, 471, 556]})


#######
# Prob 1:
# lambda ~ -1 which is about a reciprical trans
#######

logmeans = np.log(prob1_data.groupby('level').mean())
logstd = np.log(prob1_data.groupby('level').std())

logmeans = sm.add_constant(logmeans)
prob1_model = sm.OLS(logstd, logmeans).fit()

lambdaa = 1 - prob1_model.params[1]

#######
# Prob 2:
# Not desired boxcox function in python modules
# SAS says lambda ~ -1.1 which is about a reciprical trans
########
from scipy import stats

xt = stats.boxcox(prob1_data.fail_time)

######
# Prob3a:
# means model would be y_{ij} = mu_i + epsilon_{ij} where i is the index for
# group and j is the index for observations within groups 1 <= i <= 5 and
# 1 <= j <= 6
# epsilon_{ij} ~iid N(0, sigma^2)
#
# mu_i is the parameter for the mean failure time in hours of group i
# epsilon_{ij} is the error term for observation j in group i
# sigma^2 is the variance of the errors
######

########
# Prob3b:
# levene's test on original data
# The results of this test will inform a test of H0: sigma_i != sigma_j for some
# i,j but the actual test performed has the following hypotheses:
# 
# mu*_i denotes the mean absolute deviation for group i (estimated by
# sum_j |y_{ij} - bar{y}_i|/n )
# H0: mu*_i = mu*_j for all i, j
# Ha: mu*_i != mu*_j for some i, j
#
# F-stat of about 37.4 yields a p-value of .000... so we're going to reject
# the null hypothesis and conclude that the variances are not equal
#########

g1=prob1_data.query('level =="1520"').fail_time
g2=prob1_data.query('level =="1600"').fail_time
g3=prob1_data.query('level =="1680"').fail_time
g4=prob1_data.query('level =="1760"').fail_time
g5=prob1_data.query('level =="1840"').fail_time
stats.levene(g1, g2, g3, g4, g5, center='mean')

#########
# Prob3c:
# In the Q-Q plot the points don't like on the diag line => not normally distr
# In predicted value v residuals plot there is obvious fanning => no homo var
#########

import matplotlib.pyplot as plt

prob1_model3 = ols('fail_time ~ level', prob1_data).fit()
resid = prob1_model3.resid
fitted = prob1_model3.fittedvalues

fig1 = sm.qqplot(resid, line='s')
plt.show()

fig2 = plt.scatter(fitted, resid)
plt.show()

#########
# Prob3d:
# It seems pretty clear that as the mean failure time increases, so does the var
#########

prob1_data.boxplot(column = 'fail_time', by = 'level')
plt.show()

##############
# Prob3e:
# Tukey's hsd test
# It appears that the 1520 group is significantly different from the 1680, 1760,
# and 1840 groups and all other group means are not considered sig different
#############
from statsmodels.stats.multicomp import MultiComparison

tukey_test = MultiComparison(prob1_data['fail_time'], prob1_data['level'])
print tukey_test.tukeyhsd()

######
# Prob4a:
# Let w_{ij} = 1/y_{ij}
# means model would be y_{ij} = mu_i + epsilon_{ij} where i is the index for
# group and j is the index for observations within groups 1 <= i <= 5 and
# 1 <= j <= 6
# epsilon_{ij} ~iid N(0, sigma^2)
#
# mu_i is the parameter for the mean reciprical failure time in hours of group i
# epsilon_{ij} is the error term for observation j in group i
# sigma^2 is the variance of the errors
######

prob1_data['fail_time_recip'] = 1/prob1_data['fail_time']

########
# Prob4b:
# levene's test on original data
# The results of this test will inform a test of H0: sigma_i != sigma_j for some
# i,j but the actual test performed has the following hypotheses:
# 
# mu*_i denotes the mean absolute deviation for group i (estimated by
# sum_j |w_{ij} - bar{y}_i|/n )
# H0: mu*_i = mu*_j for all i, j
# Ha: mu*_i != mu*_j for some i, j
#
# F-stat of about 1.337 yields a p-value of .284, yielding little evidence that
# after taking the reciprical of the failure times that there is a difference
# in variance between groups
#########

g1_recip=prob1_data.query('level =="1520"').fail_time_recip
g2_recip=prob1_data.query('level =="1600"').fail_time_recip
g3_recip=prob1_data.query('level =="1680"').fail_time_recip
g4_recip=prob1_data.query('level =="1760"').fail_time_recip
g5_recip=prob1_data.query('level =="1840"').fail_time_recip
stats.levene(g1_recip, g2_recip, g3_recip, g4_recip, g5_recip, center='mean')

#########
# Prob4c:
# The normal prob plot looks pretty good. Resids look approx normally distr
# In predicted value v residuals plot there is obvious fanning => no homo var
#########

prob1_model4 = ols('fail_time_recip ~ level', prob1_data).fit()
resid_recip = prob1_model4.resid
fitted_recip = prob1_model4.fittedvalues

sm.qqplot(resid_recip, line='s')
plt.show()

plt.scatter(fitted_recip, resid_recip)
range_x = max(fitted_recip) - min(fitted_recip)
range_y = max(resid_recip) - min(resid_recip)
plt.xlim(min(fitted_recip) - .1*range_x, max(fitted_recip) + .1*range_x)
plt.ylim(min(resid_recip) - .1*range_y, max(resid_recip) + .1*range_y)
plt.show()

#########
# Prob4d:
# It appears that the variance remains constant across the different temp levels
#########

prob1_data.boxplot(column = 'fail_time_recip', by = 'level')
plt.show()

##############
# Prob4e:
# Tukey's hsd test
# It appears that the 1520 group is not significantly different from the 1600 
# and the 1680 group is not sig different form the 1760 group but all other
# group comparisons are considered sig diff
#############
tukey_test_recip = MultiComparison(prob1_data['fail_time_recip'], prob1_data['level'])
print tukey_test_recip.tukeyhsd()

##############
# Prob5:
# I would definitely go with the transformed data. The diagnostic plots look
# much better indicating that we can assume the model assumptions are met.
# Looking at the original boxplot of the untransformed data, it appears that
# there were actually more differences than what the Tukey comparisons
# suggested. By taking the reciprical of the failure times we have 'tamed' the
# increasing nature of spread which allows us to see more accurately differences
# in our analysis
##############

#############
# Prob6:
# Looks like theres a linear trend and some curvature but not any evidence
# for a cubic or quartic term
#############

prob1_model5 = ols('fail_time_recip ~ C(level, Poly)', prob1_data).fit()

################################################################################
# Prob7a:
# I don't think so because it appears that there is a relationship between
# mean response and variability. If there wasn't such a relationship then WLS
# would be an ok choice. Using WLS would not capture this. 
################################################################################


################################################################################
# Prob 7b:
# I woulds say this does sound reasonable as it appears that the variance is
# proportional to some power of the mean response per group. The Box-Cox
# procedure seeks to find this power.
################################################################################

############
# Prob8:
# We need a total sample size of 28. And group sample sizes of 28/4 = 7 to have
# at least power of .9
############
import math
import statsmodels.stats.power as smp

sigma = 25
mus = [60., 50., 60., 55.]
mu = np.mean(mus)
f = math.sqrt(((60 - mu)**2/4 + (50 - mu)**2/4 + (60 - mu)**2/4 + (55 - mu)**2/4)/sigma)

smp.FTestAnovaPower().solve_power(effect_size=f, alpha=0.05, power=.9, k_groups=4)
smp.FTestAnovaPower().solve_power(effect_size=f, alpha=0.05, power=.939, k_groups=4)

###########
# Prob9a:
# We need a total sample size of 36. And group sample sizes of 36/4 = 9 to have
# at least power of .9
############

sigma = 36
mus = [60., 50., 60., 55.]
mu = np.mean(mus)
f = math.sqrt(((60 - mu)**2/4 + (50 - mu)**2/4 + (60 - mu)**2/4 + (55 - mu)**2/4)/sigma)

smp.FTestAnovaPower().solve_power(effect_size=f, alpha=0.05, power=.9, k_groups=4)
smp.FTestAnovaPower().solve_power(effect_size=f, alpha=0.05, power=.92, k_groups=4)

###########
# Prob9b:
# We need a total sample size of 48. And group sample sizes of 48/4 = 12 to have
# at least power of .9
############

sigma = 49
mus = [60., 50., 60., 55.]
mu = np.mean(mus)
f = math.sqrt(((60 - mu)**2/4 + (50 - mu)**2/4 + (60 - mu)**2/4 + (55 - mu)**2/4)/sigma)

smp.FTestAnovaPower().solve_power(effect_size=f, alpha=0.05, power=.9, k_groups=4)
smp.FTestAnovaPower().solve_power(effect_size=f, alpha=0.05, power=.92, k_groups=4)

############
# Prob9c&d:
# As the variability increases, the sensitivity of an ANOVA F-test decreases.
# Thus, the need for more data to detect an actual difference is required.
# As simga increases, so should the sample size.
############
