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


logmeans = np.log(prob1_data.groupby('level').mean())
logstd = np.log(prob1_data.groupby('level').std())

# prob1_data.level = prob1_data.level.astype('category')

prob1_model = ols('logstd ~ logmeans', prob1_data).fit()

lambdaa = 1 - prob1_model.params.logmeans

prob1_data['fail_time_log'] = prob1_data.fail_time**lambdaa

prob1_model2 = ols('fail_time_log ~ level', prob1_data).fit()

