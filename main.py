from DSAE import Discriminative_SAE
import DSAE.Pre_process as Pre_process
import DSAE.To_full as To_full
import DSAE.Dropout as Dropout
import pandas as pd
import numpy as np 
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='data/test.csv')
parser.add_argument('--outputdir', type=str, default='data')
parser.add_argument('--dim1', type=int, default=600)
parser.add_argument('--dim2', type=int, default=256)
parser.add_argument('--epoch1', type=int, default=3000)
parser.add_argument('--epoch2', type=int, default=1000)
parser.add_argument('--learning_rate', type=float, default=4e-3)
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--print_step', type=int, default=200)

args = parser.parse_args()

def main(data, outdir):
    ########################    Read Data     ########################
    data_T = pd.read_csv(data, index_col=0)  
    data_raw = Dropout.main(data_T, outdir)

    adj = cosine_similarity(data_raw.values)
    print(adj) 

    ########################    Data Preprocessing    ######################
    data_raw_process, row, col, data_true_part = Pre_process.normalize(data_raw, data_T)  

    ########################        Imputation         ###################### 
    model = Discriminative_SAE(dims = [args.dim1, args.dim2],  
                            activations = ['sigmoid', 'relu'],
                            epoch = [args.epoch1, args.epoch2], 
                            loss = 'rmse',
                            lr = args.learning_rate,
                            noise = None,   
                            batch_size = args.batch, 
                            print_step = args.print_step,
                            Adj = adj)   

    model.fit(data_raw_process, data_true_part)   
    predict = model.predict(data_raw_process)     

    impute_part = pd.DataFrame(predict, index=row, columns=col)
    impute = To_full.getAll(impute_part, data_raw)  
    impute.to_csv(outdir + '/impute.csv')

    print("------------------------- The metrics of this {}x{}--------------------------- ".format(data_T.values.shape[0], data_T.values.shape[1]))
    print("Mean Absolute Error: MAE = {0:.3f}".format( mean_absolute_error(data_T, impute) ))
    print("Mean square error: MSE = {0:.3f}".format( mean_squared_error(data_T, impute) ** 0.5 ))
    print("Pearson correlation coefficient: PCC = {0:.3f}".format( pearsonr(data_true_part.reshape(-1), impute_part.values.reshape(-1))[0] ))

main(args.input, args.outputdir)
