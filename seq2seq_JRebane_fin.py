
__author__ = "Jonathan Rebane. A multivariate input extension of bitcoin timeseries prediction seq2seq RNN work by Guillaume Chevalier"
# - https://github.com/guillaume-chevalier/
__version__ = "2018-05"




from datasetup import * 
import random

generatedata = generatedata_v1


import tensorflow as tf  # Version 1.0 or 0.12
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from cryptory import Cryptory

sample_x, sample_y = generatedata(isTrain=True, batch_size=64)

# manually set hyperparameters
seq_length = sample_x.shape[0]   
batch_size = 64 
input_dim = sample_x.shape[-1]
output_dim = sample_y.shape[-1]
hidden_dim = 30 
layers_stacked_count = 2  
learning_rate = 0.007   
nb_iters = 1000  
lr_decay = 0.9  
momentum = 0.5  
lambda_l2_reg = 0.006  


try:
    tf.nn.seq2seq = tf.contrib.legacy_seq2seq
    tf.nn.rnn_cell = tf.contrib.rnn
    tf.nn.rnn_cell.GRUCell = tf.contrib.rnn.GRUCell
    print("TensorFlow's version : 1.0 (or more)")
except: 
    print("TensorFlow's version : 0.12")



predlist = list() 
allpred = list() #store predictions across random model initializations

for z in range(10): #run RNN multiple times and take mean predictions to accommodate for variability

    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    
    with tf.variable_scope('Seq2seq'):
    
        # Encoder: inputs
        enc_inp = [
            tf.placeholder(tf.float32, shape=(None, input_dim), name="inp_{}".format(t))
               for t in range(seq_length)
        ]
    
        # Decoder: expected outputs
        expected_sparse_output = [
            tf.placeholder(tf.float32, shape=(None, output_dim), name="expected_sparse_output_".format(t))
              for t in range(seq_length)
        ]
        
        # Give a "GO" token to the decoder. 
        dec_inp = [ tf.zeros_like(enc_inp[0], dtype=np.float32, name="GO") ] + enc_inp[:-1]
    
        # Create a `layers_stacked_count` of stacked RNNs (GRU cells here). 
        cells = []
        for i in range(layers_stacked_count):
            with tf.variable_scope('RNN_{}'.format(i)):
                cells.append(tf.nn.rnn_cell.GRUCell(hidden_dim))
                # cells.append(tf.nn.rnn_cell.BasicLSTMCell(...))
        cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        
        # The encoder and the decoder uses the same cell with unshared weights. 
        dec_outputs, dec_memory = tf.nn.seq2seq.basic_rnn_seq2seq(
            enc_inp, 
            dec_inp, 
            cell
        )
        
        # For reshaping the output dimensions of the seq2seq RNN: 
        w_out = tf.Variable(tf.random_normal([hidden_dim, output_dim]))
        b_out = tf.Variable(tf.random_normal([output_dim]))
        
        # Final outputs: with linear rescaling for enabling possibly large and unrestricted output values.
        output_scale_factor = tf.Variable(1.0, name="Output_ScaleFactor")
        
        reshaped_outputs = [output_scale_factor*(tf.matmul(i, w_out) + b_out) for i in dec_outputs]
    
    
    # Training loss and optimizer
    
    with tf.variable_scope('Loss'):
        # L2 loss
        output_loss = 0
        for _y, _Y in zip(reshaped_outputs, expected_sparse_output):
            output_loss += tf.reduce_mean(tf.nn.l2_loss(_y - _Y))
            
        # L2 regularization 
        reg_loss = 0
        for tf_var in tf.trainable_variables():
            if not ("Bias" in tf_var.name or "Output_" in tf_var.name):
                reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))
                
        loss = output_loss + lambda_l2_reg * reg_loss
    
    with tf.variable_scope('Optimizer'):
        optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=lr_decay, momentum=momentum)
        train_op = optimizer.minimize(loss)
    
    
    # Training
    
    def train_batch(batch_size):
        """
        Training step that optimizes the weights 
        provided some batch_size X and Y examples from the dataset. 
        """
        X, Y = generatedata(isTrain=True, batch_size=batch_size)
        feed_dict = {enc_inp[t]: X[t] for t in range(len(enc_inp))}
        feed_dict.update({expected_sparse_output[t]: Y[t] for t in range(len(expected_sparse_output))})
    
        _, loss_t = sess.run([train_op, loss], feed_dict)
        return loss_t
    
    def test_batch(batch_size):
        """
        Test step,Weights are frozen by not
        doing sess.run on the train_op. 
        """
        X, Y = generatedata(isTrain=False, batch_size=batch_size)
        feed_dict = {enc_inp[t]: X[t] for t in range(len(enc_inp))}
        feed_dict.update({expected_sparse_output[t]: Y[t] for t in range(len(expected_sparse_output))})
        loss_t = sess.run([loss], feed_dict)
        return loss_t[0]
    
    
    # Training
    train_losses = []
    test_losses = []
    
    sess.run(tf.global_variables_initializer())
    for t in range(nb_iters+1):
        train_loss = train_batch(batch_size)
        train_losses.append(train_loss)
        
        if t % 10 == 0: 
            # Tester
            test_loss = test_batch(batch_size)
            test_losses.append(test_loss)
            print("Step {}/{}, train loss: {}, \tTEST loss: {}".format(t, nb_iters, train_loss, test_loss))
    
    print("Fin. train loss: {}, \tTEST loss: {}".format(train_loss, test_loss))
    
    
 
    plt.figure(figsize=(12, 6))
    plt.plot(
        np.array(range(0, len(test_losses)))/float(len(test_losses)-1)*(len(train_losses)-1), 
        np.log(test_losses), 
        label="Test loss"
    )
    plt.plot(
        np.log(train_losses), 
        label="Train loss"
    )
    plt.title("Training errors over time (on a logarithmic scale)")
    plt.xlabel('Iteration')
    plt.ylabel('log(Loss)')
    plt.legend(loc='best')
    plt.show()
    
    logloss = np.log(test_losses)
    mintest = min(logloss)
    
    # =============================================================================
    # mintest = min(logloss)
    # =============================================================================
    
    global Y_mean
    global Y_std
    
    X, Y = generatedata_v2(isTrain=False)
    Y_std = get_std()
    Y_mean = get_mean()
    
    feed_dict = {enc_inp[t]: X[t] for t in range(seq_length)}
    outputs = np.array(sess.run([reshaped_outputs], feed_dict)[0])
    ind = Y_std.shape[0] - X.shape[1]
    errsum = 0
    stepserrdl = []
    allcumsum = list()
    
    for j in range(seq_length, seq_length+1): # or to X.shape[1] for all predictions... start at seq_length to prevent overlap of train and test set predictions
        plt.figure(figsize=(12, 3))
        for k in range(input_dim):
            past = X[:,j,k]
            expected = Y[:,j,0] * Y_std[ind+j,0,0] + Y_mean[ind+j,0,0]
            pred = outputs[:,j,0] * Y_std[ind+j,0,0] + Y_mean[ind+j,0,0]
            
            label1 = "(past) values" if k==0 else "_nolegend_"
            label2 = "True values" if k==0 else "_nolegend_"
            label3 = "Predictions" if k==0 else "_nolegend_"
            #plt.plot(range(len(past)), past, "o--b", label=label1)
            plt.plot(range(len(expected)), expected, "x--b", label=label2)
            plt.plot(range(len(pred)), pred, "o--y", label=label3)
        
        error = mean_squared_error(expected, pred)
        inderror = (expected - pred)**2
        cummulativesumdl = np.cumsum(inderror)
        stepserrdl.append(error)
        errsum = errsum + error
        print('Test MSE: %.3f' % error)
        plt.legend(loc='best')
        plt.title("Predictions v.s. true values")
        plt.show()
        pyplot.plot(cummulativesumdl)
        allcumsum.append(cummulativesumdl)
        
    
    print(errsum/(X.shape[1]-seq_length))
    pyplot.plot(stepserrdl)
    pyplot.show()
    predlist.append(allcumsum[0])
    allpred.append(np.ndarray.tolist(pred))



#####ARIMA START######

Xar = loadCurrencynormal("USD")
#X = normalizearima(X)
train = Xar[:int(len(Xar) * 0.8) + int(seq_length * 0.4)]
test = Xar[int(len(Xar) * 0.8) + int(seq_length * 0.4):]
    

model = ARIMA(train, order=(0,1,2)) # ARIMA with grid searched parameters
model_fit = model.fit(disp=0)
output = model_fit.forecast(steps=len(test))
yhat = output[0]
    


#error for first seq_length days
error = mean_squared_error(test[:seq_length], yhat[:seq_length])

inderror = (expected - np.mean(allpred, axis=0))**2
cummulativesumalldl = np.cumsum(inderror)

print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test[:seq_length])
pyplot.plot(yhat[:seq_length], color='red')
pyplot.plot(np.mean(allpred, axis=0), color='gold')
pyplot.show()

inderror = (test[:seq_length] - yhat[:seq_length])**2
cummulativesumar = np.cumsum(inderror)
pyplot.plot(cummulativesumar)



#######COMPARE MODELS#######

plt.figure(figsize=(10, 6.18))

label1 = "Seq2Seq cumulative sum of squares " 
label2 = "ARIMA cumulative sum of squares" 

plt.plot(cummulativesumalldl,label=label1)
plt.plot(cummulativesumar, label=label2)
plt.legend(loc='best')
plt.ylabel('Error')
plt.xlabel('Days')
plt.title("Model Comparison of cumulative sum of squares over %d days" % seq_length)
plt.show()




