import time
from keras.callbacks import Callback
import matplotlib.pyplot as plt
from IPython import display
import os


class TimerCallback(Callback):
    #Callback: TimerCallback()
    # this callback make sure to interrupt the training if a certain time limit is reached, 
    # saving the weigths of the last model to train again
    def __init__(self, maxExecutionTime, byBatch = False, on_interrupt=None):
        '''  
         Arguments:
            maxExecutionTime (number): Time in minutes. The model will keep training 
                                       until shortly before this limit
                                       (If you need safety, provide a time with a certain tolerance)
            byBatch (boolean)     : If True, will try to interrupt training at the end of each batch
                                    If False, will try to interrupt the model at the end of each epoch    
                                   (use `byBatch = True` only if each epoch is going to take hours)          
            on_interrupt (method)          : called when training is interrupted
                signature: func(model,elapsedTime), where...
                      model: the model being trained
                      elapsedTime: the time passed since the beginning until interruption   
        '''
        self.maxExecutionTime = maxExecutionTime * 60
        self.on_interrupt = on_interrupt
        # the same handler is used for checking each batch or each epoch
        if byBatch == True:
            #on_batch_end is called by keras every time a batch finishes
            self.on_batch_end = self.on_end_handler
        else:
            #on_epoch_end is called by keras every time an epoch finishes
            self.on_epoch_end = self.on_end_handler
    
    
    #Keras will call this when training begins
    def on_train_begin(self, logs):
        self.startTime = time.time()
        self.longestTime = 0            #time taken by the longest epoch or batch
        self.lastTime = self.startTime  #time when the last trained epoch or batch was finished
    
    
    #this is our custom handler that will be used in place of the keras methods:
        #`on_batch_end(batch,logs)` or `on_epoch_end(epoch,logs)`
    def on_end_handler(self, index, logs):
        
        currentTime      = time.time()                           
        self.elapsedTime = currentTime - self.startTime    #total time taken until now
        thisTime         = currentTime - self.lastTime     #time taken for the current epoch
                                                             #or batch to finish
        self.lastTime = currentTime
        
        #verifications will be made based on the longest epoch or batch
        if thisTime > self.longestTime:
            self.longestTime = thisTime
        
        
        #if the (assumed) time taken by the next epoch or batch is greater than the
            #remaining time, stop training
        remainingTime = self.maxExecutionTime - self.elapsedTime
        if remainingTime < self.longestTime:
            
            self.model.stop_training = True  #this tells Keras to not continue training
            print("\n\nTimerCallback: Finishing model training before it takes too much time. (Elapsed time: " + str(self.elapsedTime/60.) + " minutes )\n\n")
            
            #if we have passed the `on_interrupt` callback, call it here
            if self.on_interrupt is not None:
                self.on_interrupt(self.model, self.elapsedTime)
                
                
                
                

class PlotCurves(Callback):
    # Callback: PlotCurves
    # plot losses and accuracy after each epoch and save the weights of the best model
    
    def __init__(self, model_name):
        self.model_name = model_name
        
    def on_train_begin(self, logs={}):
        self.epoch = 0
        self.best_epoch = 0
        self.best_auc_epoch = 0
        self.x = []
        self.losses = []
        self.acc = []
        self.auc = []
        self.val_losses = []
        self.val_acc = []
        self.val_auc = []
        self.best_val_acc = 0
        self.best_val_auc = 0
        self.fig = plt.figure(figsize=(10, 5))
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.fig = plt.figure(figsize=(10, 5))
        self.logs.append(logs)
        self.x.append(self.epoch)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.auc.append(logs.get('auc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))
        self.val_auc.append(logs.get('val_auc'))
        self.epoch += 1
        
       # (Possibly) update best validation accuracy and save the network
        if self.val_acc[-1] > self.best_val_acc:
            self.best_val_acc = self.val_acc[-1]
            self.best_epoch = self.epoch
            self.model.save_weights(os.path.join('./Model/weights', self.model_name + 'best_acc_model.h5'))
            
        # (Possibly) update best validation AUC and save the network
        if self.val_auc[-1] > self.best_val_auc:
            self.best_val_auc = self.val_auc[-1]
            self.best_auc_epoch = self.epoch
            self.model.save_weights(os.path.join('./Model/weights', self.model_name + 'best_auc_model.h5'))
        
        display.clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.plot(self.x, self.acc, label="acc")
        plt.plot(self.x, self.val_acc, label="val_acc")
        plt.plot(self.x, self.auc, label="auc")
        plt.plot(self.x, self.val_auc, label="val_auc")
        plt.legend()
        plt.title('Best validation accuracy = {:.2f}% on epoch {} of {} \n' \
                  'Best validation AUC = {:.2f}% on epoch {} of {}'.format(
                        100. * self.best_val_acc, self.best_epoch, self.epoch,
                        100. * self.best_val_auc, self.best_auc_epoch, self.epoch))
        plt.show();