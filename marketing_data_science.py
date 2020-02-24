
from sklearn.metrics import confusion_matrix, auc, accuracy_score
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pickle

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties 
import seaborn as sns 

#RFM
def RFM_plot_grid(df2,frequency_label,recency_label,label):
    df3 = pd.melt(df2.drop(columns = ['orderdate','recency','frequency']), id_vars=['clientId','customer','recency_cate','frequency_cate','gender'], var_name='types', value_name='values') 
    df3['values'] = pd.to_numeric(df3['values'],errors='coerce')
    df3 = df3.dropna()
    
    fig, axes = plt.subplots(6, 6,figsize=(25,15))
    counti = 0
    for i in frequency_label[::-1]:
        count = 6
        for j in recency_label:
            count -= 1 
            if df3[(df3['recency_cate']==j) & (df3['frequency_cate']==i)].shape[0] != 0:
                sns.barplot(x="types", y="values", data=df3[(df3['recency_cate']==j) & (df3['frequency_cate']==i)], capsize=.2,ax=axes[counti, count]) #.set_title("best customers")
        
        counti += 1 
    fig.savefig('RFM_plot_grid_'+label+'.png', dpi=300)
    

def RFM_stackedplot(df2, frequency_label,recency_label,label):
    
    df3 = pd.melt(df2.drop(columns = ['orderdate','recency','frequency']), id_vars=['clientId','customer','recency_cate','frequency_cate',label], var_name='types', value_name='values')
    df3['values'] = pd.to_numeric(df3['values'],errors='coerce')
    df3 = df3.dropna()
    
    fig, axes = plt.subplots(6, 6, figsize=(25, 15))
#   plt.figlegend( [ax.legend()], 'label1', label = 'lower center', ncol=5, labelspacing=0.1 )
    
    counti = 0
    for i in frequency_label[::-1]:
        count = 6
        for j in recency_label:
            count -= 1 
            if df3[(df3['recency_cate']==j) & (df3['frequency_cate']==i)].shape[0] != 0:
                df4 = df3[(df3['recency_cate']==j) & (df3['frequency_cate']==i)]
                df4 = df4.groupby(['types', label]).agg({'values': 'sum'})
                df4 = df4.groupby(['types', label]).sum()
                df4 =df4.groupby(level=1).apply(lambda x:100 * x / float(x.sum()))
                df4 = df4.add_suffix('').reset_index() #to df
                df4=df4.pivot(label, 'types', 'values')
                
                #draw
                ax = df4.plot.bar(stacked=True,width=0.7, legend = False, ax =axes[counti, count] ,rot=0)
                ax.legend( loc= 1, fontsize =8)                
                # sns.barplot(x="types", y="values", data=df3[(df3['recency_cate']==j) & (df3['frequency_cate']==i)], capsize=.2,ax=axes[counti, count]) #.set_title("best customers")
        counti += 1 
    
    fig.savefig('RFM_stackedplot_'+label+'.png', dpi=300)
    
    return ax, fig

# logistic model
def logistic_model(X_train,y_train,X_test,y_test,
                    sales_price = 3500,
                    marketing_expense = 300,
                    product_cost = 1750,
                    plot_name = 'logistic_regression'
                   ):
    
    X_train_log = X_train.copy()
    X_train_log['intercept'] = 1
    logistic = sm.Logit(y_train,X_train_log)
    
    # fit the model
    result = logistic.fit()
    result_df = results_summary_to_dataframe(result, plot_name = plot_name)
    
    
    X_test_log = X_test.copy()
    X_test_log['intercept'] = 1
    y_test_df=pd.DataFrame(y_test)
    y_test_df[plot_name+'_pred'] = result.predict(X_test_log)
    y_test_df['pred_yn']= np.where(y_test_df[plot_name+'_pred']>=0.5, 1,0)
    
    
    conf_logist = confusion_matrix(y_test_df['buy'], y_test_df['pred_yn'])
    
    plot_confusion_matrix(conf_logist, ['no','buy'],
                          title=plot_name+"Confusion Matrix plot", cmap=plt.cm.Reds)#, cmap=plt.cm.Reds
    
    # model_profit 
    model_profit = sales_price * conf_logist[1,1] - conf_logist[::,1].sum() * marketing_expense - product_cost * conf_logist[1,1]
    
    model_profit_df = pd.DataFrame({
            'categories' : ['price/product', 'cost/product', 'marketing cost/product', 'revenue'],
            'price' : [sales_price,product_cost, marketing_expense, '-'],
            'taget audience' : [conf_logist[1,1],conf_logist[1,1], conf_logist[::,1].sum(), '-'],
            'total' : [sales_price* conf_logist[1,1], product_cost* conf_logist[1,1], marketing_expense * conf_logist[::,1].sum(),model_profit  ],
            })
    
    
    # all_profit 
    all_profit = sales_price*conf_logist[1,::].sum() - product_cost* conf_logist[1,::].sum()- marketing_expense *  conf_logist.sum()
    
    all_df = pd.DataFrame({
            'categories' : ['price/product', 'cost/product', ',arketing cost/product', 'revenue'],
            'price' : [sales_price,product_cost, marketing_expense, '-'],
            'target audience' : [conf_logist[1,::].sum(), conf_logist[1,::].sum(), conf_logist.sum(), '-'],
            'total' : [sales_price*conf_logist[1,::].sum(), product_cost* conf_logist[1,::].sum(), marketing_expense *  conf_logist.sum(),all_profit  ],
            })
    
    
    # -------single model summary--------
    
    print( "################ summary ################ ")
    
    print(confusion_matrix(y_test_df['buy'], y_test_df['pred_yn']))
#    print("____________________{}cohort analysis____________________".format(plot_name))
#    print(classification_report(y_test_df['buy'], y_test_df['pred_yn']))
    print(accuracy_score(y_test_df['buy'], y_test_df['pred_yn']))      
    
          
    # importance
    print( '------------------ variables to notice ------------------' )
    print('、'.join(result_df['variables'].tolist()))
    print('\n'.join(result_df['meaning'].tolist()))
    
    
    # profit comparison
    if model_profit - all_profit > 0 :
        print( '------------------【losing profit】------------------' )
        print( 'model earns more than all  marketing $' + str(model_profit - all_profit) )
        print( 'revenue decreases' + str( round(model_profit / all_profit, 3) ) + 'times')
        
    else:
        print( '------------------【making profit】------------------' )
        print( 'model earns less than all marketing $' + str(model_profit - all_profit) )
        print( 'revenue increases' + str( round(model_profit / all_profit, 3) ) + 'times')
    
    print( '------------------marketing revenue matrix------------------' )
    print(all_df)
    all_df
    all_df.to_csv(plot_name+'marketing_revenue_matrix.csv',encoding = 'cp950')
    print('marketing_revenue_matrix.csv saved')
    
    print( '------------------' +plot_name+ 'model marketing revenue matrix------------------' )
    print(model_profit_df)
    model_profit_df.to_csv(plot_name+'model_marketing_revenue_matrix.csv',encoding = 'cp950')
    print('model_marketing_revenue_matrix.csv saved')
    
    
    print( '------------------' +plot_name+ ' important variables------------------' )
#   print(result_df)
    result_df.to_csv(plot_name+'important_variables.csv',encoding = 'cp950')
    return all_df, model_profit_df, result_df,y_test_df

def logistic_importance(
        X_train,
        y_train,
        X_test,
        y_test,
        plot_name
                   ):
        
    X_train_log = X_train.copy()
    X_train_log['intercept'] = 1
    logistic = sm.Logit(y_train,X_train_log)
    
    # fit the model
    result = logistic.fit()
    result_df = results_summary_to_dataframe(result, plot_name = plot_name)
    return result_df 

def logistic_conf(
        X_train,
        y_train,
        X_test,
        y_test,
        plot_name
                   ):
        
    X_train_log = X_train.copy()
    X_train_log['intercept'] = 1
    logistic = sm.Logit(y_train,X_train_log)
    
    # fit the model
    result = logistic.fit()
#   result_df = results_summary_to_dataframe(result, plot_name = plot_name)
        
        
        
    X_test_log = X_test.copy()
    X_test_log['intercept'] = 1
    y_test_df=pd.DataFrame(y_test)
    y_test_df['pred'] = result.predict(X_test_log)
    y_test_df['pred_yn']= np.where(y_test_df['pred']>=0.5, 1,0)
    
    
    conf_logist = confusion_matrix(y_test_df['buy'], y_test_df['pred_yn'])
    
    plot_confusion_matrix(conf_logist, ['no','buy'],
                          title=plot_name+"Confusion Matrix plot", cmap=plt.cm.Reds)#, cmap=plt.cm.Reds
    
    
    print( "################ summary ################ ")
    print('Confusion matrix')
    print(confusion_matrix(y_test_df['buy'], y_test_df['pred_yn']))
#   print("____________________{}cohort analysis____________________".format(plot_name))
#   print(classification_report(y_test_df['buy'], y_test_df['pred_yn']))
    print("Test Accuracy = {:.3f}".format(accuracy_score(y_test_df['buy'], y_test_df['pred_yn'])))
    
    
    return conf_logist 

#confusion matrix

import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        1#print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig.savefig(title+'.png', dpi=300)
    
def model_testRF(clf, 
                 X_train,
                 y_train,
                 X_test,
                 y_test,
                 plot_name = 'logistic_regression'):
    model = clf.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:,1]
    
    y_test_df=pd.DataFrame(y_test)
    y_test_df[plot_name+'_pred'] = y_pred_prob
    
    #Confusion Matrix
    conf_logist = confusion_matrix(y_test, y_pred)
    
 
    plot_confusion_matrix(conf_logist, ['no','buy'],
                          title=plot_name+"Confusion Matrix plot", cmap=plt.cm.Reds)#, cmap=plt.cm.Reds
    
    # -------single model summary--------
    

    print( "################ summary ################ ")
    
    print(confusion_matrix(y_test, y_pred))
#   print("____________________{}cohort analysis____________________".format(plot_name))
#   print(classification_report(y_test, y_pred))
    print("Training Accuracy = {:.3f}".format(model.score(X_train, y_train)))
    print("Test Accuracy = {:.3f}".format(model.score(X_test, y_test)))
