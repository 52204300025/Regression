# -!- coding: utf-8 -!-
#----配置环境
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文(windows)
plt.rcParams['axes.unicode_minus'] = False   # 用来正常显示负号
#导入相应的模型
from  xgboost import XGBRegressor#XGBOOST
from sklearn.ensemble import RandomForestRegressor#随机森林
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
#设定文件夹
import os
os.chdir('/home/shn/ML data/RRF/RF')
#读入数据文件
data=pd.read_excel('1-Octene-1014.xlsx')
# df_test=pd.read_excel('test.xlsx').iloc[:,0:6]
# #----数据编码
# ss=data['precatalyst'].append(df_test['precatalyst'])
# m=ss.unique()
# le=LabelEncoder()
# le.fit(m)
# re = le.transform(ss)
# data['precatalyst']=re[0:len(data['precatalyst'])]
# df_test['precatalyst']=re[len(data['precatalyst']):len(ss)]

#----划分自变量和因变量
data.columns
X=data[['precatalyst', 'S/C', 'solvent', 'temperature', 'time', 'bar']]
name='linear'
y=data[name]#在这里更改需要测试的因变量

#如果你需要更改就把括号里的改成：  线性比(l:b)
#----查看相关性
df_cor=data.corr()
fig1=plt.figure()
sns.heatmap(data=df_cor,cmap="YlGnBu",robust=True,annot=True)
plt.title('Correlation matrix',fontsize=15)
plt.show()
#----划分数据
from sklearn.model_selection import train_test_split
#此时的划分比例为0.3
X_train, X_test,y_train, y_test =train_test_split(X,y,test_size=0.3,random_state=904)

r2=[]
mse_=[]
rmse_=[]
mae_=[]
mape_=[]

train_pre=pd.DataFrame(np.zeros((len(y_train),4)))
test_pre=pd.DataFrame(np.zeros((len(y_test),4)))
#----自动调参
#导入评价指标
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import median_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
#定义新的评价指标
def rmse(actual: np.ndarray, predicted: np.ndarray):
    """ Root Mean Squared Error """
    return np.sqrt(mse(actual, predicted))

names=['XGBOOST',   'Random Forest',     'SVM']

models=[XGBRegressor(),RandomForestRegressor(),SVR()]

ps={'p1':{'n_estimators': range(50, 100, 10),#树的个数
                  'max_depth':range(1, 10, 2) ,#这个值为树的最大深度。这个值也是用来避免过拟合的。max_depth越大，模型会学到更具体更局部的样本。
                  'learning_rate' :np.linspace(0.1, 0.5, 5, endpoint=True)},
    'p2':{'n_estimators': range(50, 100, 10),#树的个数
                  'max_depth':range(1, 10, 2) },
    'p3':{'C': [4, 5,6,7,8],  'gamma': [0.1,0.2,0.3],'kernel':[ 'rbf','sigmoid']}}
best_p={}
i=0
for m_name,model in zip(names,models):
   bas_model=model
   print('Training'+m_name+'Model')
   nn='p'+str(i+1)
   p=ps[nn]
    #--调节参数
   parameters = p
   grid = GridSearchCV(estimator=bas_model, param_grid=parameters)
   grid.fit(X_train,y_train)
   print(m_name,grid.best_params_)
   print(grid.best_score_)
   print(m_name+'train-R2:',r2_score(y_train,grid.predict(X_train)))
   print(m_name+'test-R2:',r2_score(y_test,grid.predict(X_test)))
   print('=======================================================================')
   best_p[m_name]=grid.best_params_#记录svm的最优参数
   
   r2.append(r2_score(y_test,grid.predict(X_test)))
   mse_.append(mse(y_test,grid.predict(X_test)))
   rmse_.append(rmse(y_test,grid.predict(X_test)))
   mae_.append(mae(y_test,grid.predict(X_test)))
   mape_.append(mape(y_test,grid.predict(X_test)))
   
   train_pre.iloc[:,i]=grid.predict(X_train)
   test_pre.iloc[:,i]=grid.predict(X_test)
   
   
   
   fig2=plt.figure()    
   sns.regplot(x=train_pre.iloc[:,i],y=y_train)
   plt.xlabel('predictions')
   plt.ylabel('actual value')
   plt.title(name+'Training sets performance plots'+'-'+m_name)
   plt.show()
    
   fig3=plt.figure()    
   sns.regplot(x=test_pre.iloc[:,i],y=y_test,color='violet')
   plt.xlabel('predictions')
   plt.ylabel('actual value')
   plt.title(name+'Test sets performance plots'+'-'+m_name)
   plt.show()
   
   i+=1
   
#----堆叠模型
#---stacking()
#from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from mlxtend.regressor import StackingCVRegressor
clf1 = XGBRegressor(learning_rate=best_p['XGBOOST']['learning_rate'],max_depth=best_p['XGBOOST']['max_depth'],n_estimators=best_p['XGBOOST']['n_estimators'])
clf2 = RandomForestRegressor(max_depth=best_p['Random Forest']['max_depth'],n_estimators=best_p['Random Forest']['n_estimators'])
clf3 = SVR(C=best_p['SVM']['C'],gamma=best_p['SVM']['gamma'])
l_model = LinearRegression()
#tree = DecisionTreeRegressor(max_depth= 5,min_samples_split= 3)
sre= StackingCVRegressor(regressors=[clf1, clf2, clf3],                                                      
                            meta_regressor=l_model,
                            random_state=42)
   
sre.fit(X_train,y_train)

print('Stacking'+'train-R2:',r2_score(y_train,sre.predict(X_train)))
print('Stacking'+'test-R2:',r2_score(y_test,sre.predict(X_test)))
print('=======================================================================')


r2.append(r2_score(y_test,sre.predict(X_test)))
rmse_.append(rmse(y_test,sre.predict(X_test)))
mae_.append(mae(y_test,sre.predict(X_test)))
mape_.append(mape(y_test,sre.predict(X_test)))
train_pre.iloc[:,3]=sre.predict(X_train)
test_pre.iloc[:,3]=sre.predict(X_test)

fig4=plt.figure()    
sns.regplot(x=train_pre.iloc[:,3],y=y_train)
plt.xlabel('predictions')
plt.ylabel('actual value')
plt.title(name+'Training sets performance plots'+'-'+'Stacking')
plt.show()
    
fig5=plt.figure()    
sns.regplot(x=test_pre.iloc[:,3],y=y_test,color='violet')
plt.xlabel('predictions')
plt.ylabel('actual value')
plt.title(name+'Test sets performance plots'+'-'+'Stacking')
plt.show()
   

#----绘制预测曲线
#-1，训练集
train_pre.columns=['XGBOOST_pre','RandomForest_pre','SVM_pre','StackingModel_pre']
fig6=plt.figure()
colors=['blue','violet','gold','tomato']
styles=['--',':','-.','dashdot']
i=0
for n in train_pre:
    plt.plot(train_pre[n],color=colors[i],label=n,linestyle=styles[i])
    plt.legend()
    i=i+1
y_train.index=list(range(len(y_train)))
plt.plot(y_train,label='True')
plt.title('training sets',fontsize=13)
plt.show()
#-2，测试集
test_pre.columns=['XGBOOST_pre','RandomForest_pre','SVM_pre','StackingModel_pre']
fig7=plt.figure()
colors=['blue','violet','gold','tomato']
styles=['--',':','-.','dashdot']
i=0
for n in test_pre:
    plt.plot(test_pre[n],color=colors[i],label=n,linestyle=styles[i])
    plt.legend()
    i=i+1
    
y_test.index=list(range(len(y_test)))
plt.plot(y_test,label='True')
plt.legend()

plt.title('test sets',fontsize=13)
plt.show()
#----输出指标

import numpy as np
zhibiao=pd.DataFrame([np.round(rmse_,4),np.round(mae_,4),np.round(mape_,4),np.round(r2,4)],index=['RMSE','MAE','MAPE','R2'],columns=['XGBOOST','RandomForest','SVM','StackingModel'])
print(zhibiao.T)


print('best parameters:',best_p)

for key in best_p:
    print(key+'best parameters:',best_p[key],sep='\n')
    
#----输出结果

train_pre.to_csv('train-pre.csv')
test_pre.to_csv('test-pre.csv')

#----新数据预测
clf1.fit(X_train,y_train)
clf2.fit(X_train,y_train)
clf3.fit(X_train,y_train)
chanlv=pd.DataFrame([clf1.predict(df_test),
                     clf2.predict(df_test),
                     clf3.predict(df_test),
                     sre.predict(df_test)]).T
chanlv.columns=['XGBOOST','RandomForest','SVM','StackingModel']
chanlv.to_csv('new-data.csv')


