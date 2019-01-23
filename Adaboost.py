'''
Adaboost example
date : 2018.12.02
author : WX
reference : https://github.com/px528/AdaboostExample

'''


import numpy as np
import matplotlib.pyplot as plt
import pandas as  pd


#加载数据
def load_data():
    data = np.array([[1,2],[2,3],[4,4],[4,5],[9,9],[7,8],[1,7]])
    label = np.array([1.0,1.0,1.0,-1.0,-1.0,-1.0,1.0])
    return data,label


'''
计算弱分类器的阈值：
stepsize为步长
依照步长来决定阈值的大小

input: data
output: thresh
输出格式: [arrary([ .. ,.. ]),......,.....]
'''
def make_thresh(data):
    rangemin = np.min(data,axis=0)
    rangemax = np.max(data,axis=0)
    stepsize = 10.0
    step = (rangemax-rangemin)/stepsize
    thresh = [ ]
    thresh.append( rangemin + step)
    for t in range(int(stepsize)+1):
        if t < int(stepsize+1):
            thresh.append(thresh[t]+step)
        else:
            break
    print('阈值{}'.format(thresh))
    return thresh

'''
弱分类器:
根据阈值来判定输出的lable是多少
分类器方式: lt(lower than) 和 gt(greater than)

input: data
output: ret, thresh 分类结果，阈值
ret:
{'lt':{'0': arrary([ .. ,.. ]),......,....., '1': ..... , ....} , 'gt':{'0': arrary([ .. ,.. ]),......,....., '1': ..... , ....}}
'''

def weak_classifier(data):
    ret = { }
    outlt = { }
    outgt = { }
    thresh = np.array(make_thresh(data))
    for j in range(data.shape[1]):
        outlt[j] = [ ]
        outgt[j] = [ ]
        for i in range(data.shape[0]):
            for k in range(thresh.shape[0]):
                predlablelt = np.ones(data.shape[0])
                predlablelt[data[:,j] <= thresh[k][j]] = -1.0
                outlt[j].append(predlablelt)
                ret['lt'] = outlt
                predlablegt = np.ones(data.shape[0])
                predlablegt[data[:,j] > thresh[k][j]] = -1.0
                outgt[j].append( predlablegt)
                ret['gt'] = outgt
            else:
                break
    #print('预测结果:{}'.format(ret))
    return ret, thresh


'''
计算分类错误
通过分类器输出与真实leble比较，与lable不符合的将权重相加即为误差

input: ret, lable, D, dim
ret: 预测结果
lable: 真实分类
D: 各点权重
dim: 数据维度

output: error
{'lt':{'0': arrary([ .. ,.. ]),......,....., '1': ..... , ....} , 'gt':{'0': arrary([ .. ,.. ]),......,....., '1': ..... , ....}}
'''

def cal_error(ret, lable, D, dim):
    erlt = { }
    ergt = { }
    error = { }
    masklt = np.ones_like(lable)
    maskgt = np.ones_like(lable)
    for i in range(dim):
        erlt[i] = [ ]
        ergt[i] = [ ]
        for j in ret['lt'][i]:
             index1 = j != lable
             maskl = masklt*index1
             erlt[i].append(np.sum(D*maskl))
             error['lt'] = erlt
        for k in ret['gt'][i]:
             index2 = k != lable
             maskg = maskgt*index2
             ergt[i].append(np.sum(D * maskg))
             error['gt'] = ergt
    #print('误差{}'.format(error))
    return error


'''
计算最小的误差并返回相应得阈值分类器
input: error

return:
ret: 误差最小的分类器预测结果
key: 键值，表示分类器采用的是lt 还是 gt
i: 分类用的数据维度是哪一维
index: 阈值的索引
out[key][i][index]: 阈值
thresh: 全部阈值
'''
def cal_e_min( error):
    ret = 100000
    for key in error:
        for i in error[key]:
            temp = min(error[key][i])
            if ret > temp:
                ret = temp
    for key in error:
        for i in error[key]:
            if ret == min(error[key][i]):
                print(ret)
                return ret, key, i, error[key][i].index(ret)


'''
训练一个弱分类器
'''
def weak(data, lable, D):
    out, thresh = weak_classifier(data)
    #print(out)
    p = cal_error(out, lable, D, data.shape[1])
    #print(p)
    ret, key, i, index = cal_e_min(p)

    #--debug--
    #print('错误值是:{}'.format(ret))
    #print('弱分类器采用的是:{}'.format(key))
    #print('分类器相应的维度:{}'.format(i))
    #print('分类器采用的阈值的索引:{}'.format(index))
    #print('采用的阈值:{}'.format(thresh[index][i]))
    #print('预测的结果是:{}'.format(out[key][i][index]))

    return ret, key, i, index, out[key][i][index],thresh


#计算各弱分类器权重
def cal_alpha(e):
    if e == 0.5:
        return 0.0001
    else:
        return 0.5 * np.log((1 - e) /(e+1e-16))

#计算各点权重
def cal_D( D, alpha, lable, pred):
    nD=np.zeros_like(D)
    nD=D * np.exp(-1*alpha * lable * pred)
    return nD/np.sum(nD)

#最终的预测结果
def cal_final_pred(i, alpha, pred, lable):
    ret = np.zeros_like(lable)
    for j in range(i + 1):
        ret += alpha[j] * pred[j]
    return np.sign(ret)

#计算最终的错误
def cal_final_e(lable, cal_final_predict):
    ret = 0
    for i in range(len(lable)):
        if lable[i] != cal_final_predict[i]:
            ret += 1
    return ret / len(lable)

# 训练boost模型
def adaboost_train(data, lable, iteration):
    D = {}
    weakcls = {}
    alpha = {}
    pred = {}
    m = data.shape[0]

    for i in range(iteration):
        D.setdefault(i)
        weakcls.setdefault(i)
        alpha.setdefault(i)
        pred.setdefault(i)

    for i in range(iteration):
        if i == 0:
            D[i] = np.ones((m)) / m
        else:
            D[i] = cal_D(D[i - 1], alpha[i - 1], lable, pred[i - 1])

        ret, key, ie, index, predict,thresh=weak(data, lable, D[i])

        # 画出弱分类器的示意图
        if int(ie) == 0:
            plt.vlines(thresh[index][0], 0,10, colors=(0.5+i/10, 0.0+i/10, 0.3+i/10), label='weak_classifier '+str(i))
        else:
            plt.hlines(thresh[index][1], 0,10, colors=(0.1+i/10, 0.5+i/10, 0.1+i/10), label='weak_classifier '+str(i))

        pred[i] = predict
        alp = cal_alpha(ret)
        alpha[i] = alp

        cal_final_predict = cal_final_pred(i, alpha, pred, lable)

        print('迭代(iteration):{}'.format(i+1))
        print('最终预测:{}'.format(cal_final_predict) )
        print('最终误差:{}'.format(cal_final_e(lable, cal_final_predict)))

        if cal_final_e(lable, cal_final_predict) == 0 or ret == 0:
            break

    plot_data(data, lable)
    plt.legend(loc='upper left')
    plt.show()
    return i+1

def plot_data(data, lable):
    m,n = data.shape
    px1 = [  ]
    px2 = [  ]
    for i in range(m):
        if lable[i] == 1.0:
            px1.append(data[i])
        else:
            px2.append(data[i])
    pl=np.array(px1)
    pll=np.array(px2)
    plt.scatter(pl[:,0],pl[:,1],color='r')
    plt.scatter(pll[:,0],pll[:,1], color='b')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Final result')
#data, lable = load_data()
#adaboost_train(data,lable,3)
#plotdata(data, lable)



data = pd.read_csv('data.csv')

#get X and y
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values
y = np.array(y, dtype=np.float64)
adaboost_train(X, y,4)

