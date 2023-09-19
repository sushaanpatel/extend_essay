import matplotlib.pyplot as plt
from matplotlib.axis import Axis
import csv
import math
import statistics
import random
from timeit import default_timer as timer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np

#! Ulity functions for formatting and generating data
def getx(arr):
    out = []
    for i in arr:
        out.append(i[0])
    return out
        
def gety(arr):
    out = []
    for i in arr:
        out.append(i[1])
    return out

def load_data(size):
    samples = []
    for i in range(0, size+1):
        print("loading ", i , " element data...")
        ran = generate(i)
        samples.append(ran)
    return samples

def load_data_tocsv(size, comp, k):
    temp = load_data(size)
    out = []
    for i in temp:
        out.append(comp(i, k))
    name = ''
    if comp == k_sorted:
        name = 'k_sorted'
    elif comp == middle_sorted:
        name = 'middle_sorted'
    elif comp== border_sorted:
        name = 'border_sorted'
    with open(f"data/{name}.csv", 'w') as f:
        csvwriter = csv.writer(f)
        # csvwriter.writerow(["Length", "Data"])
        for i in range(0, len(out)):
            csvwriter.writerow(out[i])
            
def format_csv(file, flag):
    out = []
    with open(file, 'r') as f:
        reader = csv.reader(f)
        i = 0
        for row in reader:
            temp = []
            if row != []:
                for j in row:
                    temp.append(float(j))
                i += 1
                if flag:
                    out.append([i, temp[0]])
                else:
                    out.append(temp)
    return out

def generate(high):
    out = []
    for i in range(0, high):
        num = random.randint(0, high)
        out.append(num+random.uniform(0.0, random.uniform(0.0,20.0)))
    return out

def k_sorted(arr, k):
    InsertionSort(arr, 0, len(arr)-1, 0, 0, 0)
    for j in range(len(arr)):
        ran = random.randint(max(0, j - k), min(len(arr) - 1, j + k))
        temp = arr[j]
        arr[j] = arr[ran]
        arr[ran] = temp
    return arr

def splitarr(arr):
    n = len(arr)
    bottom = math.floor(n*(1/4))-1
    top = math.floor(n*(3/4))-1
    toparr = []
    middlearr = []
    bottomarr = []
    t = 0
    while t<=bottom:
        bottomarr.append(arr[t])
        t+=1
    m = bottom+1
    while m<=top:
        middlearr.append(arr[m])
        m+=1
    b = top+1
    while b<n:
        toparr.append(arr[b])
        b+=1
    return [bottomarr, middlearr, toparr]
    
def border_sorted(arr, k):
    InsertionSort(arr, 0, len(arr)-1, 0, 0, 0)
    splited = splitarr(arr)
    toparr = splited[2]
    middlearr = splited[1]
    bottomarr = splited[0]
    random.shuffle(middlearr)
    return bottomarr + middlearr + toparr
    
def middle_sorted(arr, k):
    InsertionSort(arr, 0, len(arr)-1, 0, 0, 0)
    splited = splitarr(arr)
    toparr = splited[2]
    middlearr = splited[1]
    bottomarr = splited[0]
    random.shuffle(bottomarr)
    random.shuffle(toparr)
    return bottomarr + middlearr + toparr

def mean(arr, low, high):
    sums = 0
    c = 0
    for i in range(low, high+1):
        sums = sums + arr[i]
        c += 1
    c = 1 if c==0 else c
    return sums/c

#! Sorting functions
def InsertionSort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

def bubbleSort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                temp = arr[j]
                arr[j] = arr[j+1]
                arr[j+1] = temp

def part(arr, low, high):
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i = i + 1
            temp = arr[j]
            arr[j] = arr[i]
            arr[i] = temp
    temp2 = arr[i+1]
    arr[i+1] = pivot
    arr[high] = temp2
    return i+1

def quicksort(arr, low, high):
    if low < high:
        pi = part(arr, low, high)
        quicksort(arr, low, pi-1)
        quicksort(arr, pi+1, high)


def hybridpart(arr, low, high, n):
    pivot = None
    if n == 1:
        pivot = math.floor(mean(arr, low, high))
        high = high + 1
        arr.insert(high, pivot)
    else:
        pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] < pivot:
            i = i + 1
            temp = arr[j]
            arr[j] = arr[i]
            arr[i] = temp
    temp2 = arr[i+1]
    arr[i+1] = pivot
    arr[high] = temp2
    return i+1

def hybridIn(arr, low, high):
    for i in range(low, high+1):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

def hybrid(arr, low, high, n, k, ogpi):
    if low < high:
        pi = hybridpart(arr, low, high, n)
        if n==1:
            ogpi = pi
            high = high + 1
        n = n + 1
        if abs(high-pi) < k:
            hybridIn(arr, pi + 1, high)
        else:
            hybrid(arr, pi + 1, high, n, k, ogpi)
        if n == 2:
            arr.pop(ogpi)
        if abs(pi-low) < k: 
            hybridIn(arr, low, pi - 1)
        else:
            hybrid(arr, low, pi - 1, n, k, ogpi)
    
#! Plotting and timing functions
def getTimestocsv(dset, func, low, k, label):
    out = []
    for i in dset:
        st = timer()
        if func == hybrid:
            func(i, low, len(i) - 1, 1, k, 0)
        elif func == quicksort:
            func(i, low, len(i) - 1)
        elif func == bubbleSort:
            func(i)
        elif func == InsertionSort:
            func(i)
        ed = timer()
        out.append([float(ed - st)])
    with open(f'results/{label}.csv', 'w') as file:
        write = csv.writer(file)
        for i in out:
            write.writerow(i)
        
    
def create_plot(dset, c1, c2, num, name, xlab, ylab):    
    xt = getx(dset)
    yt = gety(dset)

    x = np.array(xt)
    y = np.array(yt)
    
    poly = PolynomialFeatures(degree=4, include_bias=False)

    #reshape data to work properly with sklearn
    poly_features = poly.fit_transform(x.reshape(-1, 1))

    #fit polynomial regression model
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(poly_features, y)
    y_predicted = poly_reg_model.predict(poly_features)
    
    f = plt.figure(num)
    plt.plot(x, y_predicted, color=c1, label=name)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend(loc ="upper left")
    # plt.yticks(np.arange(0, 0.5, 0.02))
    # plt.bar(x, y, color=c2)
    f.show()


#* Proofs and Misc research
def mean_proof():
    data = load_sdev_data(100)
    plot_withoutreg(data[0], 'red', 'green', 1, 'Median', 'Number of elements', 'Value')
    plot_withoutreg(data[1], 'blue', 'yellow', 1, 'Mean', 'Number of elements', 'Value')
    plt.show()

def threshold(size):
    #! With static arrays
    data = [(load_data(size))[size-1]]
    k = []
    for i in range(5,101):
        k.append(i)
    out = []
    for i in k:
        st = timer()
        hybrid(data[0], 0, len(data) - 1, i)
        ed = timer()
        out.append([i, (ed - st)])
    for i in range(0,10):
        out.pop(i)
    return out
    
def avg_threshold(size):
    mins=[]
    for i in range(0,10):
        out = threshold(size)
        xt = getx(out)
        yt = gety(out)

        x = np.array(xt)
        y = np.array(yt)

        poly = PolynomialFeatures(degree=2, include_bias=False)

        #reshape data to work properly with sklearn
        poly_features = poly.fit_transform(x.reshape(-1, 1))

        #fit polynomial regression model
        poly_reg_model = LinearRegression()
        poly_reg_model.fit(poly_features, y)
        y_predicted = poly_reg_model.predict(poly_features)
        miny = min(y_predicted)
        minx = x[list(y_predicted).index(miny)]
        mins.append([minx, miny])
    temp = []
    for i in mins:
        temp.append(i[0])
    return [size, statistics.mean(temp)]

def plot_avg_threshold():
    data = []
    for i in range(1000,2100,100):
        data.append(avg_threshold(i))
    xt = getx(data)
    yt = gety(data)

    plt.plot(xt, yt, color='blue', label='Threshold value')
    plt.xlabel('Number of elements')
    plt.ylabel('Mean Threshold value')
    plt.legend(loc ="upper left")
    plt.yticks(np.arange(0,100,10))
    plt.show()

def plot_withoutreg(dset, c1, c2, num, name, xlab, ylab):
    xt = getx(dset)
    yt = gety(dset)

    x = np.array(xt)
    y = np.array(yt)
    f = plt.figure(num)
    
    plt.scatter(x, y, color=c1, label=name)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend(loc ="upper left")
    f.show()
    
def growthRates(function, name, color, num):
    x = np.linspace(100,10000,100)
    y = function(x)  
    plt.plot(x, y, label=name, color=color)
    
    plt.xlabel("Number of elements")
    plt.ylabel("Space used (bytes)")
    plt.legend(loc ="upper left")
    plt.xticks(np.arange(1000, 11000, 1000))
    # plt.yticks(np.arange(1000, 11000, 1000))
    f = plt.figure(num)
    f.show()