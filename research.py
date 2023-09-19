import sys
from funcs import *
from threading import Thread
sys.setrecursionlimit(10**6)
deets = [[quicksort, "Quick", "red"], [bubbleSort, "Bubble", "black"], [hybrid, "Hybrid", "purple"], [InsertionSort, "Insertion", "blue"]]
# deets = [[InsertionSort, "Insertion", "blue"]]

def CreateData(size):
    load_data_tocsv(size, k_sorted, 3)
    load_data_tocsv(size, middle_sorted, 3)
    load_data_tocsv(size, border_sorted, 3)

def CollectTime():
    #*loading data
    ksort = format_csv('data/k_sorted.csv', False)
    msort = format_csv('data/middle_sorted.csv', False)
    bsort = format_csv('data/border_sorted.csv', False)


    #*Collecting timing data
    # det = [[quicksort, "Quick", "red"]]
    for i in deets:
        if i[0] == bubbleSort:
            th3 = Thread(target = getTimestocsv, args=(bsort, i[0], 0, 62, f"{i[1]}_bsort"))
            print(f"Starting bsort for {i[1]}")
            th3.start()
            th3.join()
        else:
            th1 = Thread(target = getTimestocsv, args=(ksort, i[0], 0, 62, f"{i[1]}_ksort"))
            th2 = Thread(target = getTimestocsv, args=(msort, i[0], 0, 62, f"{i[1]}_msort"))
            th3 = Thread(target = getTimestocsv, args=(bsort, i[0], 0, 62, f"{i[1]}_bsort"))
            
            print(f"Starting ksort for {i[1]}")
            th1.start()
            th1.join()
            print(f"Starting msort for {i[1]}")
            th2.start()
            th2.join()
            print(f"Starting bsort for {i[1]}")
            th3.start()
            th3.join()

def PlotData():
    for i in range(0,4):
        k = format_csv(f'results/{deets[i][1]}_ksort.csv', True)
        m = format_csv(f'results/{deets[i][1]}_msort.csv', True)
        b = format_csv(f'results/{deets[i][1]}_bsort.csv', True)
        
        create_plot(k, "green", '', i+1, f'{deets[i][1]} Sort for K-sorted dataset', 'Number of elements', 'Time')
        create_plot(m, "blue", '', i+1, f'{deets[i][1]} Sort for middle sorted dataset', 'Number of elements', 'Time')
        create_plot(b, "red", '', i+1, f'{deets[i][1]} Sort for border sorted dataset', 'Number of elements', 'Time')
        
    plt.show()
    
def rates():
    def I(x):
        return (8*x) + 72

    def B(x):
        return (8*x) + 88

    def Q(x):
        return (8*x) + (72*(np.log2(x))) + 8
    
    def H(x):
        return (8*x) + (248*(np.log2(x/62)))

    growthRates(I, 'Insertion sort', "purple", 1)
    growthRates(B, 'Bubble sort', "blue", 1)
    growthRates(Q, 'Quick sort', "green", 1) 
    growthRates(H, 'Hybrid sort', "red", 1)
    plt.show()
    
def comb_eval():
    datats = [["ksort", "K-sorted"], ["msort", "middle sorted"], ["bsort", "border sorted"]]
    for i in range(0,3):
            d1 = format_csv(f'results/{deets[0][1]}_{datats[i][0]}.csv', True)
            d2 = format_csv(f'results/{deets[1][1]}_{datats[i][0]}.csv', True)
            d3 = format_csv(f'results/{deets[2][1]}_{datats[i][0]}.csv', True)
            d4 = format_csv(f'results/{deets[3][1]}_{datats[i][0]}.csv', True)
            
            create_plot(d1, "green", '', i+1, f'{deets[0][1]} Sort for {datats[i][1]} dataset', 'Number of elements', 'Time')
            create_plot(d2, "blue", '', i+1, f'{deets[1][1]} Sort for {datats[i][1]} dataset', 'Number of elements', 'Time')
            create_plot(d3, "red", '', i+1, f'{deets[2][1]} Sort for {datats[i][1]} dataset', 'Number of elements', 'Time')
            create_plot(d4, "purple", '', i+1, f'{deets[3][1]} Sort for {datats[i][1]} dataset', 'Number of elements', 'Time')
            
    plt.show()
    
comb_eval()


