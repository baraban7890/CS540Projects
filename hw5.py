import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def hw5(filename):
    f = pd.read_csv(filename)
    print(f)
    year = []
    X = []
    Y = []
    for index,row in f.iterrows():
        x = []
        x.append(1)
        x.append(row['year']) 
        X.append(x)
        Y.append(row['days'])
        year.append(row['year'])
    X = np.array(X)
    Y = np.array(Y)
    
    print("Q3a:")
    print(X)
    print("Q3b:")
    print(Y)
    Z = np.dot(np.transpose(X),X)
    print(Z)
    I = np.linalg.inv(Z)
    print(I)
    PI = np.dot(I,np.transpose(X))
    print("Q3e:")
    print(PI)
    hat_beta = np.dot(PI,Y)
    print("Q3f:")
    print(hat_beta)
    y_test = hat_beta[0] + hat_beta[1] * 2021
    print("Q4: " + str(y_test))
    if(hat_beta[1] > 0):
        print("Q5a: >")
    elif(hat_beta[1] == 0):
        print("Q5a: =")
    else:
        print("Q5a: <")
    print("Q5b: This means that for each year that passes, the number of days that lake mendota is predicted to be frozen over decreases.")
    xstar = hat_beta[0] / -hat_beta[1]
    print("Q6a: " + str(xstar))
    print("Q6b: I believe xstar is a compelling prediction based on the trends because as time goes on, the regression predicts that the number of days that Mendota will be frozen over will decrease by .2, and this seems like an accurate estimation. Also, just by eyeballing the data, I can tell that the trend is downwards, as there were many more years with days frozen in the triple digits back then than there are now.")
    
    plt.plot(year,Y, color = "black")
    plt.title('# of days Lake Mendota is frozen over by year')
    plt.xlabel('Year')
    plt.ylabel('# of days Lake Mendota is frozen over')
    plt.savefig("plot.jpg")
        
if __name__ == "__main__":
    hw5(sys.argv[1])
