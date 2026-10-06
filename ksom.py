import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class Ksom:
    def __init__(self,ip,wt):
        self.ip = ip
        self.wt = wt
        self.max_possible = len(self.ip[0])*10+len(self.wt)
        print("max_possible = ",self.max_possible)
        
    def ksom_wp(self,lrate,epochs):
        self.lrate = lrate
        self.epochs = epochs

        history = []
        for init_wt in range(len(self.wt)):
            history.append([self.wt[init_wt].copy()])
        print("initial weights:",history)
        
        if len(self.wt[0]) != len(self.ip[0]):
            print("input and weight dimensions should be same !")
        elif lrate < 0 or lrate > 1:
            print("expected learning rate between 0 and 1 !")
        else:
            for epoch in range(self.epochs):
                print(f"{"___" * 10}epoch = {epoch+1}{"___"*28}")
                
                for i in self.ip:
                    print("~",i)

                    stored_distances = []
                    for j in self.wt:
                        '''print(f"{j} = {self.euclidean_distance(i,j)}") '''
                        stored_distances.append(self.euclidean_distance(i,j))

                    print(f"min = {np.min(stored_distances)} , ",end="")

                    min_index = 0
                    # weight updation
                    if self.all_equal(stored_distances):
                        min_index = np.random.randint(0,len(self.wt))
                    else:
                        min_index = stored_distances.index(np.min(stored_distances))
                    
                    print("min_index =",min_index,end=" , ")
                    self.wt[min_index] += self.lrate * (i - self.wt[min_index])
                    # self.lrate +=
                    
                    print("updated weight:",self.wt[min_index])
                    
                    # store history
                    history[min_index].append(self.wt[min_index].copy())
        '''print("\nhistory=",history)'''
        self.ksom_result = history
        
        return history
    
    def weight_summary(self,link):
        self.link = link
        split_digit=[int(d) for d in str(self.link)]
        '''print(split_digit)'''
        
        if self.link < self.max_possible:
            print(f"\nThe weight updation summary:{self.link}, max possible:",self.max_possible)
            history = self.ksom_result
            max_len = max(len(row) for row in history)
            '''print("max_len in history=",max_len)'''
            weight_size = len(history[0][0])
            
            transposed_matrix = [
                [row[i].copy() if i < len(row) else np.full(weight_size,np.nan) for row in history] for i in range(max_len)
            ]
            '''
            print("transposed = ",transposed_matrix)
            print("groups = ")
            for groups in transposed_matrix:
                print(groups)
            
            print([transposed_matrix[i][split_digit[1]-1][split_digit[0]-1] for i in range(len(transposed_matrix))])'''
    
    
            # plotting part
            y_points = [transposed_matrix[i][split_digit[1]-1][split_digit[0]-1] for i in range(len(transposed_matrix))]
            x_points = list(range(0,len(y_points)))
            print("x_points=",x_points,"y_points=",y_points)
    
            sns.lineplot(
                x=x_points,
                y=y_points,
                marker = "o",
                markerfacecolor="red",
                markeredgecolor="black"
            )
            plt.title(f"W{self.link}: weight updation graph")
            plt.xlabel("iteration step")
            plt.ylabel("weight")
            plt.grid(True)
            plt.show()
        else:
            print(f"weight updation for W{self.link} is not possible !")
            
    def euclidean_distance(self,x,w):
        d = [(x[i] - w[i])**2 for i in range(len(x))]
        return sum(d)

    def all_equal(self,lst):
        return True if len(set(lst)) == 1 else False
        
if __name__=="__main__":
    X = np.array([
        [1,0,1,0],
        [1,0,0,0],
        [1,1,1,1],
        [0,1,1,0]
    ])
    W = np.array([
        [0.3,0.5,0.7,0.2],
        [0.6,0.5,0.4,0.2]
    ])
    
    alpha = 0.6
    number_of_epochs = 3
    
    test1 = Ksom(X,W)
    test1.ksom_wp(alpha, number_of_epochs)
    test1.weight_summary(22)