import matplotlib.pyplot as plt

def display(*arg) :
    for elts in arg :
        plt.plot(list(range(1,len(elts)+1)), elts)   
    plt.show()