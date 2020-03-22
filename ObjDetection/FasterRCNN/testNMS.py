import py_cpu_nms as nms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat

def drawRectange(regs):
    fig,ax = plt.subplots()
    colors = "bgrcmyk"
    color = colors[np.random.randint(7)]

    for cur in regs:
        rect = pat.Rectangle(cur[0:2], cur[2],cur[3],
                             color=colors[np.random.randint(7)], fill=False, linewidth=3)
        ax.add_patch(rect)
    plt.axis('equal')
    plt.grid()
    plt.show()

'''
a set of rectangles with following format:
([x11,y11,x12,y12]...[xn1,yn1,xn2,yn2])
'''
def createRandRegns(num):
    x1y1 = np.random.randint(1, 20, (num, 2));
    deta = np.random.randint(1, 20, (num, 2));
    x2y2 = x1y1 + deta;
    score = np.random.random((num,1))

    return np.concatenate((x1y1,x2y2,score), axis=1)

def createRandRegns1(num):
    regs = np.array([[ 34.000, 52.000, 39.000, 69.000, 0.143],
                     [ 39.000, 68.000, 55.000, 74.000, 0.511],
                     [ 82.000, 37.000, 99.000, 55.000, 0.011],
                     [ 40.000, 46.000, 45.000, 62.000, 0.249],
                     [ 87.000, 79.000, 101.000, 89.000, 0.381]])
    return regs

if __name__ == '__main__':
    regs = createRandRegns(10)
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    print(regs)
    drawRectange(regs)
    keep = nms.py_cpu_nms(regs, 0.2)
    print("results:", keep)