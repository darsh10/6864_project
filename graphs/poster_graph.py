import numpy as np
from matplotlib import pyplot as plt
import matplotlib 

def results_plt():
    font = {'family' : 'normal',
        'size'   : 22}

    title_font = {'family' : 'normal',
        'size'   : 28}

    axis_font = {'size':'20'}

    matplotlib.rc('font', **font)

    results = [0.1, 0.28, 0.3, 0.7*0.63,0.7*0.83]
    labels = ['Lucene', 'LogReg', 'RCNN', 
    'RCNN\n + Lucene\n(Est.)', 'RCNN\n + LogReg\n(Est.)']
    N = 5
    oracle = 0.7

    fig, ax = plt.subplots(figsize=(10,8)) 
    ax.spines["top"].set_visible(False)    
    ax.spines["bottom"].set_visible(False)    
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(False)    
    plt.ylim(0,1.1)
    ax.get_xaxis().tick_bottom()    
    ax.get_yaxis().tick_left()

    ind = np.arange(N)  # the x locations for the groups
    width = 0.65      # the width of the bars
    

    plt.plot(np.arange(N+1), [oracle for i in xrange(len(ind)+1)], '--', color='k', label='RCNN, 50 candidates')
    plt.legend(loc=2, frameon=False, fontsize=22)

    rects2 = ax.bar(ind + width/2, results, width, color='#73bfe5', edgecolor='none')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Accuracy P@1')
    ax.set_title('Pairing Weak Learner with RCNN Model', **title_font)
    ax.set_xticks(ind + width)
    ax.set_xticklabels(labels,**axis_font)

    # ax.legend((rects1[0], rects2[0]), ('Men', 'Women'))

    plt.savefig('fake_results.png',bbox_inches='tight')


if __name__ == '__main__':
    results_plt()