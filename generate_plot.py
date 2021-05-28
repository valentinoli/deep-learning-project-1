import matplotlib.pyplot as plt


def performance_plot(display_vals, NUM_ROUNDS, BOOTSTRAP):
    labels= ["Naive(1)\ntrain",
             "(Naive(1)\ntest",
             "Shared\nWeightNet(1)\ntrain",
             "Shared\nWeightNet(1)\ntest",
             "Shared\nWeightNet(1)'\ntrain",
             "Shared\nWeightNet(1)'\ntest",
             "Shared\nWeightNet(2)'\ntrain",
             "Shared\nWeightNet(2)'\ntest",
             "Benchmark\ntrain",
             "Benchmark\ntest",
            ]


    fig = plt.figure(figsize=(15, 6), dpi=80)
    plt.boxplot(display_vals, labels=labels, zorder=1)

    if BOOTSTRAP:
        bootstrap_text = ': {} rounds w/ bootstrap'.format(NUM_ROUNDS)
    else:
        bootstrap_text = ': {} rounds'.format(NUM_ROUNDS)
        
        for i in range(len(display_vals)):
            plt.plot([i+1]*len(display_vals[0]),display_vals[i],".", markersize=7, label=labels[i].replace('\n', ' '))

    plt.hlines(0.85, xmin=0.5, xmax=11, ls='--', color='gray')
    plt.title("Models performance comparison" + bootstrap_text)
    plt.ylabel("Bootstrapped Accuracy [%]")
    
    plt.show()
    fig.savefig("models_accuracy_boxplot.png")    