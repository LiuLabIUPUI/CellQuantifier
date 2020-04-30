import matplotlib.pyplot as plt

def vis_learning_stats(stats, output_dir, metrics):

    fig, ax = plt.subplots(1, 3)

    print(stats.history)

    ax[0].xlabel("Epoch")
    ax[0].ylabel("Loss")
    ax[0].plot(stats.history["loss"])
    ax[0].plot(stats.history["val_loss"])
    ax[0].legend(["Training", "Validation"])

    ax[1].xlabel("Epoch")
    ax[1].ylabel("Accuracy")
    ax[1].plot(stats.history["categorical_accuracy"])
    ax[1].plot(stats.history["val_categorical_accuracy"])
    ax[1].legend(["Training", "Validation"])

    ax[2].xlabel("Epoch")
    ax[2].ylabel("Accuracy")
    ax[2].plot(stats.history["binary_accuracy"])
    ax[2].plot(stats.history["val_binary_accuracy"])
    ax[2].legend(["Training", "Validation"])

    plt.savefig(output_dir + '/train_stats.png')
