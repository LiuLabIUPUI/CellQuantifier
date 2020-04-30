import matplotlib.pyplot as plt

def vis_learning_stats(stats, output_dir, metrics):
    plt.figure()

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(stats.history["loss"])
    plt.plot(stats.history["val_loss"])
    plt.legend(["Training", "Validation"])

    plt.savefig(output_dir + "plot_loss" + '.' + out_format, format=out_format)

    plt.figure()

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(stats.history["categorical_accuracy"])
    plt.plot(stats.history["val_categorical_accuracy"])
    plt.legend(["Training", "Validation"])

    plt.savefig(output_dir + "plot_accuracy" + '.' + out_format, format=out_format)

def visualize_learning_stats_boundary_hard(stats, output_dir, metrics):
    plt.figure()

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(stats.history["loss"])
    plt.plot(stats.history["val_loss"])
    plt.legend(["Training", "Validation"])

    plt.savefig(output_dir + "plot_loss" + '.' + out_format, format=out_format)

    plt.figure()

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(stats.history["binary_accuracy"])
    plt.plot(stats.history["val_binary_accuracy"])
    plt.legend(["Training", "Validation"])

    plt.savefig(output_dir + "plot_accuracy" + '.' + out_format, format=out_format)
