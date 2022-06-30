# Values are manually added from the evaulation of many experiemtns

import numpy as np
import matplotlib.pyplot as plt

train_size = [40, 30, 20, 10]

# Train size average accuracies and standard deviations
mean_acc_fs = np.array([0.63778, 0.602775, 0.54224, 0.47778]) 
mean_acc_svm = np.array([0.52592593, 0.52962963, 0.52592593, 0.49259259])
mean_acc_rf = np.array([0.48518519, 0.47777778, 0.47407407, 0.46296296])
mean_acc_knn = np.array([0.54444444, 0.55925926, 0.52962963, 0.56296296])
std_acc_fs = np.array([0.009961777, 0.015697771, 0.041129588, 0.047128675])
std_acc_svm = np.array([0.00523783, 0.00523783, 0.02771598, 0.03186046])
std_acc_rf = np.array([0.03434674, 0.00907218, 0.00523783, 0.02916299])
std_acc_knn = np.array([0.00907218, 0.01047566, 0.01385799, 0.03777051])

t = np.array(train_size) # X ticks
# std_acc = np.array([0, 0, 0, 0])

legend = ['Few-shot', 'SVM', 'Random Forest', 'KNN']

# Std Dev Plot
fig, ax = plt.subplots()
ax.plot(t,mean_acc_fs*100, color="c",label="label")
ax.plot(t,mean_acc_svm*100, color="r",label="label")
ax.plot(t,mean_acc_rf*100, color="g",label="label")
ax.plot(t,mean_acc_knn*100, color="b",label="label")

ax.fill_between(t, (mean_acc_fs-std_acc_fs)*100, (mean_acc_fs+std_acc_fs)*100, color='c', alpha=.1)
ax.fill_between(t, (mean_acc_svm-std_acc_svm)*100, (mean_acc_svm+std_acc_svm)*100, color='r', alpha=.1)
ax.fill_between(t, (mean_acc_rf-std_acc_rf)*100, (mean_acc_rf+std_acc_rf)*100, color='g', alpha=.1)
ax.fill_between(t, (mean_acc_knn-std_acc_knn)*100, (mean_acc_knn+std_acc_knn)*100, color='b', alpha=.1)

plt.title("Testing accuracy vs number of training samples",fontsize=14)
plt.xlabel("Training samples per class",fontsize=14)
plt.ylabel("Test accuracy (%)",fontsize=14)
plt.yticks(([35, 40, 45, 50, 55, 60, 65, 70]))
plt.xticks(t)
plt.legend(legend)
plt.tight_layout()
plt.savefig("results_plot_trainsz_all.jpg")


# Noise average accuracies and standard deviations
mean_acc_fs = np.array([0.6422, 0.60835, 0.5822, 0.58668, 0.5644, 0.54222]) 
mean_acc_svm = np.array([0.53333333, 0.54074074, 0.52592593, 0.52592593, 0.52592593, 0.54074074])
mean_acc_rf = np.array([0.46666667, 0.50740741, 0.46296296, 0.52592593, 0.5037037,  0.47407407])
mean_acc_knn = np.array([0.53333333, 0.54444444, 0.54814815, 0.53703704, 0.51851852, 0.54074074])
std_acc_fs = np.array([0.004986682, 0.022787277, 0.026771814, 0.026500887, 0.028776084, 0.053513475])
std_acc_svm = np.array([0., 0.00523783, 0.01385799, 0.01385799, 0.01385799, 0.02283116])
std_acc_rf = np.array([5.55111512e-17, 1.38579903e-02, 5.23782801e-03, 3.77705149e-02, 5.23782801e-03, 4.28734700e-02])
std_acc_knn = np.array([0., 0.02400274, 0.02771598, 0.01888526, 0.01385799, 0.01385799])

# Percent noisy data
t = np.array([0, 10, 20, 30, 40, 50])

# Std Dev Plot
fig, ax = plt.subplots()
ax.plot(t,mean_acc_fs*100, color="c",label="label")
ax.plot(t,mean_acc_svm*100, color="r",label="label")
ax.plot(t,mean_acc_rf*100, color="g",label="label")
ax.plot(t,mean_acc_knn*100, color="b",label="label")

ax.fill_between(t, (mean_acc_fs-std_acc_fs)*100, (mean_acc_fs+std_acc_fs)*100, color='c', alpha=.1)
ax.fill_between(t, (mean_acc_svm-std_acc_svm)*100, (mean_acc_svm+std_acc_svm)*100, color='r', alpha=.1)
ax.fill_between(t, (mean_acc_rf-std_acc_rf)*100, (mean_acc_rf+std_acc_rf)*100, color='g', alpha=.1)
ax.fill_between(t, (mean_acc_knn-std_acc_knn)*100, (mean_acc_knn+std_acc_knn)*100, color='b', alpha=.1)

plt.title("Testing accuracy vs percentage of noisy data",fontsize=14)
plt.xlabel("Training samples per class",fontsize=14)
plt.xlabel("Percentage of noisy data (%)",fontsize=14)
plt.ylabel("Test accuracy (%)",fontsize=14)
plt.yticks([35, 40, 45, 50, 55, 60, 65, 70])
plt.xticks(t)
plt.legend(legend)
plt.tight_layout()
plt.savefig("results_plot_noise_all.jpg")