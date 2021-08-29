import matplotlib.pyplot as plt
import numpy as np

f = open("plot.txt", "r")
cross_point_errors = []
heading_erros = []
cross_point_errors = [float(f) for f in f.readline().split("CROSS_POINT_ERROR:")[1].strip().split(" ")]
print("Cross Point Mean: ", np.mean(cross_point_errors))
print("Cross Point Variance: ", np.var(cross_point_errors))
heading_erros = [float(f) for f in f.readline().split("HEADING_ERROR:")[1].strip().split(" ")]
print("Heading Error Mean: ", np.mean(heading_erros))
print("Heading Error Variance: ", np.var(heading_erros))
print("Heading error SD:", np.std(heading_erros))

plt.plot(np.arange(len(cross_point_errors)), cross_point_errors)
plt.xlabel("Readings")
plt.ylabel("Cross Point Error")
plt.title("Computation of cross point error at each interval")
plt.grid()
plt.show()

plt.plot(np.arange(len(heading_erros)), heading_erros)
plt.xlabel("Readings")
plt.ylabel("Heading Error")
plt.title("Computation of heading error at each interval")
plt.grid()
plt.show()
print(cross_point_errors)
print(heading_erros)
mean = np.mean(heading_erros)
sum = 0
for i in range(len(heading_erros)):
    sum = sum + (heading_erros[i] - mean)

print("variance ", sum/len(heading_erros))