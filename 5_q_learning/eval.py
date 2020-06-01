import pickle
import glob
import matplotlib.pyplot as plt

logs = glob.glob('pkl/0.01_*.pkl')

print(logs)

fig, axes = plt.subplots(2, len(logs), figsize=(20, 6))

for i, log in enumerate(logs):
    with open(log, 'rb') as f:
        unpickler = pickle.Unpickler(f)
        data = unpickler.load()

        step_log = data[0]
        state_log = data[1]
        num_success = data[2]
        num_fail = data[3]
        success_total_step = data[4]
        fail_total_step = data[5]

    axes[0, i].plot(step_log[:100])
    axes[1, i].plot(state_log[:100])

    print(num_success)
    print(num_fail)

    print(success_total_step)
    print(fail_total_step)

plt.show()