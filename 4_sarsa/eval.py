import pickle
import matplotlib.pyplot as plt

with open('./pkl/log_epsilon.pkl', 'rb') as f:
    step_log, agent_elog, num_success, num_fail, success_total_step, fail_total_step = pickle.load(f)

plt.plot(step_log, color='brown')
plt.show()

plt.plot(agent_elog)
plt.show()

print(f"NUM SUCCESS : {num_success}")
print(f"NUM SUCCESS : {num_fail}")
print(f"TOTAL SUCCESS STEP : {success_total_step}")
print(f"TOTAL FAIL STEP : {fail_total_step}")

# with open('log.pkl', 'rb') as f:
#     step_log, num_success, num_fail, success_total_step, fail_total_step = pickle.load(f)
#
# plt.plot(step_log, color='brown')
# plt.show()
#
# print(f"NUM SUCCESS : {num_success}")
# print(f"NUM FAIL : {num_fail}")
# print(f"TOTAL SUCCESS STEP : {success_total_step}")
# print(f"TOTAL FAIL STEP : {fail_total_step}")