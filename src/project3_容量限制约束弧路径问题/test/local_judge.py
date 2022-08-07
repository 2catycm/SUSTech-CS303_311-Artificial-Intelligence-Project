from pathlib import Path
import os
import time

time_limit_seconds = 60
random_seed = 0
solver_location = "../submit/CARP_solver.py"
data_location = "CARP_samples"
if __name__ == '__main__':
    cwd = Path('.')
    cwd = cwd.absolute()
    cwd /= data_location
    costs = []
    dataset_names = []
    times = []
    dats = list(cwd.rglob("*.dat"))
    for i, dat in enumerate(dats):
        name = dat.stem.split("_")[1]  # 要求命名按照 i_name 的形式。
        print(f"Running {i}/{len(dats) - 1} instance {name}. ")
        param = f'"{dat}" '
        param += f'-t {time_limit_seconds} '
        param += f'-s {random_seed} '
        with os.popen(f'python "{solver_location}" {param}', 'r') as p:
            start = time.time()
            result = p.readlines()  # 这一步就结束了
            time_used = time.time() - start
            times.append(time_used)
            dataset_names.append(name)
            cost = int(result[1].split()[1])
            costs.append(cost)
            print(f"Instance {name} finished with time {time_used:e}s and cost {cost}.")
            print()

    print(dataset_names)
    print(costs)
    print(f"Total cost is {sum(costs)}. Total time used is {sum(times)}s. ")

    # baseline 是        [309, 370, 6446, 4188, 344, 482, 212], 12351.
    # 使用 Ulusoy split   [309, 368, 6446, 4188, 344, 472, 212] 优化了一点点