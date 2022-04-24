import subprocess
import csv

episodes = {500, 1000, 1500, 2000, 3000, 4000, 5000, 7500, 10000, 12500, 15000, 17500, 20000}

steps = {50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2500}

alpha = {0.001, 0.005, 0.01, 0.05, 0.1}

gamma = {0.9, 0.92, 0.94, 0.96, 0.98, 0.99, 1}


header = ['episodes', 'max_steps', 'alpha', 'gamma', 'result_1', 'result_2']
res_one = None
res_two = None

file = open('results.csv', 'w')
writer = csv.writer(file)
writer.writerow(header)

for ep in episodes:
    for st in steps:
        for al in alpha:
            for ga in gamma:
                proc = subprocess.Popen(['python3', 'run_game.py', str(ep), str(st), str(al), str(ga)],
                                        stdout=subprocess.PIPE)
                while True:
                    line = proc.stdout.readline()
                    if not line:
                        break
                    if res_one is None:
                        res_one = line.rstrip().decode('UTF-8')
                    else:
                        res_two = line.rstrip().decode('UTF-8')

                    print(line.rstrip())

                row = [ep, st, al, ga, res_one, res_two]
                writer.writerow(row)

                res_one = None
                res_two = None

file.close()
