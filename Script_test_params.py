import subprocess
import csv

episodes = {500, 2000, 3000, 10000}

steps = {50, 100, 200, 300, 500, 700, 1000}

alpha = {0.01}

gamma = {0.99}

granu = {150}

header = ['episodes', 'max_steps', 'alpha', 'gamma', 'granu', 'result_1', 'result_2']
res_one = None
res_two = None

file = open('results.csv', 'w')
writer = csv.writer(file)
writer.writerow(header)

for ep in episodes:
    for st in steps:
        for al in alpha:
            for ga in gamma:
                for gr in granu:

                    proc = subprocess.Popen(['python3', 'run_game.py', str(ep), str(st), str(al), str(ga), str(gr)],
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

                    row = [ep, st, al, ga, gr, res_one, res_two]
                    writer.writerow(row)

                    res_one = None
                    res_two = None

file.close()
