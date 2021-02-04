import os
keel_datasets_path = "Unsupervised_Anomaly_Detection"
keel_datasets_path = os.path.abspath(keel_datasets_path)
for root, dirs, files in os.walk(keel_datasets_path, topdown=False):
    for name in dirs:
        data_paths = []

        data_paths.append(keel_datasets_path + "\\" + name + "\\" + name + "-5-1tra")
        data_paths.append(keel_datasets_path + "\\" + name + "\\" + name + "-5-1tst")

        for path in data_paths:
            dat = open(path + ".dat","rt")
            try:
                os.remove(path + ".arff")
            except:
                pass
            arff = open(path + ".arff",'a+')
            dat_lines = dat.readlines()
            for line in dat_lines:
                if ("@output" in line) or ("@input" in line):
                    continue
                else:
                    line = line.lower()
                    arff.write(line)
            dat.close()
            arff.close()
