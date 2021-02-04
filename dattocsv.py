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
                os.remove(path + ".csv")
            except:
                pass
            csv = open(path + ".csv",'a+')
            dat_lines = dat.readlines()
            for line in dat_lines:
                if ("@relation" in line) or ("@attribute" in line) or ("@data" in line) or ("@outputs" in line) or ("@output" in line):
                    continue
                else:
                    if "@inputs" in line:
                        line = line[7:-1] + "," + "Class\n"
                    
                    line = line.replace(" ", "")
                    csv.write(line)
            dat.close()
            csv.close()