
import os
import pandas as pd
import numpy as np

def predict(data,model,result_root):

    for file in data.list_files:
        chromosomes_iterator = data.result_dataloader(file)
        
        #Final name 
        name = os.path.split(file)[1]
        name = name.replace(".csv","").replace(".gz","")

        g_res = {"chrom":[],"res":[]}
        for ch in range(len(data.result_chromosomes)):
            #Perform on the chromosomes one by one because if not there is shuffling
            res = []
            for inp,oup in chromosomes_iterator[ch]:
                res.append(model.model(inp).detach().numpy())

            res = np.concatenate(res).flatten()
            #print(res.shape)
            g_res["chrom"].extend([data.result_chromosomes[ch]]*len(res))
            g_res["res"] = np.concatenate([g_res["res"],res])

        final_file = os.path.join(result_root,f"{name}_prediction.csv")
        print("Saving to",final_file)
        pd.DataFrame(g_res).to_csv(final_file,index=False)