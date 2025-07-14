import selfies as sf

# def convert():
#     # Add more data for better training
#     file_path = "datasets\\dataJ_250k_rndm_zinc_drugs_clean.txt"
#     with open(file_path, 'rt') as f:
#         lines = f.readlines()
#     smiles_data = [ line.strip() for line in lines ]
#     print("read", len(smiles_data))
#     selfies_data = []
#     for s in smiles_data:
#         print(s)
#         selfie = sf.encoder(s)
#         print(selfie)
#         selfies_data.append(selfie)
#     outfile = "datasets\\dataJ_250k_rndm_zinc_drugs_clean.sf.txt"
#     with open(outfile, "wt") as f:
#         f.write("\n".join(selfies_data))
   

# convert()
    
file_path = "datasets\\dataJ_250k_rndm_zinc_drugs_clean.sf.txt"
with open(file_path, "rt") as f:
    selfies_data = [line.strip() for line in f.readlines()]
