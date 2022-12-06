N_samples = 3000
base_source = 10
patch_size = 3 # including own element

elements = [2**i for i in range(3, 18)]
for i in elements.copy():
    if i < (patch_size):
        elements.remove(i)

sampling_freq = []
funcs = []
for i in elements:
    if (N_samples / (base_source * (i - (patch_size-1)))) <= 1:
        freq = int(i / (N_samples/base_source + (patch_size-1)))
        sampling_freq.append(freq) 
        funcs.append(base_source)      
        
    else:
        sampling_freq.append(1)
        funcs.append(int(N_samples / i))
        

