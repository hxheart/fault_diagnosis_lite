import numpy as np
import matplotlib.pyplot as plt
import os

num_train_begin = 4
num_train_end   = 5

# ######################################################################################

# for index_run in range(num_train_begin, num_train_end):
#     print('\n ------------------> index run (GAT 2018 ICLR):', index_run)
#     with open(f'./train_fd_GAT_ICLR_2018.py', 'r', encoding='utf-8') as file:
#         exec(file.read())
#     # ----------------------
#     np.save(f'./visual_GAT_2018_loss_{index_run}.npy',     np.array(visual_loss))
#     np.save(f'./visual_GAT_2018_accuracy_{index_run}.npy', np.array(visual_accuracy))

# # ######################################################################################

for index_run in range(num_train_begin, num_train_end):
    print('\n ------------------> index run (GATv2 2022 ICLR):', index_run)
    with open(f'./train_fd_GATv2_ICLR_2022.py', 'r', encoding='utf-8') as file:
        exec(file.read())
    # ----------------------
    np.save(f'./visual_GATv2_2022_loss_{index_run}.npy',     np.array(visual_loss))
    np.save(f'./visual_GATv2_2022_accuracy_{index_run}.npy', np.array(visual_accuracy))

# ######################################################################################

# for index_run in range(num_train_begin, num_train_end):
#     print('\n ------------------> index run (EtaGAT):', index_run)
#     with open(f'./train_fd_EtaGAT_20251014.py', 'r', encoding='utf-8') as file:
#         exec(file.read())
#     # ----------------------
#     np.save(f'./visual_EtaGAT_loss_{index_run}.npy',     np.array(visual_loss))
#     np.save(f'./visual_EtaGAT_accuracy_{index_run}.npy', np.array(visual_accuracy))

######################################################################################


# for index_run in range(num_train_begin, num_train_end):
#     print('\n ------------------> index run (EtaGAT):', index_run)
#     with open(f'./train_fd_EtaGATv2_20251025.py', 'r', encoding='utf-8') as file:
#         exec(file.read())
#     # ----------------------
#     np.save(f'./visual_EtaGATv2_loss_{index_run}.npy',     np.array(visual_loss))
#     np.save(f'./visual_EtaGATv2_accuracy_{index_run}.npy', np.array(visual_accuracy))

# ######################################################################################




