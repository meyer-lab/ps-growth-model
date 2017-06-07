import h5py
import matplotlib.pyplot as plt



        
# def walkers_info(col_num, dataset):    
#     for param_num in range(3, 17):
#         walkers_pvalues = []
#         for k in range(76):
#             one_param_one_walker = []
#             for l in range(dataset.shape[0] // 76):
#                 one_param_one_walker.append(dataset[k+l*75, param_num])
#             walkers_pvalues.append(geweke_single_chain(np.array(one_param_one_walker)))
        
#         yield param_num, walkers_pvalues

# def geweke_test(col_num = 3, file = 'first_chain.h5'):
#     file = h5py.File(file, 'r')
#     dataset = file['column' + str(col_num) + '/data']
    
#     for param_num, walkers_pvalues in walkers_info(col_num, dataset):
#         x = np.arange(1, len(walkers_pvalues)+1, 1)
#         y = np.array(walkers_pvalues)
#         plt.title('paramater ' + str(param_num))
#         plt.scatter(x, y)
#         plt.xlabel('walker number')
#         plt.ylabel('p-value')        
#         plt.show()
    
#     file.close()

# def pass_geweke_test(min_pval = .0001 , col_num = 3, file = 'first_chain.h5'):
#     file = h5py.File(file, 'r')
#     dataset = file['column' + str(col_num) + '/data']
    
#     flag = True
#     for param_num, walkers_pvalues in walkers_info(col_num, dataset):
#         for i in range(len(walkers_pvalues)):
#             if walkers_pvalues[i] < min_pval:
#                 print('Failed at parameter ' + str(param_num) + ', walker ' + str(i) + '. ' + 'P-value was ' + str(walkers_pvalues[i]))
#                 flag = False
    
#     return flag

# def all_cols_pass_geweke_test(min_pval = .0001, file = 'first_chain.h5'):
#     flag = True
#     for i in range(3, 20):
# #        print('entered column ' + str(i))
#         if not pass_geweke_test(min_pval, i, file):
#             print('Column ' + str(i) + ' failed.')
#             flag = False
#     return flag
