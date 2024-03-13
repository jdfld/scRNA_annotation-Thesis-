import data_proc_methods as dp
import model
import scanpy as sc

#adata_chunk1 = data_proc_methods.pipeline('data_chunk1.h5ad')

#X_train = adata_chunk1.obsm['X_harmony']
#y_train = adata_chunk1.obs['supercluster_term'].values

#rfc = model()
#rfc.fit(X_train,y_train)

#adata_chunk2 = data_proc_methods.pipeline('data_chunk2.h5ad')

#X_val = adata_chunk2.obsm['X_harmony']
#y_val = adata_chunk2.obs['supercluster_term'].values
#print(rfc.predict_acc(X_val,y_val))


thalamic_data = dp.pipeline('thalamic_complex24327.h5ad')
print('Loaded')
sc.pp.neighbors(thalamic_data,n_pcs=50,use_rep='X_harmony')
print('Finished')
