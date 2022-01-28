import tensorflow as tf
import numpy      as np
from   scipy      import sparse
import HiCtoolbox

R     =25000
alpha = 0.227

model = tf.keras.models.load_model(r"model.h5") #model available at https://github.com/meet-eu-21/Team-SA2/
HiCfilename=r'' #Path to the HiCfilename

#Script by Leopold Caron and Julien Mozziconaci
#Build matrix
A=np.loadtxt(HiCfilename)
A=np.int_(A)
print('Input data shape : ',np.shape(A))
A=np.concatenate((A,np.transpose(np.array([A[:,1],A[:,0],A[:,2]]))), axis=0)#build array at pb resolution
A = sparse.coo_matrix( (A[:,2], (A[:,0],A[:,1])))
binned_map=HiCtoolbox.bin2d(A,R,R) #!become csr sparse array
print('Input at the good resolution : ',np.shape(binned_map))
del A #keep space
x = np.asarray(HiCtoolbox.SCN(binned_map))**alpha #not filtered matrix
#Script by Leopold Caron and Julien Mozziconaci



#Prediction

#Create (1,33,33) shaped windows from 
print("Processing data....")
windows = []
for i in range(0,len(x)-32):
    windows.append(x[i:i+33,i:i+33])
print("Done!")

print("Assess score...")
windows = np.array(windows).reshape((len(windows),1,33,33))
ress = model.predict(windows).ravel()
pos = np.array([x for x in range(16,len(x)-16)])
output = np.vstack((pos,ress)).T
np.savetxt("prediction.txt", output, header="chrom_pos score",comments='')
print("Done, score saved.")



        
