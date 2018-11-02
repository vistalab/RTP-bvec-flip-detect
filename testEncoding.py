import src.features.build_features as feat
import src.data.preprocess as prep
import pickle
import numpy as np

encodeArr = pickle.load(open('data/processed/encodeList12.p', 'rb'))
realArr = pickle.load(open('data/processed/tensorsAndNames48to12.p', 'rb'))

encodeSample = encodeArr[0][50]
realSample = realArr[0][50]

x = np.copy(realSample)
y = np.copy(realSample)
z = np.copy(realSample)
x = feat.flipAxes(x, specificAxis=1)[0]
y = feat.flipAxes(y, specificAxis=2)[0]
z = feat.flipAxes(z, specificAxis=3)[0]

xEn = prep.getEncodeArray(x)
yEn = prep.getEncodeArray(y)
zEn = prep.getEncodeArray(z)

print(prep.getEncodeArray(realSample)[5,5,5])
print(encodeSample[5,5,5])
print(xEn[5,5,5])
print(yEn[5,5,5])
print(zEn[5,5,5])

pass
