import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
sns.set()
f,ax=plt.subplots(figsize=(12,12))
mydic = {'Acne_Vulgaris': 0, 'Actinic_solar_Damage(Actinic_Keratosis)': 1, 'Actinic_solar_Damage(Pigmentation)': 2, 'Actinic_solar_Damage(Solar_Elastosis)': 3, 'Allergic_Contact_Dermatitis': 4, 'Alopecia_Areata': 5, 'Basal_Cell_Carcinoma': 6, 'Blue_Nevus': 7, 'Compound_Nevus': 8, 'Congenital_Nevus': 9, 'Cutaneous_Horn': 10, 'Dermatofibroma': 11, 'Dyshidrosiform_Eczema': 12, 'Dysplastic_Nevus': 13, 'Eczema': 14, 'Epidermoid_Cyst': 15, 'Ichthyosis': 16, 'Inverse_Psoriasis': 17, 'Keratoacanthoma': 18, 'Malignant_Melanoma': 19, 'Nevus_Incipiens': 20, 'Onychomycosis': 21, 'Perioral_Dermatitis': 22, 'Pityrosporum_Folliculitis': 23, 'Psoriasis': 24, 'Pyogenic_Granuloma': 25, 'Rhinophyma': 26, 'Sebaceous_Gland_Hyperplasia': 27, 'Seborrheic_Dermatitis': 28, 'Seborrheic_Keratosis': 29, 'Skin_Tag': 30, 'Stasis_Dermatitis': 31, 'Stasis_Edema': 32, 'Stasis_Ulcer': 33, 'Steroid_Use_abusemisuse_Dermatitis': 34, 'Tinea_Corporis': 35, 'Tinea_Faciale': 36, 'Tinea_Manus': 37, 'Tinea_Pedis': 38, 'Tinea_Versicolor': 39}
label = []
for mm in mydic.keys():
    label.append(mm)
print(label)
y_true = [0,0,1,2,1,2,0,2,2,0,1,1,39]
y_pred = [1,0,1,2,1,0,0,2,2,0,1,1,38]
for i in range(0,40):
    y_true.append(i)
    y_pred.append(i)
print (y_pred)
print (y_true)
C2= confusion_matrix(y_true, y_pred)
print(C2) #打印出来看看
sns.heatmap(C2,annot=True,ax=ax) #画热力图

ax.set_title('confusion matrix') #标题
ax.set_xlabel('predict') #x轴
ax.set_ylabel('true') #y轴
plt.show()

def print_confusion_matrix(y_pred,y_true):
    sns.set()
    f,ax=plt.subplots(figsize=(12,12))
    C2= confusion_matrix(y_true, y_pred)
    sns.heatmap(C2,annot=True,ax=ax,cmap="greys") #画热力图
    ax.set_title('confusion matrix') #标题
    ax.set_xlabel('predict') #x轴
    ax.set_ylabel('true') #y轴
    plt.show()