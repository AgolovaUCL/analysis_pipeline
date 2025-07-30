from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

def run_pca(derivatives_base, n_features = 2):
    """
    Here we run PCA to classify the units
    """

    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=2)

    principalComponents = pca.fit_transform(x)

    principalDf = pd.DataFrame(data = principalComponents
                , columns = ['principal component 1', 'principal component 2'])
    
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    ax.grid()
    plt.show()
