
def entrenar(data_train, columna, valor):
    import pandas as pd
    import numpy as np
    
    #Numero total de datos
    N = len(data_train)

    #Numero de datos que CARAVAN = 1
    N_c1 = float(len(data_train[data_train[columna]==valor]))


    #Numero de datos que CARAVAN = 0
    N_c0 = float(len(data_train[data_train[columna]!=valor]))

    #Probabiidad de que CARAVAN = 1
    P_c1 = N_c1/N

    
    #Se crea una lista con las columnas y eliminamos la ultima
    col = list(data_train.columns)
    col.remove(columna)
    columnas = col
    
    lista_de_columnas = []
    
    
    #Se comienzan a crear las listas, su nomenclatura sera 'valores_MOSTYPE' por ejemplo
    for i in columnas:
        exec('valores_{0} = data_train["{1}"].unique()'.format(i,i))
        
    lista_Nx = []
    lista_Ncx = [] 
    lista_pcx = []
    lista_epsilon = [] 
    lista_Score = []
    lista_nombre = []
    lista_valores = []
    
    
    for i in columnas:

        l = eval('valores_{}'.format(i))
        #print(i)
        
        for j in l:

            Nx = float(len(data_train[data_train[i]==j]))
            lista_Nx.append(Nx)      
  
        for j in l:
            
            Ncx = float(len(data_train[(data_train[i]==j) & (data_train[columna]==valor)]))
            lista_Ncx.append(Ncx)
            
              
                 
        for j in l:

            Nx=float(len(data_train[data_train[i]==j]))
            Ncx = float(len(data_train[(data_train[i]==j) & (data_train[columna]==valor)]))
            
            
            Pcx = Ncx/Nx
        
            lista_pcx.append(Pcx)
           
            e = float(Nx*((Pcx-P_c1)/((Nx*(P_c1*(1-P_c1))))**(0.5)))
            lista_epsilon.append(e)
            
            
        
        for j in l:

            Ncxcero = float(len(data_train[(data_train[i]==j) & (data_train[columna]!=valor)]))   
            Ncx = float(len(data_train[(data_train[i]==j) & (data_train[columna]==valor)]))

           
            Pcx = float((Ncx+1)/(N_c1+2))

            Pcxcero = float((Ncxcero+1)/(N_c0+2))
        
            try:
                S = float(np.log(Pcx/Pcxcero))
            
            finally:
                lista_Score.append(S)
                
                
    for i in columnas:
        la = eval('valores_{}'.format(i))
        for j in la:
            lista_nombre.append(i)
            
    for i in columnas:
        lados = eval('list(valores_{})'.format(i))
        lista_valores+=lados
        
        
    diccionario = {'Variable':lista_nombre,'Valor':lista_valores,
               'Nx':lista_Nx,'Ncx':lista_Ncx,'P(C|X)':lista_pcx, 'Epsilon':lista_epsilon,'Score':lista_Score}
    
    df = pd.DataFrame(diccionario, columns = ['Variable','Valor','Nx','Ncx','P(C|X)','Epsilon','Score'])
    
    return df, P_c1


def clasificar(data_test, data_calculos, columna, valor, p_c1):
    import numpy as np
    
    lista_prob = []
    lista_tem = []
    col = list(data_test.columns)
    col.remove(columna)
    columnas = col
    
    for j in data_test.index:
        lista_tem.clear()
        
        for i in columnas:
            valor = data_test[i][j]
            #print(i,j)
            try:
                c = float(data_calculos['Score'][(data_calculos['Variable'] == i) & (data_calculos['Valor'] == valor)])
                #print(c)
            except TypeError:
                c = 0
                #print('Not found')
                #c = -2.0
        
            lista_tem.append(c)
        
        lista_prob.append(np.sum(lista_tem) + np.log(p_c1))
        
    data_test['Score'] = lista_prob
    
    return data_test


def fit(data, columna, valor, testSize, puntos, umbralE, umbralS):
    
    from sklearn.model_selection import train_test_split
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    
    data = eliminarNulos(data,-1)
    #umbralS = 3
    
    train, test = train_test_split(data, test_size = testSize)
    trained, p_c1 = entrenar(train, columna, valor)
    trained = filtrar(trained, 'Epsilon', umbralE)
    clasificados = clasificar(test, trained, columna, valor, p_c1, umbralS)
    
    vp, vn, fp, fn = cuentas(clasificados, columna, valor, umbralS)
    
    #matrizDeConfusion(vp, vn, fp, fn)
    
    tasaFP, tasaVP = roc(clasificados, columna, valor, puntos)
    
    with pd.ExcelWriter("pruebaFinal_yYa_mEVOaMimir.xlsx") as writer:
        clasificados.to_excel(writer, sheet_name="Score", index=False)
        entrenados.to_excel(writer, sheet_name="Training", index=False)

    return clasificados, trained


def fitValidacionCruzada(data, columna, valor, testSize, puntos, umbralE, umbralS):
    #from sklearn.model_selection import KFold
    from sklearn.model_selection import train_test_split
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
        
    list_ent = []
    list_clasif = []
    list_desemp = []
    
    data = eliminarNulos(data,-1)
    #umbralS = 3
    #nFolds = 2
    
    data = data.sample(frac=1)
    
    nFolds =  int (1 / testSize)
    #print (nFolds)
    
    
    #for i in range (nFolds):
    for i in range (1):
        train, test = splitVC(data, nFolds, i, testSize)
        trained, p_c1 = entrenar(train, columna, valor)
        list_ent.append(filtrar(trained, 'Epsilon', umbralE))
        list_clasif.append(clasificar(test, trained, columna, valor, p_c1, umbralS))
    
        #vp, vn, fp, fn = cuentas(clasificados, columna, valor, umbralS)
    
        #atrizDeConfusion(vp, vn, fp, fn)
        #tasaFP, tasaVP = roc(clasificados, columna, valor, puntos)
        
        #list_desemp.append(vp, vn, fp, fn, tasaFP, tasaVP)
        
        
    clasificados = pd.concat(list_clasif, ignore_index=True)
    entrenados = pd.concat(list_ent, ignore_index=True)
    entrenados.groupby(['Variable','Valor'])['Nx','Ncx','P(C|X)','Epsilon','Score'].agg('mean').reset_index()
    
    with pd.ExcelWriter("pruebaFinal_yYa_mEVOaMimir.xlsx") as writer:
        clasificados.to_excel(writer, sheet_name="Score", index=False)
        entrenados.to_excel(writer, sheet_name="Training", index=False)
    
    return clasificados, entrenados

#==================================================================================
def cuentas(data, columna, valor, umbralS):
    
    vp = float(len(data[(data[columna]==valor) & (data['Score']> umbralS)]))
    vn = float(len(data[(data[columna]!=valor) & (data['Score']<= umbralS)]))
    fp = float(len(data[(data[columna]!=valor) & (data['Score']> umbralS)]))
    fn = float(len(data[(data[columna]==valor) & (data['Score']<= umbralS)]))
    
    return vp, vn, fp, fn

def splitVC(data, nFolds, contador, test_size):
    #print(contador)
    
    import math
    
    size = math.floor(test_size*len(data))
    #print (size)
    corteInferior = contador*size
    corteSuperior = (contador+1)*size
    
    #print (corteInferior, corteSuperior, size)
    
    test = data.iloc[corteInferior:corteSuperior]
    train = data.drop(data.index[range(corteInferior, corteSuperior)])
    
    return train, test

def rocCuentas(data, columna, valor, puntos):
    import matplotlib.pyplot as plt
    import numpy as np
    #import seaborn as sns
    import pandas as pd
    
    #print(data['Score'].max(),data['Score'].min())
    cubeta = int(len(data)/puntos)
    #print(cubeta)
    #print(cubeta)
    tasaFP = []
    tasaVP = []
    vp = []
    vn = []
    fp = []
    fn = []
    s = []

    mylists = [vp, vn, fp, fn, fp]

    #rint(len(data))

    n=1
    while n <= len(data):
        #print(n)
        #print(data['Score'].nlargest(n*cubeta))
        umbralS = data['Score'].nlargest(n*cubeta).iloc[-1]
        #print(umbralS)
        s.append(umbralS)
        appenders = cuentas(data, columna, valor, umbralS)
    
        for x, lst in zip(appenders, mylists):
            lst.append(x)
        
            #vp, vn, fp, fn = cuentas(data, columna, valor, umbralS)
        
        tasaFP.append(fp[-1] / (vn[-1] + fp[-1]))
        tasaVP.append(vp[-1] / (vp[-1] + fn[-1]))
        
        n += cubeta
        #print(metricas)
    
    diccionario = {'vp':vp,'vn':vn,'fp':fp, 'fn':fn,'TasaFP':tasaFP, 'TasaVP':tasaVP,'Score':s}
    cuentasDF = pd.DataFrame(diccionario, columns = ['vp','vn','fp', 'fn','TasaFP', 'TasaVP', 'Score'])

    cuentasDF.head(10)
    return cuentasDF

def roc(data, columna, valor, puntos):
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    
    #print(data['Score'].max(),data['Score'].min())
    cubeta = int(len(data)/puntos)
    #print(cubeta)
    #print(cubeta)
    tasaFP = []
    tasaVP = []
    
    n=1
    while n <= len(data):
        #print(n)
        #print(data['Score'].nlargest(n*cubeta))
        umbralS = data['Score'].nlargest(n*cubeta).iloc[-1]
        #print(umbralS)
        vp, vn, fp, fn = cuentas(data, columna, valor, umbralS)
        tasaFP.append(fp / (vn + fp))
        tasaVP.append(vp / (vp + fn))
        #print(vp, vn, fp, fn, vp + vn + fp + fn, tasaFP, tasaVP)
        #curvaROC.append([tasaFP,tasaVP])
        #sns.set_style("darkgrid")
        n += cubeta
    plt.plot(tasaFP,tasaVP)
    #print(curvaROC)
    #return tasaFP, tasaVP
    return None

def matrizDeConfusion(vp, vn, fp, fn):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    
    cm = np.array([[vp,  fn],[fp,  vn]])
    #print('imprimiendo matriz de confusion')
    sns.heatmap(cm, cmap = sns.cm.rocket_r)
    return None

def ordenarMayor(data,columna, cuantos):

    data = data.sort_values(by=columna, ascending = False).head(cuantos)
    return data

def ordenarMenor(data,columna, cuantos):

    data = data.sort_values(by=columna, ascending = True).head(cuantos)
    return data

def filtrar(data, columna, umbral):

    data = data[(abs(data[columna]) >= umbral)]
    return data
    
#=========================================================================

def eliminarNulos(d1,valor):
    for j in d1.columns: 
        c=0
        for i in d1[j].isnull():
        
            if i==True:
                d1[j][c]=valor
            c+=1
    return d1

    
def binnearDatos (data, columna, noCubetas):
    import numpy as np
    import pandas as pd
    
    maximo = data[columna].max()
    minimo = data[(data[columna] > 0)][columna].min()
    #print(minimo, maximo)
    cortes = np.linspace(minimo ,maximo, noCubetas + 1) #cortes son los límites de los bins
    cubetas = cortes + (cortes[1] - cortes[0])/2 #cubetas guarda los valores representantes de los bins
    cubetas = cubetas[:-1].copy() 
    
    columnaBinneada = pd.cut(data[columna], cortes, labels = cubetas)
    data[columna] = columnaBinneada.astype(float, copy=True, errors='raise')
    
    return data

def binnearVariosDatos (data, columnas, noCubetas):
    import numpy as np
    import pandas as pd
    
    for columna in columnas:
        maximo = data[columna].max()
        minimo = data[(data[columna] > 0)][columna].min()
        #print(minimo, maximo)
        cortes = np.linspace(minimo ,maximo, noCubetas + 1) #cortes son los límites de los bins
        cubetas = cortes + (cortes[1] - cortes[0])/2 #cubetas guarda los valores representantes de los bins
        cubetas = cubetas[:-1].copy() 
        
        columnaBinneada = pd.cut(data[columna], cortes, labels = cubetas)
    
        data[columna] = columnaBinneada.astype(float, copy=True, errors='raise')
    
    return data


#====================Overloading some methods=======================#

def clasificar(data_test, trained, columna, valor, p_c1, umbralE, umbralS):
    import numpy as np
    
    trained = filtrar(trained, 'Epsilon', umbralE)
    
    lista_prob = []
    lista_tem = []
    col = list(data_test.columns)
    col.remove(columna)
    columnas = col
    
    for j in data_test.index:
        lista_tem.clear()
        
        for i in columnas:
            valor = data_test[i][j]
        
            try:
                c = float(trained['Score'][(trained['Valor']==valor)&(trained['Variable']== i)])
                #print(c)
            except TypeError:
                c = 0
                #print('Not found')
                #c = -2.0
        
            lista_tem.append(c)
        
        lista_prob.append(np.sum(lista_tem) + np.log(p_c1))
        
    data_test['Score'] = lista_prob
    
    return data_test

def calcularScores(data_test, trained, columna, valor, p_c1):
    import numpy as np
    
    lista_prob = []
    lista_tem = []
    col = list(data_test.columns)
    col.remove(columna)
    columnas = col
    
    for j in data_test.index:
        lista_tem.clear()
        
        for i in columnas:
            valor = data_test[i][j]
        
            try:
                c = float(trained['Score'][(trained['Valor']==valor)&(trained['Variable']== i)])
                #print(c)
            except TypeError:
                c = 0
                #print('Not found')
                #c = -2.0
        
            lista_tem.append(c)
        
        lista_prob.append(np.sum(lista_tem) + np.log(p_c1))
        
    data_test['Score'] = lista_prob
    
    return data_test


def fitValidacionCruzada(data, columna, valor, testSize, umbralE, tipo):
    #from sklearn.model_selection import KFold
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
        
    list_ent = []
    list_clasif = []
    list_desemp = []
    
    data = eliminarNulos(data,-1)
    #umbralS = 3
    #nFolds = 2
    
    data = data.sample(frac=1)
    
    nFolds =  int (1 / testSize)
    #print (nFolds)
    
    #for i in range (nFolds):
    for i in range (nFolds):
        train, test = splitVC(data, nFolds, i, testSize)
        trained, p_c1 = entrenar(train, columna, valor)
        trained = filtrar(trained, 'Epsilon', umbralE)
        list_ent.append(trained)
        list_clasif.append(calcularScores(test, trained, columna, valor, p_c1))
    
        #vp, vn, fp, fn = cuentas(clasificados, columna, valor, umbralS)
    
        #atrizDeConfusion(vp, vn, fp, fn)
        
        #list_desemp.append(vp, vn, fp, fn, tasaFP, tasaVP)
        
        
    clasificados = pd.concat(list_clasif, ignore_index=True)
    entrenados = pd.concat(list_ent, ignore_index=True)
    
    entrenados = entrenados.groupby(['Variable','Valor'])[['Nx', 'Ncx', 'P(C|X)', 'Epsilon', 'Score']].agg('mean').reset_index()
    roc=rocCuentas(clasificados, columna, valor, 100)
    #roc(clasificados, columna, valor, 100)
    
    nombreArchivo = columna +'_'+ valor + '_e' + str(umbralE)+'_'+ tipo + '.xlsx'
    
    with pd.ExcelWriter(nombreArchivo) as writer:
        entrenados.to_excel(writer, sheet_name="Training", index=False)
        clasificados.to_excel(writer, sheet_name="Score", index=False)
        roc.to_excel(writer, sheet_name="Métricas", index=False)
    return clasificados, entrenados, roc