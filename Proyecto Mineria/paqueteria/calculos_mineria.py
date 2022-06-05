

def entrenar(data_train, columna, valor):

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


def clasificar(data_test, data_calculos, columna, valor, p_c1, umbralS):
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
                c = float(data_calculos['Score'][(data_calculos['Valor']==valor)&(data_calculos['Variable']== i)])
                print(c)
            except TypeError:
                c = 0
                print('Not found')
                #c = -2.0
        
            lista_tem.append(c)
        
        lista_prob.append(np.sum(lista_tem) + np.log(p_c1))
        
    data_test['Score'] = lista_prob
    
    return data_test


def fit(data, columna, valor, testSize, puntos, umbralE, umbralS):
    
    data = eliminarNulos(data,-1)
    #umbralS = 3
    
    train, test = train_test_split(data, test_size = testSize)
    trained, p_c1 = entrenar(train, columna, valor)
    trained = filtrar(trained, 'Epsilon', umbralE)
    clasificados = clasificar(test, trained, columna, valor, p_c1, umbralS)
    
    vp, vn, fp, fn = cuentas(clasificados, columna, valor, umbralS)
    
    #matrizDeConfusion(vp, vn, fp, fn)
    
    tasaFP, tasaVP = roc(clasificados, columna, valor, puntos)
    
    
    return clasificados, trained


def fitValidacionCruzada(data, columna, valor, testSize, puntos, umbralE, umbralS):
    
    from sklearn.model_selection import train_test_split
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    #from sklearn.model_selection import KFold
    
    list_ent = []
    list_clasif = []
    list_desemp = []
    
    data = eliminarNulos(data,-1)
    #umbralS = 3
    #nFolds = 2
    nFolds =  int (1 / testSize)
    #print (nFolds)
    
    
    for i in range (nFolds):
        train, test = train_test_split(data, test_size = testSize)
        trained, p_c1 = entrenar(train, columna, valor)
        list_ent.append(filtrar(trained, 'Epsilon', umbralE))
        list_clasif.append(clasificar(test, trained, columna, valor, p_c1, umbralS))
    
        #vp, vn, fp, fn = cuentas(clasificados, columna, valor, umbralS)
    
        #matrizDeConfusion(vp, vn, fp, fn)
        #tasaFP, tasaVP = roc(clasificados, columna, valor, puntos)
        
        #list_desemp.append(vp, vn, fp, fn, tasaFP, tasaVP)
        
        
    clasificados = pd.concat(list_clasif, ignore_index=True)
    entrenados = pd.concat(list_ent, ignore_index=True)
    
    return clasificados, entrenados


def fitValidacionCruzadaSKL(data, columna, valor, testSize, puntos, umbralE, umbralS):
    
    from sklearn.model_selection import train_test_split
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    list_ent = []
    list_clasif = []
    list_desemp = []
    
    data = eliminarNulos(data,-1)
    #umbralS = 3
    #nFolds = 2
    nFolds =  int (1 / testSize)
    #print (nFolds)
    
    
    for i in range (0):
        train, test = train_test_split(data, test_size = testSize)
        trained, p_c1 = entrenar(train, columna, valor)
        list_ent.append(filtrar(trained, 'Epsilon', umbralE))
        list_clasif.append(clasificar(test, trained, columna, valor, p_c1, umbralS))
    
        #vp, vn, fp, fn = cuentas(clasificados, columna, valor, umbralS)
    
        #matrizDeConfusion(vp, vn, fp, fn)
        #tasaFP, tasaVP = roc(clasificados, columna, valor, puntos)
        
        #list_desemp.append(vp, vn, fp, fn, tasaFP, tasaVP)
        
        
    clasificados = pd.concat(list_clasif, ignore_index=True)
    entrenados = pd.concat(list_ent, ignore_index=True)
    
    return clasificados, entrenados

#==================================================================================
def cuentas(data, columna, valor, umbralS):
    
    vp = float(len(data[(data[columna]==valor) & (data['Score']> umbralS)]))
    vn = float(len(data[(data[columna]!=valor) & (data['Score']<= umbralS)]))
    fp = float(len(data[(data[columna]!=valor) & (data['Score']> umbralS)]))
    fn = float(len(data[(data[columna]==valor) & (data['Score']<= umbralS)]))
    
    return vp, vn, fp, fn

def splitVC(data, nFolds, contador, test_size):
    import math
    #dividir
    size = math.floor(test_size*len(data))
    print (size)
    corteInferior = contador*size
    corteSuperior = (contador+1)*size
    
    test = data.iloc[corteInferior:corteSuperior]
    train = data.drop(data.index[range(corteInferior1, corteSuperior)])
    #

def roc(data, columna, valor, puntos):
    
    cubeta = int(len(data)/puntos)
    #curvaROC = []
    tasaFP = []
    tasaVP = []
    for n in range(1, puntos):
        #print(n)
        umbralS = data['Score'].nlargest(n*cubeta).iloc[-1]
        vp, vn, fp, fn = cuentas(data, columna, valor, umbralS)
        tasaFP.append(fp / (vn + fp))
        tasaVP.append(vp / (vp + fn))
        #curvaROC.append([tasaFP,tasaVP])
        
        #sns.set_style("darkgrid")
    plt.plot(tasaFP,tasaVP)
    #print(curvaROC)
    return tasaFP,tasaVP

def matrizDeConfusion(vp, vn, fp, fn):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    
    #print(vp, vn, fp, fn)
    
    cm = np.array([[vp,  fn],[fp,  vn]])
    print('imprimiendo matriz de confusion')
    sns.heatmap(cm, cmap = sns.cm.rocket_r)
    return None

def ordenarMayor(data,columna, cuantos):

    data = data.sort_values(by=columna, ascending = False).head(cuantos)
    return data

def ordenarMenor(data,columna, cuantos):

    data = data.sort_values(by=columna, ascending = True).head(cuantos)
    return data

def filtrar(data, columna, umbral):

    data = data[(data[columna] >= umbral)]
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


#=========================SCORE SOLO COLUMNA==============================

def fitVC(data, columna, valor, testSize, puntos, umbralE):
    
    from sklearn.model_selection import train_test_split
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    data = eliminarNulos(data,-1)
    umbralS = 3
    
    train, test = train_test_split(data, test_size = testSize)
    trained, p_c1 = entrenar(train, columna, valor)
    trained = filtrar(trained, 'Epsilon', umbralE)
    clasificados = clasificar(test, trained, columna, valor, p_c1, umbralS)
    
    vp, vn, fp, fn = cuentas(clasificados, columna, valor, umbralS)
    
    #matrizDeConfusion(vp, vn, fp, fn)
    
    tasaFP, tasaVP = roc(clasificados, columna, valor, puntos)
    
    return clasificados



#def prediccionScore(data_test,df,rango_1,rango_2,columnas): 
 #   lista_prob = []
  #  lista_tem = []
   # for j in range(rango_1,rango_2):
    #    lista_tem.clear()
     #   for i in columnas:
      #  	valor = data_test[i][j]
       #     try:
        #        c = float(df['Score'][(df['Valor']==valor)&(df['Variable']== i)])
       	#	except TypeError:
         #   		c = -2.0
        
        #	lista_tem.append(c)
    
    #	lista_prob.append(float(np.sum(lista_tem))) 
    #data_test.insert(loc = 86 , column = 'Score', value = lista_prob) 
    #return data_test 

#==================SCORE Y DEMAS VALORES======================================


#def prediccionScoreDos(data_test2,df,rango_1,rango_2, columnas):
#	lista_prob2 = []
#	lista_tem2 = []
 #   for j in range(rango_1,rango_2):
  #  	lista_tem2.clear()
   # 	for i in columnas:
    #    	valor = data_test2[i][j]
     #   	try:
      #          c = float(df['Score'][(df['Valor']==valor)&(df['Variable']== i)])
       # 	except TypeError:
        #    	c = -2.0
        #	data_test2.loc[j,i]=c
        #	lista_tem2.append(c)
    
    #	lista_prob2.append(float(np.sum(lista_tem2))) 
	
	#data_test2.insert(loc = 86 , column = 'Score', value = lista_prob2)
	#return data_test2


