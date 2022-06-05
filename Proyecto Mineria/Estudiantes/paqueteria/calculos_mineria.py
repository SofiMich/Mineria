def calculo(data_train, columna, valor):
    
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
        
    
    #print(lista_de_columnas)
        
    lista_Nx = []
    lista_Ncx = [] 
    lista_pcx = []
    lista_epsilon = [] 
    lista_Score = []
    lista_nombre = []
    lista_valores = []
    
    
    for i in columnas:
        
        #exec('print(valores_{0})'.format(i))
        l = eval('valores_{}'.format(i))
        
        for j in l:

            Nx = float(len(data_train[data_train[i]==j]))
            lista_Nx.append(Nx)
            
            
            #try:
             #   parametro = float(j)
              #  print(i,j)
               # print(type(j))
                #Nx = eval('float(len(data_train[data_train["{0}"]=={1}]))'.format(i,j))
                #lista_Nx.append(Nx)
             
            
            #except ValueError:
            
             #   Nx = eval('float(len(data_train[data_train["{0}"]=="{1}"]))'.format(i,j))
              #  lista_Nx.append(Nx)
             
                
                
  
        for j in l:
            
            Ncx = float(len(data_train[(data_train[i]==j) & (data_train[columna]==valor)]))
            lista_Ncx.append(Ncx)
            
            #try:
             #   parametro = float(j)

              #  Ncx = eval('float(len(data_train[(data_train["{0}"]=={1}) & (data_train[columna]==valor)]))'.format(i,j,i,j))
               # lista_Ncx.append(Ncx)

            #except ValueError:  
             #   Ncx = eval('float(len(data_train[(data_train["{0}"]=="{1}") & (data_train[columna]==valor)]))'.format(i,j,i,j))
              #  lista_Ncx.append(Ncx)
               
            
        
        for j in l:

            
            Nx=float(len(data_train[data_train[i]==j]))
            Ncx = float(len(data_train[(data_train[i]==j) & (data_train[columna]==valor)]))

            
            #try:
             #   parametro = float(j)            
              #  Nx=eval('float(len(data_train[data_train["{0}"]=={1}]))'.format(i,j))
               # Ncx = eval('float(len(data_train[(data_train["{0}"]=={1}) & (data_train[columna]==valor)]))'.format(i,j,i,j))           
                
            #except ValueError:
                #Nx=eval('float(len(data_train[data_train["{0}"]=="{1}"]))'.format(i,j))
                #Ncx = eval('float(len(data_train[(data_train["{0}"]=="{1}") & (data_train[columna]==valor)]))'.format(i,j,i,j))
               

            Pcx = Ncx/Nx
        
            lista_pcx.append(Pcx)
           
            e = float(Nx*((Pcx-P_c1)/((Nx*(P_c1*(1-P_c1))))**(0.5)))
            lista_epsilon.append(e)
            
            
        
        for j in l:

            Ncxcero = float(len(data_train[(data_train[i]==j) & (data_train[columna]!=valor)]))   
            Ncx = float(len(data_train[(data_train[i]==j) & (data_train[i]==j)]))

            
            
            #try:
             #   parametro = float(j)
                
              #  Ncxcero = eval('float(len(data_train[(data_train["{0}"]=={1}) & (data_train[columna]!=valor)]))'.format(i,j,i,j))    
               # Ncx = eval('float(len(data_train[(data_train["{0}"]=={1}) & (data_train["{2}"]=={3})]))'.format(i,j,i,j))
            
            #except ValueError:
              
             #   Ncxcero = eval('float(len(data_train[(data_train["{0}"]=="{1}") & (data_train[columna]!=valor)]))'.format(i,j,i,j))    
              #  Ncx = eval('float(len(data_train[(data_train["{0}"]=="{1}") & (data_train[columna]==valor)]))'.format(i,j,i,j))
            
           
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
    
    return df



#==================================================================================


def ordenarMayor(data,columna, cuantos):

    data = data.sort_values(by=columna, ascending = False).head(cuantos)
    return data

def ordenarMenor(data,columna, cuantos):

    data = data.sort_values(by=columna, ascending = True).head(cuantos)
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






