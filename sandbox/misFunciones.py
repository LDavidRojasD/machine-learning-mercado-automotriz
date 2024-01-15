import pandas as pd

# Funcion para calcular la impureza de gini en una variable independiente
def impurezaGini(caracteristica, clase, datos):
    """str, str, DataFrame -> float"""    
    atributoClase = datos.groupby([caracteristica, clase])[clase].count()
    atributo = datos.groupby([caracteristica])[clase].count()
    procesados = pd.merge(atributoClase, atributo, on = [caracteristica], 
                          suffixes = ('_individual', '_total')) 
    procesados["combinacion"] = (procesados[clase + "_individual"]/
                                 procesados[clase + "_total"])**2
    giniCombinacion = 1 - procesados.groupby([caracteristica, clase + "_total"])["combinacion"].sum()
    giniPesado = (giniCombinacion * atributo) / atributo.sum() 
    return giniPesado.sum()