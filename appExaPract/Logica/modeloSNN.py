from django.urls import reverse
import pandas as pd
from tensorflow.python.keras.models import load_model
from keras import backend as K
#from appCreditoBanco.Logica import modeloSNN
#import pickle
#import json
#from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import tokenizer_from_json
from keras_preprocessing.sequence import pad_sequences
import numpy as np
from PIL import Image

class modeloSNN():
    #Función para cargar red neuronal 
    def cargarNN(self,nombreArchivo):
        model = load_model(nombreArchivo+'.h5')    
        print("Red Neuronal Cargada desde Archivo") 
        return model

    def predecirNUevoCliente(self,ruta):
        dic = {0:'BAS' , 1:'EBO', 2:'EOS',3:'KSC' , 4:'LYA', 5:'LYT',6:'MMZ' , 7:'MOB', 8:'MOM',9:'MYB' , 10:'MYO', 11:'NGB',12:'NGS' , 13:'PMB', 14:'PMO'}
        imagen = ruta
        imagen2 = imagen.resize((32, 32))
        imagen2 = imagen2.convert("RGB")
        pixeles = np.array(imagen2).flatten()
        fila = pd.Series(pixeles)
        dfnew = pd.DataFrame(fila)
        dfnew = dfnew.T
        x = dfnew.values
        x = x.reshape(1, 32, 32, 3)
        modelo=self.cargarNN(self,'Recursos/modeloRedNeuronalOptimizada')
        pred = modelo.predict(x)
        pred_labels = np.argmax(pred, axis=1)
        ClaseMayorProbabilidad=np.argmax(pred)
        prob = pred.tolist()[0][ClaseMayorProbabilidad]
        prob = str(round(prob*100, 4)) + '%'
        salida = {"clase":dic[int (pred_labels)],"certeza":prob }
        print(salida)
        return salida