from django.shortcuts import render
from PIL import Image
from appExaPract.Logica import modeloSNN
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from django.http import JsonResponse
import base64
import io

class Clasificacion():
    def determinarAprobacion(request):
        return render(request, "VistaComentarios.html")
    
    @api_view(['GET', 'POST'])
    def predecir(request):
        try:
            imagen = request.FILES.get('imagen')
            if imagen:
                imagen_pil = Image.open(imagen)
                resul = modeloSNN.modeloSNN.predecirNUevoCliente(modeloSNN.modeloSNN,imagen_pil)

                imagen_jpg = imagen_pil.convert("RGB")
                buffer = io.BytesIO()
                imagen_jpg.save(buffer, format='JPEG')
                imagen_bytes = buffer.getvalue()
                imagen_codificada = base64.b64encode(imagen_bytes).decode('utf-8')

            else:
                resul = 'No se ha seleccionado ninguna imagen'
                imagen_codificada = ''
        except:
            resul = 'Error al procesar la imagen'
            imagen_codificada = ''

        return render(request, "Informe.html", {"e": resul, "imagen_codificada": imagen_codificada})