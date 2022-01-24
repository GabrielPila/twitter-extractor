# twitter-extractor

Extractor de Twitter

## Instrucciones:

1. Clonar el repositorio: `git clone git@github.com:GabrielPila/twitter-extractor.git`
2. Ingresar al repositorio: `cd twitter-extractor`
3. Crear un ambiente virtual: `virtualenv venv`
4. Activar ambiente virtual: `source venv/bin/activate`
5. Instalar requirements: `pip install -r requirements.txt`
6. Crear un archivo `.env`: `touch .env` 
7. colocar la credencial `BEARER_TOKEN` de la siguiente manera en el archivo `.env`:
```bash
BEARER_TOKEN=AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
``` 
6. Configurar detalles en `params.py`
7. Ejecutar `python api/main.py`