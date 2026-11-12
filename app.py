import io
import zipfile
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import base64
import json
import streamlit.components.v1 as components

st.set_page_config(page_title='GoPro 360', layout='wide')
st.title('GoPro 360 · Fotografía y panorámicas')
st.caption('GoPro HERO4 Black · Corrección, ensamblaje y visor TIFF de Hugin')
st.warning('La corrección radial es visual y aproximada; no reemplaza una calibración geométrica de lente. Para PIX4Dmatic utiliza los JPG originales.')

files = st.file_uploader('Selecciona fotografías JPG', type=['jpg','jpeg','png'], accept_multiple_files=True)
strength = st.slider('Intensidad aproximada de corrección gran angular', 0.0, 0.8, 0.35, 0.05)
crop_pct = st.slider('Recorte de bordes después de corregir (%)', 0, 20, 0, 1)
max_side = st.select_slider('Resolución máxima para panorámica (px)', options=[800,1200,1600,2000], value=1200)

def decode(upload):
    data = np.frombuffer(upload.getvalue(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f'No se pudo leer {upload.name}')
    return img

def resize(img, limit):
    h,w = img.shape[:2]
    scale = min(1.0, limit/max(h,w))
    return cv2.resize(img, (round(w*scale),round(h*scale)), interpolation=cv2.INTER_AREA) if scale<1 else img

def approximate_correction(img, amount):
    # Visual radial remap only; not a calibrated camera model.
    if amount == 0:
        return img.copy()
    h,w = img.shape[:2]
    yy,xx = np.indices((h,w), dtype=np.float32)
    x = (xx-(w-1)/2)/((w-1)/2)
    y = (yy-(h-1)/2)/((h-1)/2)
    r2 = x*x+y*y
    factor = 1 + amount*r2
    mx = ((x/factor)*((w-1)/2)+(w-1)/2).astype(np.float32)
    my = ((y/factor)*((h-1)/2)+(h-1)/2).astype(np.float32)
    return cv2.remap(img,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)

def apply_crop(img, pct):
    if pct <= 0:
        return img
    h, w = img.shape[:2]
    dx, dy = round(w * pct / 100), round(h * pct / 100)
    return img[dy:h-dy, dx:w-dx].copy()

if files:
    st.success(f'{len(files)} fotografías cargadas')
    try:
        first=decode(files[0]); corrected=apply_crop(approximate_correction(first,strength),crop_pct)
        a,b=st.columns(2)
        a.image(cv2.cvtColor(resize(first,900),cv2.COLOR_BGR2RGB),caption='Original')
        b.image(cv2.cvtColor(resize(corrected,900),cv2.COLOR_BGR2RGB),caption='Corrección aproximada')
    except Exception as e:
        st.error(str(e)); st.stop()

    if st.button('Preparar ZIP de fotografías corregidas'):
        with st.spinner('Corrigiendo fotografías...'):
            buffer=io.BytesIO()
            with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
                for f in files:
                    img=apply_crop(approximate_correction(decode(f),strength),crop_pct)
                    ok, encoded=cv2.imencode('.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,92])
                    if ok:
                        z.writestr('corregidas/'+f.name.rsplit('.',1)[0]+'_corregida.jpg',encoded.tobytes())
            st.download_button('Descargar ZIP',buffer.getvalue(),file_name='gopro360_corregidas.zip',mime='application/zip')

    st.subheader('Ensamblaje panorámico experimental')
    st.info('OpenCV puede fallar con escenas de poca textura, ojo de pez intenso o fotografías que no se solapan. El resultado no garantiza cobertura esférica completa.')
    if st.button('Intentar crear panorámica'):
        if len(files)<2:
            st.warning('Necesitas al menos dos fotografías.')
        else:
            with st.spinner('Buscando correspondencias y ensamblando...'):
                try:
                    imgs=[resize(decode(f),max_side) for f in files]
                    stitcher=cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
                    status,pano=stitcher.stitch(imgs)
                    if status==cv2.Stitcher_OK:
                        st.image(cv2.cvtColor(pano,cv2.COLOR_BGR2RGB),caption='Panorámica ensamblada')
                        ok, encoded=cv2.imencode('.jpg',pano,[cv2.IMWRITE_JPEG_QUALITY,93])
                        if ok:
                            st.download_button('Descargar panorámica JPG',encoded.tobytes(),file_name='panoramica_gopro.jpg',mime='image/jpeg')
                    else:
                        st.error(f'No se pudo ensamblar (código OpenCV: {status}). Prueba con menos imágenes consecutivas o utiliza Hugin para controlar la lente y los puntos de control.')
                except Exception as e:
                    st.error(f'Error durante el ensamblaje: {e}')
else:
    st.info('Carga las fotografías originales para comenzar.')

st.divider()
st.caption('Para PIX4Dmatic utiliza los JPG originales, no los corregidos. Esta herramienta no genera una nube de puntos 3D.')


st.divider()
st.header("Visor panorámico 360° · TIFF de Hugin")
st.write("Carga la panorámica final exportada por Hugin (TIFF, JPG o PNG), no las 21 fotos individuales.")
pano_file = st.file_uploader("Panorámica ensamblada", type=["tif", "tiff", "jpg", "jpeg", "png"], key="pano_viewer")
if pano_file is not None:
    try:
        with Image.open(io.BytesIO(pano_file.getvalue())) as source:
            pano_image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = pano_image.size
        st.caption(f"Imagen original: {width} × {height} píxeles · Relación {width / height:.2f}:1")
        limit = st.select_slider("Ancho máximo para visualizar (menor consume menos memoria)", options=[1024, 2048, 3072, 4096], value=2048)
        if width > limit:
            pano_image.thumbnail((limit, limit * 2), Image.Resampling.LANCZOS)
        with io.BytesIO() as buffer:
            pano_image.save(buffer, format="JPEG", quality=86, optimize=True)
            jpg_data = buffer.getvalue()
        st.download_button("Descargar panorámica convertida a JPG", jpg_data, file_name="panoramica_hugin.jpg", mime="image/jpeg")
        st.info("El visor supone proyección equirectangular; si Hugin exportó una proyección cilíndrica, vuelve a exportar en equirectangular para evitar deformaciones. Para una panorámica equirectangular completa de 360° × 180°, la imagen suele tener proporción 2:1. Una panorámica parcial también puede verse, pero debes indicar el campo horizontal y vertical real. No fuerces 360° si no cubre toda la vuelta.")
        c1,c2,c3=st.columns(3)
        with c1: haov=st.slider("Cobertura horizontal (grados)",30,360,360,5)
        with c2: vaov=st.slider("Cobertura vertical (grados)",20,180,100,5)
        with c3: pitch=st.slider("Inclinación inicial (grados)",-60,60,0,5)
        encoded=base64.b64encode(jpg_data).decode("ascii")
        config={"type":"equirectangular", "panorama":"data:image/jpeg;base64,"+encoded,
                "autoLoad":True,"showControls":True,"mouseZoom":True,"draggable":True,
                "haov":haov,"vaov":vaov,"vOffset":0,"pitch":pitch,"hfov":min(90,haov)}
        # Pannellum renders a cylindrical-style interactive look-around from an equirectangular image.
        html='''<!DOCTYPE html><html><head><meta charset="utf-8"/>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/pannellum@2.5.6/build/pannellum.css"/>
<script src="https://cdn.jsdelivr.net/npm/pannellum@2.5.6/build/pannellum.js"></script>
<style>html,body{margin:0;height:100%;background:#111}#panorama{height:100vh;width:100%}</style></head>
<body><div id="panorama"></div><script>pannellum.viewer('panorama', CONFIG);</script></body></html>'''.replace("CONFIG", json.dumps(config))
        components.html(html, height=520, scrolling=False)
        st.caption("El visor necesita internet para cargar Pannellum desde CDN. El archivo TIFF se procesa localmente y no se envía a un servidor por esta aplicación.")
    except Exception as exc:
        st.error(f"No se pudo leer la panorámica: {exc}")
