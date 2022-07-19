![](imagenes/UC_FMRI.jpg)

---

---

***Andres Eduardo Aracena Rangel***

*Estudiante del programa del Magister en Física Médica*

---

---

El siguiente Script de Python forma parte del trabajo especial de grado.

Profesora Guía:

*PhD María Daniela Cornejo*

---

---

Imagenes de fMRI extraidas de OpenNeuro:

- [ds002422](https://openneuro.org/datasets/ds002422/versions/1.1.0)

---

---

Con referencia a:

 - [Nipype tutorial de Michael Notter](https://miykael.github.io/nipype_tutorial/)

Acronimos:

- CSF: Cerebrospinal Fluid (*líquido cefalorraquídeo*)
- GM: Gray Matter (*materia gris*)
- WM: White Matter (*materia blanca*)

# Flujo Trabajo de Preprocesamiento - con iterable de mask_ref

## Importamos Librerias


```python
import time # medir el tiempo de ejecución de nuestros programas
start = time.process_time()
inicio = time.time()
```


```python
import os # El módulo os nos permite acceder a funcionalidades dependientes del Sistema Operativo
from os.path import join as opj # Este método concatena varios componentes de ruta con exactamente un separador de directorio(‘/’)

from nipype import Node, Workflow, Function, SelectFiles, DataSink

from nipype.interfaces.fsl import ExtractROI, MCFLIRT, BET, FAST, Threshold, FLIRT, SliceTimer
from nipype.interfaces.spm import SliceTiming, Smooth, Realign
from nipype.algorithms.rapidart import ArtifactDetect
```

    220521-05:29:56,596 nipype.utils WARNING:
    	 A newer version (1.8.1) of nipy/nipype is available. You are using 1.7.0



```python
from niflow.nipype1.workflows.fmri.fsl import create_susan_smooth
```

**nota:** En Nipype, puede escribir flujos de trabajo anidados , donde un flujo de trabajo secundario puede tomar el lugar de un nodo en un script determinado.

Nipype incluye flujos de trabajos preempaquetados, entre los que se encuentra `Susan` el cual realiza un proceso de filtrado gaussiano.

Al tratar de importar el paquete de `Susan` con

    from nipype.workflows.fmri.fsl import create_susan_smooth

en mi caso, me produce el siguiente error

    'NoneType' object is not iterable
    
por esta razon, importamos mediante

    from niflow.nipype1.workflows.fmri.fsl import create_susan_smooth


```python
# MATLAB: especifique la ruta al SPM actual y el modo predeterminado de MATLAB
from nipype.interfaces.matlab import MatlabCommand
MatlabCommand.set_default_paths('/home/aracena/login/spm12')
MatlabCommand.set_default_matlab_cmd("matlab -nodesktop -nosplash")
```

Es importamte especificar la ruta del modulo SPM a utilizar, sino, al usar el modulo dara el siguiente error:

    stty: 'standard input': Inappropriate ioctl for device

## Definimos parámetros


```python
# Ruta del directorio de la data
path_data = '/home/aracena/data/' 

'''
imagenes de OpenNeuro ds002422
'''
# Ruta donde reposan las imagenes de OpenNeuro ds002422
image_dir = opj(path_data,'ds002422')

# Ruta de archivo .json
path_json = opj(image_dir,'sub-01','func','sub-01_task-rest_bold.json')

'''
Ruta donde reposa el archivo bbr.sch de FSL
'''
path_bbr = '/home/aracena/fsl/etc/flirtsch/bbr.sch'

'''
Ruta donde se guardaran los resultados
'''
experiment_new = '/home/aracena/thesis_ds002422/'
path_output = opj(experiment_new,'01_fase1_extraccion_mask_brain','output')

'''
Ruta donde se guardaran los resultados del flujo de trabajo
'''
path_wf = opj(path_output, 'output_workflow')

# Crear la carpeta de salida
os.system('mkdir -p %s'%path_output)

'''
Indicamos los mapas de probabilidad o máscaras que vamos a procesar.
En este notebook, los mapas de probabilidad se extraen con la interfaz FAST de fsl; el resultado de esta
interfaz es una lista de archivos, es decir, [[pve0], [pve1], [pv2]], siendo

    pve0 = CSF
    pve1 = GM
    pve2 = WM

Debemos seleccionar solo los mapas de probabilidad que deseamos procesar, indicando las 
máscara de referencia (mask_ref), que normalmente es la la máscara de wm, y la máscara de extracción (mask_ext),
es decir, la máscara que se aplicara a la imagen funcional para extraer los datos. Debemos indicar 0,1,2 para 
las máscaras de CSF,GM,WM respectivamente. Por ejemplo, queremos mask_ref = WM y mask_extr = CSF, se ingresa:
    
    mask_ref = 2
    mask_ext = 0
'''
mask_ref = 2

# Indicamos en una lista, las mascaras externas que deseamos para la extracción
mask_ext = [0,1,2]
```

## Definimos funciones

### Función para extraer orden de adquisión de los cortes de la imagen


```python
'''
Funcion para extraer el orden de adquisión de los cortes de la imagen.

Inputs:

- json_arch: archivo .json

Output:

- slice_order: orden de adqusión de los cortes
- TR: tiempo de repetición
- number_of_slices: numero de slices
'''
 
def order_slice(json_arch):
    import json as json
    with open(json_arch, 'rt') as fp:
        task_info = json.load(fp)
    
    '''
    Extraemos información del arhivo .json
    '''
    # tiempo de repeticion
    TR = task_info['RepetitionTime']
    # tiempo de adquisión de cortes
    slice_timing = task_info['SliceTiming']
    
    '''
    Procesamos data extraida del archivo .json
    '''
    # Numero de slices
    number_of_slices = len(slice_timing)
    # Tiempo en adquirir primer corte
    time_first = TR/number_of_slices
    # Valor minimo de slice_timing
    mini = min(slice_timing)
    # Valor maximo de slice_timing
    maxi = max(slice_timing)
    # Primer valor de slice_timing
    prim = slice_timing[0]
    # Segundo valor de slice_timing
    segu = slice_timing[1]

    if prim == mini:
        if segu == mini+time_first:
          print('Orden de adquisición de cortes secuenciales ascendente')
          slice_order = list(range(1, number_of_slices+1, 1))      
        else:
          print('Orden de adquisición de cortes intercalados 1-1')
          slice_order = list(range(1, number_of_slices+1, 2)) + list(range(2, number_of_slices+1, 2))      
    else:
        if segu == maxi - time_first:
          print('Orden de adquisición de cortes secuenciales descendente')
          slice_order = list(range(snumber_of_slices,0 , -1))
        else:
          print('Orden de adquisición de cortes intercalados 1 no 1')
          slice_order = list(range(2, number_of_slices+1, 2))+list(range(1, number_of_slices+1, 2))
    
    print(slice_timing)
    
    return slice_order,TR, number_of_slices
```

## Especificar flujo de entrada y salida

Especificamos dónde se pueden buscar y recopilar archivos del disco duro como datos de entrada, y cómo etiquetralos y guardar los datos de salida. 

En este notebook utilizaremos:

    recopilar --> SelectFiles
    almacenar --> DataSink


### Nodo SelectFiles


```python
'''
Plantilla de cadena con cadenas basadas en {}
'''

# La entrada template indica la plantilla de cadena que debe coincidir en el directorio indicado a buscar
templates = {'anat': '{subject_id}/anat/sub-01_T1w.nii', 
             'func': '{subject_id}/func/sub-01_task-rest_bold.nii.gz'}

'''
Creamos el nodo SelectFiles
'''
selectfiles = Node(SelectFiles(templates),
          name='selectfiles')

'''
Inputs
- Ubicación de la carpeta del conjunto de datos
- Cadenas de marcador de posición {}
'''
# La entrada base_directory indica en que directorio buscar
selectfiles.inputs.base_directory = image_dir

#Ingresamos la(s) cadena(s) de marcador de posición {} con valores
selectfiles.inputs.subject_id = 'sub-01'
```

### Nodo DataSink


```python
datasink = Node(DataSink(base_directory = path_output,
                         container = "datasink"),
                name="datasink")

# Definir cadenas de sustitución
substitutions = [('_mask_ext_0', 'mask_ext_csf'),
                 ('_mask_ext_1', 'mask_ext_gm'),
                 ('_mask_ext_2', 'mask_ext_wm'),
                 ('detrend.nii.gz', 'fmri_rest_prepro.nii.gz'),
                 ('_smooth0', 'smoooth'),
                 ('_mask_func0', 'mask_func'),                 
                 ('_thresh', 'threshold')
                ]

# Alimente las cadenas de sustitución al nodo DataSink
datasink.inputs.substitutions = substitutions
```

## Flujo de trabajo - preparación imagen funcional

### Nodo Function (Descomprimir archivos .gz)

Por lo general, la data que se encuentra alojada en OpenNeuro se encuentra en formato comprimido  '.gz', sin embargo, la data para ser procesada en la canalización de preprocesamiento debe estar descomprimida y en formato  '.nifti'. Se crea una función con modulo de Nipype Gunzip, donde verifica la extensión y descomprime si es '.gz'.


```python
def gunzip_func(file):
    from nipype import Node
    from nipype.algorithms.misc import Gunzip
    veri = file
    a = veri.find('.gz')
    if a < 0:
        out_file = file
    else:
        gunzip = Gunzip(in_file=veri)
        out_file = gunzip.run().outputs.out_file
    return out_file

gunzip_func = Node(Function(input_file=["file"],
                       output_names=["out_file"],
                       function=gunzip_func),
              name='gunzip_func')
```


```python
def gunzip_anat(file):
    from nipype import Node
    from nipype.algorithms.misc import Gunzip
    veri = file
    a = veri.find('.gz')
    if a < 0:
        out_file = file
    else:
        gunzip = Gunzip(in_file=veri)
        out_file = gunzip.run().outputs.out_file
    return out_file

gunzip_anat = Node(Function(input_file=["file"],
                       output_names=["out_file"],
                       function=gunzip_anat),
              name='gunzip_anat')
```


```python
from nipype.interfaces.fsl import ExtractROI, MCFLIRT, BET, FAST, Threshold, FLIRT, SliceTimer
```

### Nodo ExtractROI (Eliminar escaneos ficticios)

Las imágenes funcionales son obtenidas como protocolo con una cantidad de escaneos ficticios al principio, los cuales deben ser extraidos/eliminados del conjunto de imagenes funcionales adquiridas.


```python
extract = Node(ExtractROI(t_min=4, t_size=-1, output_type='NIFTI'),
               name="extract")
```

### Nodo SliceTiming (Corrección del tiempo de corte )


Para corregir la adquisición de los cortes de los volúmenes, utilizaremos SliceTiming. Los datos de entrada, inputs,

    num_slices,
    ref_slice,
    slice_order,
    time_repetition,
    time_acquisition
    
son extraídos del archivo '.json' al ingresarlo en la función

    def order_slice(json_arch)



```python
# Funcion para extraer el orden de adquisión de las cortes de la imagen
res_fun = order_slice(path_json)

slice_order = res_fun[0]
TR = res_fun[1]
number_of_slices = res_fun[2]

# SliceTimer - correct for slice wise acquisition
slicetime = Node(SliceTimer(index_dir=False,
                             interleaved=True,
                             output_type='NIFTI',
                             time_repetition=TR),
                  name="slicetime")
```

    Orden de adquisición de cortes intercalados 1 no 1
    [1.5375, 0, 1.6225, 0.085, 1.7075, 0.1725, 1.7925, 0.2575, 1.8775, 0.3425, 1.9625, 0.4275, 2.05, 0.5125, 2.135, 0.5975, 2.22, 0.6825, 2.305, 0.77, 2.39, 0.855, 2.475, 0.94, 2.56, 1.025, 2.6475, 1.11, 2.7325, 1.195, 2.8175, 1.28, 2.9025, 1.3675, 2.9875, 1.4525]


### Nodo MCFLIRT - VOL (Corrección de movimiento)

Para corregir el movimiento en el escáner, usaremos FSL MCFLIRT.

**NOTA:** Al ejecutar el nodo MCFLIRT necesitamos dos outputs para el procesamiento, los cuales son:

    out_file: nos entrega una matriz 4D con la corrección del movimiento
    mean_img: nos entrega un volumen, matriz 3D, promedio con la corrección del movimiento

Si la versión de `FSL` instalado no es la versión 5, al ejecutar el flujo de trabajo conducirá a un error que nos indica que no se encuentra el archivo de `mean_img`, arrojando un error como:

    FileNotFoundError: No such file or directory /opt/home/aracena/thesis_practica/tips_nipype/16_workflow_preprocesamiento/output_prefunc/output_workflow/work_preproc_func/mcflirt/sub-01_task-rest_bold_roi_st_mcf_mean_reg.nii.gz' for output 'mean_img' of a MCFLIRT interface
    
Esto se debe, como en mi caso que tengo instalado FSL versión 6.0, al sobrescribir el archivo de procesamiento de mean_img, NO suprime la extensión previa, creando un archivo con la extensión copiado dos veces, como se observa en la figura:

![](imagenes/error_MCFLIRT.png)

Este es un resultado previo a este notebook y arrojando el error. En este caso, marcaremos en negrita lo previamente explicado

   sub-01-task-rest_bold_roi_st_mcf**.nii.gz**_mean_reg**.nii.gz**

La solución para obtener estos dos outputs, fue obtenerlos separadamente, donde el output `out_file` lo obtendremos mediante el nodo MCFLIRT, y el output `mean_img` lo obtendremos con una nodo Function MCFLIRT. Detalles sobre cómo se elaboro este nodo Function MCFLIRT, puede revisar el notebook **aplicacion_NODO_funcion_MCFLIRT.ipynb**


    


```python
#MCFLIRT.help()
```


```python
'''
Creamos el Nodo MCFLIRT - VOL
'''

mcflirt_vol = Node(MCFLIRT(), name="mcflirt_vol")

# inputs
mcflirt_vol.inputs.save_plots=True
```

### Función MCFLIRT-MEAN


```python
def crear_mean_img(archivo,img_func,path_wf): 
    import os, pathlib, re
    from os.path import join as opj
    from nipype.interfaces.fsl import MCFLIRT
    
    '''
    Sustraemos extensión de archivo
    '''
    print('-----\n', archivo)
    ext = '.gz'
    img_modi = archivo

    while ext != '':
        # extraemos la extencion del archivo
        path = pathlib.Path(img_modi)
        ext = path.suffix
        #realizamos la sustracción de la extensión del archivo
        img_modi = re.sub(ext,"",img_modi)
        ext = pathlib.Path(img_modi).suffix
        new_name = pathlib.Path(img_modi).name

    '''
    creamos el archivo sin extension y su ruta
    '''
    path_out_file = opj(path_wf,new_name)
    os.system('touch %s'%path_out_file)
        
    '''
    ejecutamos la interfaz MCFLRIT - MEAN
    '''
    mfmean = MCFLIRT()
    mfmean.inputs.in_file = img_func
    mfmean.inputs.out_file = path_out_file
    mfmean.inputs.mean_vol=True
    out_file = mfmean.run().outputs.mean_img
          
    return out_file

'''
Creamos el nodo Fuction de MCFLIRT-MEAN
'''

mcflirt_mean = Node(Function(input_file=['archivo','img_func', 'path_wf'],
                            output_names=['mean_img'],
                            function=crear_mean_img),
              name='mcflirt_mean')

mcflirt_mean.inputs.path_wf = path_wf
```

### Nodo ArtifactDetect (Detección de artefactos)

La herramienta `ArtifactDetect`es útil para detectar valores atípicos de movimiento e intensidad en las imágenes funcionales. Los inputs significa lo siguiente:

- **norm_threshold** - Umbral para usar para detectar valores atípicos relacionados con el movimiento cuando se usa movimiento compuesto
- **zintensity_threshold** - Uso del umbral Z de intensidad para detectar imágenes que se desvían de la media
- **mask_type** - Tipo de máscara que debe usarse para enmascarar los datos funcionales. spm_global utiliza un cálculo similar a spm_global para determinar la máscara cerebral
- **parameter_source** - Fuente de parámetros de movimiento
- **use_differences** - Si desea utilizar diferencias entre estimaciones de movimiento sucesivo (primer elemento) y parámetro de intensidad (segundo elemento) para determinar valores atípicos


```python
# Artifact Detection - determines outliers in functional images
art = Node(ArtifactDetect(norm_threshold=2,
                          zintensity_threshold=3,
                          mask_type='spm_global',
                          parameter_source='FSL',
                          use_differences=[True, False],
                          plot_type='svg'),
           name="art")
```

### Nodo Smooth

La herramienta `Smooth` es útil para el suavizado gaussiano 3D de los volúmenes de imagen.


```python
# Smooth - image smoothing
smooth = Node(Smooth(), name="smooth")
#smooth.iterables = ("fwhm", fwhm)
```

    stty: 'standard input': Inappropriate ioctl for device


## Flujo de trabajo - preparacion imagen anatomica

### Nodo BET

Extracción del craneo de la imagen anatomica.

Con los notebook previos, *workflow_BET_iterable_frac.ipynb* y *workflow_BET_iterable_robust.ipynb* usaremos los siguientes inputs:


```python
'''
Creamos el nodo BET
'''
skullstrip = Node(BET(output_type='NIFTI_GZ'),
                name="skullstrip")
# Inputs Nodo
skullstrip.inputs.robust = True
skullstrip.inputs.frac = 0.6
```

### Nodo FAST (Segmentación de imagen anatómica)

Para crear los mapas de probabilidad para la CSF, GM y WM, usamos la interfaz `NewSegment` de SPM.


```python
#FAST.help()
```


```python
'''
Creamos el nodo FAST
'''

segmentation = Node(FAST(output_type='NIFTI_GZ'),
                name="segmentation")
```

## Flujo de trabajo - Corregistro

### Nodo Function (Selección de mapas de probabilidad)

Declaramos una función auxiliar para seleccionar el mapa de probabilidad de CSF y WM. Debido a que el campo de salida del nodo *segmentation* nos da una lista de archivos, es decir, [[pve0], [pve1], [pv2]], siendo

    pve0 = CSF
    pve1 = GM
    pve2 = WM

Por lo tanto, usando las siguiente función, podemos seleccionar solo los mapas de probabilidad que desamos.

#### Seleccionar mapa de probabilidad (mp) de REFERENCIA


```python
# Seleccione el mp de CSF de la salida del nodo segmentation
def get_ref(files, mask_ref):
    return files[mask_ref]

# Creamos Nodo
get_mask_ref = Node(Function(input_names=['files', 'mask_ref'],
                       output_names=["output_file"],
                       function=get_ref),
              name='get_mask_ref')

get_mask_ref.inputs.mask_ref = mask_ref
```

#### Seleccionar mapa de probabilidad (mp) de EXTRACCION


```python
# Seleccione el mp de WM de la salida del nodo segmentation
def get_ext(files, mask_ext):
    return files[mask_ext]

# Creamos Nodo
get_mask_ext = Node(Function(input_names=['files', 'mask_ext'],
                       output_names=['output_file'],
                       function=get_ext),
              name='get_mask_ext')

get_mask_ext.iterables = ('mask_ext', mask_ext)
```

### Nodo Threshold (Creación de mascara binaria)

Creamos una máscara binaria a partir del mapa de probabilidad de WM. Esto se puede hacer mediante la interfaz `Threshold` de FSL.

Esta mascara binaria servirá para guiar el corregistro entre la imagen anatomica y funcional, mediante el nodo `coreg_bbr` que se definira más adelante.


```python
# Threshold - Threshold WM probability image
threshold = Node(Threshold(thresh=0.5,
                           args='-bin',
                           output_type='NIFTI_GZ'),
                name="threshold")
```


```python
#Threshold.help()
```

### Nodo FLIRT (Calcular Matriz de Corregistro)

Como siguiente paso, nos aseguraremos de que las imágenes funcionales se registren conjuntamente con la imagen anatómica. Para ello, usaremos la interfaz `FLIRT` de FSL . Como acabamos de crear un mapa de probabilidad de CSF, podemos usarlo junto con la función de costo del registro basado en límites (BBR) para optimizar el corregistro de la imagen. Como algunas notas útiles ...

    utilizar un grado de libertad  (dof) de 6
    especificar la función de costo (cost) como bbr
    utilizar el schedule = path_bbr (definida en los parámetros)


```python
# FLIRT - pre-alineación de imágenes funcionales a imágenes anatómicas
coreg_pre = Node(FLIRT(dof=6, output_type='NIFTI_GZ'),
                 name="coreg_pre")

# FLIRT - corregistro de imágenes funcionales a imágenes anatómicas con BBR
coreg_bbr = Node(FLIRT(dof=6,
                       cost='bbr',
                       schedule=path_bbr,
                       output_type='NIFTI_GZ'),
                 name="coreg_bbr")
```

### Nodo FLIRT 2 (Aplicar Matriz de Corregistro a la imagen funcional)

Ahora que conocemos la matriz de coregistro para superponer correctamente la imagen media funcional en la anatomía específica del sujeto, debemos aplicar el coregistro a toda la serie de tiempo. Esto se puede lograr con la interface `FLIRT` de FSL de la siguiente manera:


```python
# Especificar el voxel resolución isométrica que desea después de coregistration 
desired_voxel_iso = 4

# Aplicar coregistration warp a imágenes funcionales 
applywarp = Node(FLIRT(interp='spline',
                       apply_isoxfm=desired_voxel_iso,
                       output_type='NIFTI'),
                 name="applywarp")
```

**Importante :** como puede observar, se especifica una variable `desired_voxel_iso`. Esto es muy importante en esta etapa; de lo contrario `FLIRT`, transformará sus imágenes funcionales a una resolución de la imagen anatómica, lo que aumentará drásticamente el tamaño del archivo (por ejemplo, de 1 a 10 GB por archivo). Si no desea cambiar la resolución del vóxel, use el parámetro adicional `no_resample=True`. Importante, para que esto funcione, aún necesita definir `apply_isoxfm=desired_voxel_iso`.

### Nodo Susan (Suavizado)

El siguiente paso es suavizar la imagen. La forma más sencilla de hacer esto es utilizar la función `Smooth` de `FSL` o `SPM`. Tenga en cuenta que esta vez, estamos importando un flujo de trabajo en lugar de una interfaz.


```python
susan = create_susan_smooth(name='susan')
susan.inputs.inputnode.fwhm = 4
```


```python
susan.outputs
```




    
    inputnode = 
    fwhm = None
    in_files = None
    mask_file = None
    
    mask = 
    out_file = None
    
    meanfunc2 = 
    out_file = None
    
    median = 
    out_stat = None
    
    merge = 
    out = None
    
    multi_inputs = 
    cart_btthresh = None
    cart_fwhm = None
    cart_in_file = None
    cart_usans = None
    
    outputnode = 
    smoothed_files = None
    
    smooth = 
    smoothed_file = None





```python
#create_susan_smooth?
```

## Flujo de trabajo - Crear máscara binaria

Hay muchos enfoques posibles sobre cómo enmascarar sus imágenes funcionales. Uno de ellos no lo es en absoluto, uno tiene una simple máscara cerebral y uno que solo considera cierto tipo de tejido cerebral, por ejemplo, la materia gris.

Para este Script, queremos crear una máscara de liquido cefalorraquídeo dilatada. Para ello necesitamos:

- Vuelva a muestrear el mapa de probabilidad de csf a la misma resolución que las imágenes funcionales
- Umbral de este mapa de probabilidad remuestreado en un valor específico
- Dilata esta máscara con algunos voxels para hacer que la máscara sea menos conservadora y más inclusiva.

El primer paso se puede realizar de muchas formas (por ejemplo, utilizando `mri_convert` de freesurfer, nibabel) en nuestro caso, usaremos `FLIRT` de FSL. El truco consiste en utilizar la máscara de probabilidad, como archivo de entrada y como archivo de referencia.

### Nodo FLIRT  3 (Máscara csf)


```python
# Initiate resample node
resample = Node(FLIRT(apply_isoxfm=desired_voxel_iso,
                      output_type='NIFTI'),
                name="resample")
```

### Nodo Threshold 2 (Mascara csf)

Afortunadamente, el segundo y tercer paso se pueden realizar con un solo nodo. Podemos tomar casi el mismo nodo `Threshold` que el anterior. Solo necesitamos agregar otro argumento adicional: `-dilF-` que aplica un filtrado máximo de todos los vóxeles.


```python
# Threshold - Imagen de probabilidad de CSF umbral 
mask_EXT  =  Node ( Threshold (args = '-bin -dilF',
                               output_type = 'NIFTI' ), 
                name = "mask_EXT" )

mask_EXT.iterables = ('thresh', [0.5,0.95,0.97,0.99])
```

### Nodo Function (selección de mapa de probabilidad de CSF)

Función `get_csf(files)` definida en Flujo de trabajo-Corregistro

## Flujo de trabajo - Aplicar la máscara binaria

Para aplicar la máscara a las imágenes funcionales suavizadas, usaremos la interfaz `ApplyMask` de FSL.

**Importante:** el flujo de trabajo de Susan proporciona una lista de archivos, es decir, [smoothed_func.nii], en lugar de solo el nombre del archivo directamente. Si usáramos un NodO para `ApplyMask`, se produciría el siguiente error:

    TraitError: The 'in_file' trait of an ApplyMaskInput instance must be an existing file name, but a value of ['/output/work_preproc/susan/smooth/mapflow/_smooth0/asub-07_ses-test_task-fingerfootlips_bold_mcf_flirt_smooth.nii.gz'] <class 'list'> was specified.


Para evitar esto , usaremos a `MapNode` y especificaremos `in_file` en el input `iterfield`.Así, el nodo es capaz de manejar una lista de entradas, ya que sabrá que tiene que aplicarse de forma iterativa a la lista de entradas.

### Nodo ApplyMask


```python
from nipype import MapNode
from nipype.interfaces.fsl import ApplyMask

mask_func = MapNode(ApplyMask(output_type='NIFTI'),
                    name="mask_func",
                    iterfield=["in_file"])
```

### Nodo TSNR (Eliminar tendencias lineales en imágenes funcionales)

Usemos el módulo `TSNR` de Nipype para eliminar tendencias lineales y cuadráticas en las imágenes funcionales suavizadas. Para ello, solo tienes que especificar el parámetro `regress_poly` en el inicio del nodo.


```python
from nipype.algorithms.confounds import TSNR

detrend = Node(TSNR(regress_poly=2), name="detrend")
```

#### Para la fMRI


```python
from nipype.algorithms.confounds import TSNR

detrend_fmri = Node(TSNR(regress_poly=2), name="detrend_fmri")
```

## Definimos los Flujos de trabajo


```python
# Flujo de trabajo de Preparacion - imagen func
prefunc = Workflow(name = 'work_preproc_func', base_dir = path_wf)

# Flujo de trabajo de Preparación - imagen anat
preanat = Workflow(name = 'work_preproc_anat', base_dir = path_wf)

# Flujo de trabajo de corregistro
prereg = Workflow(name = 'work_preproc_corregistro', base_dir = path_wf)
```

### Conectamos Nodos


```python
prefunc.connect(gunzip_func, 'out_file', extract, 'in_file')
prefunc.connect(extract, 'roi_file', slicetime, 'in_file')
prefunc.connect(slicetime,'slice_time_corrected_file', mcflirt_vol, 'in_file')
prefunc.connect(slicetime,'slice_time_corrected_file', mcflirt_mean, 'img_func')
prefunc.connect(mcflirt_vol, 'out_file', mcflirt_mean, 'archivo')
```


```python
preanat.connect(gunzip_anat, 'out_file', skullstrip, 'in_file')
preanat.connect(skullstrip, 'out_file', segmentation, 'in_files')
# creacion mascara wm-anat
preanat.connect(segmentation, 'partial_volume_files', get_mask_ref, 'files'),
preanat.connect(get_mask_ref, 'output_file', threshold, 'in_file')
```


```python
prereg.connect(preanat, 'gunzip_anat.out_file', coreg_bbr, 'reference')
prereg.connect(preanat, 'gunzip_anat.out_file', applywarp, 'reference')
prereg.connect(preanat, 'skullstrip.out_file', coreg_pre, 'reference')
prereg.connect(preanat, 'threshold.out_file', coreg_bbr, 'wm_seg')
prereg.connect(prefunc, 'mcflirt_vol.out_file', applywarp, 'in_file')
prereg.connect(prefunc, 'mcflirt_mean.mean_img', coreg_pre, 'in_file')
prereg.connect(prefunc, 'mcflirt_mean.mean_img', coreg_bbr, 'in_file')
prereg.connect(coreg_pre, 'out_matrix_file', coreg_bbr, 'in_matrix_file')
prereg.connect(coreg_bbr, 'out_matrix_file', applywarp, 'in_matrix_file')
prereg.connect(applywarp, 'out_file', susan, 'inputnode.in_files')
#creacion mascara extraccion-anat
prereg.connect(preanat, 'segmentation.partial_volume_files', get_mask_ext, 'files')
prereg.connect(get_mask_ext, 'output_file', resample, 'in_file')
prereg.connect(get_mask_ext, 'output_file', resample, 'reference')
prereg.connect(resample, 'out_file', mask_EXT, 'in_file')
prereg.connect(mask_EXT, 'out_file', susan, 'inputnode.mask_file')
```


```python
prereg.connect(selectfiles, 'anat', preanat, 'gunzip_anat.file')
prereg.connect(selectfiles, 'func', prefunc, 'gunzip_func.file')

prereg.connect(susan, 'outputnode.smoothed_files', mask_func, 'in_file')
prereg.connect(mask_EXT, 'out_file', mask_func, 'mask_file')
prereg.connect(mask_func, 'out_file', detrend, 'in_file')
prereg.connect(detrend, 'detrended_file', datasink, 'masks_brain')
prereg.connect(susan, 'outputnode.smoothed_files', datasink, 'fmri_smooth')
prereg.connect(applywarp, 'out_file', datasink, 'fmri_sin_smooth')
prereg.connect(mask_func, 'out_file', datasink, 'masks_brain_sin_detrend')

prereg.connect(susan, 'outputnode.smoothed_files', detrend_fmri, 'in_file')
prereg.connect(detrend_fmri, 'detrended_file', datasink, 'fmri_detrend')



prereg.connect(preanat, 'segmentation.partial_volume_files', datasink, 'mask_files')
```

### Visualizamos el Flujo de trabajo


```python
# Create preproc output graph
prereg.write_graph(graph2use='colored', format='png', simple_form=True)

# Visualize the graph
from IPython.display import Image
Image(filename=path_wf+'/work_preproc_corregistro/graph.png')
```

    220521-05:30:09,49 nipype.workflow INFO:
    	 Generated workflow graph: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/graph.png (graph2use=colored, simple_form=True).





    
![png](output_84_1.png)
    




```python
# Otra visualización del flujo de trabajo

prereg.write_graph(graph2use='flat', format='png', simple_form=True)
Image(filename=path_wf+'/work_preproc_corregistro/graph_detailed.png')
```

    220521-05:30:09,879 nipype.workflow INFO:
    	 Generated workflow graph: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/graph.png (graph2use=flat, simple_form=True).





    
![png](output_85_1.png)
    



### Ejecutamos el flujo de trabajo


```python
prereg.run('MultiProc', plugin_args={'n_procs': 8})
```

    220521-05:30:09,923 nipype.workflow INFO:
    	 Workflow work_preproc_corregistro settings: ['check', 'execution', 'logging', 'monitoring']
    220521-05:30:10,45 nipype.workflow INFO:
    	 Running in parallel.
    220521-05:30:10,51 nipype.workflow INFO:
    	 [MultiProc] Running 0 tasks, and 1 jobs ready. Free memory (GB): 13.99/13.99, Free processors: 8/8.
    220521-05:30:10,121 nipype.workflow INFO:
    	 [Node] Setting-up "work_preproc_corregistro.selectfiles" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/selectfiles".
    220521-05:30:10,127 nipype.workflow INFO:
    	 [Node] Executing "selectfiles" <nipype.interfaces.io.SelectFiles>
    220521-05:30:10,132 nipype.workflow INFO:
    	 [Node] Finished "selectfiles", elapsed time 0.001645s.
    220521-05:30:12,53 nipype.workflow INFO:
    	 [Job 0] Completed (work_preproc_corregistro.selectfiles).
    220521-05:30:12,57 nipype.workflow INFO:
    	 [MultiProc] Running 0 tasks, and 2 jobs ready. Free memory (GB): 13.99/13.99, Free processors: 8/8.
    220521-05:30:12,143 nipype.workflow INFO:
    	 [Job 1] Cached (work_preproc_corregistro.work_preproc_func.gunzip_func).
    220521-05:30:12,146 nipype.workflow INFO:
    	 [Job 6] Cached (work_preproc_corregistro.work_preproc_anat.gunzip_anat).
    220521-05:30:14,112 nipype.workflow INFO:
    	 [Job 2] Cached (work_preproc_corregistro.work_preproc_func.extract).
    220521-05:30:14,116 nipype.workflow INFO:
    	 [Job 7] Cached (work_preproc_corregistro.work_preproc_anat.skullstrip).
    220521-05:30:16,120 nipype.workflow INFO:
    	 [Job 3] Cached (work_preproc_corregistro.work_preproc_func.slicetime).
    220521-05:30:16,124 nipype.workflow INFO:
    	 [Job 8] Cached (work_preproc_corregistro.work_preproc_anat.segmentation).
    220521-05:30:18,61 nipype.workflow INFO:
    	 [MultiProc] Running 0 tasks, and 5 jobs ready. Free memory (GB): 13.99/13.99, Free processors: 8/8.
    220521-05:30:18,143 nipype.workflow INFO:
    	 [Job 4] Cached (work_preproc_corregistro.work_preproc_func.mcflirt_vol).
    220521-05:30:18,150 nipype.workflow INFO:
    	 [Job 9] Cached (work_preproc_corregistro.get_mask_ext).
    220521-05:30:18,154 nipype.workflow INFO:
    	 [Job 15] Cached (work_preproc_corregistro.get_mask_ext).
    220521-05:30:18,157 nipype.workflow INFO:
    	 [Job 21] Cached (work_preproc_corregistro.get_mask_ext).
    220521-05:30:18,160 nipype.workflow INFO:
    	 [Job 27] Cached (work_preproc_corregistro.work_preproc_anat.get_mask_ref).
    220521-05:30:20,136 nipype.workflow INFO:
    	 [Job 5] Cached (work_preproc_corregistro.work_preproc_func.mcflirt_mean).
    220521-05:30:20,140 nipype.workflow INFO:
    	 [Job 10] Cached (work_preproc_corregistro.resample).
    220521-05:30:20,143 nipype.workflow INFO:
    	 [Job 16] Cached (work_preproc_corregistro.resample).
    220521-05:30:20,147 nipype.workflow INFO:
    	 [Job 22] Cached (work_preproc_corregistro.resample).
    220521-05:30:20,150 nipype.workflow INFO:
    	 [Job 28] Cached (work_preproc_corregistro.work_preproc_anat.threshold).
    220521-05:30:22,66 nipype.workflow INFO:
    	 [MultiProc] Running 0 tasks, and 13 jobs ready. Free memory (GB): 13.99/13.99, Free processors: 8/8.
    220521-05:30:22,149 nipype.workflow INFO:
    	 [Job 11] Cached (work_preproc_corregistro.mask_EXT).
    220521-05:30:22,155 nipype.workflow INFO:
    	 [Job 12] Cached (work_preproc_corregistro.mask_EXT).
    220521-05:30:22,163 nipype.workflow INFO:
    	 [Job 13] Cached (work_preproc_corregistro.mask_EXT).
    220521-05:30:22,170 nipype.workflow INFO:
    	 [Job 14] Cached (work_preproc_corregistro.mask_EXT).
    220521-05:30:22,177 nipype.workflow INFO:
    	 [Job 17] Cached (work_preproc_corregistro.mask_EXT).
    220521-05:30:22,185 nipype.workflow INFO:
    	 [Job 18] Cached (work_preproc_corregistro.mask_EXT).
    220521-05:30:22,193 nipype.workflow INFO:
    	 [Job 19] Cached (work_preproc_corregistro.mask_EXT).
    220521-05:30:22,202 nipype.workflow INFO:
    	 [Job 20] Cached (work_preproc_corregistro.mask_EXT).
    220521-05:30:24,68 nipype.workflow INFO:
    	 [MultiProc] Running 0 tasks, and 5 jobs ready. Free memory (GB): 13.99/13.99, Free processors: 8/8.
    220521-05:30:24,151 nipype.workflow INFO:
    	 [Job 23] Cached (work_preproc_corregistro.mask_EXT).
    220521-05:30:24,154 nipype.workflow INFO:
    	 [Job 24] Cached (work_preproc_corregistro.mask_EXT).
    220521-05:30:24,159 nipype.workflow INFO:
    	 [Job 25] Cached (work_preproc_corregistro.mask_EXT).
    220521-05:30:24,164 nipype.workflow INFO:
    	 [Job 26] Cached (work_preproc_corregistro.mask_EXT).
    220521-05:30:24,170 nipype.workflow INFO:
    	 [Job 29] Cached (work_preproc_corregistro.coreg_pre).
    220521-05:30:26,70 nipype.workflow INFO:
    	 [MultiProc] Running 0 tasks, and 1 jobs ready. Free memory (GB): 13.99/13.99, Free processors: 8/8.
    220521-05:30:26,153 nipype.workflow INFO:
    	 [Job 30] Cached (work_preproc_corregistro.coreg_bbr).
    220521-05:30:28,149 nipype.workflow INFO:
    	 [Job 31] Cached (work_preproc_corregistro.applywarp).
    220521-05:30:30,73 nipype.workflow INFO:
    	 [MultiProc] Running 0 tasks, and 24 jobs ready. Free memory (GB): 13.99/13.99, Free processors: 8/8.
    220521-05:30:30,160 nipype.workflow INFO:
    	 [Job 32] Cached (work_preproc_corregistro.susan.median).
    220521-05:30:30,171 nipype.workflow INFO:
    	 [Job 33] Cached (work_preproc_corregistro.susan.mask).
    220521-05:30:30,183 nipype.workflow INFO:
    	 [Job 42] Cached (work_preproc_corregistro.susan.median).
    220521-05:30:30,196 nipype.workflow INFO:
    	 [Job 43] Cached (work_preproc_corregistro.susan.mask).
    220521-05:30:30,211 nipype.workflow INFO:
    	 [Job 52] Cached (work_preproc_corregistro.susan.median).
    220521-05:30:30,226 nipype.workflow INFO:
    	 [Job 53] Cached (work_preproc_corregistro.susan.mask).
    220521-05:30:30,237 nipype.workflow INFO:
    	 [Job 62] Cached (work_preproc_corregistro.susan.median).
    220521-05:30:30,249 nipype.workflow INFO:
    	 [Job 63] Cached (work_preproc_corregistro.susan.mask).
    220521-05:30:32,74 nipype.workflow INFO:
    	 [MultiProc] Running 0 tasks, and 20 jobs ready. Free memory (GB): 13.99/13.99, Free processors: 8/8.
    220521-05:30:32,145 nipype.workflow INFO:
    	 [Job 34] Cached (work_preproc_corregistro.susan.meanfunc2).
    220521-05:30:32,151 nipype.workflow INFO:
    	 [Job 44] Cached (work_preproc_corregistro.susan.meanfunc2).
    220521-05:30:32,157 nipype.workflow INFO:
    	 [Job 54] Cached (work_preproc_corregistro.susan.meanfunc2).
    220521-05:30:32,165 nipype.workflow INFO:
    	 [Job 64] Cached (work_preproc_corregistro.susan.meanfunc2).
    220521-05:30:32,172 nipype.workflow INFO:
    	 [Job 72] Cached (work_preproc_corregistro.susan.median).
    220521-05:30:32,179 nipype.workflow INFO:
    	 [Job 73] Cached (work_preproc_corregistro.susan.mask).
    220521-05:30:32,194 nipype.workflow INFO:
    	 [Job 82] Cached (work_preproc_corregistro.susan.median).
    220521-05:30:32,202 nipype.workflow INFO:
    	 [Job 83] Cached (work_preproc_corregistro.susan.mask).
    220521-05:30:34,77 nipype.workflow INFO:
    	 [MultiProc] Running 0 tasks, and 18 jobs ready. Free memory (GB): 13.99/13.99, Free processors: 8/8.
    220521-05:30:34,155 nipype.workflow INFO:
    	 [Job 35] Cached (work_preproc_corregistro.susan.merge).
    220521-05:30:34,161 nipype.workflow INFO:
    	 [Job 45] Cached (work_preproc_corregistro.susan.merge).
    220521-05:30:34,167 nipype.workflow INFO:
    	 [Job 55] Cached (work_preproc_corregistro.susan.merge).
    220521-05:30:34,172 nipype.workflow INFO:
    	 [Job 65] Cached (work_preproc_corregistro.susan.merge).
    220521-05:30:34,181 nipype.workflow INFO:
    	 [Job 74] Cached (work_preproc_corregistro.susan.meanfunc2).
    220521-05:30:34,188 nipype.workflow INFO:
    	 [Job 84] Cached (work_preproc_corregistro.susan.meanfunc2).
    220521-05:30:34,200 nipype.workflow INFO:
    	 [Job 92] Cached (work_preproc_corregistro.susan.median).
    220521-05:30:34,207 nipype.workflow INFO:
    	 [Job 93] Cached (work_preproc_corregistro.susan.mask).
    220521-05:30:36,78 nipype.workflow INFO:
    	 [MultiProc] Running 0 tasks, and 17 jobs ready. Free memory (GB): 13.99/13.99, Free processors: 8/8.
    220521-05:30:36,158 nipype.workflow INFO:
    	 [Job 36] Cached (work_preproc_corregistro.susan.multi_inputs).
    220521-05:30:36,167 nipype.workflow INFO:
    	 [Job 46] Cached (work_preproc_corregistro.susan.multi_inputs).
    220521-05:30:36,175 nipype.workflow INFO:
    	 [Job 56] Cached (work_preproc_corregistro.susan.multi_inputs).
    220521-05:30:36,184 nipype.workflow INFO:
    	 [Job 66] Cached (work_preproc_corregistro.susan.multi_inputs).
    220521-05:30:36,191 nipype.workflow INFO:
    	 [Job 75] Cached (work_preproc_corregistro.susan.merge).
    220521-05:30:36,198 nipype.workflow INFO:
    	 [Job 85] Cached (work_preproc_corregistro.susan.merge).
    220521-05:30:36,216 nipype.workflow INFO:
    	 [Job 94] Cached (work_preproc_corregistro.susan.meanfunc2).
    220521-05:30:36,223 nipype.workflow INFO:
    	 [Job 102] Cached (work_preproc_corregistro.susan.median).
    220521-05:30:38,81 nipype.workflow INFO:
    	 [MultiProc] Running 0 tasks, and 16 jobs ready. Free memory (GB): 13.99/13.99, Free processors: 8/8.
    220521-05:30:38,168 nipype.workflow INFO:
    	 [Job 37] Cached (work_preproc_corregistro.susan.smooth).
    220521-05:30:38,179 nipype.workflow INFO:
    	 [Job 47] Cached (work_preproc_corregistro.susan.smooth).
    220521-05:30:38,194 nipype.workflow INFO:
    	 [Job 57] Cached (work_preproc_corregistro.susan.smooth).
    220521-05:30:38,208 nipype.workflow INFO:
    	 [Job 67] Cached (work_preproc_corregistro.susan.smooth).
    220521-05:30:38,217 nipype.workflow INFO:
    	 [Job 76] Cached (work_preproc_corregistro.susan.multi_inputs).
    220521-05:30:38,227 nipype.workflow INFO:
    	 [Job 86] Cached (work_preproc_corregistro.susan.multi_inputs).
    220521-05:30:38,240 nipype.workflow INFO:
    	 [Job 95] Cached (work_preproc_corregistro.susan.merge).
    220521-05:30:38,249 nipype.workflow INFO:
    	 [Job 103] Cached (work_preproc_corregistro.susan.mask).
    220521-05:30:40,82 nipype.workflow INFO:
    	 [MultiProc] Running 0 tasks, and 20 jobs ready. Free memory (GB): 13.99/13.99, Free processors: 8/8.
    220521-05:30:40,157 nipype.workflow INFO:
    	 [Node] Setting-up "work_preproc_corregistro.detrend_fmri" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/_mask_ext_2/_thresh_0.99/detrend_fmri".
    220521-05:30:40,161 nipype.workflow INFO:
    	 [Job 39] Cached (work_preproc_corregistro.mask_func).
    220521-05:30:40,169 nipype.workflow INFO:
    	 [Node] Executing "detrend_fmri" <nipype.algorithms.confounds.TSNR>
    220521-05:30:40,182 nipype.workflow INFO:
    	 [Node] Setting-up "work_preproc_corregistro.detrend_fmri" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/_mask_ext_2/_thresh_0.97/detrend_fmri".
    220521-05:30:40,187 nipype.workflow INFO:
    	 [Job 49] Cached (work_preproc_corregistro.mask_func).
    220521-05:30:40,188 nipype.workflow INFO:
    	 [Node] Executing "detrend_fmri" <nipype.algorithms.confounds.TSNR>
    220521-05:30:40,214 nipype.workflow INFO:
    	 [Node] Setting-up "work_preproc_corregistro.detrend_fmri" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/_mask_ext_2/_thresh_0.95/detrend_fmri".
    220521-05:30:40,217 nipype.workflow INFO:
    	 [Job 59] Cached (work_preproc_corregistro.mask_func).
    220521-05:30:40,221 nipype.workflow INFO:
    	 [Node] Executing "detrend_fmri" <nipype.algorithms.confounds.TSNR>
    220521-05:30:40,258 nipype.workflow INFO:
    	 [Node] Setting-up "work_preproc_corregistro.detrend_fmri" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/_mask_ext_2/_thresh_0.5/detrend_fmri".
    220521-05:30:40,274 nipype.workflow INFO:
    	 [Node] Executing "detrend_fmri" <nipype.algorithms.confounds.TSNR>
    220521-05:30:40,295 nipype.workflow INFO:
    	 [Node] Setting-up "_mask_func0" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/_mask_ext_2/_thresh_0.5/mask_func/mapflow/_mask_func0".
    220521-05:30:40,311 nipype.workflow INFO:
    	 [Node] Executing "_mask_func0" <nipype.interfaces.fsl.maths.ApplyMask>
    220521-05:30:41,600 nipype.workflow INFO:
    	 [Node] Finished "_mask_func0", elapsed time 1.282697s.
    220521-05:30:42,86 nipype.workflow INFO:
    	 [Job 69] Completed (work_preproc_corregistro.mask_func).
    220521-05:30:42,95 nipype.workflow INFO:
    	 [MultiProc] Running 4 tasks, and 16 jobs ready. Free memory (GB): 13.19/13.99, Free processors: 4/8.
                         Currently running:
                           * work_preproc_corregistro.detrend_fmri
                           * work_preproc_corregistro.detrend_fmri
                           * work_preproc_corregistro.detrend_fmri
                           * work_preproc_corregistro.detrend_fmri
    220521-05:30:42,409 nipype.workflow INFO:
    	 [Node] Setting-up "work_preproc_corregistro.detrend" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/_mask_ext_2/_thresh_0.99/detrend".
    220521-05:30:42,433 nipype.workflow INFO:
    	 [Node] Executing "detrend" <nipype.algorithms.confounds.TSNR>
    220521-05:30:42,438 nipype.workflow INFO:
    	 [Node] Setting-up "work_preproc_corregistro.detrend" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/_mask_ext_2/_thresh_0.97/detrend".
    220521-05:30:42,453 nipype.workflow INFO:
    	 [Node] Executing "detrend" <nipype.algorithms.confounds.TSNR>
    220521-05:30:42,492 nipype.workflow INFO:
    	 [Node] Setting-up "work_preproc_corregistro.detrend" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/_mask_ext_2/_thresh_0.95/detrend".
    220521-05:30:42,535 nipype.workflow INFO:
    	 [Node] Setting-up "work_preproc_corregistro.detrend" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/_mask_ext_2/_thresh_0.5/detrend".
    220521-05:30:42,558 nipype.workflow INFO:
    	 [Node] Executing "detrend" <nipype.algorithms.confounds.TSNR>
    220521-05:30:42,579 nipype.workflow INFO:
    	 [Node] Executing "detrend" <nipype.algorithms.confounds.TSNR>
    220521-05:30:44,91 nipype.workflow INFO:
    	 [MultiProc] Running 8 tasks, and 12 jobs ready. Free memory (GB): 12.39/13.99, Free processors: 0/8.
                         Currently running:
                           * work_preproc_corregistro.detrend
                           * work_preproc_corregistro.detrend
                           * work_preproc_corregistro.detrend
                           * work_preproc_corregistro.detrend
                           * work_preproc_corregistro.detrend_fmri
                           * work_preproc_corregistro.detrend_fmri
                           * work_preproc_corregistro.detrend_fmri
                           * work_preproc_corregistro.detrend_fmri
    220521-05:30:50,23 nipype.workflow INFO:
    	 [Node] Finished "detrend", elapsed time 7.45121s.
    220521-05:30:50,91 nipype.workflow INFO:
    	 [Job 60] Completed (work_preproc_corregistro.detrend).
    220521-05:30:50,95 nipype.workflow INFO:
    	 [MultiProc] Running 7 tasks, and 12 jobs ready. Free memory (GB): 12.59/13.99, Free processors: 1/8.
                         Currently running:
                           * work_preproc_corregistro.detrend
                           * work_preproc_corregistro.detrend
                           * work_preproc_corregistro.detrend
                           * work_preproc_corregistro.detrend_fmri
                           * work_preproc_corregistro.detrend_fmri
                           * work_preproc_corregistro.detrend_fmri
                           * work_preproc_corregistro.detrend_fmri
    220521-05:30:50,364 nipype.workflow INFO:
    	 [Job 77] Cached (work_preproc_corregistro.susan.smooth).
    220521-05:30:50,689 nipype.workflow INFO:
    	 [Node] Finished "detrend", elapsed time 8.080028s.
    220521-05:30:50,868 nipype.workflow INFO:
    	 [Node] Finished "detrend", elapsed time 8.424977s.
    220521-05:30:51,4 nipype.workflow INFO:
    	 [Node] Finished "detrend", elapsed time 8.520896s.
    220521-05:30:51,185 nipype.workflow INFO:
    	 [Node] Finished "detrend_fmri", elapsed time 10.960027s.
    220521-05:30:51,429 nipype.workflow INFO:
    	 [Node] Finished "detrend_fmri", elapsed time 11.253223s.
    220521-05:30:51,638 nipype.workflow INFO:
    	 [Node] Finished "detrend_fmri", elapsed time 11.438519s.
    220521-05:30:51,794 nipype.workflow INFO:
    	 [Node] Finished "detrend_fmri", elapsed time 11.484664s.
    220521-05:30:52,92 nipype.workflow INFO:
    	 [Job 38] Completed (work_preproc_corregistro.detrend_fmri).
    220521-05:30:52,94 nipype.workflow INFO:
    	 [Job 48] Completed (work_preproc_corregistro.detrend_fmri).
    220521-05:30:52,98 nipype.workflow INFO:
    	 [Job 58] Completed (work_preproc_corregistro.detrend_fmri).
    220521-05:30:52,101 nipype.workflow INFO:
    	 [Job 68] Completed (work_preproc_corregistro.detrend_fmri).
    220521-05:30:52,104 nipype.workflow INFO:
    	 [Job 40] Completed (work_preproc_corregistro.detrend).
    220521-05:30:52,107 nipype.workflow INFO:
    	 [Job 50] Completed (work_preproc_corregistro.detrend).
    220521-05:30:52,111 nipype.workflow INFO:
    	 [Job 70] Completed (work_preproc_corregistro.detrend).
    220521-05:30:52,118 nipype.workflow INFO:
    	 [MultiProc] Running 0 tasks, and 17 jobs ready. Free memory (GB): 13.99/13.99, Free processors: 8/8.
    220521-05:30:52,215 nipype.workflow INFO:
    	 [Node] Setting-up "work_preproc_corregistro.datasink" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/_mask_ext_2/_thresh_0.99/datasink".
    220521-05:30:52,215 nipype.workflow INFO:
    	 [Node] Setting-up "work_preproc_corregistro.datasink" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/_mask_ext_2/_thresh_0.97/datasink".
    220521-05:30:52,221 nipype.workflow INFO:
    	 [Node] Setting-up "work_preproc_corregistro.datasink" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/_mask_ext_2/_thresh_0.95/datasink".
    220521-05:30:52,221 nipype.workflow INFO:
    	 [Node] Setting-up "work_preproc_corregistro.detrend_fmri" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/_mask_ext_1/_thresh_0.99/detrend_fmri".
    220521-05:30:52,221 nipype.workflow INFO:
    	 [Node] Setting-up "work_preproc_corregistro.datasink" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/_mask_ext_2/_thresh_0.5/datasink".
    220521-05:30:52,223 nipype.workflow INFO:
    	 [Node] Executing "datasink" <nipype.interfaces.io.DataSink>
    220521-05:30:52,224 nipype.workflow INFO:
    	 [Node] Executing "datasink" <nipype.interfaces.io.DataSink>
    220521-05:30:52,225 nipype.workflow INFO:
    	 [Node] Executing "detrend_fmri" <nipype.algorithms.confounds.TSNR>
    220521-05:30:52,226 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain_sin_detrend/_mask_ext_2/_thresh_0.97/_mask_func0/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain_sin_detrend/mask_ext_wm/threshold_0.97/mask_func/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
    220521-05:30:52,227 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain_sin_detrend/_mask_ext_2/_thresh_0.99/_mask_func0/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain_sin_detrend/mask_ext_wm/threshold_0.99/mask_func/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
    220521-05:30:52,233 nipype.workflow INFO:
    	 [Job 87] Cached (work_preproc_corregistro.susan.smooth).
    220521-05:30:52,235 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_detrend/_mask_ext_2/_thresh_0.97/detrend.nii.gz -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_detrend/mask_ext_wm/threshold_0.97/fmri_rest_prepro.nii.gz
    220521-05:30:52,239 nipype.workflow INFO:
    	 [Job 96] Cached (work_preproc_corregistro.susan.multi_inputs).
    220521-05:30:52,239 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_detrend/_mask_ext_2/_thresh_0.99/detrend.nii.gz -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_detrend/mask_ext_wm/threshold_0.99/fmri_rest_prepro.nii.gz
    220521-05:30:52,246 nipype.workflow INFO:
    	 [Node] Executing "datasink" <nipype.interfaces.io.DataSink>
    220521-05:30:52,243 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain/_mask_ext_2/_thresh_0.99/detrend.nii.gz -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain/mask_ext_wm/threshold_0.99/fmri_rest_prepro.nii.gz
    220521-05:30:52,238 nipype.workflow INFO:
    	 [Node] Setting-up "_mask_func0" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/_mask_ext_1/_thresh_0.99/mask_func/mapflow/_mask_func0".
    220521-05:30:52,238 nipype.workflow INFO:
    	 [Node] Executing "datasink" <nipype.interfaces.io.DataSink>
    220521-05:30:52,251 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_smooth/_mask_ext_2/_thresh_0.99/_smooth0/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_smooth/mask_ext_wm/threshold_0.99/smoooth/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz
    220521-05:30:52,237 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain/_mask_ext_2/_thresh_0.97/detrend.nii.gz -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain/mask_ext_wm/threshold_0.97/fmri_rest_prepro.nii.gz
    220521-05:30:52,254 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain_sin_detrend/_mask_ext_2/_thresh_0.95/_mask_func0/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain_sin_detrend/mask_ext_wm/threshold_0.95/mask_func/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
    220521-05:30:52,255 nipype.workflow INFO:
    	 [Node] Executing "_mask_func0" <nipype.interfaces.fsl.maths.ApplyMask>
    220521-05:30:52,263 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_detrend/_mask_ext_2/_thresh_0.95/detrend.nii.gz -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_detrend/mask_ext_wm/threshold_0.95/fmri_rest_prepro.nii.gz
    220521-05:30:52,263 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_smooth/_mask_ext_2/_thresh_0.97/_smooth0/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_smooth/mask_ext_wm/threshold_0.97/smoooth/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz
    220521-05:30:52,263 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain_sin_detrend/_mask_ext_2/_thresh_0.5/_mask_func0/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain_sin_detrend/mask_ext_wm/threshold_0.5/mask_func/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
    220521-05:30:52,266 nipype.workflow INFO:
    	 [Node] Finished "datasink", elapsed time 0.039061s.
    220521-05:30:52,267 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain/_mask_ext_2/_thresh_0.95/detrend.nii.gz -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain/mask_ext_wm/threshold_0.95/fmri_rest_prepro.nii.gz
    220521-05:30:52,268 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_detrend/_mask_ext_2/_thresh_0.5/detrend.nii.gz -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_detrend/mask_ext_wm/threshold_0.5/fmri_rest_prepro.nii.gz
    220521-05:30:52,273 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_smooth/_mask_ext_2/_thresh_0.95/_smooth0/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_smooth/mask_ext_wm/threshold_0.95/smoooth/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz
    220521-05:30:52,271 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain/_mask_ext_2/_thresh_0.5/detrend.nii.gz -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain/mask_ext_wm/threshold_0.5/fmri_rest_prepro.nii.gz
    220521-05:30:52,276 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_smooth/_mask_ext_2/_thresh_0.5/_smooth0/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_smooth/mask_ext_wm/threshold_0.5/smoooth/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz
    220521-05:30:52,265 nipype.workflow INFO:
    	 [Node] Finished "datasink", elapsed time 0.039859s.
    220521-05:30:52,277 nipype.workflow INFO:
    	 [Node] Finished "datasink", elapsed time 0.023507s.
    220521-05:30:52,282 nipype.workflow INFO:
    	 [Node] Finished "datasink", elapsed time 0.01867s.
    220521-05:30:53,122 nipype.workflow INFO:
    	 [Node] Finished "_mask_func0", elapsed time 0.864822s.
    220521-05:30:54,95 nipype.workflow INFO:
    	 [Job 41] Completed (work_preproc_corregistro.datasink).
    220521-05:30:54,96 nipype.workflow INFO:
    	 [Job 51] Completed (work_preproc_corregistro.datasink).
    220521-05:30:54,97 nipype.workflow INFO:
    	 [Job 61] Completed (work_preproc_corregistro.datasink).
    220521-05:30:54,99 nipype.workflow INFO:
    	 [Job 71] Completed (work_preproc_corregistro.datasink).
    220521-05:30:54,101 nipype.workflow INFO:
    	 [Job 79] Completed (work_preproc_corregistro.mask_func).
    220521-05:30:54,104 nipype.workflow INFO:
    	 [MultiProc] Running 1 tasks, and 13 jobs ready. Free memory (GB): 13.79/13.99, Free processors: 7/8.
                         Currently running:
                           * work_preproc_corregistro.detrend_fmri
    220521-05:30:54,195 nipype.workflow INFO:
    	 [Node] Setting-up "work_preproc_corregistro.detrend" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/_mask_ext_1/_thresh_0.99/detrend".
    220521-05:30:54,195 nipype.workflow INFO:
    	 [Node] Setting-up "work_preproc_corregistro.detrend_fmri" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/_mask_ext_1/_thresh_0.97/detrend_fmri".
    220521-05:30:54,200 nipype.workflow INFO:
    	 [Node] Executing "detrend" <nipype.algorithms.confounds.TSNR>
    220521-05:30:54,199 nipype.workflow INFO:
    	 [Node] Executing "detrend_fmri" <nipype.algorithms.confounds.TSNR>
    220521-05:30:54,204 nipype.workflow INFO:
    	 [Job 97] Cached (work_preproc_corregistro.susan.smooth).
    220521-05:30:54,206 nipype.workflow INFO:
    	 [Node] Setting-up "_mask_func0" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/_mask_ext_1/_thresh_0.97/mask_func/mapflow/_mask_func0".
    220521-05:30:54,209 nipype.workflow INFO:
    	 [Job 104] Cached (work_preproc_corregistro.susan.meanfunc2).
    220521-05:30:54,218 nipype.workflow INFO:
    	 [Node] Executing "_mask_func0" <nipype.interfaces.fsl.maths.ApplyMask>
    220521-05:30:54,225 nipype.workflow INFO:
    	 [Node] Setting-up "_median0" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/susan/_mask_ext_0/_thresh_0.99/median/mapflow/_median0".
    220521-05:30:54,233 nipype.workflow INFO:
    	 [Node] Setting-up "_mask0" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/susan/_mask_ext_0/_thresh_0.99/mask/mapflow/_mask0".
    220521-05:30:54,237 nipype.workflow INFO:
    	 [Node] Executing "_median0" <nipype.interfaces.fsl.utils.ImageStats>
    220521-05:30:54,242 nipype.workflow INFO:
    	 [Node] Executing "_mask0" <nipype.interfaces.fsl.utils.ImageMaths>
    220521-05:30:55,363 nipype.workflow INFO:
    	 [Node] Finished "_mask_func0", elapsed time 1.141616s.
    220521-05:30:55,490 nipype.workflow INFO:
    	 [Node] Finished "_median0", elapsed time 1.248534s.
    220521-05:30:55,548 nipype.workflow INFO:
    	 [Node] Finished "_mask0", elapsed time 1.288444s.
    220521-05:30:56,97 nipype.workflow INFO:
    	 [Job 89] Completed (work_preproc_corregistro.mask_func).
    220521-05:30:56,100 nipype.workflow INFO:
    	 [Job 112] Completed (work_preproc_corregistro.susan.median).
    220521-05:30:56,103 nipype.workflow INFO:
    	 [Job 113] Completed (work_preproc_corregistro.susan.mask).
    220521-05:30:56,109 nipype.workflow INFO:
    	 [MultiProc] Running 3 tasks, and 11 jobs ready. Free memory (GB): 13.39/13.99, Free processors: 5/8.
                         Currently running:
                           * work_preproc_corregistro.detrend_fmri
                           * work_preproc_corregistro.detrend
                           * work_preproc_corregistro.detrend_fmri
    220521-05:30:56,396 nipype.workflow INFO:
    	 [Node] Setting-up "work_preproc_corregistro.detrend_fmri" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/_mask_ext_1/_thresh_0.95/detrend_fmri".
    220521-05:30:56,394 nipype.workflow INFO:
    	 [Node] Setting-up "work_preproc_corregistro.detrend" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/_mask_ext_1/_thresh_0.97/detrend".
    220521-05:30:56,408 nipype.workflow INFO:
    	 [Job 105] Cached (work_preproc_corregistro.susan.merge).
    220521-05:30:56,414 nipype.workflow INFO:
    	 [Node] Setting-up "_mask_func0" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/_mask_ext_1/_thresh_0.95/mask_func/mapflow/_mask_func0".
    220521-05:30:56,408 nipype.workflow INFO:
    	 [Node] Executing "detrend_fmri" <nipype.algorithms.confounds.TSNR>
    220521-05:30:56,438 nipype.workflow INFO:
    	 [Node] Executing "_mask_func0" <nipype.interfaces.fsl.maths.ApplyMask>
    220521-05:30:56,439 nipype.workflow INFO:
    	 [Node] Setting-up "_meanfunc20" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/susan/_mask_ext_0/_thresh_0.99/meanfunc2/mapflow/_meanfunc20".
    220521-05:30:56,430 nipype.workflow INFO:
    	 [Node] Executing "detrend" <nipype.algorithms.confounds.TSNR>
    220521-05:30:56,462 nipype.workflow INFO:
    	 [Node] Executing "_meanfunc20" <nipype.interfaces.fsl.utils.ImageMaths>
    220521-05:30:57,885 nipype.workflow INFO:
    	 [Node] Finished "_meanfunc20", elapsed time 1.41928s.
    220521-05:30:58,123 nipype.workflow INFO:
    	 [Job 114] Completed (work_preproc_corregistro.susan.meanfunc2).
    220521-05:30:58,152 nipype.workflow INFO:
    	 [MultiProc] Running 6 tasks, and 8 jobs ready. Free memory (GB): 12.79/13.99, Free processors: 2/8.
                         Currently running:
                           * work_preproc_corregistro.mask_func
                           * work_preproc_corregistro.detrend_fmri
                           * work_preproc_corregistro.detrend
                           * work_preproc_corregistro.detrend_fmri
                           * work_preproc_corregistro.detrend
                           * work_preproc_corregistro.detrend_fmri
    220521-05:30:58,235 nipype.workflow INFO:
    	 [Node] Finished "_mask_func0", elapsed time 1.757196s.
    220521-05:30:58,411 nipype.workflow INFO:
    	 [Job 106] Cached (work_preproc_corregistro.susan.multi_inputs).
    220521-05:30:58,425 nipype.workflow INFO:
    	 [Node] Setting-up "work_preproc_corregistro.susan.merge" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/susan/_mask_ext_0/_thresh_0.99/merge".
    220521-05:30:58,430 nipype.workflow INFO:
    	 [Node] Executing "merge" <nipype.interfaces.utility.base.Merge>
    220521-05:30:58,433 nipype.workflow INFO:
    	 [Node] Finished "merge", elapsed time 0.000282s.
    220521-05:30:59,302 nipype.workflow INFO:
    	 [Node] Finished "detrend", elapsed time 5.0993s.
    220521-05:30:59,582 nipype.workflow INFO:
    	 [Node] Finished "detrend_fmri", elapsed time 7.355572s.
    220521-05:31:00,117 nipype.workflow INFO:
    	 [Job 78] Completed (work_preproc_corregistro.detrend_fmri).
    220521-05:31:00,122 nipype.workflow INFO:
    	 [Job 80] Completed (work_preproc_corregistro.detrend).
    220521-05:31:00,126 nipype.workflow INFO:
    	 [Job 99] Completed (work_preproc_corregistro.mask_func).
    220521-05:31:00,130 nipype.workflow INFO:
    	 [Job 115] Completed (work_preproc_corregistro.susan.merge).
    220521-05:31:00,137 nipype.workflow INFO:
    	 [MultiProc] Running 3 tasks, and 10 jobs ready. Free memory (GB): 13.39/13.99, Free processors: 5/8.
                         Currently running:
                           * work_preproc_corregistro.detrend_fmri
                           * work_preproc_corregistro.detrend
                           * work_preproc_corregistro.detrend_fmri
    220521-05:31:00,335 nipype.workflow INFO:
    	 [Node] Setting-up "work_preproc_corregistro.datasink" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/_mask_ext_1/_thresh_0.99/datasink".
    220521-05:31:00,336 nipype.workflow INFO:
    	 [Node] Setting-up "work_preproc_corregistro.detrend" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/_mask_ext_1/_thresh_0.95/detrend".
    220521-05:31:00,340 nipype.workflow INFO:
    	 [Node] Executing "detrend" <nipype.algorithms.confounds.TSNR>
    220521-05:31:00,344 nipype.workflow INFO:
    	 [Node] Executing "datasink" <nipype.interfaces.io.DataSink>
    220521-05:31:00,347 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain_sin_detrend/_mask_ext_1/_thresh_0.99/_mask_func0/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain_sin_detrend/mask_ext_gm/threshold_0.99/mask_func/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
    220521-05:31:00,350 nipype.workflow INFO:
    	 [Node] Setting-up "work_preproc_corregistro.susan.multi_inputs" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/susan/_mask_ext_0/_thresh_0.99/multi_inputs".
    220521-05:31:00,353 nipype.workflow INFO:
    	 [Node] Setting-up "_smooth0" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/susan/_mask_ext_1/_thresh_0.5/smooth/mapflow/_smooth0".
    220521-05:31:00,358 nipype.workflow INFO:
    	 [Node] Executing "multi_inputs" <nipype.interfaces.utility.wrappers.Function>
    220521-05:31:00,359 nipype.workflow INFO:
    	 [Node] Finished "detrend", elapsed time 3.893157s.
    220521-05:31:00,350 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_detrend/_mask_ext_1/_thresh_0.99/detrend.nii.gz -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_detrend/mask_ext_gm/threshold_0.99/fmri_rest_prepro.nii.gz
    220521-05:31:00,367 nipype.workflow INFO:
    	 [Node] Setting-up "_median0" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/susan/_mask_ext_0/_thresh_0.97/median/mapflow/_median0".
    220521-05:31:00,367 nipype.workflow INFO:
    	 [Node] Executing "_smooth0" <nipype.interfaces.fsl.preprocess.SUSAN>
    220521-05:31:00,372 nipype.workflow INFO:
    	 [Node] Finished "multi_inputs", elapsed time 0.001605s.
    220521-05:31:00,365 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain/_mask_ext_1/_thresh_0.99/detrend.nii.gz -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain/mask_ext_gm/threshold_0.99/fmri_rest_prepro.nii.gz
    220521-05:31:00,376 nipype.workflow INFO:
    	 [Node] Executing "_median0" <nipype.interfaces.fsl.utils.ImageStats>
    220521-05:31:00,396 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_smooth/_mask_ext_1/_thresh_0.99/_smooth0/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_smooth/mask_ext_gm/threshold_0.99/smoooth/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz
    220521-05:31:00,401 nipype.workflow INFO:
    	 [Node] Finished "datasink", elapsed time 0.054433s.
    220521-05:31:01,233 nipype.workflow INFO:
    	 [Node] Finished "_median0", elapsed time 0.834976s.
    220521-05:31:01,800 nipype.workflow INFO:
    	 [Node] Finished "detrend_fmri", elapsed time 7.598167s.
    220521-05:31:02,116 nipype.workflow INFO:
    	 [Job 88] Completed (work_preproc_corregistro.detrend_fmri).
    220521-05:31:02,118 nipype.workflow INFO:
    	 [Job 90] Completed (work_preproc_corregistro.detrend).
    220521-05:31:02,119 nipype.workflow INFO:
    	 [Job 81] Completed (work_preproc_corregistro.datasink).
    220521-05:31:02,122 nipype.workflow INFO:
    	 [Job 116] Completed (work_preproc_corregistro.susan.multi_inputs).
    220521-05:31:02,124 nipype.workflow INFO:
    	 [Job 122] Completed (work_preproc_corregistro.susan.median).
    220521-05:31:02,128 nipype.workflow INFO:
    	 [MultiProc] Running 3 tasks, and 7 jobs ready. Free memory (GB): 13.39/13.99, Free processors: 5/8.
                         Currently running:
                           * work_preproc_corregistro.susan.smooth
                           * work_preproc_corregistro.detrend
                           * work_preproc_corregistro.detrend_fmri
    220521-05:31:02,232 nipype.workflow INFO:
    	 [Node] Setting-up "work_preproc_corregistro.datasink" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/_mask_ext_1/_thresh_0.97/datasink".
    220521-05:31:02,244 nipype.workflow INFO:
    	 [Node] Setting-up "_smooth0" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/susan/_mask_ext_0/_thresh_0.99/smooth/mapflow/_smooth0".
    220521-05:31:02,245 nipype.workflow INFO:
    	 [Node] Executing "datasink" <nipype.interfaces.io.DataSink>
    220521-05:31:02,249 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain_sin_detrend/_mask_ext_1/_thresh_0.97/_mask_func0/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain_sin_detrend/mask_ext_gm/threshold_0.97/mask_func/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
    220521-05:31:02,252 nipype.workflow INFO:
    	 [Node] Setting-up "_mask0" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/susan/_mask_ext_0/_thresh_0.97/mask/mapflow/_mask0".
    220521-05:31:02,253 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_detrend/_mask_ext_1/_thresh_0.97/detrend.nii.gz -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_detrend/mask_ext_gm/threshold_0.97/fmri_rest_prepro.nii.gz
    220521-05:31:02,265 nipype.workflow INFO:
    	 [Node] Setting-up "_median0" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/susan/_mask_ext_0/_thresh_0.95/median/mapflow/_median0".
    220521-05:31:02,264 nipype.workflow INFO:
    	 [Node] Executing "_mask0" <nipype.interfaces.fsl.utils.ImageMaths>
    220521-05:31:02,269 nipype.workflow INFO:
    	 [Node] Executing "_median0" <nipype.interfaces.fsl.utils.ImageStats>
    220521-05:31:02,262 nipype.workflow INFO:
    	 [Node] Executing "_smooth0" <nipype.interfaces.fsl.preprocess.SUSAN>
    220521-05:31:02,269 nipype.workflow INFO:
    	 [Node] Setting-up "_mask0" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/susan/_mask_ext_0/_thresh_0.95/mask/mapflow/_mask0".
    220521-05:31:02,288 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain/_mask_ext_1/_thresh_0.97/detrend.nii.gz -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain/mask_ext_gm/threshold_0.97/fmri_rest_prepro.nii.gz
    220521-05:31:02,310 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_smooth/_mask_ext_1/_thresh_0.97/_smooth0/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_smooth/mask_ext_gm/threshold_0.97/smoooth/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz
    220521-05:31:02,323 nipype.workflow INFO:
    	 [Node] Finished "datasink", elapsed time 0.074111s.
    220521-05:31:02,311 nipype.workflow INFO:
    	 [Node] Executing "_mask0" <nipype.interfaces.fsl.utils.ImageMaths>
    220521-05:31:03,561 nipype.workflow INFO:
    	 [Node] Finished "_median0", elapsed time 1.289021s.
    220521-05:31:03,605 nipype.workflow INFO:
    	 [Node] Finished "_mask0", elapsed time 1.316119s.
    220521-05:31:03,655 nipype.workflow INFO:
    	 [Node] Finished "_mask0", elapsed time 1.329253s.
    220521-05:31:03,710 nipype.workflow INFO:
    	 [Node] Finished "detrend_fmri", elapsed time 7.288639s.
    220521-05:31:03,748 nipype.workflow INFO:
    	 [Node] Finished "detrend", elapsed time 3.406126s.
    220521-05:31:04,119 nipype.workflow INFO:
    	 [Job 98] Completed (work_preproc_corregistro.detrend_fmri).
    220521-05:31:04,120 nipype.workflow INFO:
    	 [Job 100] Completed (work_preproc_corregistro.detrend).
    220521-05:31:04,122 nipype.workflow INFO:
    	 [Job 91] Completed (work_preproc_corregistro.datasink).
    220521-05:31:04,123 nipype.workflow INFO:
    	 [Job 123] Completed (work_preproc_corregistro.susan.mask).
    220521-05:31:04,125 nipype.workflow INFO:
    	 [Job 132] Completed (work_preproc_corregistro.susan.median).
    220521-05:31:04,126 nipype.workflow INFO:
    	 [Job 133] Completed (work_preproc_corregistro.susan.mask).
    220521-05:31:04,128 nipype.workflow INFO:
    	 [MultiProc] Running 2 tasks, and 5 jobs ready. Free memory (GB): 13.59/13.99, Free processors: 6/8.
                         Currently running:
                           * work_preproc_corregistro.susan.smooth
                           * work_preproc_corregistro.susan.smooth
    220521-05:31:04,212 nipype.workflow INFO:
    	 [Node] Setting-up "work_preproc_corregistro.datasink" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/_mask_ext_1/_thresh_0.95/datasink".
    220521-05:31:04,222 nipype.workflow INFO:
    	 [Node] Executing "datasink" <nipype.interfaces.io.DataSink>
    220521-05:31:04,222 nipype.workflow INFO:
    	 [Node] Setting-up "_meanfunc20" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/susan/_mask_ext_0/_thresh_0.97/meanfunc2/mapflow/_meanfunc20".
    220521-05:31:04,223 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain_sin_detrend/_mask_ext_1/_thresh_0.95/_mask_func0/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain_sin_detrend/mask_ext_gm/threshold_0.95/mask_func/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
    220521-05:31:04,225 nipype.workflow INFO:
    	 [Node] Executing "_meanfunc20" <nipype.interfaces.fsl.utils.ImageMaths>
    220521-05:31:04,225 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_detrend/_mask_ext_1/_thresh_0.95/detrend.nii.gz -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_detrend/mask_ext_gm/threshold_0.95/fmri_rest_prepro.nii.gz
    220521-05:31:04,229 nipype.workflow INFO:
    	 [Node] Setting-up "_meanfunc20" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/susan/_mask_ext_0/_thresh_0.95/meanfunc2/mapflow/_meanfunc20".
    220521-05:31:04,237 nipype.workflow INFO:
    	 [Node] Setting-up "_median0" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/susan/_mask_ext_0/_thresh_0.5/median/mapflow/_median0".
    220521-05:31:04,233 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain/_mask_ext_1/_thresh_0.95/detrend.nii.gz -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain/mask_ext_gm/threshold_0.95/fmri_rest_prepro.nii.gz
    220521-05:31:04,245 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_smooth/_mask_ext_1/_thresh_0.95/_smooth0/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_smooth/mask_ext_gm/threshold_0.95/smoooth/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz
    220521-05:31:04,247 nipype.workflow INFO:
    	 [Node] Executing "_meanfunc20" <nipype.interfaces.fsl.utils.ImageMaths>
    220521-05:31:04,248 nipype.workflow INFO:
    	 [Node] Finished "datasink", elapsed time 0.024128s.
    220521-05:31:04,250 nipype.workflow INFO:
    	 [Node] Executing "_median0" <nipype.interfaces.fsl.utils.ImageStats>
    220521-05:31:04,244 nipype.workflow INFO:
    	 [Node] Setting-up "_mask0" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/susan/_mask_ext_0/_thresh_0.5/mask/mapflow/_mask0".
    220521-05:31:04,260 nipype.workflow INFO:
    	 [Node] Executing "_mask0" <nipype.interfaces.fsl.utils.ImageMaths>
    220521-05:31:05,405 nipype.workflow INFO:
    	 [Node] Finished "_median0", elapsed time 1.143064s.
    220521-05:31:05,519 nipype.workflow INFO:
    	 [Node] Finished "_meanfunc20", elapsed time 1.288089s.
    220521-05:31:05,597 nipype.workflow INFO:
    	 [Node] Finished "_mask0", elapsed time 1.320255s.
    220521-05:31:05,709 nipype.workflow INFO:
    	 [Node] Finished "_meanfunc20", elapsed time 1.4598309999999999s.
    220521-05:31:06,121 nipype.workflow INFO:
    	 [Job 101] Completed (work_preproc_corregistro.datasink).
    220521-05:31:06,123 nipype.workflow INFO:
    	 [Job 124] Completed (work_preproc_corregistro.susan.meanfunc2).
    220521-05:31:06,124 nipype.workflow INFO:
    	 [Job 134] Completed (work_preproc_corregistro.susan.meanfunc2).
    220521-05:31:06,126 nipype.workflow INFO:
    	 [Job 142] Completed (work_preproc_corregistro.susan.median).
    220521-05:31:06,128 nipype.workflow INFO:
    	 [Job 143] Completed (work_preproc_corregistro.susan.mask).
    220521-05:31:06,131 nipype.workflow INFO:
    	 [MultiProc] Running 2 tasks, and 3 jobs ready. Free memory (GB): 13.59/13.99, Free processors: 6/8.
                         Currently running:
                           * work_preproc_corregistro.susan.smooth
                           * work_preproc_corregistro.susan.smooth
    220521-05:31:06,217 nipype.workflow INFO:
    	 [Node] Setting-up "work_preproc_corregistro.susan.merge" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/susan/_mask_ext_0/_thresh_0.97/merge".
    220521-05:31:06,220 nipype.workflow INFO:
    	 [Node] Setting-up "work_preproc_corregistro.susan.merge" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/susan/_mask_ext_0/_thresh_0.95/merge".
    220521-05:31:06,222 nipype.workflow INFO:
    	 [Node] Executing "merge" <nipype.interfaces.utility.base.Merge>
    220521-05:31:06,224 nipype.workflow INFO:
    	 [Node] Executing "merge" <nipype.interfaces.utility.base.Merge>
    220521-05:31:06,226 nipype.workflow INFO:
    	 [Node] Finished "merge", elapsed time 0.000301s.
    220521-05:31:06,226 nipype.workflow INFO:
    	 [Node] Finished "merge", elapsed time 0.000284s.
    220521-05:31:06,227 nipype.workflow INFO:
    	 [Node] Setting-up "_meanfunc20" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/susan/_mask_ext_0/_thresh_0.5/meanfunc2/mapflow/_meanfunc20".
    220521-05:31:06,230 nipype.workflow INFO:
    	 [Node] Executing "_meanfunc20" <nipype.interfaces.fsl.utils.ImageMaths>
    220521-05:31:07,239 nipype.workflow INFO:
    	 [Node] Finished "_meanfunc20", elapsed time 1.007685s.
    220521-05:31:08,123 nipype.workflow INFO:
    	 [Job 125] Completed (work_preproc_corregistro.susan.merge).
    220521-05:31:08,124 nipype.workflow INFO:
    	 [Job 135] Completed (work_preproc_corregistro.susan.merge).
    220521-05:31:08,126 nipype.workflow INFO:
    	 [Job 144] Completed (work_preproc_corregistro.susan.meanfunc2).
    220521-05:31:08,129 nipype.workflow INFO:
    	 [MultiProc] Running 2 tasks, and 3 jobs ready. Free memory (GB): 13.59/13.99, Free processors: 6/8.
                         Currently running:
                           * work_preproc_corregistro.susan.smooth
                           * work_preproc_corregistro.susan.smooth
    220521-05:31:08,202 nipype.workflow INFO:
    	 [Node] Setting-up "work_preproc_corregistro.susan.multi_inputs" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/susan/_mask_ext_0/_thresh_0.95/multi_inputs".
    220521-05:31:08,201 nipype.workflow INFO:
    	 [Node] Setting-up "work_preproc_corregistro.susan.multi_inputs" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/susan/_mask_ext_0/_thresh_0.97/multi_inputs".
    220521-05:31:08,203 nipype.workflow INFO:
    	 [Node] Setting-up "work_preproc_corregistro.susan.merge" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/susan/_mask_ext_0/_thresh_0.5/merge".
    220521-05:31:08,207 nipype.workflow INFO:
    	 [Node] Executing "multi_inputs" <nipype.interfaces.utility.wrappers.Function>
    220521-05:31:08,207 nipype.workflow INFO:
    	 [Node] Executing "multi_inputs" <nipype.interfaces.utility.wrappers.Function>
    220521-05:31:08,209 nipype.workflow INFO:
    	 [Node] Finished "multi_inputs", elapsed time 0.000731s.
    220521-05:31:08,210 nipype.workflow INFO:
    	 [Node] Executing "merge" <nipype.interfaces.utility.base.Merge>
    220521-05:31:08,211 nipype.workflow INFO:
    	 [Node] Finished "multi_inputs", elapsed time 0.001244s.
    220521-05:31:08,213 nipype.workflow INFO:
    	 [Node] Finished "merge", elapsed time 0.000274s.
    220521-05:31:10,124 nipype.workflow INFO:
    	 [Job 126] Completed (work_preproc_corregistro.susan.multi_inputs).
    220521-05:31:10,126 nipype.workflow INFO:
    	 [Job 136] Completed (work_preproc_corregistro.susan.multi_inputs).
    220521-05:31:10,128 nipype.workflow INFO:
    	 [Job 145] Completed (work_preproc_corregistro.susan.merge).
    220521-05:31:10,131 nipype.workflow INFO:
    	 [MultiProc] Running 2 tasks, and 3 jobs ready. Free memory (GB): 13.59/13.99, Free processors: 6/8.
                         Currently running:
                           * work_preproc_corregistro.susan.smooth
                           * work_preproc_corregistro.susan.smooth
    220521-05:31:10,220 nipype.workflow INFO:
    	 [Node] Setting-up "work_preproc_corregistro.susan.multi_inputs" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/susan/_mask_ext_0/_thresh_0.5/multi_inputs".
    220521-05:31:10,220 nipype.workflow INFO:
    	 [Node] Setting-up "_smooth0" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/susan/_mask_ext_0/_thresh_0.97/smooth/mapflow/_smooth0".
    220521-05:31:10,224 nipype.workflow INFO:
    	 [Node] Executing "_smooth0" <nipype.interfaces.fsl.preprocess.SUSAN>
    220521-05:31:10,225 nipype.workflow INFO:
    	 [Node] Setting-up "_smooth0" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/susan/_mask_ext_0/_thresh_0.95/smooth/mapflow/_smooth0".
    220521-05:31:10,226 nipype.workflow INFO:
    	 [Node] Executing "multi_inputs" <nipype.interfaces.utility.wrappers.Function>
    220521-05:31:10,237 nipype.workflow INFO:
    	 [Node] Finished "multi_inputs", elapsed time 0.000952s.
    220521-05:31:10,238 nipype.workflow INFO:
    	 [Node] Executing "_smooth0" <nipype.interfaces.fsl.preprocess.SUSAN>
    220521-05:31:12,127 nipype.workflow INFO:
    	 [Job 146] Completed (work_preproc_corregistro.susan.multi_inputs).
    220521-05:31:12,130 nipype.workflow INFO:
    	 [MultiProc] Running 4 tasks, and 1 jobs ready. Free memory (GB): 13.19/13.99, Free processors: 4/8.
                         Currently running:
                           * work_preproc_corregistro.susan.smooth
                           * work_preproc_corregistro.susan.smooth
                           * work_preproc_corregistro.susan.smooth
                           * work_preproc_corregistro.susan.smooth
    220521-05:31:12,211 nipype.workflow INFO:
    	 [Node] Setting-up "_smooth0" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/susan/_mask_ext_0/_thresh_0.5/smooth/mapflow/_smooth0".
    220521-05:31:12,216 nipype.workflow INFO:
    	 [Node] Executing "_smooth0" <nipype.interfaces.fsl.preprocess.SUSAN>
    220521-05:31:14,130 nipype.workflow INFO:
    	 [MultiProc] Running 5 tasks, and 0 jobs ready. Free memory (GB): 12.99/13.99, Free processors: 3/8.
                         Currently running:
                           * work_preproc_corregistro.susan.smooth
                           * work_preproc_corregistro.susan.smooth
                           * work_preproc_corregistro.susan.smooth
                           * work_preproc_corregistro.susan.smooth
                           * work_preproc_corregistro.susan.smooth
    220521-05:31:27,122 nipype.workflow INFO:
    	 [Node] Finished "_smooth0", elapsed time 26.743948s.
    220521-05:31:28,143 nipype.workflow INFO:
    	 [Job 107] Completed (work_preproc_corregistro.susan.smooth).
    220521-05:31:28,146 nipype.workflow INFO:
    	 [MultiProc] Running 4 tasks, and 2 jobs ready. Free memory (GB): 13.19/13.99, Free processors: 4/8.
                         Currently running:
                           * work_preproc_corregistro.susan.smooth
                           * work_preproc_corregistro.susan.smooth
                           * work_preproc_corregistro.susan.smooth
                           * work_preproc_corregistro.susan.smooth
    220521-05:31:28,220 nipype.workflow INFO:
    	 [Node] Setting-up "work_preproc_corregistro.detrend_fmri" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/_mask_ext_1/_thresh_0.5/detrend_fmri".
    220521-05:31:28,223 nipype.workflow INFO:
    	 [Node] Executing "detrend_fmri" <nipype.algorithms.confounds.TSNR>
    220521-05:31:28,227 nipype.workflow INFO:
    	 [Node] Setting-up "_mask_func0" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/_mask_ext_1/_thresh_0.5/mask_func/mapflow/_mask_func0".
    220521-05:31:28,233 nipype.workflow INFO:
    	 [Node] Executing "_mask_func0" <nipype.interfaces.fsl.maths.ApplyMask>
    220521-05:31:29,389 nipype.workflow INFO:
    	 [Node] Finished "_mask_func0", elapsed time 1.143726s.
    220521-05:31:30,145 nipype.workflow INFO:
    	 [Job 109] Completed (work_preproc_corregistro.mask_func).
    220521-05:31:30,148 nipype.workflow INFO:
    	 [MultiProc] Running 5 tasks, and 1 jobs ready. Free memory (GB): 12.99/13.99, Free processors: 3/8.
                         Currently running:
                           * work_preproc_corregistro.detrend_fmri
                           * work_preproc_corregistro.susan.smooth
                           * work_preproc_corregistro.susan.smooth
                           * work_preproc_corregistro.susan.smooth
                           * work_preproc_corregistro.susan.smooth
    220521-05:31:30,221 nipype.workflow INFO:
    	 [Node] Setting-up "work_preproc_corregistro.detrend" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/_mask_ext_1/_thresh_0.5/detrend".
    220521-05:31:30,225 nipype.workflow INFO:
    	 [Node] Executing "detrend" <nipype.algorithms.confounds.TSNR>
    220521-05:31:32,148 nipype.workflow INFO:
    	 [MultiProc] Running 6 tasks, and 0 jobs ready. Free memory (GB): 12.79/13.99, Free processors: 2/8.
                         Currently running:
                           * work_preproc_corregistro.detrend
                           * work_preproc_corregistro.detrend_fmri
                           * work_preproc_corregistro.susan.smooth
                           * work_preproc_corregistro.susan.smooth
                           * work_preproc_corregistro.susan.smooth
                           * work_preproc_corregistro.susan.smooth
    220521-05:31:33,87 nipype.workflow INFO:
    	 [Node] Finished "detrend", elapsed time 2.8598879999999998s.
    220521-05:31:33,858 nipype.workflow INFO:
    	 [Node] Finished "detrend_fmri", elapsed time 5.629442s.
    220521-05:31:34,149 nipype.workflow INFO:
    	 [Job 108] Completed (work_preproc_corregistro.detrend_fmri).
    220521-05:31:34,151 nipype.workflow INFO:
    	 [Job 110] Completed (work_preproc_corregistro.detrend).
    220521-05:31:34,154 nipype.workflow INFO:
    	 [MultiProc] Running 4 tasks, and 1 jobs ready. Free memory (GB): 13.19/13.99, Free processors: 4/8.
                         Currently running:
                           * work_preproc_corregistro.susan.smooth
                           * work_preproc_corregistro.susan.smooth
                           * work_preproc_corregistro.susan.smooth
                           * work_preproc_corregistro.susan.smooth
    220521-05:31:34,226 nipype.workflow INFO:
    	 [Node] Setting-up "work_preproc_corregistro.datasink" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/_mask_ext_1/_thresh_0.5/datasink".
    220521-05:31:34,237 nipype.workflow INFO:
    	 [Node] Executing "datasink" <nipype.interfaces.io.DataSink>
    220521-05:31:34,240 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain_sin_detrend/_mask_ext_1/_thresh_0.5/_mask_func0/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain_sin_detrend/mask_ext_gm/threshold_0.5/mask_func/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
    220521-05:31:34,243 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_detrend/_mask_ext_1/_thresh_0.5/detrend.nii.gz -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_detrend/mask_ext_gm/threshold_0.5/fmri_rest_prepro.nii.gz
    220521-05:31:34,250 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain/_mask_ext_1/_thresh_0.5/detrend.nii.gz -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain/mask_ext_gm/threshold_0.5/fmri_rest_prepro.nii.gz
    220521-05:31:34,270 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_smooth/_mask_ext_1/_thresh_0.5/_smooth0/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_smooth/mask_ext_gm/threshold_0.5/smoooth/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz
    220521-05:31:34,274 nipype.workflow INFO:
    	 [Node] Finished "datasink", elapsed time 0.033868s.
    220521-05:31:35,697 nipype.workflow INFO:
    	 [Node] Finished "_smooth0", elapsed time 33.41463s.
    220521-05:31:36,151 nipype.workflow INFO:
    	 [Job 117] Completed (work_preproc_corregistro.susan.smooth).
    220521-05:31:36,153 nipype.workflow INFO:
    	 [Job 111] Completed (work_preproc_corregistro.datasink).
    220521-05:31:36,155 nipype.workflow INFO:
    	 [MultiProc] Running 3 tasks, and 2 jobs ready. Free memory (GB): 13.39/13.99, Free processors: 5/8.
                         Currently running:
                           * work_preproc_corregistro.susan.smooth
                           * work_preproc_corregistro.susan.smooth
                           * work_preproc_corregistro.susan.smooth
    220521-05:31:36,232 nipype.workflow INFO:
    	 [Node] Setting-up "work_preproc_corregistro.detrend_fmri" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/_mask_ext_0/_thresh_0.99/detrend_fmri".
    220521-05:31:36,235 nipype.workflow INFO:
    	 [Node] Executing "detrend_fmri" <nipype.algorithms.confounds.TSNR>
    220521-05:31:36,239 nipype.workflow INFO:
    	 [Node] Setting-up "_mask_func0" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/_mask_ext_0/_thresh_0.99/mask_func/mapflow/_mask_func0".
    220521-05:31:36,242 nipype.workflow INFO:
    	 [Node] Executing "_mask_func0" <nipype.interfaces.fsl.maths.ApplyMask>
    220521-05:31:37,125 nipype.workflow INFO:
    	 [Node] Finished "_smooth0", elapsed time 26.884908s.
    220521-05:31:37,332 nipype.workflow INFO:
    	 [Node] Finished "_mask_func0", elapsed time 1.08751s.
    220521-05:31:37,925 nipype.workflow INFO:
    	 [Node] Finished "_smooth0", elapsed time 27.698802s.
    220521-05:31:38,153 nipype.workflow INFO:
    	 [Job 127] Completed (work_preproc_corregistro.susan.smooth).
    220521-05:31:38,155 nipype.workflow INFO:
    	 [Job 137] Completed (work_preproc_corregistro.susan.smooth).
    220521-05:31:38,156 nipype.workflow INFO:
    	 [Job 119] Completed (work_preproc_corregistro.mask_func).
    220521-05:31:38,159 nipype.workflow INFO:
    	 [MultiProc] Running 2 tasks, and 5 jobs ready. Free memory (GB): 13.59/13.99, Free processors: 6/8.
                         Currently running:
                           * work_preproc_corregistro.detrend_fmri
                           * work_preproc_corregistro.susan.smooth
    220521-05:31:38,251 nipype.workflow INFO:
    	 [Node] Setting-up "work_preproc_corregistro.detrend" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/_mask_ext_0/_thresh_0.99/detrend".
    220521-05:31:38,251 nipype.workflow INFO:
    	 [Node] Setting-up "work_preproc_corregistro.detrend_fmri" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/_mask_ext_0/_thresh_0.97/detrend_fmri".
    220521-05:31:38,254 nipype.workflow INFO:
    	 [Node] Executing "detrend_fmri" <nipype.algorithms.confounds.TSNR>
    220521-05:31:38,254 nipype.workflow INFO:
    	 [Node] Executing "detrend" <nipype.algorithms.confounds.TSNR>
    220521-05:31:38,255 nipype.workflow INFO:
    	 [Node] Setting-up "work_preproc_corregistro.detrend_fmri" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/_mask_ext_0/_thresh_0.95/detrend_fmri".
    220521-05:31:38,258 nipype.workflow INFO:
    	 [Node] Executing "detrend_fmri" <nipype.algorithms.confounds.TSNR>
    220521-05:31:38,261 nipype.workflow INFO:
    	 [Node] Setting-up "_mask_func0" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/_mask_ext_0/_thresh_0.95/mask_func/mapflow/_mask_func0".
    220521-05:31:38,264 nipype.workflow INFO:
    	 [Node] Executing "_mask_func0" <nipype.interfaces.fsl.maths.ApplyMask>
    220521-05:31:38,259 nipype.workflow INFO:
    	 [Node] Setting-up "_mask_func0" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/_mask_ext_0/_thresh_0.97/mask_func/mapflow/_mask_func0".
    220521-05:31:38,301 nipype.workflow INFO:
    	 [Node] Executing "_mask_func0" <nipype.interfaces.fsl.maths.ApplyMask>
    220521-05:31:39,726 nipype.workflow INFO:
    	 [Node] Finished "_mask_func0", elapsed time 1.396178s.
    220521-05:31:39,866 nipype.workflow INFO:
    	 [Node] Finished "_mask_func0", elapsed time 1.600449s.
    220521-05:31:40,166 nipype.workflow INFO:
    	 [Job 129] Completed (work_preproc_corregistro.mask_func).
    220521-05:31:40,191 nipype.workflow INFO:
    	 [Job 139] Completed (work_preproc_corregistro.mask_func).
    220521-05:31:40,200 nipype.workflow INFO:
    	 [MultiProc] Running 5 tasks, and 2 jobs ready. Free memory (GB): 12.99/13.99, Free processors: 3/8.
                         Currently running:
                           * work_preproc_corregistro.detrend_fmri
                           * work_preproc_corregistro.detrend_fmri
                           * work_preproc_corregistro.detrend
                           * work_preproc_corregistro.detrend_fmri
                           * work_preproc_corregistro.susan.smooth
    220521-05:31:40,542 nipype.workflow INFO:
    	 [Node] Setting-up "work_preproc_corregistro.detrend" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/_mask_ext_0/_thresh_0.95/detrend".
    220521-05:31:40,540 nipype.workflow INFO:
    	 [Node] Setting-up "work_preproc_corregistro.detrend" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/_mask_ext_0/_thresh_0.97/detrend".
    220521-05:31:40,555 nipype.workflow INFO:
    	 [Node] Executing "detrend" <nipype.algorithms.confounds.TSNR>
    220521-05:31:40,553 nipype.workflow INFO:
    	 [Node] Executing "detrend" <nipype.algorithms.confounds.TSNR>
    220521-05:31:42,166 nipype.workflow INFO:
    	 [MultiProc] Running 7 tasks, and 0 jobs ready. Free memory (GB): 12.59/13.99, Free processors: 1/8.
                         Currently running:
                           * work_preproc_corregistro.detrend
                           * work_preproc_corregistro.detrend
                           * work_preproc_corregistro.detrend_fmri
                           * work_preproc_corregistro.detrend_fmri
                           * work_preproc_corregistro.detrend
                           * work_preproc_corregistro.detrend_fmri
                           * work_preproc_corregistro.susan.smooth
    220521-05:31:43,844 nipype.workflow INFO:
    	 [Node] Finished "detrend_fmri", elapsed time 7.606808s.
    220521-05:31:44,167 nipype.workflow INFO:
    	 [Job 118] Completed (work_preproc_corregistro.detrend_fmri).
    220521-05:31:44,173 nipype.workflow INFO:
    	 [MultiProc] Running 6 tasks, and 0 jobs ready. Free memory (GB): 12.79/13.99, Free processors: 2/8.
                         Currently running:
                           * work_preproc_corregistro.detrend
                           * work_preproc_corregistro.detrend
                           * work_preproc_corregistro.detrend_fmri
                           * work_preproc_corregistro.detrend_fmri
                           * work_preproc_corregistro.detrend
                           * work_preproc_corregistro.susan.smooth
    220521-05:31:44,283 nipype.workflow INFO:
    	 [Node] Finished "detrend", elapsed time 6.008654s.
    220521-05:31:45,197 nipype.workflow INFO:
    	 [Node] Finished "detrend", elapsed time 4.637421s.
    220521-05:31:45,373 nipype.workflow INFO:
    	 [Node] Finished "detrend", elapsed time 4.79353s.
    220521-05:31:46,143 nipype.workflow INFO:
    	 [Node] Finished "detrend_fmri", elapsed time 7.887137s.
    220521-05:31:46,169 nipype.workflow INFO:
    	 [Job 120] Completed (work_preproc_corregistro.detrend).
    220521-05:31:46,171 nipype.workflow INFO:
    	 [Job 128] Completed (work_preproc_corregistro.detrend_fmri).
    220521-05:31:46,172 nipype.workflow INFO:
    	 [Job 130] Completed (work_preproc_corregistro.detrend).
    220521-05:31:46,174 nipype.workflow INFO:
    	 [Job 140] Completed (work_preproc_corregistro.detrend).
    220521-05:31:46,177 nipype.workflow INFO:
    	 [MultiProc] Running 2 tasks, and 2 jobs ready. Free memory (GB): 13.59/13.99, Free processors: 6/8.
                         Currently running:
                           * work_preproc_corregistro.detrend_fmri
                           * work_preproc_corregistro.susan.smooth
    220521-05:31:46,271 nipype.workflow INFO:
    	 [Node] Setting-up "work_preproc_corregistro.datasink" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/_mask_ext_0/_thresh_0.99/datasink".
    220521-05:31:46,275 nipype.workflow INFO:
    	 [Node] Setting-up "work_preproc_corregistro.datasink" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/_mask_ext_0/_thresh_0.97/datasink".
    220521-05:31:46,279 nipype.workflow INFO:
    	 [Node] Executing "datasink" <nipype.interfaces.io.DataSink>
    220521-05:31:46,281 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain_sin_detrend/_mask_ext_0/_thresh_0.99/_mask_func0/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain_sin_detrend/mask_ext_csf/threshold_0.99/mask_func/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
    220521-05:31:46,282 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_detrend/_mask_ext_0/_thresh_0.99/detrend.nii.gz -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_detrend/mask_ext_csf/threshold_0.99/fmri_rest_prepro.nii.gz
    220521-05:31:46,283 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain/_mask_ext_0/_thresh_0.99/detrend.nii.gz -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain/mask_ext_csf/threshold_0.99/fmri_rest_prepro.nii.gz
    220521-05:31:46,284 nipype.workflow INFO:
    	 [Node] Executing "datasink" <nipype.interfaces.io.DataSink>
    220521-05:31:46,286 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain_sin_detrend/_mask_ext_0/_thresh_0.97/_mask_func0/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain_sin_detrend/mask_ext_csf/threshold_0.97/mask_func/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
    220521-05:31:46,288 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_detrend/_mask_ext_0/_thresh_0.97/detrend.nii.gz -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_detrend/mask_ext_csf/threshold_0.97/fmri_rest_prepro.nii.gz
    220521-05:31:46,288 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_smooth/_mask_ext_0/_thresh_0.99/_smooth0/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_smooth/mask_ext_csf/threshold_0.99/smoooth/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz
    220521-05:31:46,291 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain/_mask_ext_0/_thresh_0.97/detrend.nii.gz -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain/mask_ext_csf/threshold_0.97/fmri_rest_prepro.nii.gz
    220521-05:31:46,295 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_smooth/_mask_ext_0/_thresh_0.97/_smooth0/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_smooth/mask_ext_csf/threshold_0.97/smoooth/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz
    220521-05:31:46,294 nipype.workflow INFO:
    	 [Node] Finished "datasink", elapsed time 0.012243s.
    220521-05:31:46,308 nipype.workflow INFO:
    	 [Node] Finished "datasink", elapsed time 0.022057s.
    220521-05:31:46,608 nipype.workflow INFO:
    	 [Node] Finished "detrend_fmri", elapsed time 8.347283000000001s.
    220521-05:31:48,170 nipype.workflow INFO:
    	 [Job 138] Completed (work_preproc_corregistro.detrend_fmri).
    220521-05:31:48,172 nipype.workflow INFO:
    	 [Job 121] Completed (work_preproc_corregistro.datasink).
    220521-05:31:48,173 nipype.workflow INFO:
    	 [Job 131] Completed (work_preproc_corregistro.datasink).
    220521-05:31:48,176 nipype.workflow INFO:
    	 [MultiProc] Running 1 tasks, and 1 jobs ready. Free memory (GB): 13.79/13.99, Free processors: 7/8.
                         Currently running:
                           * work_preproc_corregistro.susan.smooth
    220521-05:31:48,247 nipype.workflow INFO:
    	 [Node] Setting-up "work_preproc_corregistro.datasink" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/_mask_ext_0/_thresh_0.95/datasink".
    220521-05:31:48,255 nipype.workflow INFO:
    	 [Node] Executing "datasink" <nipype.interfaces.io.DataSink>
    220521-05:31:48,257 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain_sin_detrend/_mask_ext_0/_thresh_0.95/_mask_func0/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain_sin_detrend/mask_ext_csf/threshold_0.95/mask_func/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
    220521-05:31:48,259 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_detrend/_mask_ext_0/_thresh_0.95/detrend.nii.gz -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_detrend/mask_ext_csf/threshold_0.95/fmri_rest_prepro.nii.gz
    220521-05:31:48,261 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain/_mask_ext_0/_thresh_0.95/detrend.nii.gz -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain/mask_ext_csf/threshold_0.95/fmri_rest_prepro.nii.gz
    220521-05:31:48,262 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_smooth/_mask_ext_0/_thresh_0.95/_smooth0/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_smooth/mask_ext_csf/threshold_0.95/smoooth/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz
    220521-05:31:48,264 nipype.workflow INFO:
    	 [Node] Finished "datasink", elapsed time 0.006441s.
    220521-05:31:49,412 nipype.workflow INFO:
    	 [Node] Finished "_smooth0", elapsed time 37.192452s.
    220521-05:31:50,172 nipype.workflow INFO:
    	 [Job 147] Completed (work_preproc_corregistro.susan.smooth).
    220521-05:31:50,174 nipype.workflow INFO:
    	 [Job 141] Completed (work_preproc_corregistro.datasink).
    220521-05:31:50,180 nipype.workflow INFO:
    	 [MultiProc] Running 0 tasks, and 2 jobs ready. Free memory (GB): 13.99/13.99, Free processors: 8/8.
    220521-05:31:50,279 nipype.workflow INFO:
    	 [Node] Setting-up "work_preproc_corregistro.detrend_fmri" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/_mask_ext_0/_thresh_0.5/detrend_fmri".
    220521-05:31:50,282 nipype.workflow INFO:
    	 [Node] Executing "detrend_fmri" <nipype.algorithms.confounds.TSNR>
    220521-05:31:50,285 nipype.workflow INFO:
    	 [Node] Setting-up "_mask_func0" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/_mask_ext_0/_thresh_0.5/mask_func/mapflow/_mask_func0".
    220521-05:31:50,287 nipype.workflow INFO:
    	 [Node] Executing "_mask_func0" <nipype.interfaces.fsl.maths.ApplyMask>
    220521-05:31:51,129 nipype.workflow INFO:
    	 [Node] Finished "_mask_func0", elapsed time 0.840595s.
    220521-05:31:52,174 nipype.workflow INFO:
    	 [Job 149] Completed (work_preproc_corregistro.mask_func).
    220521-05:31:52,177 nipype.workflow INFO:
    	 [MultiProc] Running 1 tasks, and 1 jobs ready. Free memory (GB): 13.79/13.99, Free processors: 7/8.
                         Currently running:
                           * work_preproc_corregistro.detrend_fmri
    220521-05:31:52,245 nipype.workflow INFO:
    	 [Node] Setting-up "work_preproc_corregistro.detrend" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/_mask_ext_0/_thresh_0.5/detrend".
    220521-05:31:52,247 nipype.workflow INFO:
    	 [Node] Executing "detrend" <nipype.algorithms.confounds.TSNR>
    220521-05:31:54,177 nipype.workflow INFO:
    	 [MultiProc] Running 2 tasks, and 0 jobs ready. Free memory (GB): 13.59/13.99, Free processors: 6/8.
                         Currently running:
                           * work_preproc_corregistro.detrend
                           * work_preproc_corregistro.detrend_fmri
    220521-05:31:54,846 nipype.workflow INFO:
    	 [Node] Finished "detrend", elapsed time 2.597481s.
    220521-05:31:55,170 nipype.workflow INFO:
    	 [Node] Finished "detrend_fmri", elapsed time 4.886229s.
    220521-05:31:56,179 nipype.workflow INFO:
    	 [Job 148] Completed (work_preproc_corregistro.detrend_fmri).
    220521-05:31:56,181 nipype.workflow INFO:
    	 [Job 150] Completed (work_preproc_corregistro.detrend).
    220521-05:31:56,183 nipype.workflow INFO:
    	 [MultiProc] Running 0 tasks, and 1 jobs ready. Free memory (GB): 13.99/13.99, Free processors: 8/8.
    220521-05:31:56,285 nipype.workflow INFO:
    	 [Node] Setting-up "work_preproc_corregistro.datasink" in "/opt/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/output_workflow/work_preproc_corregistro/_mask_ext_0/_thresh_0.5/datasink".
    220521-05:31:56,295 nipype.workflow INFO:
    	 [Node] Executing "datasink" <nipype.interfaces.io.DataSink>
    220521-05:31:56,298 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain_sin_detrend/_mask_ext_0/_thresh_0.5/_mask_func0/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain_sin_detrend/mask_ext_csf/threshold_0.5/mask_func/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
    220521-05:31:56,299 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_detrend/_mask_ext_0/_thresh_0.5/detrend.nii.gz -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_detrend/mask_ext_csf/threshold_0.5/fmri_rest_prepro.nii.gz
    220521-05:31:56,302 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain/_mask_ext_0/_thresh_0.5/detrend.nii.gz -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/masks_brain/mask_ext_csf/threshold_0.5/fmri_rest_prepro.nii.gz
    220521-05:31:56,304 nipype.interface INFO:
    	 sub: /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_smooth/_mask_ext_0/_thresh_0.5/_smooth0/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz -> /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/datasink/fmri_smooth/mask_ext_csf/threshold_0.5/smoooth/sub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz
    220521-05:31:56,314 nipype.workflow INFO:
    	 [Node] Finished "datasink", elapsed time 0.015561s.
    220521-05:31:58,181 nipype.workflow INFO:
    	 [Job 151] Completed (work_preproc_corregistro.datasink).
    220521-05:31:58,187 nipype.workflow INFO:
    	 [MultiProc] Running 0 tasks, and 0 jobs ready. Free memory (GB): 13.99/13.99, Free processors: 8/8.





    <networkx.classes.digraph.DiGraph at 0x7efe7c8418e0>



## Visualizamos resultado en carpeta ineterna


```python
print(path_output)
```

    /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output



```python
! tree /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/
```

    [01;34m/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/[00m
    ├── [01;34mdatasink[00m
    │   ├── [01;34mfmri_detrend[00m
    │   │   ├── [01;34mmask_ext_csf[00m
    │   │   │   ├── [01;34mthreshold_0.5[00m
    │   │   │   │   └── [01;31mfmri_rest_prepro.nii.gz[00m
    │   │   │   ├── [01;34mthreshold_0.95[00m
    │   │   │   │   └── [01;31mfmri_rest_prepro.nii.gz[00m
    │   │   │   ├── [01;34mthreshold_0.97[00m
    │   │   │   │   └── [01;31mfmri_rest_prepro.nii.gz[00m
    │   │   │   └── [01;34mthreshold_0.99[00m
    │   │   │       └── [01;31mfmri_rest_prepro.nii.gz[00m
    │   │   ├── [01;34mmask_ext_gm[00m
    │   │   │   ├── [01;34mthreshold_0.5[00m
    │   │   │   │   └── [01;31mfmri_rest_prepro.nii.gz[00m
    │   │   │   ├── [01;34mthreshold_0.95[00m
    │   │   │   │   └── [01;31mfmri_rest_prepro.nii.gz[00m
    │   │   │   ├── [01;34mthreshold_0.97[00m
    │   │   │   │   └── [01;31mfmri_rest_prepro.nii.gz[00m
    │   │   │   └── [01;34mthreshold_0.99[00m
    │   │   │       └── [01;31mfmri_rest_prepro.nii.gz[00m
    │   │   └── [01;34mmask_ext_wm[00m
    │   │       ├── [01;34mthreshold_0.5[00m
    │   │       │   └── [01;31mfmri_rest_prepro.nii.gz[00m
    │   │       ├── [01;34mthreshold_0.95[00m
    │   │       │   └── [01;31mfmri_rest_prepro.nii.gz[00m
    │   │       ├── [01;34mthreshold_0.97[00m
    │   │       │   └── [01;31mfmri_rest_prepro.nii.gz[00m
    │   │       └── [01;34mthreshold_0.99[00m
    │   │           └── [01;31mfmri_rest_prepro.nii.gz[00m
    │   ├── [01;34mfmri_sin_smooth[00m
    │   │   ├── sub-01_task-rest_bold_roi_st_mcf_flirt.mat
    │   │   └── sub-01_task-rest_bold_roi_st_mcf_flirt.nii
    │   ├── [01;34mfmri_smooth[00m
    │   │   ├── [01;34mmask_ext_csf[00m
    │   │   │   ├── [01;34mthreshold_0.5[00m
    │   │   │   │   └── [01;34msmoooth[00m
    │   │   │   │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz[00m
    │   │   │   ├── [01;34mthreshold_0.95[00m
    │   │   │   │   └── [01;34msmoooth[00m
    │   │   │   │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz[00m
    │   │   │   ├── [01;34mthreshold_0.97[00m
    │   │   │   │   └── [01;34msmoooth[00m
    │   │   │   │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz[00m
    │   │   │   └── [01;34mthreshold_0.99[00m
    │   │   │       └── [01;34msmoooth[00m
    │   │   │           └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz[00m
    │   │   ├── [01;34mmask_ext_gm[00m
    │   │   │   ├── [01;34mthreshold_0.5[00m
    │   │   │   │   └── [01;34msmoooth[00m
    │   │   │   │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz[00m
    │   │   │   ├── [01;34mthreshold_0.95[00m
    │   │   │   │   └── [01;34msmoooth[00m
    │   │   │   │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz[00m
    │   │   │   ├── [01;34mthreshold_0.97[00m
    │   │   │   │   └── [01;34msmoooth[00m
    │   │   │   │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz[00m
    │   │   │   └── [01;34mthreshold_0.99[00m
    │   │   │       └── [01;34msmoooth[00m
    │   │   │           └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz[00m
    │   │   └── [01;34mmask_ext_wm[00m
    │   │       ├── [01;34mthreshold_0.5[00m
    │   │       │   └── [01;34msmoooth[00m
    │   │       │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz[00m
    │   │       ├── [01;34mthreshold_0.95[00m
    │   │       │   └── [01;34msmoooth[00m
    │   │       │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz[00m
    │   │       ├── [01;34mthreshold_0.97[00m
    │   │       │   └── [01;34msmoooth[00m
    │   │       │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz[00m
    │   │       └── [01;34mthreshold_0.99[00m
    │   │           └── [01;34msmoooth[00m
    │   │               └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz[00m
    │   ├── [01;34mmask_files[00m
    │   │   ├── [01;31msub-01_T1w_brain_pve_0.nii.gz[00m
    │   │   ├── [01;31msub-01_T1w_brain_pve_1.nii.gz[00m
    │   │   └── [01;31msub-01_T1w_brain_pve_2.nii.gz[00m
    │   ├── [01;34mmasks_brain[00m
    │   │   ├── [01;34mmask_ext_csf[00m
    │   │   │   ├── [01;34mthreshold_0.5[00m
    │   │   │   │   └── [01;31mfmri_rest_prepro.nii.gz[00m
    │   │   │   ├── [01;34mthreshold_0.95[00m
    │   │   │   │   └── [01;31mfmri_rest_prepro.nii.gz[00m
    │   │   │   ├── [01;34mthreshold_0.97[00m
    │   │   │   │   └── [01;31mfmri_rest_prepro.nii.gz[00m
    │   │   │   └── [01;34mthreshold_0.99[00m
    │   │   │       └── [01;31mfmri_rest_prepro.nii.gz[00m
    │   │   ├── [01;34mmask_ext_gm[00m
    │   │   │   ├── [01;34mthreshold_0.5[00m
    │   │   │   │   └── [01;31mfmri_rest_prepro.nii.gz[00m
    │   │   │   ├── [01;34mthreshold_0.95[00m
    │   │   │   │   └── [01;31mfmri_rest_prepro.nii.gz[00m
    │   │   │   ├── [01;34mthreshold_0.97[00m
    │   │   │   │   └── [01;31mfmri_rest_prepro.nii.gz[00m
    │   │   │   └── [01;34mthreshold_0.99[00m
    │   │   │       └── [01;31mfmri_rest_prepro.nii.gz[00m
    │   │   └── [01;34mmask_ext_wm[00m
    │   │       ├── [01;34mthreshold_0.5[00m
    │   │       │   └── [01;31mfmri_rest_prepro.nii.gz[00m
    │   │       ├── [01;34mthreshold_0.95[00m
    │   │       │   └── [01;31mfmri_rest_prepro.nii.gz[00m
    │   │       ├── [01;34mthreshold_0.97[00m
    │   │       │   └── [01;31mfmri_rest_prepro.nii.gz[00m
    │   │       └── [01;34mthreshold_0.99[00m
    │   │           └── [01;31mfmri_rest_prepro.nii.gz[00m
    │   └── [01;34mmasks_brain_sin_detrend[00m
    │       ├── [01;34mmask_ext_csf[00m
    │       │   ├── [01;34mthreshold_0.5[00m
    │       │   │   └── [01;34mmask_func[00m
    │       │   │       └── sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
    │       │   ├── [01;34mthreshold_0.95[00m
    │       │   │   └── [01;34mmask_func[00m
    │       │   │       └── sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
    │       │   ├── [01;34mthreshold_0.97[00m
    │       │   │   └── [01;34mmask_func[00m
    │       │   │       └── sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
    │       │   └── [01;34mthreshold_0.99[00m
    │       │       └── [01;34mmask_func[00m
    │       │           └── sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
    │       ├── [01;34mmask_ext_gm[00m
    │       │   ├── [01;34mthreshold_0.5[00m
    │       │   │   └── [01;34mmask_func[00m
    │       │   │       └── sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
    │       │   ├── [01;34mthreshold_0.95[00m
    │       │   │   └── [01;34mmask_func[00m
    │       │   │       └── sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
    │       │   ├── [01;34mthreshold_0.97[00m
    │       │   │   └── [01;34mmask_func[00m
    │       │   │       └── sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
    │       │   └── [01;34mthreshold_0.99[00m
    │       │       └── [01;34mmask_func[00m
    │       │           └── sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
    │       └── [01;34mmask_ext_wm[00m
    │           ├── [01;34mthreshold_0.5[00m
    │           │   └── [01;34mmask_func[00m
    │           │       └── sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
    │           ├── [01;34mthreshold_0.95[00m
    │           │   └── [01;34mmask_func[00m
    │           │       └── sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
    │           ├── [01;34mthreshold_0.97[00m
    │           │   └── [01;34mmask_func[00m
    │           │       └── sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
    │           └── [01;34mthreshold_0.99[00m
    │               └── [01;34mmask_func[00m
    │                   └── sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
    └── [01;34moutput_workflow[00m
        ├── sub-01_task-rest_bold_roi_st_mcf
        ├── [01;31msub-01_task-rest_bold_roi_st_mcf_mean_reg.nii.gz[00m
        ├── [01;31msub-01_task-rest_bold_roi_st_mcf.nii.gz[00m
        └── [01;34mwork_preproc_corregistro[00m
            ├── [01;34mapplywarp[00m
            │   ├── _0x0e80cae05f73811716bb0729834375a8.json
            │   ├── command.txt
            │   ├── _inputs.pklz
            │   ├── _node.pklz
            │   ├── [01;34m_report[00m
            │   │   └── report.rst
            │   ├── result_applywarp.pklz
            │   ├── sub-01_task-rest_bold_roi_st_mcf_flirt.mat
            │   └── sub-01_task-rest_bold_roi_st_mcf_flirt.nii
            ├── [01;34mcoreg_bbr[00m
            │   ├── _0xb83866fe259ac311f62ebdc9a77be70e.json
            │   ├── command.txt
            │   ├── _inputs.pklz
            │   ├── _node.pklz
            │   ├── [01;34m_report[00m
            │   │   └── report.rst
            │   ├── result_coreg_bbr.pklz
            │   └── sub-01_task-rest_bold_roi_st_mcf_mean_reg_flirt.mat
            ├── [01;34mcoreg_pre[00m
            │   ├── _0xefefd2677d9b2074ab856af6be0160f7.json
            │   ├── command.txt
            │   ├── _inputs.pklz
            │   ├── _node.pklz
            │   ├── [01;34m_report[00m
            │   │   └── report.rst
            │   ├── result_coreg_pre.pklz
            │   └── sub-01_task-rest_bold_roi_st_mcf_mean_reg_flirt.mat
            ├── d3.js
            ├── graph1.json
            ├── graph_detailed.dot
            ├── [01;35mgraph_detailed.png[00m
            ├── graph.dot
            ├── graph.json
            ├── [01;35mgraph.png[00m
            ├── index.html
            ├── [01;34m_mask_ext_0[00m
            │   ├── [01;34mget_mask_ext[00m
            │   │   ├── _0xb6376aa6409b43b0ab1d2f5a93618a8a.json
            │   │   ├── _inputs.pklz
            │   │   ├── _node.pklz
            │   │   ├── [01;34m_report[00m
            │   │   │   └── report.rst
            │   │   └── result_get_mask_ext.pklz
            │   ├── [01;34mresample[00m
            │   │   ├── _0x84f0aa0dda8bc78e984f2dcaf8c29b7c.json
            │   │   ├── command.txt
            │   │   ├── _inputs.pklz
            │   │   ├── _node.pklz
            │   │   ├── [01;34m_report[00m
            │   │   │   └── report.rst
            │   │   ├── result_resample.pklz
            │   │   ├── sub-01_T1w_brain_pve_0_flirt.mat
            │   │   └── sub-01_T1w_brain_pve_0_flirt.nii
            │   ├── [01;34m_thresh_0.5[00m
            │   │   ├── [01;34mdatasink[00m
            │   │   │   ├── _0x2a9a1fbc0fbbdd97364aa9287dd1a738.json
            │   │   │   ├── _inputs.pklz
            │   │   │   ├── _node.pklz
            │   │   │   ├── [01;34m_report[00m
            │   │   │   │   └── report.rst
            │   │   │   └── result_datasink.pklz
            │   │   ├── [01;34mdetrend[00m
            │   │   │   ├── _0x1784c89f8f66dd18dfdf1b1e18e04cc8.json
            │   │   │   ├── [01;31mdetrend.nii.gz[00m
            │   │   │   ├── _inputs.pklz
            │   │   │   ├── _node.pklz
            │   │   │   ├── [01;34m_report[00m
            │   │   │   │   └── report.rst
            │   │   │   └── result_detrend.pklz
            │   │   ├── [01;34mdetrend_fmri[00m
            │   │   │   ├── _0xc85818eb435719b886f160ae991f985f.json
            │   │   │   ├── [01;31mdetrend.nii.gz[00m
            │   │   │   ├── _inputs.pklz
            │   │   │   ├── _node.pklz
            │   │   │   ├── [01;34m_report[00m
            │   │   │   │   └── report.rst
            │   │   │   └── result_detrend_fmri.pklz
            │   │   ├── [01;34mmask_EXT[00m
            │   │   │   ├── _0xa8242118485d11b38a574f43e3ef15dd.json
            │   │   │   ├── command.txt
            │   │   │   ├── _inputs.pklz
            │   │   │   ├── _node.pklz
            │   │   │   ├── [01;34m_report[00m
            │   │   │   │   └── report.rst
            │   │   │   ├── result_mask_EXT.pklz
            │   │   │   └── sub-01_T1w_brain_pve_0_flirt_thresh.nii
            │   │   └── [01;34mmask_func[00m
            │   │       ├── _0x5d5db8f3b24a2829f5f246f10ab10edd.json
            │   │       ├── _inputs.pklz
            │   │       ├── [01;34mmapflow[00m
            │   │       │   └── [01;34m_mask_func0[00m
            │   │       │       ├── _0x5970ab50f35cf4914334af6ab0d8486d.json
            │   │       │       ├── command.txt
            │   │       │       ├── _inputs.pklz
            │   │       │       ├── _node.pklz
            │   │       │       ├── [01;34m_report[00m
            │   │       │       │   └── report.rst
            │   │       │       ├── result__mask_func0.pklz
            │   │       │       └── sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
            │   │       ├── _node.pklz
            │   │       ├── [01;34m_report[00m
            │   │       │   └── report.rst
            │   │       └── result_mask_func.pklz
            │   ├── [01;34m_thresh_0.95[00m
            │   │   ├── [01;34mdatasink[00m
            │   │   │   ├── _0x5fc80f909b26a0c833f09864efb37a00.json
            │   │   │   ├── _inputs.pklz
            │   │   │   ├── _node.pklz
            │   │   │   ├── [01;34m_report[00m
            │   │   │   │   └── report.rst
            │   │   │   └── result_datasink.pklz
            │   │   ├── [01;34mdetrend[00m
            │   │   │   ├── _0xa6ecc3bdbaba72a404bfc33f69f614cd.json
            │   │   │   ├── [01;31mdetrend.nii.gz[00m
            │   │   │   ├── _inputs.pklz
            │   │   │   ├── _node.pklz
            │   │   │   ├── [01;34m_report[00m
            │   │   │   │   └── report.rst
            │   │   │   └── result_detrend.pklz
            │   │   ├── [01;34mdetrend_fmri[00m
            │   │   │   ├── _0x28bc218e2dcede2c6eda754b02f7ed8e.json
            │   │   │   ├── [01;31mdetrend.nii.gz[00m
            │   │   │   ├── _inputs.pklz
            │   │   │   ├── _node.pklz
            │   │   │   ├── [01;34m_report[00m
            │   │   │   │   └── report.rst
            │   │   │   └── result_detrend_fmri.pklz
            │   │   ├── [01;34mmask_EXT[00m
            │   │   │   ├── _0xb983115d10b94de6808c35017ad2eb37.json
            │   │   │   ├── command.txt
            │   │   │   ├── _inputs.pklz
            │   │   │   ├── _node.pklz
            │   │   │   ├── [01;34m_report[00m
            │   │   │   │   └── report.rst
            │   │   │   ├── result_mask_EXT.pklz
            │   │   │   └── sub-01_T1w_brain_pve_0_flirt_thresh.nii
            │   │   └── [01;34mmask_func[00m
            │   │       ├── _0x1556acb3c74d551b45ebdbb8bae9f0b7.json
            │   │       ├── _inputs.pklz
            │   │       ├── [01;34mmapflow[00m
            │   │       │   └── [01;34m_mask_func0[00m
            │   │       │       ├── _0xed734bbb225b935e7133689bf6c52127.json
            │   │       │       ├── command.txt
            │   │       │       ├── _inputs.pklz
            │   │       │       ├── _node.pklz
            │   │       │       ├── [01;34m_report[00m
            │   │       │       │   └── report.rst
            │   │       │       ├── result__mask_func0.pklz
            │   │       │       └── sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
            │   │       ├── _node.pklz
            │   │       ├── [01;34m_report[00m
            │   │       │   └── report.rst
            │   │       └── result_mask_func.pklz
            │   ├── [01;34m_thresh_0.97[00m
            │   │   ├── [01;34mdatasink[00m
            │   │   │   ├── _0x24ffa4e806e30d9f18ddbe1df136dd12.json
            │   │   │   ├── _inputs.pklz
            │   │   │   ├── _node.pklz
            │   │   │   ├── [01;34m_report[00m
            │   │   │   │   └── report.rst
            │   │   │   └── result_datasink.pklz
            │   │   ├── [01;34mdetrend[00m
            │   │   │   ├── _0x37740eb15e8bea5060bf38c72941e05f.json
            │   │   │   ├── [01;31mdetrend.nii.gz[00m
            │   │   │   ├── _inputs.pklz
            │   │   │   ├── _node.pklz
            │   │   │   ├── [01;34m_report[00m
            │   │   │   │   └── report.rst
            │   │   │   └── result_detrend.pklz
            │   │   ├── [01;34mdetrend_fmri[00m
            │   │   │   ├── _0x17f75aa3174eeb51fa3492dba56bb0db.json
            │   │   │   ├── [01;31mdetrend.nii.gz[00m
            │   │   │   ├── _inputs.pklz
            │   │   │   ├── _node.pklz
            │   │   │   ├── [01;34m_report[00m
            │   │   │   │   └── report.rst
            │   │   │   └── result_detrend_fmri.pklz
            │   │   ├── [01;34mmask_EXT[00m
            │   │   │   ├── _0xcf4e68b9761d789fb45213d07b5f7853.json
            │   │   │   ├── command.txt
            │   │   │   ├── _inputs.pklz
            │   │   │   ├── _node.pklz
            │   │   │   ├── [01;34m_report[00m
            │   │   │   │   └── report.rst
            │   │   │   ├── result_mask_EXT.pklz
            │   │   │   └── sub-01_T1w_brain_pve_0_flirt_thresh.nii
            │   │   └── [01;34mmask_func[00m
            │   │       ├── _0xe404a68311bb771069c90bbc153cfc90.json
            │   │       ├── _inputs.pklz
            │   │       ├── [01;34mmapflow[00m
            │   │       │   └── [01;34m_mask_func0[00m
            │   │       │       ├── _0x650607b65b1bee50cc29e85293fe09dc.json
            │   │       │       ├── command.txt
            │   │       │       ├── _inputs.pklz
            │   │       │       ├── _node.pklz
            │   │       │       ├── [01;34m_report[00m
            │   │       │       │   └── report.rst
            │   │       │       ├── result__mask_func0.pklz
            │   │       │       └── sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
            │   │       ├── _node.pklz
            │   │       ├── [01;34m_report[00m
            │   │       │   └── report.rst
            │   │       └── result_mask_func.pklz
            │   └── [01;34m_thresh_0.99[00m
            │       ├── [01;34mdatasink[00m
            │       │   ├── _0x6c7b9ce387a04519840791c3dc67adeb.json
            │       │   ├── _inputs.pklz
            │       │   ├── _node.pklz
            │       │   ├── [01;34m_report[00m
            │       │   │   └── report.rst
            │       │   └── result_datasink.pklz
            │       ├── [01;34mdetrend[00m
            │       │   ├── _0x4746a408dbee49152305111d1be2ff27.json
            │       │   ├── [01;31mdetrend.nii.gz[00m
            │       │   ├── _inputs.pklz
            │       │   ├── _node.pklz
            │       │   ├── [01;34m_report[00m
            │       │   │   └── report.rst
            │       │   └── result_detrend.pklz
            │       ├── [01;34mdetrend_fmri[00m
            │       │   ├── _0xf21f63e2580b8c0bb4c35639a98a8c0d.json
            │       │   ├── [01;31mdetrend.nii.gz[00m
            │       │   ├── _inputs.pklz
            │       │   ├── _node.pklz
            │       │   ├── [01;34m_report[00m
            │       │   │   └── report.rst
            │       │   └── result_detrend_fmri.pklz
            │       ├── [01;34mmask_EXT[00m
            │       │   ├── _0xf092ad9e079f8744c951516d099bac0f.json
            │       │   ├── command.txt
            │       │   ├── _inputs.pklz
            │       │   ├── _node.pklz
            │       │   ├── [01;34m_report[00m
            │       │   │   └── report.rst
            │       │   ├── result_mask_EXT.pklz
            │       │   └── sub-01_T1w_brain_pve_0_flirt_thresh.nii
            │       └── [01;34mmask_func[00m
            │           ├── _0xdcd14a4b17cd09642a7d81de1523db0b.json
            │           ├── _inputs.pklz
            │           ├── [01;34mmapflow[00m
            │           │   └── [01;34m_mask_func0[00m
            │           │       ├── _0x996f666c18905963962644f21c9b2220.json
            │           │       ├── command.txt
            │           │       ├── _inputs.pklz
            │           │       ├── _node.pklz
            │           │       ├── [01;34m_report[00m
            │           │       │   └── report.rst
            │           │       ├── result__mask_func0.pklz
            │           │       └── sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
            │           ├── _node.pklz
            │           ├── [01;34m_report[00m
            │           │   └── report.rst
            │           └── result_mask_func.pklz
            ├── [01;34m_mask_ext_1[00m
            │   ├── [01;34mget_mask_ext[00m
            │   │   ├── _0xde8934cf4456ad89ce55b11d29e1d99a.json
            │   │   ├── _inputs.pklz
            │   │   ├── _node.pklz
            │   │   ├── [01;34m_report[00m
            │   │   │   └── report.rst
            │   │   └── result_get_mask_ext.pklz
            │   ├── [01;34mresample[00m
            │   │   ├── _0x5b393694019e76f4ed997568e5d39d5e.json
            │   │   ├── command.txt
            │   │   ├── _inputs.pklz
            │   │   ├── _node.pklz
            │   │   ├── [01;34m_report[00m
            │   │   │   └── report.rst
            │   │   ├── result_resample.pklz
            │   │   ├── sub-01_T1w_brain_pve_1_flirt.mat
            │   │   └── sub-01_T1w_brain_pve_1_flirt.nii
            │   ├── [01;34m_thresh_0.5[00m
            │   │   ├── [01;34mdatasink[00m
            │   │   │   ├── _0x4e2d1e1ffc8820396faf83de0cfe8c37.json
            │   │   │   ├── _inputs.pklz
            │   │   │   ├── _node.pklz
            │   │   │   ├── [01;34m_report[00m
            │   │   │   │   └── report.rst
            │   │   │   └── result_datasink.pklz
            │   │   ├── [01;34mdetrend[00m
            │   │   │   ├── _0xc2f1f31d0f8a55f6251b488263f30db7.json
            │   │   │   ├── [01;31mdetrend.nii.gz[00m
            │   │   │   ├── _inputs.pklz
            │   │   │   ├── _node.pklz
            │   │   │   ├── [01;34m_report[00m
            │   │   │   │   └── report.rst
            │   │   │   └── result_detrend.pklz
            │   │   ├── [01;34mdetrend_fmri[00m
            │   │   │   ├── _0x41d2668af0a2b4ddc935907197f31045.json
            │   │   │   ├── [01;31mdetrend.nii.gz[00m
            │   │   │   ├── _inputs.pklz
            │   │   │   ├── _node.pklz
            │   │   │   ├── [01;34m_report[00m
            │   │   │   │   └── report.rst
            │   │   │   └── result_detrend_fmri.pklz
            │   │   ├── [01;34mmask_EXT[00m
            │   │   │   ├── _0xf65bfedf12f2d6dbcb0a214645adcca6.json
            │   │   │   ├── command.txt
            │   │   │   ├── _inputs.pklz
            │   │   │   ├── _node.pklz
            │   │   │   ├── [01;34m_report[00m
            │   │   │   │   └── report.rst
            │   │   │   ├── result_mask_EXT.pklz
            │   │   │   └── sub-01_T1w_brain_pve_1_flirt_thresh.nii
            │   │   └── [01;34mmask_func[00m
            │   │       ├── _0x90c7d5e0b548bb1c4c8575fa5e7c50ce.json
            │   │       ├── _inputs.pklz
            │   │       ├── [01;34mmapflow[00m
            │   │       │   └── [01;34m_mask_func0[00m
            │   │       │       ├── _0xc0ed4b400658febe3cc5a3d647514f92.json
            │   │       │       ├── command.txt
            │   │       │       ├── _inputs.pklz
            │   │       │       ├── _node.pklz
            │   │       │       ├── [01;34m_report[00m
            │   │       │       │   └── report.rst
            │   │       │       ├── result__mask_func0.pklz
            │   │       │       └── sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
            │   │       ├── _node.pklz
            │   │       ├── [01;34m_report[00m
            │   │       │   └── report.rst
            │   │       └── result_mask_func.pklz
            │   ├── [01;34m_thresh_0.95[00m
            │   │   ├── [01;34mdatasink[00m
            │   │   │   ├── _0x24d93b7c42721b186b69ccc65edcee7e.json
            │   │   │   ├── _inputs.pklz
            │   │   │   ├── _node.pklz
            │   │   │   ├── [01;34m_report[00m
            │   │   │   │   └── report.rst
            │   │   │   └── result_datasink.pklz
            │   │   ├── [01;34mdetrend[00m
            │   │   │   ├── _0x9f0995c0f36ea6cc0f69775606e3cac4.json
            │   │   │   ├── [01;31mdetrend.nii.gz[00m
            │   │   │   ├── _inputs.pklz
            │   │   │   ├── _node.pklz
            │   │   │   ├── [01;34m_report[00m
            │   │   │   │   └── report.rst
            │   │   │   └── result_detrend.pklz
            │   │   ├── [01;34mdetrend_fmri[00m
            │   │   │   ├── _0xb63a28f79226e622abcdc5a96accf54a.json
            │   │   │   ├── [01;31mdetrend.nii.gz[00m
            │   │   │   ├── _inputs.pklz
            │   │   │   ├── _node.pklz
            │   │   │   ├── [01;34m_report[00m
            │   │   │   │   └── report.rst
            │   │   │   └── result_detrend_fmri.pklz
            │   │   ├── [01;34mmask_EXT[00m
            │   │   │   ├── _0x536a210b13a5a18e981e0836239b6596.json
            │   │   │   ├── command.txt
            │   │   │   ├── _inputs.pklz
            │   │   │   ├── _node.pklz
            │   │   │   ├── [01;34m_report[00m
            │   │   │   │   └── report.rst
            │   │   │   ├── result_mask_EXT.pklz
            │   │   │   └── sub-01_T1w_brain_pve_1_flirt_thresh.nii
            │   │   └── [01;34mmask_func[00m
            │   │       ├── _0xa409327b5b7cf8050cb975b668a842b4.json
            │   │       ├── _inputs.pklz
            │   │       ├── [01;34mmapflow[00m
            │   │       │   └── [01;34m_mask_func0[00m
            │   │       │       ├── _0x482c0cbf63338c922c320a911acc74ae.json
            │   │       │       ├── command.txt
            │   │       │       ├── _inputs.pklz
            │   │       │       ├── _node.pklz
            │   │       │       ├── [01;34m_report[00m
            │   │       │       │   └── report.rst
            │   │       │       ├── result__mask_func0.pklz
            │   │       │       └── sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
            │   │       ├── _node.pklz
            │   │       ├── [01;34m_report[00m
            │   │       │   └── report.rst
            │   │       └── result_mask_func.pklz
            │   ├── [01;34m_thresh_0.97[00m
            │   │   ├── [01;34mdatasink[00m
            │   │   │   ├── _0x61da7f603ba973577e7d12ede3cf0b4e.json
            │   │   │   ├── _inputs.pklz
            │   │   │   ├── _node.pklz
            │   │   │   ├── [01;34m_report[00m
            │   │   │   │   └── report.rst
            │   │   │   └── result_datasink.pklz
            │   │   ├── [01;34mdetrend[00m
            │   │   │   ├── _0x40ed13c049fcdc158285891ae1304e71.json
            │   │   │   ├── [01;31mdetrend.nii.gz[00m
            │   │   │   ├── _inputs.pklz
            │   │   │   ├── _node.pklz
            │   │   │   ├── [01;34m_report[00m
            │   │   │   │   └── report.rst
            │   │   │   └── result_detrend.pklz
            │   │   ├── [01;34mdetrend_fmri[00m
            │   │   │   ├── _0xf9074e63beb5ecb011f95f870d1631a5.json
            │   │   │   ├── [01;31mdetrend.nii.gz[00m
            │   │   │   ├── _inputs.pklz
            │   │   │   ├── _node.pklz
            │   │   │   ├── [01;34m_report[00m
            │   │   │   │   └── report.rst
            │   │   │   └── result_detrend_fmri.pklz
            │   │   ├── [01;34mmask_EXT[00m
            │   │   │   ├── _0x5ad345cb21cf895cf7949690dcad152f.json
            │   │   │   ├── command.txt
            │   │   │   ├── _inputs.pklz
            │   │   │   ├── _node.pklz
            │   │   │   ├── [01;34m_report[00m
            │   │   │   │   └── report.rst
            │   │   │   ├── result_mask_EXT.pklz
            │   │   │   └── sub-01_T1w_brain_pve_1_flirt_thresh.nii
            │   │   └── [01;34mmask_func[00m
            │   │       ├── _0x28757e78b963b3e396c0df813bdeaf1e.json
            │   │       ├── _inputs.pklz
            │   │       ├── [01;34mmapflow[00m
            │   │       │   └── [01;34m_mask_func0[00m
            │   │       │       ├── _0x4617400c2d3f66d4545c6a2792fd38a4.json
            │   │       │       ├── command.txt
            │   │       │       ├── _inputs.pklz
            │   │       │       ├── _node.pklz
            │   │       │       ├── [01;34m_report[00m
            │   │       │       │   └── report.rst
            │   │       │       ├── result__mask_func0.pklz
            │   │       │       └── sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
            │   │       ├── _node.pklz
            │   │       ├── [01;34m_report[00m
            │   │       │   └── report.rst
            │   │       └── result_mask_func.pklz
            │   └── [01;34m_thresh_0.99[00m
            │       ├── [01;34mdatasink[00m
            │       │   ├── _0x0226d4d7fbb86a7709bb14268905afbf.json
            │       │   ├── _inputs.pklz
            │       │   ├── _node.pklz
            │       │   ├── [01;34m_report[00m
            │       │   │   └── report.rst
            │       │   └── result_datasink.pklz
            │       ├── [01;34mdetrend[00m
            │       │   ├── _0x82e02ae1397102eb23474dac4fcf9ed2.json
            │       │   ├── [01;31mdetrend.nii.gz[00m
            │       │   ├── _inputs.pklz
            │       │   ├── _node.pklz
            │       │   ├── [01;34m_report[00m
            │       │   │   └── report.rst
            │       │   └── result_detrend.pklz
            │       ├── [01;34mdetrend_fmri[00m
            │       │   ├── _0x9b71556fafb6f6f503dce0013afb567f.json
            │       │   ├── [01;31mdetrend.nii.gz[00m
            │       │   ├── _inputs.pklz
            │       │   ├── _node.pklz
            │       │   ├── [01;34m_report[00m
            │       │   │   └── report.rst
            │       │   └── result_detrend_fmri.pklz
            │       ├── [01;34mmask_EXT[00m
            │       │   ├── _0xcafa73b4154d47952890ec890011278e.json
            │       │   ├── command.txt
            │       │   ├── _inputs.pklz
            │       │   ├── _node.pklz
            │       │   ├── [01;34m_report[00m
            │       │   │   └── report.rst
            │       │   ├── result_mask_EXT.pklz
            │       │   └── sub-01_T1w_brain_pve_1_flirt_thresh.nii
            │       └── [01;34mmask_func[00m
            │           ├── _0xfcc886385f53e0d3ff6922f98569545e.json
            │           ├── _inputs.pklz
            │           ├── [01;34mmapflow[00m
            │           │   └── [01;34m_mask_func0[00m
            │           │       ├── _0x77d6f73e901395033dbf92e719f909dd.json
            │           │       ├── command.txt
            │           │       ├── _inputs.pklz
            │           │       ├── _node.pklz
            │           │       ├── [01;34m_report[00m
            │           │       │   └── report.rst
            │           │       ├── result__mask_func0.pklz
            │           │       └── sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
            │           ├── _node.pklz
            │           ├── [01;34m_report[00m
            │           │   └── report.rst
            │           └── result_mask_func.pklz
            ├── [01;34m_mask_ext_2[00m
            │   ├── [01;34mget_mask_ext[00m
            │   │   ├── _0xe849306eccf5c589c21f2f3f4ef30171.json
            │   │   ├── _inputs.pklz
            │   │   ├── _node.pklz
            │   │   ├── [01;34m_report[00m
            │   │   │   └── report.rst
            │   │   └── result_get_mask_ext.pklz
            │   ├── [01;34mresample[00m
            │   │   ├── _0x625016eb59bbbed7c793c0aef7f183e6.json
            │   │   ├── command.txt
            │   │   ├── _inputs.pklz
            │   │   ├── _node.pklz
            │   │   ├── [01;34m_report[00m
            │   │   │   └── report.rst
            │   │   ├── result_resample.pklz
            │   │   ├── sub-01_T1w_brain_pve_2_flirt.mat
            │   │   └── sub-01_T1w_brain_pve_2_flirt.nii
            │   ├── [01;34m_thresh_0.5[00m
            │   │   ├── [01;34mdatasink[00m
            │   │   │   ├── _0x510239005e2a13c21de27a97882f8fd4.json
            │   │   │   ├── _inputs.pklz
            │   │   │   ├── _node.pklz
            │   │   │   ├── [01;34m_report[00m
            │   │   │   │   └── report.rst
            │   │   │   └── result_datasink.pklz
            │   │   ├── [01;34mdetrend[00m
            │   │   │   ├── _0xaf2b89a6cd7e9ac9e343daa192aa33c2.json
            │   │   │   ├── [01;31mdetrend.nii.gz[00m
            │   │   │   ├── _inputs.pklz
            │   │   │   ├── _node.pklz
            │   │   │   ├── [01;34m_report[00m
            │   │   │   │   └── report.rst
            │   │   │   └── result_detrend.pklz
            │   │   ├── [01;34mdetrend_fmri[00m
            │   │   │   ├── _0x68895a1802c102a6e5b5f8a0b2dcaf9d.json
            │   │   │   ├── [01;31mdetrend.nii.gz[00m
            │   │   │   ├── _inputs.pklz
            │   │   │   ├── _node.pklz
            │   │   │   ├── [01;34m_report[00m
            │   │   │   │   └── report.rst
            │   │   │   └── result_detrend_fmri.pklz
            │   │   ├── [01;34mmask_EXT[00m
            │   │   │   ├── _0x94f5007295ce55462d307a3469381d4b.json
            │   │   │   ├── command.txt
            │   │   │   ├── _inputs.pklz
            │   │   │   ├── _node.pklz
            │   │   │   ├── [01;34m_report[00m
            │   │   │   │   └── report.rst
            │   │   │   ├── result_mask_EXT.pklz
            │   │   │   └── sub-01_T1w_brain_pve_2_flirt_thresh.nii
            │   │   └── [01;34mmask_func[00m
            │   │       ├── _0xb1c922fc06eb07976e81f67eaf73aebc.json
            │   │       ├── _inputs.pklz
            │   │       ├── [01;34mmapflow[00m
            │   │       │   └── [01;34m_mask_func0[00m
            │   │       │       ├── _0xcb4074a36ea5066f21371f6bbd38a527.json
            │   │       │       ├── command.txt
            │   │       │       ├── _inputs.pklz
            │   │       │       ├── _node.pklz
            │   │       │       ├── [01;34m_report[00m
            │   │       │       │   └── report.rst
            │   │       │       ├── result__mask_func0.pklz
            │   │       │       └── sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
            │   │       ├── _node.pklz
            │   │       ├── [01;34m_report[00m
            │   │       │   └── report.rst
            │   │       └── result_mask_func.pklz
            │   ├── [01;34m_thresh_0.95[00m
            │   │   ├── [01;34mdatasink[00m
            │   │   │   ├── _0x1ee99d863cdedf58896e1bb641c058ab.json
            │   │   │   ├── _inputs.pklz
            │   │   │   ├── _node.pklz
            │   │   │   ├── [01;34m_report[00m
            │   │   │   │   └── report.rst
            │   │   │   └── result_datasink.pklz
            │   │   ├── [01;34mdetrend[00m
            │   │   │   ├── _0x10779fa187edeb2b5c9083e18c9e77de.json
            │   │   │   ├── [01;31mdetrend.nii.gz[00m
            │   │   │   ├── _inputs.pklz
            │   │   │   ├── _node.pklz
            │   │   │   ├── [01;34m_report[00m
            │   │   │   │   └── report.rst
            │   │   │   └── result_detrend.pklz
            │   │   ├── [01;34mdetrend_fmri[00m
            │   │   │   ├── _0xec6a0d3a9a84eb117be84910e10876c6.json
            │   │   │   ├── [01;31mdetrend.nii.gz[00m
            │   │   │   ├── _inputs.pklz
            │   │   │   ├── _node.pklz
            │   │   │   ├── [01;34m_report[00m
            │   │   │   │   └── report.rst
            │   │   │   └── result_detrend_fmri.pklz
            │   │   ├── [01;34mmask_EXT[00m
            │   │   │   ├── _0xc8fa36c7f6b5271d7b5296f9795b60ce.json
            │   │   │   ├── command.txt
            │   │   │   ├── _inputs.pklz
            │   │   │   ├── _node.pklz
            │   │   │   ├── [01;34m_report[00m
            │   │   │   │   └── report.rst
            │   │   │   ├── result_mask_EXT.pklz
            │   │   │   └── sub-01_T1w_brain_pve_2_flirt_thresh.nii
            │   │   └── [01;34mmask_func[00m
            │   │       ├── _0x1ee41d75a18efd3f9a5871e8dcea6c52.json
            │   │       ├── _inputs.pklz
            │   │       ├── [01;34mmapflow[00m
            │   │       │   └── [01;34m_mask_func0[00m
            │   │       │       ├── _0x62b1f7f8d99e593d065cf634bbe40509.json
            │   │       │       ├── command.txt
            │   │       │       ├── _inputs.pklz
            │   │       │       ├── _node.pklz
            │   │       │       ├── [01;34m_report[00m
            │   │       │       │   └── report.rst
            │   │       │       ├── result__mask_func0.pklz
            │   │       │       └── sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
            │   │       ├── _node.pklz
            │   │       ├── [01;34m_report[00m
            │   │       │   └── report.rst
            │   │       └── result_mask_func.pklz
            │   ├── [01;34m_thresh_0.97[00m
            │   │   ├── [01;34mdatasink[00m
            │   │   │   ├── _0x3ba90e6105bb456be5ee13deffb17775.json
            │   │   │   ├── _inputs.pklz
            │   │   │   ├── _node.pklz
            │   │   │   ├── [01;34m_report[00m
            │   │   │   │   └── report.rst
            │   │   │   └── result_datasink.pklz
            │   │   ├── [01;34mdetrend[00m
            │   │   │   ├── _0x9a7a6d4de3a72714e198cce3a89d3546.json
            │   │   │   ├── [01;31mdetrend.nii.gz[00m
            │   │   │   ├── _inputs.pklz
            │   │   │   ├── _node.pklz
            │   │   │   ├── [01;34m_report[00m
            │   │   │   │   └── report.rst
            │   │   │   └── result_detrend.pklz
            │   │   ├── [01;34mdetrend_fmri[00m
            │   │   │   ├── _0xf1e41c51147debaec2206d20d0351e6f.json
            │   │   │   ├── [01;31mdetrend.nii.gz[00m
            │   │   │   ├── _inputs.pklz
            │   │   │   ├── _node.pklz
            │   │   │   ├── [01;34m_report[00m
            │   │   │   │   └── report.rst
            │   │   │   └── result_detrend_fmri.pklz
            │   │   ├── [01;34mmask_EXT[00m
            │   │   │   ├── _0x5293cbd3c85699a86a3b106189258445.json
            │   │   │   ├── command.txt
            │   │   │   ├── _inputs.pklz
            │   │   │   ├── _node.pklz
            │   │   │   ├── [01;34m_report[00m
            │   │   │   │   └── report.rst
            │   │   │   ├── result_mask_EXT.pklz
            │   │   │   └── sub-01_T1w_brain_pve_2_flirt_thresh.nii
            │   │   └── [01;34mmask_func[00m
            │   │       ├── _0x94a0047db8ce220af72ce0d14347cb24.json
            │   │       ├── _inputs.pklz
            │   │       ├── [01;34mmapflow[00m
            │   │       │   └── [01;34m_mask_func0[00m
            │   │       │       ├── _0x13ccdd97476a9a59d834e69d055f0590.json
            │   │       │       ├── command.txt
            │   │       │       ├── _inputs.pklz
            │   │       │       ├── _node.pklz
            │   │       │       ├── [01;34m_report[00m
            │   │       │       │   └── report.rst
            │   │       │       ├── result__mask_func0.pklz
            │   │       │       └── sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
            │   │       ├── _node.pklz
            │   │       ├── [01;34m_report[00m
            │   │       │   └── report.rst
            │   │       └── result_mask_func.pklz
            │   └── [01;34m_thresh_0.99[00m
            │       ├── [01;34mdatasink[00m
            │       │   ├── _0x28c1f76d01cf8d6554ff63f247b4da88.json
            │       │   ├── _inputs.pklz
            │       │   ├── _node.pklz
            │       │   ├── [01;34m_report[00m
            │       │   │   └── report.rst
            │       │   └── result_datasink.pklz
            │       ├── [01;34mdetrend[00m
            │       │   ├── _0x158cd835d3116392002ea18b373af3be.json
            │       │   ├── [01;31mdetrend.nii.gz[00m
            │       │   ├── _inputs.pklz
            │       │   ├── _node.pklz
            │       │   ├── [01;34m_report[00m
            │       │   │   └── report.rst
            │       │   └── result_detrend.pklz
            │       ├── [01;34mdetrend_fmri[00m
            │       │   ├── _0xec6a0d3a9a84eb117be84910e10876c6.json
            │       │   ├── [01;31mdetrend.nii.gz[00m
            │       │   ├── _inputs.pklz
            │       │   ├── _node.pklz
            │       │   ├── [01;34m_report[00m
            │       │   │   └── report.rst
            │       │   └── result_detrend_fmri.pklz
            │       ├── [01;34mmask_EXT[00m
            │       │   ├── _0x7512158bac2b357b744cf0b4758cff6a.json
            │       │   ├── command.txt
            │       │   ├── _inputs.pklz
            │       │   ├── _node.pklz
            │       │   ├── [01;34m_report[00m
            │       │   │   └── report.rst
            │       │   ├── result_mask_EXT.pklz
            │       │   └── sub-01_T1w_brain_pve_2_flirt_thresh.nii
            │       └── [01;34mmask_func[00m
            │           ├── _0xc5af521bfd67fa56c63ad8dd899d54e0.json
            │           ├── _inputs.pklz
            │           ├── [01;34mmapflow[00m
            │           │   └── [01;34m_mask_func0[00m
            │           │       ├── _0xc82607d8fbc3e7e055c43dd875ef0d35.json
            │           │       ├── command.txt
            │           │       ├── _inputs.pklz
            │           │       ├── _node.pklz
            │           │       ├── [01;34m_report[00m
            │           │       │   └── report.rst
            │           │       ├── result__mask_func0.pklz
            │           │       └── sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
            │           ├── _node.pklz
            │           ├── [01;34m_report[00m
            │           │   └── report.rst
            │           └── result_mask_func.pklz
            ├── [01;34mselectfiles[00m
            │   ├── _0x2c88730a4521099a0c2234ca5158ecf2.json
            │   ├── _inputs.pklz
            │   ├── _node.pklz
            │   ├── [01;34m_report[00m
            │   │   └── report.rst
            │   └── result_selectfiles.pklz
            ├── [01;34msusan[00m
            │   ├── [01;34m_mask_ext_0[00m
            │   │   ├── [01;34m_thresh_0.5[00m
            │   │   │   ├── [01;34mmask[00m
            │   │   │   │   ├── _0xf3c966b1e6fd17021174bace6bec7bbc.json
            │   │   │   │   ├── _inputs.pklz
            │   │   │   │   ├── [01;34mmapflow[00m
            │   │   │   │   │   └── [01;34m_mask0[00m
            │   │   │   │   │       ├── _0xc29879beadb157411346dfd196bb210b.json
            │   │   │   │   │       ├── command.txt
            │   │   │   │   │       ├── _inputs.pklz
            │   │   │   │   │       ├── _node.pklz
            │   │   │   │   │       ├── [01;34m_report[00m
            │   │   │   │   │       │   └── report.rst
            │   │   │   │   │       ├── result__mask0.pklz
            │   │   │   │   │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_mask.nii.gz[00m
            │   │   │   │   ├── _node.pklz
            │   │   │   │   ├── [01;34m_report[00m
            │   │   │   │   │   └── report.rst
            │   │   │   │   └── result_mask.pklz
            │   │   │   ├── [01;34mmeanfunc2[00m
            │   │   │   │   ├── _0xc61b103aa7e94847a204e6a04cfc7d6c.json
            │   │   │   │   ├── _inputs.pklz
            │   │   │   │   ├── [01;34mmapflow[00m
            │   │   │   │   │   └── [01;34m_meanfunc20[00m
            │   │   │   │   │       ├── _0x95f50890820830ca032a3cc67f09bbc1.json
            │   │   │   │   │       ├── command.txt
            │   │   │   │   │       ├── _inputs.pklz
            │   │   │   │   │       ├── _node.pklz
            │   │   │   │   │       ├── [01;34m_report[00m
            │   │   │   │   │       │   └── report.rst
            │   │   │   │   │       ├── result__meanfunc20.pklz
            │   │   │   │   │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_mask_mean.nii.gz[00m
            │   │   │   │   ├── _node.pklz
            │   │   │   │   ├── [01;34m_report[00m
            │   │   │   │   │   └── report.rst
            │   │   │   │   └── result_meanfunc2.pklz
            │   │   │   ├── [01;34mmedian[00m
            │   │   │   │   ├── _0xd527e6f3e41e11bacc9333e5d5f95edd.json
            │   │   │   │   ├── _inputs.pklz
            │   │   │   │   ├── [01;34mmapflow[00m
            │   │   │   │   │   └── [01;34m_median0[00m
            │   │   │   │   │       ├── _0xd5be0b38d1d0b282427652c8fe63353d.json
            │   │   │   │   │       ├── command.txt
            │   │   │   │   │       ├── _inputs.pklz
            │   │   │   │   │       ├── _node.pklz
            │   │   │   │   │       ├── [01;34m_report[00m
            │   │   │   │   │       │   └── report.rst
            │   │   │   │   │       └── result__median0.pklz
            │   │   │   │   ├── _node.pklz
            │   │   │   │   ├── [01;34m_report[00m
            │   │   │   │   │   └── report.rst
            │   │   │   │   └── result_median.pklz
            │   │   │   ├── [01;34mmerge[00m
            │   │   │   │   ├── _0x6e46ab87d3d624ae064fef4da2b2aa43.json
            │   │   │   │   ├── _inputs.pklz
            │   │   │   │   ├── _node.pklz
            │   │   │   │   ├── [01;34m_report[00m
            │   │   │   │   │   └── report.rst
            │   │   │   │   └── result_merge.pklz
            │   │   │   ├── [01;34mmulti_inputs[00m
            │   │   │   │   ├── _0xc3d9cf6c570f7204de1e297d081914ae.json
            │   │   │   │   ├── _inputs.pklz
            │   │   │   │   ├── _node.pklz
            │   │   │   │   ├── [01;34m_report[00m
            │   │   │   │   │   └── report.rst
            │   │   │   │   └── result_multi_inputs.pklz
            │   │   │   └── [01;34msmooth[00m
            │   │   │       ├── _0x020fc8416fe397c27d9c15cd20208ea3.json
            │   │   │       ├── _inputs.pklz
            │   │   │       ├── [01;34mmapflow[00m
            │   │   │       │   └── [01;34m_smooth0[00m
            │   │   │       │       ├── _0x1f375cd3253c2a51b4cadbfb46597035.json
            │   │   │       │       ├── command.txt
            │   │   │       │       ├── _inputs.pklz
            │   │   │       │       ├── _node.pklz
            │   │   │       │       ├── [01;34m_report[00m
            │   │   │       │       │   └── report.rst
            │   │   │       │       ├── result__smooth0.pklz
            │   │   │       │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz[00m
            │   │   │       ├── _node.pklz
            │   │   │       ├── [01;34m_report[00m
            │   │   │       │   └── report.rst
            │   │   │       └── result_smooth.pklz
            │   │   ├── [01;34m_thresh_0.95[00m
            │   │   │   ├── [01;34mmask[00m
            │   │   │   │   ├── _0xce9c9088c05fd6abbec3bce25b18fad3.json
            │   │   │   │   ├── _inputs.pklz
            │   │   │   │   ├── [01;34mmapflow[00m
            │   │   │   │   │   └── [01;34m_mask0[00m
            │   │   │   │   │       ├── _0xe0ea96988938e32833b7982f28eaf3bd.json
            │   │   │   │   │       ├── command.txt
            │   │   │   │   │       ├── _inputs.pklz
            │   │   │   │   │       ├── _node.pklz
            │   │   │   │   │       ├── [01;34m_report[00m
            │   │   │   │   │       │   └── report.rst
            │   │   │   │   │       ├── result__mask0.pklz
            │   │   │   │   │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_mask.nii.gz[00m
            │   │   │   │   ├── _node.pklz
            │   │   │   │   ├── [01;34m_report[00m
            │   │   │   │   │   └── report.rst
            │   │   │   │   └── result_mask.pklz
            │   │   │   ├── [01;34mmeanfunc2[00m
            │   │   │   │   ├── _0x36583b767ba1866658412d2193e352a7.json
            │   │   │   │   ├── _inputs.pklz
            │   │   │   │   ├── [01;34mmapflow[00m
            │   │   │   │   │   └── [01;34m_meanfunc20[00m
            │   │   │   │   │       ├── _0x9a5a711c0c67ffa0ac2e8828676279a2.json
            │   │   │   │   │       ├── command.txt
            │   │   │   │   │       ├── _inputs.pklz
            │   │   │   │   │       ├── _node.pklz
            │   │   │   │   │       ├── [01;34m_report[00m
            │   │   │   │   │       │   └── report.rst
            │   │   │   │   │       ├── result__meanfunc20.pklz
            │   │   │   │   │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_mask_mean.nii.gz[00m
            │   │   │   │   ├── _node.pklz
            │   │   │   │   ├── [01;34m_report[00m
            │   │   │   │   │   └── report.rst
            │   │   │   │   └── result_meanfunc2.pklz
            │   │   │   ├── [01;34mmedian[00m
            │   │   │   │   ├── _0xbd9d6facda5d8e79a30912f7e1f9275b.json
            │   │   │   │   ├── _inputs.pklz
            │   │   │   │   ├── [01;34mmapflow[00m
            │   │   │   │   │   └── [01;34m_median0[00m
            │   │   │   │   │       ├── _0x2ed04a5f4118538f3d83ab9fa367621d.json
            │   │   │   │   │       ├── command.txt
            │   │   │   │   │       ├── _inputs.pklz
            │   │   │   │   │       ├── _node.pklz
            │   │   │   │   │       ├── [01;34m_report[00m
            │   │   │   │   │       │   └── report.rst
            │   │   │   │   │       └── result__median0.pklz
            │   │   │   │   ├── _node.pklz
            │   │   │   │   ├── [01;34m_report[00m
            │   │   │   │   │   └── report.rst
            │   │   │   │   └── result_median.pklz
            │   │   │   ├── [01;34mmerge[00m
            │   │   │   │   ├── _0x87cd1a8e4e275b6b8e79d698eec38fbc.json
            │   │   │   │   ├── _inputs.pklz
            │   │   │   │   ├── _node.pklz
            │   │   │   │   ├── [01;34m_report[00m
            │   │   │   │   │   └── report.rst
            │   │   │   │   └── result_merge.pklz
            │   │   │   ├── [01;34mmulti_inputs[00m
            │   │   │   │   ├── _0xd2ca387434426da6eab5359199a1b52e.json
            │   │   │   │   ├── _inputs.pklz
            │   │   │   │   ├── _node.pklz
            │   │   │   │   ├── [01;34m_report[00m
            │   │   │   │   │   └── report.rst
            │   │   │   │   └── result_multi_inputs.pklz
            │   │   │   └── [01;34msmooth[00m
            │   │   │       ├── _0x90371d6dcdd944f054780e6cb60c313c.json
            │   │   │       ├── _inputs.pklz
            │   │   │       ├── [01;34mmapflow[00m
            │   │   │       │   └── [01;34m_smooth0[00m
            │   │   │       │       ├── _0x1b1f14666f7c8facfefc9710d47af639.json
            │   │   │       │       ├── command.txt
            │   │   │       │       ├── _inputs.pklz
            │   │   │       │       ├── _node.pklz
            │   │   │       │       ├── [01;34m_report[00m
            │   │   │       │       │   └── report.rst
            │   │   │       │       ├── result__smooth0.pklz
            │   │   │       │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz[00m
            │   │   │       ├── _node.pklz
            │   │   │       ├── [01;34m_report[00m
            │   │   │       │   └── report.rst
            │   │   │       └── result_smooth.pklz
            │   │   ├── [01;34m_thresh_0.97[00m
            │   │   │   ├── [01;34mmask[00m
            │   │   │   │   ├── _0x8301ad2ed388df217c7ba5ef32c71870.json
            │   │   │   │   ├── _inputs.pklz
            │   │   │   │   ├── [01;34mmapflow[00m
            │   │   │   │   │   └── [01;34m_mask0[00m
            │   │   │   │   │       ├── _0xbf43e1601d5dda26a527172578a2ddbb.json
            │   │   │   │   │       ├── command.txt
            │   │   │   │   │       ├── _inputs.pklz
            │   │   │   │   │       ├── _node.pklz
            │   │   │   │   │       ├── [01;34m_report[00m
            │   │   │   │   │       │   └── report.rst
            │   │   │   │   │       ├── result__mask0.pklz
            │   │   │   │   │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_mask.nii.gz[00m
            │   │   │   │   ├── _node.pklz
            │   │   │   │   ├── [01;34m_report[00m
            │   │   │   │   │   └── report.rst
            │   │   │   │   └── result_mask.pklz
            │   │   │   ├── [01;34mmeanfunc2[00m
            │   │   │   │   ├── _0x0de1c38843e6b6fdbf5dc9211ad2ca35.json
            │   │   │   │   ├── _inputs.pklz
            │   │   │   │   ├── [01;34mmapflow[00m
            │   │   │   │   │   └── [01;34m_meanfunc20[00m
            │   │   │   │   │       ├── _0xbfae6de6ec03b70cf38bded42c9a135f.json
            │   │   │   │   │       ├── command.txt
            │   │   │   │   │       ├── _inputs.pklz
            │   │   │   │   │       ├── _node.pklz
            │   │   │   │   │       ├── [01;34m_report[00m
            │   │   │   │   │       │   └── report.rst
            │   │   │   │   │       ├── result__meanfunc20.pklz
            │   │   │   │   │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_mask_mean.nii.gz[00m
            │   │   │   │   ├── _node.pklz
            │   │   │   │   ├── [01;34m_report[00m
            │   │   │   │   │   └── report.rst
            │   │   │   │   └── result_meanfunc2.pklz
            │   │   │   ├── [01;34mmedian[00m
            │   │   │   │   ├── _0xecbb32a5199b32446b0045bdda29fcde.json
            │   │   │   │   ├── _inputs.pklz
            │   │   │   │   ├── [01;34mmapflow[00m
            │   │   │   │   │   └── [01;34m_median0[00m
            │   │   │   │   │       ├── _0x85adfd7550c201375dd8e67273d57c9c.json
            │   │   │   │   │       ├── command.txt
            │   │   │   │   │       ├── _inputs.pklz
            │   │   │   │   │       ├── _node.pklz
            │   │   │   │   │       ├── [01;34m_report[00m
            │   │   │   │   │       │   └── report.rst
            │   │   │   │   │       └── result__median0.pklz
            │   │   │   │   ├── _node.pklz
            │   │   │   │   ├── [01;34m_report[00m
            │   │   │   │   │   └── report.rst
            │   │   │   │   └── result_median.pklz
            │   │   │   ├── [01;34mmerge[00m
            │   │   │   │   ├── _0x412bae23dd6801c85b947133798c6692.json
            │   │   │   │   ├── _inputs.pklz
            │   │   │   │   ├── _node.pklz
            │   │   │   │   ├── [01;34m_report[00m
            │   │   │   │   │   └── report.rst
            │   │   │   │   └── result_merge.pklz
            │   │   │   ├── [01;34mmulti_inputs[00m
            │   │   │   │   ├── _0xadca42a55418035fdf7548d715c17086.json
            │   │   │   │   ├── _inputs.pklz
            │   │   │   │   ├── _node.pklz
            │   │   │   │   ├── [01;34m_report[00m
            │   │   │   │   │   └── report.rst
            │   │   │   │   └── result_multi_inputs.pklz
            │   │   │   └── [01;34msmooth[00m
            │   │   │       ├── _0xf2ae3e1d6034285b220e2f41ceb1123b.json
            │   │   │       ├── _inputs.pklz
            │   │   │       ├── [01;34mmapflow[00m
            │   │   │       │   └── [01;34m_smooth0[00m
            │   │   │       │       ├── _0x269424a0c8650f2fdad06e9d62dbec9e.json
            │   │   │       │       ├── command.txt
            │   │   │       │       ├── _inputs.pklz
            │   │   │       │       ├── _node.pklz
            │   │   │       │       ├── [01;34m_report[00m
            │   │   │       │       │   └── report.rst
            │   │   │       │       ├── result__smooth0.pklz
            │   │   │       │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz[00m
            │   │   │       ├── _node.pklz
            │   │   │       ├── [01;34m_report[00m
            │   │   │       │   └── report.rst
            │   │   │       └── result_smooth.pklz
            │   │   └── [01;34m_thresh_0.99[00m
            │   │       ├── [01;34mmask[00m
            │   │       │   ├── _0x3729f41ebf52c95d96be46497ae99a76.json
            │   │       │   ├── _inputs.pklz
            │   │       │   ├── [01;34mmapflow[00m
            │   │       │   │   └── [01;34m_mask0[00m
            │   │       │   │       ├── _0x620db48bdb888e80a99f1b0569beefb6.json
            │   │       │   │       ├── command.txt
            │   │       │   │       ├── _inputs.pklz
            │   │       │   │       ├── _node.pklz
            │   │       │   │       ├── [01;34m_report[00m
            │   │       │   │       │   └── report.rst
            │   │       │   │       ├── result__mask0.pklz
            │   │       │   │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_mask.nii.gz[00m
            │   │       │   ├── _node.pklz
            │   │       │   ├── [01;34m_report[00m
            │   │       │   │   └── report.rst
            │   │       │   └── result_mask.pklz
            │   │       ├── [01;34mmeanfunc2[00m
            │   │       │   ├── _0xa809ed4910505a65c3f3d737a7369b5f.json
            │   │       │   ├── _inputs.pklz
            │   │       │   ├── [01;34mmapflow[00m
            │   │       │   │   └── [01;34m_meanfunc20[00m
            │   │       │   │       ├── _0x97ca59b6fd6a1bf5717279ddb4ec16a2.json
            │   │       │   │       ├── command.txt
            │   │       │   │       ├── _inputs.pklz
            │   │       │   │       ├── _node.pklz
            │   │       │   │       ├── [01;34m_report[00m
            │   │       │   │       │   └── report.rst
            │   │       │   │       ├── result__meanfunc20.pklz
            │   │       │   │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_mask_mean.nii.gz[00m
            │   │       │   ├── _node.pklz
            │   │       │   ├── [01;34m_report[00m
            │   │       │   │   └── report.rst
            │   │       │   └── result_meanfunc2.pklz
            │   │       ├── [01;34mmedian[00m
            │   │       │   ├── _0xae0c9a18d5a41ceb0e05ccd5daa3013e.json
            │   │       │   ├── _inputs.pklz
            │   │       │   ├── [01;34mmapflow[00m
            │   │       │   │   └── [01;34m_median0[00m
            │   │       │   │       ├── _0x283afe068ef6ba255e640477f80a028d.json
            │   │       │   │       ├── command.txt
            │   │       │   │       ├── _inputs.pklz
            │   │       │   │       ├── _node.pklz
            │   │       │   │       ├── [01;34m_report[00m
            │   │       │   │       │   └── report.rst
            │   │       │   │       └── result__median0.pklz
            │   │       │   ├── _node.pklz
            │   │       │   ├── [01;34m_report[00m
            │   │       │   │   └── report.rst
            │   │       │   └── result_median.pklz
            │   │       ├── [01;34mmerge[00m
            │   │       │   ├── _0xbfe989871b57e653456a9c5b009fcb42.json
            │   │       │   ├── _inputs.pklz
            │   │       │   ├── _node.pklz
            │   │       │   ├── [01;34m_report[00m
            │   │       │   │   └── report.rst
            │   │       │   └── result_merge.pklz
            │   │       ├── [01;34mmulti_inputs[00m
            │   │       │   ├── _0x8e52b73a5815f66c3979ebdc3c47e56a.json
            │   │       │   ├── _inputs.pklz
            │   │       │   ├── _node.pklz
            │   │       │   ├── [01;34m_report[00m
            │   │       │   │   └── report.rst
            │   │       │   └── result_multi_inputs.pklz
            │   │       └── [01;34msmooth[00m
            │   │           ├── _0xce59d4d3579705600488fa1199c60263.json
            │   │           ├── _inputs.pklz
            │   │           ├── [01;34mmapflow[00m
            │   │           │   └── [01;34m_smooth0[00m
            │   │           │       ├── _0x139fb859777ce05d84f6e1f6b34a68cb.json
            │   │           │       ├── command.txt
            │   │           │       ├── _inputs.pklz
            │   │           │       ├── _node.pklz
            │   │           │       ├── [01;34m_report[00m
            │   │           │       │   └── report.rst
            │   │           │       ├── result__smooth0.pklz
            │   │           │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz[00m
            │   │           ├── _node.pklz
            │   │           ├── [01;34m_report[00m
            │   │           │   └── report.rst
            │   │           └── result_smooth.pklz
            │   ├── [01;34m_mask_ext_1[00m
            │   │   ├── [01;34m_thresh_0.5[00m
            │   │   │   ├── [01;34mmask[00m
            │   │   │   │   ├── _0xa349cef355cd3151efa32af5f411f195.json
            │   │   │   │   ├── _inputs.pklz
            │   │   │   │   ├── [01;34mmapflow[00m
            │   │   │   │   │   └── [01;34m_mask0[00m
            │   │   │   │   │       ├── _0xcbe8db883db6b9f036c33a0b7e94e846.json
            │   │   │   │   │       ├── command.txt
            │   │   │   │   │       ├── _inputs.pklz
            │   │   │   │   │       ├── _node.pklz
            │   │   │   │   │       ├── [01;34m_report[00m
            │   │   │   │   │       │   └── report.rst
            │   │   │   │   │       ├── result__mask0.pklz
            │   │   │   │   │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_mask.nii.gz[00m
            │   │   │   │   ├── _node.pklz
            │   │   │   │   ├── [01;34m_report[00m
            │   │   │   │   │   └── report.rst
            │   │   │   │   └── result_mask.pklz
            │   │   │   ├── [01;34mmeanfunc2[00m
            │   │   │   │   ├── _0x1526f28c18bd5de895e8617d4478e4f0.json
            │   │   │   │   ├── _inputs.pklz
            │   │   │   │   ├── [01;34mmapflow[00m
            │   │   │   │   │   └── [01;34m_meanfunc20[00m
            │   │   │   │   │       ├── _0xf9ba43b0ee8ff785b9a292625a0b0352.json
            │   │   │   │   │       ├── command.txt
            │   │   │   │   │       ├── _inputs.pklz
            │   │   │   │   │       ├── _node.pklz
            │   │   │   │   │       ├── [01;34m_report[00m
            │   │   │   │   │       │   └── report.rst
            │   │   │   │   │       ├── result__meanfunc20.pklz
            │   │   │   │   │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_mask_mean.nii.gz[00m
            │   │   │   │   ├── _node.pklz
            │   │   │   │   ├── [01;34m_report[00m
            │   │   │   │   │   └── report.rst
            │   │   │   │   └── result_meanfunc2.pklz
            │   │   │   ├── [01;34mmedian[00m
            │   │   │   │   ├── _0x08608c77fae583f2d625a7dfb79456d3.json
            │   │   │   │   ├── _inputs.pklz
            │   │   │   │   ├── [01;34mmapflow[00m
            │   │   │   │   │   └── [01;34m_median0[00m
            │   │   │   │   │       ├── _0x15a1a0485ad43e487c28fc8c4cd05a22.json
            │   │   │   │   │       ├── command.txt
            │   │   │   │   │       ├── _inputs.pklz
            │   │   │   │   │       ├── _node.pklz
            │   │   │   │   │       ├── [01;34m_report[00m
            │   │   │   │   │       │   └── report.rst
            │   │   │   │   │       └── result__median0.pklz
            │   │   │   │   ├── _node.pklz
            │   │   │   │   ├── [01;34m_report[00m
            │   │   │   │   │   └── report.rst
            │   │   │   │   └── result_median.pklz
            │   │   │   ├── [01;34mmerge[00m
            │   │   │   │   ├── _0x2f3a9632ee58c0d20123d86d6acc2826.json
            │   │   │   │   ├── _inputs.pklz
            │   │   │   │   ├── _node.pklz
            │   │   │   │   ├── [01;34m_report[00m
            │   │   │   │   │   └── report.rst
            │   │   │   │   └── result_merge.pklz
            │   │   │   ├── [01;34mmulti_inputs[00m
            │   │   │   │   ├── _0x6f6a765360437bc0bf60e44f12904521.json
            │   │   │   │   ├── _inputs.pklz
            │   │   │   │   ├── _node.pklz
            │   │   │   │   ├── [01;34m_report[00m
            │   │   │   │   │   └── report.rst
            │   │   │   │   └── result_multi_inputs.pklz
            │   │   │   └── [01;34msmooth[00m
            │   │   │       ├── _0x4cd3d33eeee3c479579d9a08b9dd6fbf.json
            │   │   │       ├── _inputs.pklz
            │   │   │       ├── [01;34mmapflow[00m
            │   │   │       │   └── [01;34m_smooth0[00m
            │   │   │       │       ├── _0xa82a47b9e19d103e575c78e7a0a369a1.json
            │   │   │       │       ├── command.txt
            │   │   │       │       ├── _inputs.pklz
            │   │   │       │       ├── _node.pklz
            │   │   │       │       ├── [01;34m_report[00m
            │   │   │       │       │   └── report.rst
            │   │   │       │       ├── result__smooth0.pklz
            │   │   │       │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz[00m
            │   │   │       ├── _node.pklz
            │   │   │       ├── [01;34m_report[00m
            │   │   │       │   └── report.rst
            │   │   │       └── result_smooth.pklz
            │   │   ├── [01;34m_thresh_0.95[00m
            │   │   │   ├── [01;34mmask[00m
            │   │   │   │   ├── _0x9a27334e2a32750e7456b603c6b2e71b.json
            │   │   │   │   ├── _inputs.pklz
            │   │   │   │   ├── [01;34mmapflow[00m
            │   │   │   │   │   └── [01;34m_mask0[00m
            │   │   │   │   │       ├── _0x9359fb363cb69973f81e1e16495a9167.json
            │   │   │   │   │       ├── command.txt
            │   │   │   │   │       ├── _inputs.pklz
            │   │   │   │   │       ├── _node.pklz
            │   │   │   │   │       ├── [01;34m_report[00m
            │   │   │   │   │       │   └── report.rst
            │   │   │   │   │       ├── result__mask0.pklz
            │   │   │   │   │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_mask.nii.gz[00m
            │   │   │   │   ├── _node.pklz
            │   │   │   │   ├── [01;34m_report[00m
            │   │   │   │   │   └── report.rst
            │   │   │   │   └── result_mask.pklz
            │   │   │   ├── [01;34mmeanfunc2[00m
            │   │   │   │   ├── _0xba2e687cd9ecfcf40ea5cbbc0bd8b96e.json
            │   │   │   │   ├── _inputs.pklz
            │   │   │   │   ├── [01;34mmapflow[00m
            │   │   │   │   │   └── [01;34m_meanfunc20[00m
            │   │   │   │   │       ├── _0xc7ae6d791eff8a3840ba99cf3e0e2558.json
            │   │   │   │   │       ├── command.txt
            │   │   │   │   │       ├── _inputs.pklz
            │   │   │   │   │       ├── _node.pklz
            │   │   │   │   │       ├── [01;34m_report[00m
            │   │   │   │   │       │   └── report.rst
            │   │   │   │   │       ├── result__meanfunc20.pklz
            │   │   │   │   │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_mask_mean.nii.gz[00m
            │   │   │   │   ├── _node.pklz
            │   │   │   │   ├── [01;34m_report[00m
            │   │   │   │   │   └── report.rst
            │   │   │   │   └── result_meanfunc2.pklz
            │   │   │   ├── [01;34mmedian[00m
            │   │   │   │   ├── _0x01683ca38ef2ad7c3f61b01a99ff8c9e.json
            │   │   │   │   ├── _inputs.pklz
            │   │   │   │   ├── [01;34mmapflow[00m
            │   │   │   │   │   └── [01;34m_median0[00m
            │   │   │   │   │       ├── _0xe6bd9c5df2f55602d32831e8ce23317a.json
            │   │   │   │   │       ├── command.txt
            │   │   │   │   │       ├── _inputs.pklz
            │   │   │   │   │       ├── _node.pklz
            │   │   │   │   │       ├── [01;34m_report[00m
            │   │   │   │   │       │   └── report.rst
            │   │   │   │   │       └── result__median0.pklz
            │   │   │   │   ├── _node.pklz
            │   │   │   │   ├── [01;34m_report[00m
            │   │   │   │   │   └── report.rst
            │   │   │   │   └── result_median.pklz
            │   │   │   ├── [01;34mmerge[00m
            │   │   │   │   ├── _0xddbb68cb8fd3f4a8c9bff42fdc22087e.json
            │   │   │   │   ├── _inputs.pklz
            │   │   │   │   ├── _node.pklz
            │   │   │   │   ├── [01;34m_report[00m
            │   │   │   │   │   └── report.rst
            │   │   │   │   └── result_merge.pklz
            │   │   │   ├── [01;34mmulti_inputs[00m
            │   │   │   │   ├── _0x8cfe3c894d6252522feb119375bb9cb0.json
            │   │   │   │   ├── _inputs.pklz
            │   │   │   │   ├── _node.pklz
            │   │   │   │   ├── [01;34m_report[00m
            │   │   │   │   │   └── report.rst
            │   │   │   │   └── result_multi_inputs.pklz
            │   │   │   └── [01;34msmooth[00m
            │   │   │       ├── _0xeb81aa7b6fb60e1c37f6092997a474c9.json
            │   │   │       ├── _inputs.pklz
            │   │   │       ├── [01;34mmapflow[00m
            │   │   │       │   └── [01;34m_smooth0[00m
            │   │   │       │       ├── _0x1620e96961446ee3a6020a26074c7830.json
            │   │   │       │       ├── command.txt
            │   │   │       │       ├── _inputs.pklz
            │   │   │       │       ├── _node.pklz
            │   │   │       │       ├── [01;34m_report[00m
            │   │   │       │       │   └── report.rst
            │   │   │       │       ├── result__smooth0.pklz
            │   │   │       │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz[00m
            │   │   │       ├── _node.pklz
            │   │   │       ├── [01;34m_report[00m
            │   │   │       │   └── report.rst
            │   │   │       └── result_smooth.pklz
            │   │   ├── [01;34m_thresh_0.97[00m
            │   │   │   ├── [01;34mmask[00m
            │   │   │   │   ├── _0xe5a5c37fe6c0ebad1e42f8d203403d80.json
            │   │   │   │   ├── _inputs.pklz
            │   │   │   │   ├── [01;34mmapflow[00m
            │   │   │   │   │   └── [01;34m_mask0[00m
            │   │   │   │   │       ├── _0xeb301121c472bdb5e1d6d61996e46005.json
            │   │   │   │   │       ├── command.txt
            │   │   │   │   │       ├── _inputs.pklz
            │   │   │   │   │       ├── _node.pklz
            │   │   │   │   │       ├── [01;34m_report[00m
            │   │   │   │   │       │   └── report.rst
            │   │   │   │   │       ├── result__mask0.pklz
            │   │   │   │   │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_mask.nii.gz[00m
            │   │   │   │   ├── _node.pklz
            │   │   │   │   ├── [01;34m_report[00m
            │   │   │   │   │   └── report.rst
            │   │   │   │   └── result_mask.pklz
            │   │   │   ├── [01;34mmeanfunc2[00m
            │   │   │   │   ├── _0xcfdeccb4755c1752f1f4a82980866515.json
            │   │   │   │   ├── _inputs.pklz
            │   │   │   │   ├── [01;34mmapflow[00m
            │   │   │   │   │   └── [01;34m_meanfunc20[00m
            │   │   │   │   │       ├── _0xe4abfbe0c5b0c58c7cfa360eb3adcf5b.json
            │   │   │   │   │       ├── command.txt
            │   │   │   │   │       ├── _inputs.pklz
            │   │   │   │   │       ├── _node.pklz
            │   │   │   │   │       ├── [01;34m_report[00m
            │   │   │   │   │       │   └── report.rst
            │   │   │   │   │       ├── result__meanfunc20.pklz
            │   │   │   │   │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_mask_mean.nii.gz[00m
            │   │   │   │   ├── _node.pklz
            │   │   │   │   ├── [01;34m_report[00m
            │   │   │   │   │   └── report.rst
            │   │   │   │   └── result_meanfunc2.pklz
            │   │   │   ├── [01;34mmedian[00m
            │   │   │   │   ├── _0x6317f5a1e11d4cc230482f0f8e4e1e29.json
            │   │   │   │   ├── _inputs.pklz
            │   │   │   │   ├── [01;34mmapflow[00m
            │   │   │   │   │   └── [01;34m_median0[00m
            │   │   │   │   │       ├── _0xf7a0abbc186892543fa2d613f81b94a6.json
            │   │   │   │   │       ├── command.txt
            │   │   │   │   │       ├── _inputs.pklz
            │   │   │   │   │       ├── _node.pklz
            │   │   │   │   │       ├── [01;34m_report[00m
            │   │   │   │   │       │   └── report.rst
            │   │   │   │   │       └── result__median0.pklz
            │   │   │   │   ├── _node.pklz
            │   │   │   │   ├── [01;34m_report[00m
            │   │   │   │   │   └── report.rst
            │   │   │   │   └── result_median.pklz
            │   │   │   ├── [01;34mmerge[00m
            │   │   │   │   ├── _0xc952b1066c1485714a697993de514db7.json
            │   │   │   │   ├── _inputs.pklz
            │   │   │   │   ├── _node.pklz
            │   │   │   │   ├── [01;34m_report[00m
            │   │   │   │   │   └── report.rst
            │   │   │   │   └── result_merge.pklz
            │   │   │   ├── [01;34mmulti_inputs[00m
            │   │   │   │   ├── _0xf5a16f88ba1415d1ee359d9f42925acf.json
            │   │   │   │   ├── _inputs.pklz
            │   │   │   │   ├── _node.pklz
            │   │   │   │   ├── [01;34m_report[00m
            │   │   │   │   │   └── report.rst
            │   │   │   │   └── result_multi_inputs.pklz
            │   │   │   └── [01;34msmooth[00m
            │   │   │       ├── _0x8ea45c7abe599a4ae813378c82f79bcf.json
            │   │   │       ├── _inputs.pklz
            │   │   │       ├── [01;34mmapflow[00m
            │   │   │       │   └── [01;34m_smooth0[00m
            │   │   │       │       ├── _0x9936480884ecef764f72e426885a5243.json
            │   │   │       │       ├── command.txt
            │   │   │       │       ├── _inputs.pklz
            │   │   │       │       ├── _node.pklz
            │   │   │       │       ├── [01;34m_report[00m
            │   │   │       │       │   └── report.rst
            │   │   │       │       ├── result__smooth0.pklz
            │   │   │       │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz[00m
            │   │   │       ├── _node.pklz
            │   │   │       ├── [01;34m_report[00m
            │   │   │       │   └── report.rst
            │   │   │       └── result_smooth.pklz
            │   │   └── [01;34m_thresh_0.99[00m
            │   │       ├── [01;34mmask[00m
            │   │       │   ├── _0xe8a42cbf260e2d317b298030abb34aba.json
            │   │       │   ├── _inputs.pklz
            │   │       │   ├── [01;34mmapflow[00m
            │   │       │   │   └── [01;34m_mask0[00m
            │   │       │   │       ├── _0x7f3c7b5af054c918d0e1f46da7bfc7b9.json
            │   │       │   │       ├── command.txt
            │   │       │   │       ├── _inputs.pklz
            │   │       │   │       ├── _node.pklz
            │   │       │   │       ├── [01;34m_report[00m
            │   │       │   │       │   └── report.rst
            │   │       │   │       ├── result__mask0.pklz
            │   │       │   │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_mask.nii.gz[00m
            │   │       │   ├── _node.pklz
            │   │       │   ├── [01;34m_report[00m
            │   │       │   │   └── report.rst
            │   │       │   └── result_mask.pklz
            │   │       ├── [01;34mmeanfunc2[00m
            │   │       │   ├── _0x540f86b998b3121eee9f8800e08ac45e.json
            │   │       │   ├── _inputs.pklz
            │   │       │   ├── [01;34mmapflow[00m
            │   │       │   │   └── [01;34m_meanfunc20[00m
            │   │       │   │       ├── _0x43172a96369cac555bc25ac888830bf7.json
            │   │       │   │       ├── command.txt
            │   │       │   │       ├── _inputs.pklz
            │   │       │   │       ├── _node.pklz
            │   │       │   │       ├── [01;34m_report[00m
            │   │       │   │       │   └── report.rst
            │   │       │   │       ├── result__meanfunc20.pklz
            │   │       │   │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_mask_mean.nii.gz[00m
            │   │       │   ├── _node.pklz
            │   │       │   ├── [01;34m_report[00m
            │   │       │   │   └── report.rst
            │   │       │   └── result_meanfunc2.pklz
            │   │       ├── [01;34mmedian[00m
            │   │       │   ├── _0x24d5ea4c92fc3c914afe9f075f5656f5.json
            │   │       │   ├── _inputs.pklz
            │   │       │   ├── [01;34mmapflow[00m
            │   │       │   │   └── [01;34m_median0[00m
            │   │       │   │       ├── _0x436e75c44e426589d13fbbc012ffcc9a.json
            │   │       │   │       ├── command.txt
            │   │       │   │       ├── _inputs.pklz
            │   │       │   │       ├── _node.pklz
            │   │       │   │       ├── [01;34m_report[00m
            │   │       │   │       │   └── report.rst
            │   │       │   │       └── result__median0.pklz
            │   │       │   ├── _node.pklz
            │   │       │   ├── [01;34m_report[00m
            │   │       │   │   └── report.rst
            │   │       │   └── result_median.pklz
            │   │       ├── [01;34mmerge[00m
            │   │       │   ├── _0x2543190c31565c50b573e4aee7a6a24b.json
            │   │       │   ├── _inputs.pklz
            │   │       │   ├── _node.pklz
            │   │       │   ├── [01;34m_report[00m
            │   │       │   │   └── report.rst
            │   │       │   └── result_merge.pklz
            │   │       ├── [01;34mmulti_inputs[00m
            │   │       │   ├── _0xd698fb38aae2cf83dd8ea8b33b2b6342.json
            │   │       │   ├── _inputs.pklz
            │   │       │   ├── _node.pklz
            │   │       │   ├── [01;34m_report[00m
            │   │       │   │   └── report.rst
            │   │       │   └── result_multi_inputs.pklz
            │   │       └── [01;34msmooth[00m
            │   │           ├── _0xdd5d3264f179d9b0332c4fad11680ee3.json
            │   │           ├── _inputs.pklz
            │   │           ├── [01;34mmapflow[00m
            │   │           │   └── [01;34m_smooth0[00m
            │   │           │       ├── _0xf19bc2fdc60ba8655ff8a913e41c1f9f.json
            │   │           │       ├── command.txt
            │   │           │       ├── _inputs.pklz
            │   │           │       ├── _node.pklz
            │   │           │       ├── [01;34m_report[00m
            │   │           │       │   └── report.rst
            │   │           │       ├── result__smooth0.pklz
            │   │           │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz[00m
            │   │           ├── _node.pklz
            │   │           ├── [01;34m_report[00m
            │   │           │   └── report.rst
            │   │           └── result_smooth.pklz
            │   └── [01;34m_mask_ext_2[00m
            │       ├── [01;34m_thresh_0.5[00m
            │       │   ├── [01;34mmask[00m
            │       │   │   ├── _0x5195448c442f9148217d3d82ee5bebe2.json
            │       │   │   ├── _inputs.pklz
            │       │   │   ├── [01;34mmapflow[00m
            │       │   │   │   └── [01;34m_mask0[00m
            │       │   │   │       ├── _0x3ccd7dc4d5186f15f0ce1796c35fa78d.json
            │       │   │   │       ├── command.txt
            │       │   │   │       ├── _inputs.pklz
            │       │   │   │       ├── _node.pklz
            │       │   │   │       ├── [01;34m_report[00m
            │       │   │   │       │   └── report.rst
            │       │   │   │       ├── result__mask0.pklz
            │       │   │   │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_mask.nii.gz[00m
            │       │   │   ├── _node.pklz
            │       │   │   ├── [01;34m_report[00m
            │       │   │   │   └── report.rst
            │       │   │   └── result_mask.pklz
            │       │   ├── [01;34mmeanfunc2[00m
            │       │   │   ├── _0xfe37af05bb6361f13921923332a79426.json
            │       │   │   ├── _inputs.pklz
            │       │   │   ├── [01;34mmapflow[00m
            │       │   │   │   └── [01;34m_meanfunc20[00m
            │       │   │   │       ├── _0x5f29975bd73378e568f9068a792c6425.json
            │       │   │   │       ├── command.txt
            │       │   │   │       ├── _inputs.pklz
            │       │   │   │       ├── _node.pklz
            │       │   │   │       ├── [01;34m_report[00m
            │       │   │   │       │   └── report.rst
            │       │   │   │       ├── result__meanfunc20.pklz
            │       │   │   │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_mask_mean.nii.gz[00m
            │       │   │   ├── _node.pklz
            │       │   │   ├── [01;34m_report[00m
            │       │   │   │   └── report.rst
            │       │   │   └── result_meanfunc2.pklz
            │       │   ├── [01;34mmedian[00m
            │       │   │   ├── _0xf53fcf56a1b7d237eba1dd51dc632a94.json
            │       │   │   ├── _inputs.pklz
            │       │   │   ├── [01;34mmapflow[00m
            │       │   │   │   └── [01;34m_median0[00m
            │       │   │   │       ├── _0x383e28ec2078d2cba4bb6668a5d318d2.json
            │       │   │   │       ├── command.txt
            │       │   │   │       ├── _inputs.pklz
            │       │   │   │       ├── _node.pklz
            │       │   │   │       ├── [01;34m_report[00m
            │       │   │   │       │   └── report.rst
            │       │   │   │       └── result__median0.pklz
            │       │   │   ├── _node.pklz
            │       │   │   ├── [01;34m_report[00m
            │       │   │   │   └── report.rst
            │       │   │   └── result_median.pklz
            │       │   ├── [01;34mmerge[00m
            │       │   │   ├── _0xd64a8d03238a0066d41004f2a2f706a3.json
            │       │   │   ├── _inputs.pklz
            │       │   │   ├── _node.pklz
            │       │   │   ├── [01;34m_report[00m
            │       │   │   │   └── report.rst
            │       │   │   └── result_merge.pklz
            │       │   ├── [01;34mmulti_inputs[00m
            │       │   │   ├── _0x9ca96b96ab38f4a95bb0c1c39de3ac10.json
            │       │   │   ├── _inputs.pklz
            │       │   │   ├── _node.pklz
            │       │   │   ├── [01;34m_report[00m
            │       │   │   │   └── report.rst
            │       │   │   └── result_multi_inputs.pklz
            │       │   └── [01;34msmooth[00m
            │       │       ├── _0xd55655da741622e1549e2de7cae4c2cf.json
            │       │       ├── _inputs.pklz
            │       │       ├── [01;34mmapflow[00m
            │       │       │   └── [01;34m_smooth0[00m
            │       │       │       ├── _0xc590caadeb4851b6d857722afa8d665a.json
            │       │       │       ├── command.txt
            │       │       │       ├── _inputs.pklz
            │       │       │       ├── _node.pklz
            │       │       │       ├── [01;34m_report[00m
            │       │       │       │   └── report.rst
            │       │       │       ├── result__smooth0.pklz
            │       │       │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz[00m
            │       │       ├── _node.pklz
            │       │       ├── [01;34m_report[00m
            │       │       │   └── report.rst
            │       │       └── result_smooth.pklz
            │       ├── [01;34m_thresh_0.95[00m
            │       │   ├── [01;34mmask[00m
            │       │   │   ├── _0x4b8ac69ddc3dccbe6b028804c441a90b.json
            │       │   │   ├── _inputs.pklz
            │       │   │   ├── [01;34mmapflow[00m
            │       │   │   │   └── [01;34m_mask0[00m
            │       │   │   │       ├── _0xcc0a521ac8ad82d7132cc5746e75c351.json
            │       │   │   │       ├── command.txt
            │       │   │   │       ├── _inputs.pklz
            │       │   │   │       ├── _node.pklz
            │       │   │   │       ├── [01;34m_report[00m
            │       │   │   │       │   └── report.rst
            │       │   │   │       ├── result__mask0.pklz
            │       │   │   │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_mask.nii.gz[00m
            │       │   │   ├── _node.pklz
            │       │   │   ├── [01;34m_report[00m
            │       │   │   │   └── report.rst
            │       │   │   └── result_mask.pklz
            │       │   ├── [01;34mmeanfunc2[00m
            │       │   │   ├── _0x393bacb6e3611a80f345e651dd9972c1.json
            │       │   │   ├── _inputs.pklz
            │       │   │   ├── [01;34mmapflow[00m
            │       │   │   │   └── [01;34m_meanfunc20[00m
            │       │   │   │       ├── _0xeed00c2b5cc34e21f6bfa237a3b9d0c0.json
            │       │   │   │       ├── command.txt
            │       │   │   │       ├── _inputs.pklz
            │       │   │   │       ├── _node.pklz
            │       │   │   │       ├── [01;34m_report[00m
            │       │   │   │       │   └── report.rst
            │       │   │   │       ├── result__meanfunc20.pklz
            │       │   │   │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_mask_mean.nii.gz[00m
            │       │   │   ├── _node.pklz
            │       │   │   ├── [01;34m_report[00m
            │       │   │   │   └── report.rst
            │       │   │   └── result_meanfunc2.pklz
            │       │   ├── [01;34mmedian[00m
            │       │   │   ├── _0xeb425f08fb8d15db218b95e9f9ebe6ac.json
            │       │   │   ├── _inputs.pklz
            │       │   │   ├── [01;34mmapflow[00m
            │       │   │   │   └── [01;34m_median0[00m
            │       │   │   │       ├── _0x9910d0e9c0e2ea891a1b898ed8c99c0e.json
            │       │   │   │       ├── command.txt
            │       │   │   │       ├── _inputs.pklz
            │       │   │   │       ├── _node.pklz
            │       │   │   │       ├── [01;34m_report[00m
            │       │   │   │       │   └── report.rst
            │       │   │   │       └── result__median0.pklz
            │       │   │   ├── _node.pklz
            │       │   │   ├── [01;34m_report[00m
            │       │   │   │   └── report.rst
            │       │   │   └── result_median.pklz
            │       │   ├── [01;34mmerge[00m
            │       │   │   ├── _0x48450d98a8ecd9d98dca902ac3390e6e.json
            │       │   │   ├── _inputs.pklz
            │       │   │   ├── _node.pklz
            │       │   │   ├── [01;34m_report[00m
            │       │   │   │   └── report.rst
            │       │   │   └── result_merge.pklz
            │       │   ├── [01;34mmulti_inputs[00m
            │       │   │   ├── _0xc8f243bceae759c23160e277c72b2abe.json
            │       │   │   ├── _inputs.pklz
            │       │   │   ├── _node.pklz
            │       │   │   ├── [01;34m_report[00m
            │       │   │   │   └── report.rst
            │       │   │   └── result_multi_inputs.pklz
            │       │   └── [01;34msmooth[00m
            │       │       ├── _0x6493cfe96354cf40a11c1d6daf42f223.json
            │       │       ├── _inputs.pklz
            │       │       ├── [01;34mmapflow[00m
            │       │       │   └── [01;34m_smooth0[00m
            │       │       │       ├── _0x221cf7d14630ab2a801a513aeab7cedd.json
            │       │       │       ├── command.txt
            │       │       │       ├── _inputs.pklz
            │       │       │       ├── _node.pklz
            │       │       │       ├── [01;34m_report[00m
            │       │       │       │   └── report.rst
            │       │       │       ├── result__smooth0.pklz
            │       │       │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz[00m
            │       │       ├── _node.pklz
            │       │       ├── [01;34m_report[00m
            │       │       │   └── report.rst
            │       │       └── result_smooth.pklz
            │       ├── [01;34m_thresh_0.97[00m
            │       │   ├── [01;34mmask[00m
            │       │   │   ├── _0x38df8ab4e3d93b2bcaba3995301c1623.json
            │       │   │   ├── _inputs.pklz
            │       │   │   ├── [01;34mmapflow[00m
            │       │   │   │   └── [01;34m_mask0[00m
            │       │   │   │       ├── _0xb09c4dd6e75def3a5786ff2d468e5d21.json
            │       │   │   │       ├── command.txt
            │       │   │   │       ├── _inputs.pklz
            │       │   │   │       ├── _node.pklz
            │       │   │   │       ├── [01;34m_report[00m
            │       │   │   │       │   └── report.rst
            │       │   │   │       ├── result__mask0.pklz
            │       │   │   │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_mask.nii.gz[00m
            │       │   │   ├── _node.pklz
            │       │   │   ├── [01;34m_report[00m
            │       │   │   │   └── report.rst
            │       │   │   └── result_mask.pklz
            │       │   ├── [01;34mmeanfunc2[00m
            │       │   │   ├── _0xc536f67fdfbdb2ca1246e350b8361ce9.json
            │       │   │   ├── _inputs.pklz
            │       │   │   ├── [01;34mmapflow[00m
            │       │   │   │   └── [01;34m_meanfunc20[00m
            │       │   │   │       ├── _0xccb7d0042b30743260c79ee277a71ce1.json
            │       │   │   │       ├── command.txt
            │       │   │   │       ├── _inputs.pklz
            │       │   │   │       ├── _node.pklz
            │       │   │   │       ├── [01;34m_report[00m
            │       │   │   │       │   └── report.rst
            │       │   │   │       ├── result__meanfunc20.pklz
            │       │   │   │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_mask_mean.nii.gz[00m
            │       │   │   ├── _node.pklz
            │       │   │   ├── [01;34m_report[00m
            │       │   │   │   └── report.rst
            │       │   │   └── result_meanfunc2.pklz
            │       │   ├── [01;34mmedian[00m
            │       │   │   ├── _0xdaa7da1e5f60a369daabff9a168e69ed.json
            │       │   │   ├── _inputs.pklz
            │       │   │   ├── [01;34mmapflow[00m
            │       │   │   │   └── [01;34m_median0[00m
            │       │   │   │       ├── _0x150f9ffb3a6b10d74d3f860c73bc8764.json
            │       │   │   │       ├── command.txt
            │       │   │   │       ├── _inputs.pklz
            │       │   │   │       ├── _node.pklz
            │       │   │   │       ├── [01;34m_report[00m
            │       │   │   │       │   └── report.rst
            │       │   │   │       └── result__median0.pklz
            │       │   │   ├── _node.pklz
            │       │   │   ├── [01;34m_report[00m
            │       │   │   │   └── report.rst
            │       │   │   └── result_median.pklz
            │       │   ├── [01;34mmerge[00m
            │       │   │   ├── _0xb38707c8d5e327bbba41da837ce063da.json
            │       │   │   ├── _inputs.pklz
            │       │   │   ├── _node.pklz
            │       │   │   ├── [01;34m_report[00m
            │       │   │   │   └── report.rst
            │       │   │   └── result_merge.pklz
            │       │   ├── [01;34mmulti_inputs[00m
            │       │   │   ├── _0x37675359af89b98713cef7ae7d806bf4.json
            │       │   │   ├── _inputs.pklz
            │       │   │   ├── _node.pklz
            │       │   │   ├── [01;34m_report[00m
            │       │   │   │   └── report.rst
            │       │   │   └── result_multi_inputs.pklz
            │       │   └── [01;34msmooth[00m
            │       │       ├── _0x5ddeb3de4584be9882b1b14c01aee151.json
            │       │       ├── _inputs.pklz
            │       │       ├── [01;34mmapflow[00m
            │       │       │   └── [01;34m_smooth0[00m
            │       │       │       ├── _0xb9605f3b440df01231d1cbff9b24a512.json
            │       │       │       ├── command.txt
            │       │       │       ├── _inputs.pklz
            │       │       │       ├── _node.pklz
            │       │       │       ├── [01;34m_report[00m
            │       │       │       │   └── report.rst
            │       │       │       ├── result__smooth0.pklz
            │       │       │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz[00m
            │       │       ├── _node.pklz
            │       │       ├── [01;34m_report[00m
            │       │       │   └── report.rst
            │       │       └── result_smooth.pklz
            │       └── [01;34m_thresh_0.99[00m
            │           ├── [01;34mmask[00m
            │           │   ├── _0xf958f40fead78e24ded1961fe35cd49e.json
            │           │   ├── _inputs.pklz
            │           │   ├── [01;34mmapflow[00m
            │           │   │   └── [01;34m_mask0[00m
            │           │   │       ├── _0x13a9045af46b655e937607e086ac3eb3.json
            │           │   │       ├── command.txt
            │           │   │       ├── _inputs.pklz
            │           │   │       ├── _node.pklz
            │           │   │       ├── [01;34m_report[00m
            │           │   │       │   └── report.rst
            │           │   │       ├── result__mask0.pklz
            │           │   │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_mask.nii.gz[00m
            │           │   ├── _node.pklz
            │           │   ├── [01;34m_report[00m
            │           │   │   └── report.rst
            │           │   └── result_mask.pklz
            │           ├── [01;34mmeanfunc2[00m
            │           │   ├── _0xfee8af85841ecda0f5e47929b50e81fd.json
            │           │   ├── _inputs.pklz
            │           │   ├── [01;34mmapflow[00m
            │           │   │   └── [01;34m_meanfunc20[00m
            │           │   │       ├── _0x9efd1d66ae5ecc720966f2dfa0ac5ec8.json
            │           │   │       ├── command.txt
            │           │   │       ├── _inputs.pklz
            │           │   │       ├── _node.pklz
            │           │   │       ├── [01;34m_report[00m
            │           │   │       │   └── report.rst
            │           │   │       ├── result__meanfunc20.pklz
            │           │   │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_mask_mean.nii.gz[00m
            │           │   ├── _node.pklz
            │           │   ├── [01;34m_report[00m
            │           │   │   └── report.rst
            │           │   └── result_meanfunc2.pklz
            │           ├── [01;34mmedian[00m
            │           │   ├── _0x5f41043b09347e5b87c76bd9b1d51e5a.json
            │           │   ├── _inputs.pklz
            │           │   ├── [01;34mmapflow[00m
            │           │   │   └── [01;34m_median0[00m
            │           │   │       ├── _0x615971451c3f49735bdd2cbb7b395c49.json
            │           │   │       ├── command.txt
            │           │   │       ├── _inputs.pklz
            │           │   │       ├── _node.pklz
            │           │   │       ├── [01;34m_report[00m
            │           │   │       │   └── report.rst
            │           │   │       └── result__median0.pklz
            │           │   ├── _node.pklz
            │           │   ├── [01;34m_report[00m
            │           │   │   └── report.rst
            │           │   └── result_median.pklz
            │           ├── [01;34mmerge[00m
            │           │   ├── _0x5c00b1233e95768c08ef6443bfe08123.json
            │           │   ├── _inputs.pklz
            │           │   ├── _node.pklz
            │           │   ├── [01;34m_report[00m
            │           │   │   └── report.rst
            │           │   └── result_merge.pklz
            │           ├── [01;34mmulti_inputs[00m
            │           │   ├── _0x2e91037fb9319175b7a6ef334a5cc39d.json
            │           │   ├── _inputs.pklz
            │           │   ├── _node.pklz
            │           │   ├── [01;34m_report[00m
            │           │   │   └── report.rst
            │           │   └── result_multi_inputs.pklz
            │           └── [01;34msmooth[00m
            │               ├── _0x6683837f3fc0e652846e9de8f89c45f8.json
            │               ├── _inputs.pklz
            │               ├── [01;34mmapflow[00m
            │               │   └── [01;34m_smooth0[00m
            │               │       ├── _0x51c599f9f5b56ee8fa00436f15ad7814.json
            │               │       ├── command.txt
            │               │       ├── _inputs.pklz
            │               │       ├── _node.pklz
            │               │       ├── [01;34m_report[00m
            │               │       │   └── report.rst
            │               │       ├── result__smooth0.pklz
            │               │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz[00m
            │               ├── _node.pklz
            │               ├── [01;34m_report[00m
            │               │   └── report.rst
            │               └── result_smooth.pklz
            ├── [01;34mwork_preproc_anat[00m
            │   ├── [01;34mget_mask_ref[00m
            │   │   ├── _0xd3207a8c7937a28c15cae5a53eef0410.json
            │   │   ├── _inputs.pklz
            │   │   ├── _node.pklz
            │   │   ├── [01;34m_report[00m
            │   │   │   └── report.rst
            │   │   └── result_get_mask_ref.pklz
            │   ├── [01;34mgunzip_anat[00m
            │   │   ├── _0xb2f717e2ee2d0bdb606903ad07d14cef.json
            │   │   ├── _inputs.pklz
            │   │   ├── _node.pklz
            │   │   ├── [01;34m_report[00m
            │   │   │   └── report.rst
            │   │   └── result_gunzip_anat.pklz
            │   ├── [01;34msegmentation[00m
            │   │   ├── _0xdd45117b85ee12711243b7e8cb985c00.json
            │   │   ├── command.txt
            │   │   ├── _inputs.pklz
            │   │   ├── _node.pklz
            │   │   ├── [01;34m_report[00m
            │   │   │   └── report.rst
            │   │   ├── result_segmentation.pklz
            │   │   ├── [01;31msub-01_T1w_brain_pve_0.nii.gz[00m
            │   │   ├── [01;31msub-01_T1w_brain_pve_1.nii.gz[00m
            │   │   └── [01;31msub-01_T1w_brain_pve_2.nii.gz[00m
            │   ├── [01;34mskullstrip[00m
            │   │   ├── _0x82e8cdfcce754f90b04040fa89929a8b.json
            │   │   ├── command.txt
            │   │   ├── _inputs.pklz
            │   │   ├── _node.pklz
            │   │   ├── [01;34m_report[00m
            │   │   │   └── report.rst
            │   │   ├── result_skullstrip.pklz
            │   │   └── [01;31msub-01_T1w_brain.nii.gz[00m
            │   └── [01;34mthreshold[00m
            │       ├── _0x09b1f1af0691b7513cd8409bbcbe50f9.json
            │       ├── command.txt
            │       ├── _inputs.pklz
            │       ├── _node.pklz
            │       ├── [01;34m_report[00m
            │       │   └── report.rst
            │       ├── result_threshold.pklz
            │       └── [01;31msub-01_T1w_brain_pve_2_thresh.nii.gz[00m
            └── [01;34mwork_preproc_func[00m
                ├── [01;34mextract[00m
                │   ├── _0x928862c935f0d57b21fcf6a19185213a.json
                │   ├── command.txt
                │   ├── _inputs.pklz
                │   ├── _node.pklz
                │   ├── [01;34m_report[00m
                │   │   └── report.rst
                │   ├── result_extract.pklz
                │   └── sub-01_task-rest_bold_roi.nii
                ├── [01;34mgunzip_func[00m
                │   ├── _0xd902cb81eedae4e3727bdcfc0e31b847.json
                │   ├── _inputs.pklz
                │   ├── _node.pklz
                │   ├── [01;34m_report[00m
                │   │   └── report.rst
                │   ├── result_gunzip_func.pklz
                │   └── sub-01_task-rest_bold.nii
                ├── [01;34mmcflirt_mean[00m
                │   ├── _0xc8c03165a7c1aa88c9424d5d56edc9b2.json
                │   ├── _inputs.pklz
                │   ├── _node.pklz
                │   ├── [01;34m_report[00m
                │   │   └── report.rst
                │   └── result_mcflirt_mean.pklz
                ├── [01;34mmcflirt_vol[00m
                │   ├── _0xe65de07c6fe5d565c63a9be10b6c48f9.json
                │   ├── command.txt
                │   ├── _inputs.pklz
                │   ├── _node.pklz
                │   ├── [01;34m_report[00m
                │   │   └── report.rst
                │   ├── result_mcflirt_vol.pklz
                │   └── [01;31msub-01_task-rest_bold_roi_st_mcf.nii.gz[00m
                └── [01;34mslicetime[00m
                    ├── _0x3f428e2adba53482082d26c92d08921a.json
                    ├── command.txt
                    ├── _inputs.pklz
                    ├── _node.pklz
                    ├── [01;34m_report[00m
                    │   └── report.rst
                    ├── result_slicetime.pklz
                    └── sub-01_task-rest_bold_roi_st.nii
    
    610 directories, 1311 files


### Elimininar todos los archivos temporales


```python
# eliminar todos los archivos temporales almacenados en la carpeta 'workingdir_reconflow'
os.system('rm -rf %s'%prereg.base_dir)
```




    0




```python
! tree /home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/
```

    [01;34m/home/aracena/thesis_ds002422/01_fase1_extraccion_mask_brain/output/[00m
    └── [01;34mdatasink[00m
        ├── [01;34mfmri_detrend[00m
        │   ├── [01;34mmask_ext_csf[00m
        │   │   ├── [01;34mthreshold_0.5[00m
        │   │   │   └── [01;31mfmri_rest_prepro.nii.gz[00m
        │   │   ├── [01;34mthreshold_0.95[00m
        │   │   │   └── [01;31mfmri_rest_prepro.nii.gz[00m
        │   │   ├── [01;34mthreshold_0.97[00m
        │   │   │   └── [01;31mfmri_rest_prepro.nii.gz[00m
        │   │   └── [01;34mthreshold_0.99[00m
        │   │       └── [01;31mfmri_rest_prepro.nii.gz[00m
        │   ├── [01;34mmask_ext_gm[00m
        │   │   ├── [01;34mthreshold_0.5[00m
        │   │   │   └── [01;31mfmri_rest_prepro.nii.gz[00m
        │   │   ├── [01;34mthreshold_0.95[00m
        │   │   │   └── [01;31mfmri_rest_prepro.nii.gz[00m
        │   │   ├── [01;34mthreshold_0.97[00m
        │   │   │   └── [01;31mfmri_rest_prepro.nii.gz[00m
        │   │   └── [01;34mthreshold_0.99[00m
        │   │       └── [01;31mfmri_rest_prepro.nii.gz[00m
        │   └── [01;34mmask_ext_wm[00m
        │       ├── [01;34mthreshold_0.5[00m
        │       │   └── [01;31mfmri_rest_prepro.nii.gz[00m
        │       ├── [01;34mthreshold_0.95[00m
        │       │   └── [01;31mfmri_rest_prepro.nii.gz[00m
        │       ├── [01;34mthreshold_0.97[00m
        │       │   └── [01;31mfmri_rest_prepro.nii.gz[00m
        │       └── [01;34mthreshold_0.99[00m
        │           └── [01;31mfmri_rest_prepro.nii.gz[00m
        ├── [01;34mfmri_sin_smooth[00m
        │   ├── sub-01_task-rest_bold_roi_st_mcf_flirt.mat
        │   └── sub-01_task-rest_bold_roi_st_mcf_flirt.nii
        ├── [01;34mfmri_smooth[00m
        │   ├── [01;34mmask_ext_csf[00m
        │   │   ├── [01;34mthreshold_0.5[00m
        │   │   │   └── [01;34msmoooth[00m
        │   │   │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz[00m
        │   │   ├── [01;34mthreshold_0.95[00m
        │   │   │   └── [01;34msmoooth[00m
        │   │   │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz[00m
        │   │   ├── [01;34mthreshold_0.97[00m
        │   │   │   └── [01;34msmoooth[00m
        │   │   │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz[00m
        │   │   └── [01;34mthreshold_0.99[00m
        │   │       └── [01;34msmoooth[00m
        │   │           └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz[00m
        │   ├── [01;34mmask_ext_gm[00m
        │   │   ├── [01;34mthreshold_0.5[00m
        │   │   │   └── [01;34msmoooth[00m
        │   │   │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz[00m
        │   │   ├── [01;34mthreshold_0.95[00m
        │   │   │   └── [01;34msmoooth[00m
        │   │   │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz[00m
        │   │   ├── [01;34mthreshold_0.97[00m
        │   │   │   └── [01;34msmoooth[00m
        │   │   │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz[00m
        │   │   └── [01;34mthreshold_0.99[00m
        │   │       └── [01;34msmoooth[00m
        │   │           └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz[00m
        │   └── [01;34mmask_ext_wm[00m
        │       ├── [01;34mthreshold_0.5[00m
        │       │   └── [01;34msmoooth[00m
        │       │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz[00m
        │       ├── [01;34mthreshold_0.95[00m
        │       │   └── [01;34msmoooth[00m
        │       │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz[00m
        │       ├── [01;34mthreshold_0.97[00m
        │       │   └── [01;34msmoooth[00m
        │       │       └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz[00m
        │       └── [01;34mthreshold_0.99[00m
        │           └── [01;34msmoooth[00m
        │               └── [01;31msub-01_task-rest_bold_roi_st_mcf_flirt_smooth.nii.gz[00m
        ├── [01;34mmask_files[00m
        │   ├── [01;31msub-01_T1w_brain_pve_0.nii.gz[00m
        │   ├── [01;31msub-01_T1w_brain_pve_1.nii.gz[00m
        │   └── [01;31msub-01_T1w_brain_pve_2.nii.gz[00m
        ├── [01;34mmasks_brain[00m
        │   ├── [01;34mmask_ext_csf[00m
        │   │   ├── [01;34mthreshold_0.5[00m
        │   │   │   └── [01;31mfmri_rest_prepro.nii.gz[00m
        │   │   ├── [01;34mthreshold_0.95[00m
        │   │   │   └── [01;31mfmri_rest_prepro.nii.gz[00m
        │   │   ├── [01;34mthreshold_0.97[00m
        │   │   │   └── [01;31mfmri_rest_prepro.nii.gz[00m
        │   │   └── [01;34mthreshold_0.99[00m
        │   │       └── [01;31mfmri_rest_prepro.nii.gz[00m
        │   ├── [01;34mmask_ext_gm[00m
        │   │   ├── [01;34mthreshold_0.5[00m
        │   │   │   └── [01;31mfmri_rest_prepro.nii.gz[00m
        │   │   ├── [01;34mthreshold_0.95[00m
        │   │   │   └── [01;31mfmri_rest_prepro.nii.gz[00m
        │   │   ├── [01;34mthreshold_0.97[00m
        │   │   │   └── [01;31mfmri_rest_prepro.nii.gz[00m
        │   │   └── [01;34mthreshold_0.99[00m
        │   │       └── [01;31mfmri_rest_prepro.nii.gz[00m
        │   └── [01;34mmask_ext_wm[00m
        │       ├── [01;34mthreshold_0.5[00m
        │       │   └── [01;31mfmri_rest_prepro.nii.gz[00m
        │       ├── [01;34mthreshold_0.95[00m
        │       │   └── [01;31mfmri_rest_prepro.nii.gz[00m
        │       ├── [01;34mthreshold_0.97[00m
        │       │   └── [01;31mfmri_rest_prepro.nii.gz[00m
        │       └── [01;34mthreshold_0.99[00m
        │           └── [01;31mfmri_rest_prepro.nii.gz[00m
        └── [01;34mmasks_brain_sin_detrend[00m
            ├── [01;34mmask_ext_csf[00m
            │   ├── [01;34mthreshold_0.5[00m
            │   │   └── [01;34mmask_func[00m
            │   │       └── sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
            │   ├── [01;34mthreshold_0.95[00m
            │   │   └── [01;34mmask_func[00m
            │   │       └── sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
            │   ├── [01;34mthreshold_0.97[00m
            │   │   └── [01;34mmask_func[00m
            │   │       └── sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
            │   └── [01;34mthreshold_0.99[00m
            │       └── [01;34mmask_func[00m
            │           └── sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
            ├── [01;34mmask_ext_gm[00m
            │   ├── [01;34mthreshold_0.5[00m
            │   │   └── [01;34mmask_func[00m
            │   │       └── sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
            │   ├── [01;34mthreshold_0.95[00m
            │   │   └── [01;34mmask_func[00m
            │   │       └── sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
            │   ├── [01;34mthreshold_0.97[00m
            │   │   └── [01;34mmask_func[00m
            │   │       └── sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
            │   └── [01;34mthreshold_0.99[00m
            │       └── [01;34mmask_func[00m
            │           └── sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
            └── [01;34mmask_ext_wm[00m
                ├── [01;34mthreshold_0.5[00m
                │   └── [01;34mmask_func[00m
                │       └── sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
                ├── [01;34mthreshold_0.95[00m
                │   └── [01;34mmask_func[00m
                │       └── sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
                ├── [01;34mthreshold_0.97[00m
                │   └── [01;34mmask_func[00m
                │       └── sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
                └── [01;34mthreshold_0.99[00m
                    └── [01;34mmask_func[00m
                        └── sub-01_task-rest_bold_roi_st_mcf_flirt_smooth_masked.nii
    
    91 directories, 53 files


## Tiempo de ejecución


```python
fin = time.time()
end = time.process_time()
tiempo = fin - inicio
tiempo2 = end - start

print('-----------------------------\n', 
      'tiempo de ejecución\n\n', tiempo, 'seg\n', tiempo/60, 'min\n',      
     '-----------------------------\n')
print('---------------------------------------\n', 
      'tiempo de ejecución del sistema y CPU\n\n', tiempo2, 'seg\n', tiempo2/60, 'min\n',   
     '---------------------------------------\n')
```

    -----------------------------
     tiempo de ejecución
    
     127.3383538722992 seg
     2.1223058978716534 min
     -----------------------------
    
    ---------------------------------------
     tiempo de ejecución del sistema y CPU
    
     9.585817185 seg
     0.15976361975 min
     ---------------------------------------
    


## Fin
