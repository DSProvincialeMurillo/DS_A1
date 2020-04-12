"""
Algoritmo para multiplicar dos matrices de tamaño M*N x N*L 
de forma secuencial o de forma paralela en IBM Cloud utilizando pywren.

Fecha:
    12/04/2020
Autores: 
        Marc Provinciale Isach
        Aaron Murillo Lort
    * Ambos autores han colaborado de forma equitativa y han implementado
      entre los dos todo el código del algoritmo. 
"""
from cos_backend import COSBackend
import pywren_ibm_cloud as pywren
import numpy as np
import pickle as p
import time

WORKERS = 100
RANGO = 10 #Rango valores de [-RANGO, RANGO]
N = 10
M = 10
L = 10

PAQUETES = False

BUCKET = 'marc-provinciale-bucket'

# Inicializacion de matrices y subida a la nube según si es secuencial o paralelo
# subiendo todo B
def matrix_ini(x, n, m, l, iterdata): 
    cos = COSBackend()
    np.random.seed()
    A = np.random.randint(2*x, size=(m, n)) - x
    B = np.random.randint(2*x, size=(n, l)) - x

    #Subida de datos de forma secuencial
    if WORKERS == 1:
        cos.put_object(BUCKET,'/secuencial/A', p.dumps(A, p.HIGHEST_PROTOCOL))
        cos.put_object(BUCKET,'/secuencial/B', p.dumps(B, p.HIGHEST_PROTOCOL))
    
    #Subida de datos de forma paralela
    else:
        #Dividir matriz A en paquetes según el número de workers
        for i in iterdata:
            i = str(i).split('|')
            #Obtener posición de inicio del worker
            op_ini = i[1].split(',')
            op_ini[0] = int(op_ini[0])
            #Obtener posición final del worker
            op_fi = i[2].split(',')
            op_fi[0] = int(op_fi[0])+1
            cos.put_object(BUCKET,'/paralelo/f'+i[0], p.dumps(A[op_ini[0]:op_fi[0],:], p.HIGHEST_PROTOCOL))
        #Subir matriz B entera
        cos.put_object(BUCKET,'/secuencial/B', p.dumps(B, p.HIGHEST_PROTOCOL))

# Inicializacion de matrices y subida a la nube según si es secuencial o paralelo
# sin subir todo B a veces
def matrix_ini_paquetes(x, n, m, l, w, iterdata):
    cos = COSBackend()
    np.random.seed()
    A = np.random.randint(2*x, size=(m, n)) - x
    B = np.random.randint(2*x, size=(n, l)) - x

    #Subida de datos de forma secuencial
    if WORKERS == 1:
        cos.put_object(BUCKET,'/secuencial/A', p.dumps(A, p.HIGHEST_PROTOCOL))
        cos.put_object(BUCKET,'/secuencial/B', p.dumps(B, p.HIGHEST_PROTOCOL))

    #Subida de datos de forma paralela
    else:
        #Dividir matrices en filas y columnas
        for i in range(0, w):
            rang = str(iterdata[i]).split('|')
        
            op_ini = rang[1].split(',')
            op_ini[0] = int(op_ini[0])
            op_ini[1] = int(op_ini[1])

            op_fi = rang[2].split(',')
            op_fi[0] = int(op_fi[0])
            op_fi[1] = int(op_fi[1])

            Apar = np.zeros((op_fi[0] - op_ini[0] + 1, n), int)

            if (op_fi[0] - op_ini[0]) >= 1:     # Si la operación necesita dos filas o más
                if op_ini[1] < op_fi[1] or (op_fi[0] - op_ini[0]) > 1 or (op_ini[1] == op_fi[1] and op_fi[0] - op_ini[0] == 1): # Si todas las columnas se ven afectadas  
                    cos.put_object(BUCKET, '/paralelo/B'+str(i), p.dumps(B, p.HIGHEST_PROTOCOL))
                else:   # Si no todas las columnas se ven afectadas a pesar de necesitar dos o más filas
                    Bpar = np.zeros((n, ((op_fi[1] - op_ini[1]) % l) + 1), int)
                    j = 0
                    while op_ini[1] != op_fi[1]:    # Subimos aquellas que sean necesarias
                        Bpar[:,j] = B[:,op_ini[1]]
                        op_ini[1] = (op_ini[1] + 1) % L
                        j = j + 1
                    Bpar[:,j] = B[:,op_ini[1]]
                    cos.put_object(BUCKET, '/paralelo/B'+str(i), p.dumps(Bpar, p.HIGHEST_PROTOCOL))
            else:   # Subimos únicamente las columnas necesarias
                Bpar = np.zeros((n, ((op_fi[1] - op_ini[1]) % l) + 1), int)
                j = 0
                while op_ini[1] != op_fi[1]:
                    Bpar[:,j] = B[:,op_ini[1]]
                    op_ini[1] = (op_ini[1] + 1) % L
                    j = j + 1
                Bpar[:,j] = B[:,op_ini[1]]
                cos.put_object(BUCKET, '/paralelo/B'+str(i), p.dumps(Bpar, p.HIGHEST_PROTOCOL))    
            
            j = 0
            while op_ini[0] <= op_fi[0]:    # Subimos las filas necesarias para la operación
                Apar[j,:] = A[op_ini[0],:]
                op_ini[0] = op_ini[0] + 1
                j = j + 1
            cos.put_object(BUCKET, '/paralelo/A'+str(i), p.dumps(Apar, p.HIGHEST_PROTOCOL))


#Calculo de la matriz C según si es de forma secuencial o paralela
def matrix_mult(x):
    cos = COSBackend()
    x = str(x).split('|')

    #Calculo de forma secuencial
    if WORKERS == 1:
        A = p.loads(cos.get_object(BUCKET, '/secuencial/A'))
        B = p.loads(cos.get_object(BUCKET, '/secuencial/B'))
        results = np.dot(A, B)

    #Calculo de forma paralela que hará cada worker con su parte correspondiente
    else:
        results = []
        
        op_ini = x[1].split(',')
        op_ini[0] = int(op_ini[0])
        op_ini[1] = int(op_ini[1])

        op_fi = x[2].split(',')
        op_fi[0] = int(op_fi[0])
        op_fi[1] = int(op_fi[1])

        A = p.loads(cos.get_object(BUCKET, '/paralelo/f'+x[0]))
        B = p.loads(cos.get_object(BUCKET, '/secuencial/B'))

        rango=op_ini[0]

        while op_ini <= op_fi:
            #Calculo de la posición C[f_act-f_ini, c_act]
            results.append(A[op_ini[0]-rango].dot(B[:,op_ini[1]]))
            op_ini[1] = op_ini[1] + 1
            #Saltamos de fila de C
            if (op_ini[1] >= L):
                op_ini[0] = op_ini[0] + 1
                op_ini[1] = 0
        
    return results

# Cálculo de la matriz C según si es de forma secuencial o paralela
# descargando B entera o por paquetes según el caso
def matrix_mult_paquetes(x):
    cos = COSBackend()

    # Cálculo de forma secuencial
    if WORKERS == 1:
        A = p.loads(cos.get_object(BUCKET, '/secuencial/A'))
        B = p.loads(cos.get_object(BUCKET, '/secuencial/B'))
        results = np.dot(A, B)

    # Cálculo de forma paralela que hará cada worker con su parte correspondiente
    else:
        x = str(x).split('|')
        results = []

        worker = int(x[0])
        A = p.loads(cos.get_object(BUCKET, '/paralelo/A'+str(worker)))  # Descargamos los paquetes del worker
        B = p.loads(cos.get_object(BUCKET, '/paralelo/B'+str(worker)))

        op_ini = x[1].split(',')
        op_ini[0] = int(op_ini[0])
        op_ini[1] = int(op_ini[1])
 
        op_fi = x[2].split(',')
        op_fi[0] = int(op_fi[0])
        op_fi[1] = int(op_fi[1])

        f = 0       

        if (M * L / WORKERS) >= L:      # Si el paquete de B descargado incluye todo B
            while op_ini <= op_fi:     # Cálculo del worker con B entera
                results.append(A[f].dot(B[:,op_ini[1]]))
                op_ini[1] = op_ini[1] + 1
                if (op_ini[1] >= L):
                    op_ini[0] = op_ini[0] + 1
                    f = f + 1
                    op_ini[1] = 0
        else:
            c = 0

        while op_ini <= op_fi:     # Cálculo del worker siguiendo el orden de las columnas en Bw
            results.append(A[f].dot(B[:,c]))
            op_ini[1] = op_ini[1] + 1
            c = c + 1
            if (op_ini[1] >= L):
                op_ini[0] = op_ini[0] + 1
                f = f + 1
                op_ini[1] = 0

    return results


#Función de agrupar los resultados de los workers
def matrix_reduce(results): 
    #Agrupar los resultados de forma paralela
    if WORKERS != 1:
        C = np.zeros((M,L), int)
        f = 0
        c = 0
        # Para cada worker cojemos los resultados
        for map_results in results:
            # Para cada resultado del worker lo guardamos en C[f,c]
            for i in map_results:
                C[f][c] = i
                c = c + 1
                # Siguiente fila de C
                if c >= L:
                    c = 0
                    f = f + 1
        return C
    #Agrupar los resultados de forma secuencial
    else:
        return results

#Inicializar iterdata. Tendrá el número de worker y el rango de posiciones ambas inclusivas que va a calcular
def matrix_iterdata(m, l):
    # Número de operaciones que tiene que hacer en formato entero
    op_int = (int)((m * l) /WORKERS)
    #Residuo de las operaciones que tiene que hacer
    op_mod = (m * l) %WORKERS
    f = 0
    c = 0
    iterdata = []
    #Calcular posición inicio y posición fin (ambas incluidas) que va a calcular el worker
    for i in range(0, WORKERS):
        subiterdata = str(i)
        subiterdata = subiterdata + '|'+str(f)+','+str(c)
        # Si el worker actual tiene que hacer una operación de más
        if i < op_mod:
            n_op_act = op_int + 1
        else:
            n_op_act = op_int
        f = f + (int)((n_op_act / l) + (c / l))
        c = (c + n_op_act) % (l)
        if c == 0:
            subiterdata = subiterdata+'|'+str(f - 1)+','+str((c - 1) % l)
        else: 
            subiterdata = subiterdata+'|'+str(f)+','+str((c - 1) % l)
        iterdata.append(subiterdata)
    return iterdata

#Borrar datos del cloud
def clean():
    cos = COSBackend()
    print('Cleaning...',end='')
    if WORKERS != 1:
        cos.delete_object(BUCKET, '/secuencial/B')
        for i in range(0, WORKERS):
            print('.',end='')
            cos.delete_object(BUCKET, '/paralelo/f'+str(i))
    else:
        cos.delete_object(BUCKET, '/secuencial/A')
        cos.delete_object(BUCKET, '/secuencial/B')
    print('.',end='\n')


if __name__ == '__main__':
    if WORKERS > 100:
        WORKERS = 100
    if WORKERS > M*L:
        print ("No se puede dividir la matriz con un número mayor de workers que de filas.")
    else:
        pw = pywren.ibm_cf_executor()
        if WORKERS != 1:
            iterdata = matrix_iterdata(M, L)
        else: 
            iterdata = [1]
            
        if PAQUETES:
            pw.call_async(matrix_ini_paquetes, [RANGO, N, M, L, iterdata])
            pw.wait()
            start_time = time.time()
            pw.wait(pw.map_reduce(matrix_mult_paquetes, iterdata, matrix_reduce))
        else: 
            pw.call_async(matrix_ini, [RANGO, N, M, L, iterdata])
            pw.wait()
            start_time = time.time()
            pw.wait(pw.map_reduce(matrix_mult, iterdata, matrix_reduce))

        elapsed_time = time.time() - start_time
        #print(pw.get_result())
        print(elapsed_time)
        pw.clean()
        clean()
        print('######################################################')
        print('END')