"""Module with data type"""
from typing import Union
import numpy as np


Position1 = np.dtype("(3,)float")               # координаты в трёхмерном пространстве
Position = np.ndarray                           # Position1
PositionsArray = np.ndarray                     # np.ndarray(:)[Position1]
Positions2DArray = np.ndarray                   # np.ndarray(:,:)[Position1]


RotationMatrixDtype = np.dtype("(3,3)float")    # Матрица поворота в трехмерном пространстве
RotationMatrix = np.ndarray                     # RotationMatrixDtype
RotationMatrixArray = np.ndarray                # np.ndarray(:)[RotationMatrixDtype]

QuadraticFormDType = np.dtype("(3,3)float")
QuadraticForm = np.ndarray                      # QuadraticFormDType
QuadraticFormArray = np.ndarray                 # np.ndarray(:)[QuadraticFormDType]

FloatsArray = np.ndarray                        # np.ndarray(:)[float]
BoolsArray = np.ndarray                         # np.ndarray(:)[bool]



Epoch = float

EpochMulti = np.ndarray
VectorMulti3 = np.ndarray
Vector3 = np.ndarray
Matrix33 = np.ndarray

StateVector = np.ndarray
"""StateVector consists of all variable parameters of satellite
StateVector[ 0] = x
StateVector[ 1] = y
StateVector[ 2] = z
StateVector[ 3] = vx
StateVector[ 4] = vy
StateVector[ 5] = vz
StateVector[ 6] = mass
StateVector[ 7] = area
StateVector[ 8] = reflection coefficient
StateVector[ 9] = drag coefficient
StateVector[10] = orientation x
StateVector[11] = orientation y
StateVector[12] = orientation z
"""
