#!/usr/bin/env python3
def matrix_shape(matrix):
    mat = []
    mat.append(len(matrix))
    if type(matrix[0]) == list:
        mat.append(len(matrix[0]))
        if type(matrix[0][0]) == list:
            mat.append(len(matrix[0][0]))
    return mat
                
mat1 = [[1, 2], [3, 4]]
print(matrix_shape(mat1))
mat2 = [[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
        [[16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]]
print(matrix_shape(mat2))
