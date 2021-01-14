#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 17:39:37 2020

@author: kaifeiwang
"""

import numpy as np
import math

np.set_printoptions(formatter={'float': '{: 0.6f}'.format})

def get_matrix(filename):
    """从filename中读取矩阵"""
    matrix = np.loadtxt(filename, delimiter = ' ')
    return matrix

def PLU_decompose(matrix, out_path):
    """矩阵分解PLU"""
    xx = matrix.copy()
    m = np.size(matrix, 0)
    n = np.size(matrix, 1)
    L = np.zeros_like(matrix, dtype = "float")  #下三角矩阵
    P = np.eye(m, dtype='float')  #部分主元法行交换矩阵
    i = 0
    base = 0
    while i < m:
        P_temp = np.eye(m, dtype = 'float')  #本次高斯消元的行交换矩阵
        A = np.eye(m, dtype = 'float')  #高斯变换矩阵
        temp = np.fabs(matrix[i:, i + base])  #部分主元法考虑的列元素（前面的行已经被确定）
        max_line = temp.argmax()  #得到要交换到首位的行号
        if max_line != 0:  #需要交换
            P_temp[i, i] = 0
            P_temp[i, i + max_line] = 1
            P_temp[i + max_line, i + max_line] = 0
            P_temp[i + max_line, i] = 1
        matrix = np.matmul(P_temp, matrix) #交换原矩阵
        L = np.matmul(P_temp, L) #交换下三角矩阵
        P = np.matmul(P_temp, P) #存储行交换变换
        if math.fabs(matrix[i, i + base]) >= 1e-15:
            for j in range(i + 1, m):  #求此次高斯变换
                L[j, i] = matrix[j, i] / matrix[i, i]
                A[j, i] = - matrix[j, i] / matrix[i, i]
            i += 1
        else:
            base += 1
        matrix = np.matmul(A, matrix) #原矩阵高斯变换
        if i + base >= n:
            break
    L = L + np.eye(m, dtype = "float")  #在对角线位置填1
    with open(out_path, 'w') as file_object:
        file_object.write("P:\n")
        for i in range(np.size(P, 0)):
            line = str(P[i, :]) + '\n'
            file_object.write(line)
        file_object.write("L:\n")
        for i in range(np.size(L, 0)):
            line = str(L[i, :]) + '\n'
            file_object.write(line)
        file_object.write("U:\n")
        for i in range(np.size(L, 0)):
            line = str(matrix[i, :]) + '\n'
            file_object.write(line)
        file_object.close()
    print("P:\n", P, "\nL:\n", L, "\nU:\n", matrix)
    print(P.dot(xx))
    print(L.dot(matrix))
    return P, L, matrix

def PU_decompose(matrix):
    """矩阵的初等变换PA = U
    简化阶梯形矩阵"""
    m = np.size(matrix, 0)
    n = np.size(matrix, 1)
    P = np.eye(m, dtype='float')  #部分主元法高斯变换矩阵
    base = 0
    i = 0
    while i < m:
        P_temp = np.eye(m, dtype = 'float')  #本次的高斯变换矩阵
        temp = np.fabs(matrix[i:, i + base])  #部分主元法考虑的列元素（前面的行已经被确定）
        max_line = temp.argmax()  #得到要交换到首位的行号
        if max_line != 0:  #需要交换
            P_temp[i, i] = 0
            P_temp[i, i + max_line] = 1
            P_temp[i + max_line, i + max_line] = 0
            P_temp[i + max_line, i] = 1
            matrix = np.matmul(P_temp, matrix) #交换原矩阵
            P = np.matmul(P_temp, P) #P矩阵累积变换
            P_temp = np.eye(m, dtype = 'float')  #还原成单位矩阵
            
        if math.fabs(matrix[i, i + base]) >= 1e-15:
#            matrix[i, :] = matrix[i, :] / matrix[i, i + base]#该行归一
            P_temp[i, i] = 1 / matrix[i, i + base] #该行归一
            for j in range(0, m):  #求此次高斯变换
                if j == i:
                    continue
                P_temp[j, i] = -matrix[j, i + base] / matrix[i, i + base]
            i += 1
        else:
            base += 1
        P = np.matmul(P_temp, P) #P矩阵累积变换
        matrix = np.matmul(P_temp, matrix) #原矩阵高斯变换
        P_temp = np.eye(m, dtype = 'float')  #还原成单位矩阵
        if i + base >= n:
            break
    return P, matrix
        
def QR_Gram_Schmidt(matrix, out_path):
    """矩阵QR分解"""
    Q = np.zeros_like(matrix)
    R = np.zeros_like(matrix)
    for k in range(0, np.size(matrix, 0)):  #求每一步确定的向量
        Q[:, k] = matrix[:, k].copy()
        for j in range(0, k):  #求R矩阵中的值
            R[j, k] = np.dot(Q[:, j].T, matrix[:, k])  #第k列在第j个向量上的分量
            Q[:, k] -= R[j, k] * Q[:, j]  #减去该分量
        R[k, k] = np.linalg.norm(Q[:, k]) #求第k个向量的模长
        Q[:, k] = Q[:, k] / R[k,k]
    with open(out_path, 'w') as file_object:
        file_object.write("Q:\n")
        for i in range(np.size(Q, 0)):
            line = str(Q[i, :]) + '\n'
            file_object.write(line)
        file_object.write("R:\n")
        for i in range(np.size(R, 0)):
            line = str(R[i, :]) + '\n'
            file_object.write(line)
        file_object.close()
    print("Q:\n", Q, "\nR:\n", R)
    return Q, R
        
def Householder(matrix, out_path):
    """矩阵Householder约减"""
    """需要记录Q的更新，就是每次算出一个Rx，都乘进去
    需要记录R的更新，就是每次算出一个Rx，都乘一次，需要先乘A"""
    m = np.size(matrix, 0)  #待处理矩阵的size m*n
    Q = np.eye(m)  #Q矩阵是m*m
    R = matrix.copy()   #R矩阵是m*n，首先copy matrix
    zero_c = np.zeros(m) #m维0向量，用于扩充列
    for i in range(0, m - 1):  #有m-1个Rx矩阵
        mod = np.linalg.norm(R[i:,i])
        e1 = np.zeros_like(R[i:,i].reshape(-1,1))
        e1[0, 0] = 1
        w = R[i:,i].reshape(-1,1) - mod * e1
        w = w / np.linalg.norm(w)
        I = np.eye(np.size(w))
        Rx = I - 2 * np.dot(w, w.T)
        #接下来需要将Rx添加行、列变成m*m增广矩阵
        zero_r = np.zeros((1, np.size(Rx, 1))) #零向量
        #行添加零向量
        m_new = np.size(Rx, 0)
        for k in range(m_new, np.size(matrix, 0)):
            Rx = np.r_[zero_r, Rx]
        #列添加零向量
        for k in range(m_new, np.size(matrix, 0)):
            Rx = np.c_[zero_c, Rx]
        #增广的对角线位置置1
        for i in range(0, m - m_new):
            Rx[i, i] = 1
        R = Rx.dot(R)  #将Rx左乘变换
        Q = Q.dot(Rx.T)
    with open(out_path, 'w') as file_object:
        file_object.write("Q:\n")
        for i in range(np.size(Q, 0)):
            line = str(Q[i, :]) + '\n'
            file_object.write(line)
        file_object.write("R:\n")
        for i in range(np.size(R, 0)):
            line = str(R[i, :]) + '\n'
            file_object.write(line)
        file_object.close()
    print("Q:\n", Q, "\nR:\n", R)
    return Q, R
        
def Givens(matrix, out_path):
    """矩阵Givens约减"""
    """其实和Householder类似，
    需要记录Q的更新，就是每次算出一个P，都乘进去
    需要记录R的更新，就是每次算出一个Rx，都乘一次，需要先乘A"""
    m = np.size(matrix, 0) #m行
    n = np.size(matrix, 1) #n列
    R = matrix.copy()   #R矩阵是m*n，首先copy matrix
    Q = np.eye(m)  #Q矩阵是m*m
    for i in range(0, n): #约减每一列（只留下对角线元素及上面的）
        for j in range(i + 1, m): #在该列的第i个元素要和下面的所有元素约减
            cos = R[i, i] / math.sqrt(R[j, i] ** 2 + R[i, i] ** 2)
            sin = R[j, i] / math.sqrt(R[j, i] ** 2 + R[i, i] ** 2)
            P = np.eye(m)
            P[i, i] = cos
            P[j, j] = cos
            P[j, i] = -sin
            P[i, j] = sin
            R = P.dot(R)
            Q = Q.dot(P.T)
    with open(out_path, 'w') as file_object:
        file_object.write("Q:\n")
        for i in range(np.size(Q, 0)):
            line = str(Q[i, :]) + '\n'
            file_object.write(line)
        file_object.write("R:\n")
        for i in range(np.size(R, 0)):
            line = str(R[i, :]) + '\n'
            file_object.write(line)
        file_object.close()
    print("Q:\n", Q, "\nR:\n", R)
    return Q, R

def Gram(matrix):
    """Gram-Schmidt正交化"""
    n = np.size(matrix, 1)  #列
    m = np.size(matrix, 0)  #行
    for i in range(0, n):
        u = np.zeros(m)
        for j in range(0, i):
            u += matrix[:, j].dot(matrix[:, i]) * matrix[:, j]
        matrix[:, i] -= u
        matrix[:, i] = matrix[:, i] / np.linalg.norm(matrix[:, i])
    return matrix

def URV(matrix, out_path, err = 1e-15):  
    """矩阵的URV分解 Am*n = Um*m Rm*n Vn*n
    U: R(A)的基 + N(A.T)的基
    V: R(A.T)的基 + N(A)的基
    R = U.T A V
    先将A进行高斯变换，PA = U
    """   
    m = np.size(matrix, 0) #m行
    n = np.size(matrix, 1) #n列
    U = np.zeros((m, m))
    R = np.zeros((m, n))
    V = np.zeros((n, n))
    P1, U1 = PU_decompose(matrix) #高斯约旦方法化简最简型
    basic_col = [] #基本列列号
    none_zero_row = [] #非零行行号
    for i in range(0, m): #遍历所有行
        for j in range(i, n):
            if(math.fabs(U1[i, j]) > err):
                basic_col.append(j) #添加基本列
                none_zero_row.append(i)  #添加非零行
                break
    rank = len(basic_col) #矩阵的秩
    for k in range(0, rank):
        #基本列进入U
        U[:, k] = matrix[:, basic_col[k]].copy() / np.linalg.norm(matrix[:, basic_col[k]])
        #非零行进入V
        V[:, k] = U1[none_zero_row[k], :].copy() / np.linalg.norm(U1[none_zero_row[k], :])
    U[:, 0:rank] =  Gram(U[:, 0:rank]) #R(A)的正交基
    V[:, 0:rank] =  Gram(V[:, 0:rank]) #R(A.T)的正交基
    
    free_vs = [] #自由变量
    non_free_vs = {} #存主元位置
    for i in range(0, m):
        for j in range(0, n):
            if math.fabs(U1[i, j]) >= 1e-15:  #i行的第一个非零元素
                non_free_vs[j] = i
                break
    for i in range(0, n):
        if i not in non_free_vs:
            free_vs.append(i)
    
    sp_zero = np.zeros((n, len(free_vs)))  #零空间
    for i in range(0, len(free_vs)):
        sp_zero[free_vs[i], i] = 1
    for i in non_free_vs:  #遍历字典键（列）
        for j in range(0, len(free_vs)):
            sp_zero[i, j] = -U1[non_free_vs[i], free_vs[j]]
    for i in range(0, np.size(sp_zero, 1)):
        V[:, i + rank] = sp_zero[:, i] / np.linalg.norm(sp_zero[:, i])  #将A零空间进入V
    V[:, rank : m] = Gram(V[:, rank : m])
    
    for i in range(rank, m):
        U[:, i] = (P1[i, :] / np.linalg.norm(P1[i, :])) #AT零空间进入U
    U[:, rank: m] = Gram(U[:, rank: m])  #正交化
    R = U.T.dot(matrix).dot(V)  #求R
    with open(out_path, 'w') as file_object:
        file_object.write("U:\n")
        for i in range(np.size(U, 0)):
            line = str(U[i, :]) + '\n'
            file_object.write(line)
        file_object.write("R:\n")
        for i in range(np.size(R, 0)):
            line = str(R[i, :]) + '\n'
            file_object.write(line)
        file_object.write("V:\n")
        for i in range(np.size(V, 0)):
            line = str(V[i, :]) + '\n'
            file_object.write(line)
        file_object.close()
    print("U:\n", U, "\nR:\n", R, "\nV:\n", V)
    print("rank of input matrix: ", rank)
    print("VV.T")
    print(V.dot(V.T))
    print("UU.T")
    print(U.dot(U.T))
    print("URV.T")
    print(U.dot(R).dot(V.T))
    return U, R, V
        
        
    

            
                


if __name__ == "__main__":
    switch = {1:PLU_decompose, 2:QR_Gram_Schmidt, 3:Householder, 4:Givens, 5:URV}
    print("Please input the filepath where store the matrix you want to manage:")
    # "/Users/kaifeiwang/Desktop/filename.txt" "/Users/kaifeiwang/Desktop/plu_out.txt"
    input_path = input()
    print("Please input the filepath where store the results:")
    out_path = input()
    print("Please choose the function you want to do:\n1: PA=LU分解\n2: QR_Gram_Schmidt分解\n3: Householder reduction\n4: Givens reduction\n5: URV分解")
    fun = int(input())
    matrix = get_matrix(input_path)
    try:
        switch[fun](matrix, out_path) #执行相应的方法。
    except KeyError:
        print("Please check your input number\n")
    except:
        print("Please cheak the format of matrix!!")
        
        
    