//
//  main.c
//  cifar10-CNN-single
//
//  Created by Nomura Ryoji on 2016/07/30.
//  Copyright © 2016年 Nomura Ryoji. All rights reserved.
//

//
//  main.c
//  mnist_CNN
//
//  Created by Nomura Ryoji on 2016/07/09.
//  Copyright © 2016年 Nomura Ryoji. All rights reserved.
//  matrix counting: col incremental first


#include <stdio.h>
#include <stdlib.h> /* rand, malloc */
#include <time.h>
#include <math.h>

#define EPOCH 10 //epoch of learning
#define BatchSize 5//batch size for each epoch
#define DataCount 10000 //data size of learning
#define TestCount 10000 //data size of learning
#define ImageSize 1024 //dimension of each data 28x28=784
#define ClassCount 10 //dimension of output
#define ETA 0.005 //learning rate
#define LAM 0.1 //learning rate
#define FilterCount 4 //number of fiter
#define DimFil 13 //dimension of fiter
#define Pad 2 //dimension of padding
#define Pol 3 //pooling mesh
#define Str 2 //pooling stride

int num, num2, row, col, scan, rowC, colC, ep; // for repeating
void SubMat(double *, double *, double *, int, int);
void ColNorm(double *, int , int);
void Softmax(double *, int, int);
void dSoftmax(double *, double *, int, int);
void RandMat(double *, double, int, int);
void NormConst(double *, double *, int, int, int);

void ZeroPadding(double *, double *, int, int, int, int);
void Relu(double *, double *, int *, int, int, int);
void Convolution(double *, double *, double *, int, int, int, int, int);
void MaxPooling(double *, double *, int *, int, int, int, int, int);
void AllConn(double *, double *, double *, double *, int, int, int, int);

int main(void){
    FILE *fp;
    char *fname = "/Users/nomuraryoji/Documents/program/C/cifar-10-batches-bin/data_batch_1.bin";
    int PP[BatchSize]; //pick up number
    unsigned char buf[ImageSize], bufL[1];
    int label[BatchSize], ImSize, LbSize;
    long int offset;
    srand((unsigned int)time(NULL));
    int DimX = (int)sqrt(ImageSize);
    int Dzpd = DimX + Pad * 2;
    int DCon = Dzpd - DimFil + 1;
    int dimF = DimFil * DimFil;
    int dimC = DCon * DCon;
    int dimZP = Dzpd * Dzpd; //zero padding
    int DPol = (int)((DCon-Pol) / Str) + 1;
    int dimPL = DPol * DPol;
//    int CorCount = 0;
    
    /*allocate memory for each matrix*/
    double *X_In, *T_Out, *Y_Out, *D_Out;
    double *X_zpd, *H_cov, *B_cov, *X_cov, *dX_cov, *X_rel, *dX_rel, *X_pol, *dX_pol, *W_con, *B_con;
    int *Pos_rel, *W_pol;
    X_In = (double*)malloc(sizeof(double) * ImageSize * BatchSize); //Input data matrix [dimxBatchSize]
    T_Out = (double*)malloc(sizeof(double) * ClassCount * BatchSize);
    Y_Out = (double*)malloc(sizeof(double) * ClassCount * BatchSize);
    D_Out = (double*)malloc(sizeof(double) * ClassCount * BatchSize);
    
    X_zpd = (double*)malloc(sizeof(double) * dimZP * BatchSize);
    H_cov = (double*)malloc(sizeof(double) * dimF * FilterCount);
    B_cov = (double*)malloc(sizeof(double) * 1 * FilterCount);
    X_cov = (double*)malloc(sizeof(double) * dimC * BatchSize * FilterCount);
    dX_cov = (double*)malloc(sizeof(double) * dimC * BatchSize * FilterCount);
    X_rel = (double*)malloc(sizeof(double) * dimC * BatchSize * FilterCount);
    dX_rel = (double*)malloc(sizeof(double) * dimC * BatchSize * FilterCount);
    X_pol = (double*)malloc(sizeof(double) * dimPL * BatchSize * FilterCount);
    dX_pol = (double*)malloc(sizeof(double) * dimPL * BatchSize * FilterCount);
    
    Pos_rel = (int*)malloc(sizeof(int) * dimC * BatchSize * FilterCount);
    W_pol = (int*)malloc(sizeof(int) * dimPL * dimC * BatchSize * FilterCount);
    W_con = (double*)malloc(sizeof(double) * ClassCount * dimPL * FilterCount);
    B_con = (double*)malloc(sizeof(double) * ClassCount);
    
    RandMat(H_cov, 0.1, dimF, FilterCount);
    RandMat(B_cov, 0.1, 1, FilterCount);
    
    RandMat(W_con, 0.1, ClassCount * dimPL, FilterCount);
    RandMat(B_con, 0.1, 1, ClassCount);
    
    /*learning*/
    for(ep=0; ep<EPOCH; ep++){
        //read image values
        fp = fopen(fname, "rb");
        if(fp == NULL){
            printf("%sファイルが開けません¥n", fname);
            return -1;
        }
        //Batch calculation
        for(scan=0; scan<BatchSize; scan++){
            PP[scan] = rand() % DataCount + 1;
        }
        ImSize = 0;
        for(col=0; col<BatchSize; col++){
            offset = (PP[col] - 1) * (ImageSize * 3 + 1);
            fseek(fp, offset, SEEK_SET);
            LbSize += fread(bufL, sizeof(unsigned char), 1, fp);
            label[col] = bufL[0];
            
            offset = (PP[col] - 1) * (ImageSize * 3 + 1) + 1;
            fseek(fp, offset, SEEK_SET);
            ImSize += fread(buf, sizeof(unsigned char), ImageSize, fp);
            for(row=0 ; row<ImageSize; row++){
                X_In[row + col * ImageSize] = (double)buf[row]/255;
            }
        }
        fclose(fp);
        
        //initialize Y_Out 1-of-k
        for (col=0; col<BatchSize; col++){
            for (row=0; row<ClassCount; row++){
                if(row == label[col]){
                    T_Out[row + col * ClassCount] = 1;
                }else{
                    T_Out[row + col * ClassCount] = 0;
                }
            }
        }
        
        //forward propagation
        //first forward propagation
        ZeroPadding(X_In, X_zpd, DimX, Pad, 1, BatchSize);
        Convolution(X_zpd, H_cov, X_cov, Dzpd, DimFil, 1, FilterCount, BatchSize);
        Relu(X_cov, X_rel, Pos_rel, DCon, FilterCount, BatchSize);
        MaxPooling(X_rel, X_pol, W_pol, Pol, Str, DCon, FilterCount, BatchSize);
        
//        ColNorm(X_pol, BatchSize * FilterCount, dimPL);
        AllConn(X_pol, W_con, B_con, Y_Out, DPol, ClassCount, FilterCount, BatchSize);
        Softmax(Y_Out, ClassCount, BatchSize);
        SubMat(Y_Out, T_Out, D_Out, ClassCount, BatchSize);
        
        for (scan=0; scan<BatchSize; scan++){
            for (num=0; num<FilterCount; num++){
                for(row=0; row<dimPL; row++){
                    dX_pol[row + (num + scan * FilterCount) * dimPL] = 0.0;
                    
                }
            }
        }
        for (scan=0; scan<BatchSize; scan++){
            for (num=0; num<FilterCount; num++){
                for(row=0; row<dimPL; row++){
                    for(col=0; col<ClassCount; col++ ){
                        dX_pol[row + (num + scan * FilterCount) * dimPL]
                        += W_con[col + (row + num * dimPL) * ClassCount]
                        * D_Out[col + scan * ClassCount];
                    }
                }
            }
        }
        
        for (scan=0; scan<BatchSize; scan++){
            for (num=0; num<FilterCount; num++){
                for(row=0; row<dimC; row++){
                    dX_rel[row + (num + scan * FilterCount) * dimC] = 0.0;
                }
            }
        }
        for (scan=0; scan<BatchSize; scan++){
            for (num=0; num<FilterCount; num++){
                for(row=0; row<dimC; row++){
                    for(col=0; col<dimPL; col++ ){
                        dX_rel[row + (num + scan * FilterCount) * dimC]
                        += W_pol[col + (row + (num + scan * FilterCount) * dimC) * dimPL]
                        * dX_pol[col + (num + scan * FilterCount) * dimPL];
                    }
                }
            }
        }
        
        for (scan=0; scan<BatchSize; scan++){
            for (num=0; num<FilterCount; num++){
                for(row=0; row<dimC; row++){
                    dX_cov[row + (num + scan * FilterCount) * dimC] = 0.0;
                }
            }
        }
        for (scan=0; scan<BatchSize; scan++){
            for (num=0; num<FilterCount; num++){
                for(colC=0; colC<DCon; colC++ ){
                    for(rowC=0; rowC<DCon; rowC++ ){
                        dX_cov[rowC + colC * DCon + (num + scan * FilterCount) * dimC]
                        += Pos_rel[colC + rowC * DCon + (num + scan * FilterCount) * dimC]
                        * dX_rel[rowC + colC * DCon+ (num + scan * FilterCount) * dimC];
                    }
                }
            }
        }
        
        for (scan=0; scan<BatchSize; scan++){
            for (num=0; num<FilterCount; num++){
                for(row=0; row<DCon; row++){
                    for(col=0; col<DCon; col++ ){
                        for(rowC=0; rowC<DimFil; rowC++){
                            for(colC=0; colC<DimFil; colC++ ){
                                H_cov[rowC + colC * DimFil + num * dimF]
                                -= ETA * dX_cov[row + col * DCon + (num + scan * FilterCount) * dimC]
                                * X_zpd[(row + rowC) + (col + colC) * Dzpd + scan * dimZP] / BatchSize;
                            }
                        }
                    }
                }
            }
        }
        
        for (scan=0; scan<BatchSize; scan++){ //updating W_con
            for (num=0; num<FilterCount; num++){
                for(row=0; row<ClassCount; row++ ){
                    B_con[row + scan * ClassCount] -= ETA * D_Out[row + scan * ClassCount] / BatchSize;
                    for(col=0; col<dimPL; col++ ){
                        W_con[row + col * ClassCount + num * ClassCount * dimPL]
                        -= ETA * D_Out[row + scan * ClassCount] * X_pol[col + (num + scan * FilterCount) * dimPL] / BatchSize;
                    }
                }
            }
        }
    }
    

//checking image
    
    for(scan=0; scan<BatchSize; scan++){
        printf("Input batch No.%d\n", scan);
        for(row=0; row<DimX; row++ ){
            for(col=0; col<DimX; col++ ){
                printf("%02d", (int)(X_In[row + col * DimX + scan * ImageSize] * 25.5));
            }
            printf("\n");
        }
    }
    
    for(scan=0; scan<BatchSize; scan++){
        printf("Input batch No.%d\n", scan);
        for(row=0; row<Dzpd; row++ ){
            for(col=0; col<Dzpd; col++ ){
                printf("%02d", (int)(X_zpd[row + col * Dzpd + scan * dimZP]*25.5));
            }
            printf("\n");
        }
    }
    
    for(num=0; num<FilterCount; num++){
        printf("Conv filter No.%d\n", num);
        for(row=0; row<DimFil; row++ ){
            for(col=0; col<DimFil; col++ ){
                printf("%.4f", (H_cov[row + col * DimFil + num * dimF]));
            }
            printf("\n");
        }
    }
    
    for(scan=0; scan<BatchSize; scan++){
        for(num=0; num<FilterCount; num++){
            printf("convolution, batch:%d, filter:%d\n", scan, num);
            for(row=0; row<DCon; row++){
                for(col=0; col<DCon; col++){
                    printf("%.2f", (X_cov[row + col * DCon + (num + scan * FilterCount) * dimC]));
                }
                printf("\n");
            }
        }
    }
    
    for(scan=0; scan<BatchSize; scan++){
        for(num=0; num<FilterCount; num++){
            printf("Pos_rel:%d, filter:%d\n", scan, num);
            for(row=0; row<DCon; row++ ){
                for(col=0; col<DCon; col++ ){
                    printf("%02d", Pos_rel[row + col * DCon + (num + scan * FilterCount) * dimC]);
                }
                printf("\n");
            }
        }
    }
    
    for(scan=0; scan<BatchSize; scan++){
        for(num=0; num<FilterCount; num++){
            printf("pooling, batch*%d, filter:%d\n", scan, num);
            for(row=0; row<DPol; row++ ){
                for(col=0; col<DPol; col++ ){
                    printf("%.2f", (X_pol[row + col * DPol + (num + scan * FilterCount) * dimPL]));
                }
                printf("\n");
            }
        }
    }
    
    printf("WP_header\n");
    for(row=0; row<20; row++ ){
        for(col=0; col<20; col++ ){
            printf("%02d", W_pol[row + col * dimPL]);
        }
        printf("\n");
    }
    printf("W_Con_header\n");
    for(col=0; col<5; col++ ){
        for(row=0; row<ClassCount; row++ ){
            printf("%f ", W_con[row + col * ClassCount]);
        }
        printf("\n");
    }

    printf("\n");
    printf("\n");
    
    for(row=0; row<ClassCount; row++ ){
        for(col=0; col<BatchSize; col++ ){
            printf("%f %f ", Y_Out[row + col * ClassCount], T_Out[row + col * ClassCount]);
        }
        printf("\n");
    }
    printf("\n");
    


/*
//checking with test data
    int epoch = TestCount / BatchSize;
    //    char *fname = "/Users/nomuraryoji/Documents/program/C/cifar-10-batches-bin/data_batch_1.bin";
    for(ep=0; ep<epoch; ep++){
        for(scan=0; scan<BatchSize; scan++){
            PP[scan] = scan + ep * BatchSize;
        }
        fp = fopen(fname, "rb");
        if(fp == NULL){
            printf("%sファイルが開けません¥n", fname);
            return -1;
        }
        ImSize = 0;
        for(col=0; col<BatchSize; col++){
            offset = (PP[col] - 1) * (ImageSize * 3 + 1);
            fseek(fp, offset, SEEK_SET);
            LbSize += fread(bufL, sizeof(unsigned char), 1, fp);
            label[col] = bufL[0];
 
            offset = (PP[col] - 1) * (ImageSize * 3 + 1) + 1;
            fseek(fp, offset, SEEK_SET);
            ImSize += fread(buf, sizeof(unsigned char), ImageSize, fp);
            for(row=0 ; row<ImageSize; row++){
                X_In[row + col * ImageSize] = (double)buf[row]/255;
            }
        }
        fclose(fp);
        
        for (scan=0; scan<BatchSize; scan++){
            ZeroPadding(X_In, X_zpd, scan, DimX, Pad);
            //ZeroPadding(double *A, double *padA, int batch, int dimx, int pad)
        }
        for (scan=0; scan<BatchSize; scan++){
            for (num=0; num<FilterCount; num++){
                Convolution(X_zpd, H_cov, X_cov, Dzpd, DimFil, num, scan);
                //Convolution(double *X, double *H, double *ConvX, int dimx, int dimh, int numh, int batch)
            }
        }
        for (scan=0; scan<BatchSize; scan++){
            for (num=0; num<FilterCount; num++){
                Relu(X_cov, X_rel, Pos_rel, DCon, num, scan);
                //Relu(double *A, double *reluA, int *pos, int dimx, int numfil, int batch)
            }
        }
        for (scan=0; scan<BatchSize; scan++){
            for (num=0; num<FilterCount; num++){
                MaxPooling(X_rel, X_pol, W_pol, Pol, Str, DCon, num, scan);
                //MaxPooling(double *A, double *poolA, int *WP, int pl, int st, int dimx, int numfil, int batch)
            }
        }
        
        ColNorm(X_pol, BatchSize * FilterCount, dimPL);
        AllConn(X_pol, W_con, B_con, Y_Out, DPol, ClassCount, FilterCount, BatchSize);
        Softmax(Y_Out, ClassCount, BatchSize);
        
    }
    
    for (col=0; col<BatchSize; col++){
        for (row=0; row<ClassCount; row++){
            if(row == label[col]){
                T_Out[row + col * ClassCount] = 1;
            }else{
                T_Out[row + col * ClassCount] = 0;
            }
        }
    }

    for(row=0; row<ClassCount; row++ ){
        for(col=0; col<10; col++ ){
            printf("%f %d ", Y_Out[row + col * ClassCount], (int)T_Out[row + col * ClassCount]);
        }
        printf( "\n");
    }
*/
    printf( "\n");
    
    free(X_In);
    free(T_Out);
    free(Y_Out);
    free(D_Out);
    
    free(X_zpd);
    free(H_cov);
    free(B_cov);
    free(X_cov);
    free(dX_cov);
    free(X_rel);
    free(dX_rel);
    free(X_pol);
    free(dX_pol);
    free(W_con);
    free(B_con);
    free(Pos_rel);
    free(W_pol);
    
    return 0;
}

void SubMat(double *trn, double *out, double *Del, int I, int K){
    //Del[KxI]=Out[KxI]-Trn[KxI]
    for (row=0; row<I; row++){
        for (col=0; col<K; col++){
            Del[row + col * I] =trn[row + col * I] - out[row + col * I];
        }
    }
}

void ColNorm(double *A, int I, int K){
    for (col=0; col<I; col++){
        double colmax = 0.0;
        for (row=0; row<K; row++){
            if(fabs(A[row + col * K]) > colmax){
                colmax = fabs(A[row + col * K]);
            }
        }
        for (row=0; row<K; row++){
            if(colmax != 0.0){A[col * K + row] /= colmax;}
        }
    }
}

void Softmax(double *A, int K, int I){
    for (col=0; col<I; col++){
        double ColTotal = 0.0;
        for (row=0; row<K; row++){
            ColTotal += exp(A[col * K + row]);
        }
        for (row=0; row<K; row++){
            A[col * K + row] = exp(A[col * K + row]) / ColTotal;
        }
    }
}

void dSoftmax(double *A, double *dA, int I, int K){
    for (col=0; col<I; col++){
        for (row=0; row<K; row++){
            dA[col * K + row] = (A[col * K + row]) * (1 - A[col * K + row]);
        }
    }
}

void RandMat(double *A, double B, int Row, int Col){
    for (col=0; col<Col; col++){
        for (row=0; row<Row; row++){
            A[row + col * Row] = B * (((double)rand()+1.0)/((double)RAND_MAX+2.0)-0.5);
        }
    }
}

void ZeroPadding(double *A, double *padA, int dimx, int pad, int chan, int batch){
    int dimp = dimx + pad * 2;
    int dim2p = dimp * dimp;
    int dim2x = dimx * dimx;
    for (scan=0; scan<batch; scan++){
        for (num=0; num<chan; num++){
            for (row=0; row<dimp; row++){
                for (col=0; col<dimp; col++){
                    padA[row + col * dimp + (num + scan * chan) * dim2p] = 0.0;
                }
            }
        }
    }
    for (scan=0; scan<batch; scan++){
        for (num=0; num<chan; num++){
            for (row=pad; row<(dimp-pad); row++){
                for (col=pad; col<(dimp-pad); col++){
                    padA[row + col * dimp + (num + scan * chan) * dim2p] = A[(row-pad) + (col-pad) * dimx + (num + scan * chan) * dim2x];
                }
            }
        }
    }
}

void Relu(double *A, double *reluA, int *pos, int dimx, int chan, int batch){
    int dim2x = dimx * dimx;
    for (scan=0; scan<batch; scan++){
        for (num=0; num<chan; num++){
            for (col=0; col<dimx; col++){
                for (row=0; row<dimx; row++){
                    if(A[col * dimx + row + (num + scan * chan) * dim2x]>0){
                        reluA[row + col * dimx + (num + scan * chan) * dim2x]
                        = A[row + col * dimx + (num + scan * chan) * dim2x];
                        pos[row + col * dimx + (num + scan * chan) * dim2x] = 1;
                    }else{
                        reluA[row + col * dimx + (num + scan * chan) * dim2x] = 0.0;
                        pos[row + col * dimx + (num + scan * chan) * dim2x] = 0;
                    }
                }
            }
        }
    }
}

void Convolution(double *X, double *H, double *ConvX, int dimx, int dimh, int chan, int chan2, int batch){
    int dim2x = dimx * dimx;
    int dim2h = dimh * dimh;
    int dimc = dimx - dimh + 1;
    int dim2c = dimc * dimc;
    
    for (scan=0; scan<batch; scan++){
        for (num2=0; num2<chan2; num2++){
            for (col=0; col<dimc; col++){
                for (row=0; row<dimc; row++){
                    ConvX[row + col * dimc + (num2 + scan * chan2) * dim2c] = 0.0;
                }
            }
        }
    }
    for (scan=0; scan<batch; scan++){
        for (num=0; num<chan; num++){
            for (num2=0; num2<chan2; num2++){
                for (col=0; col<dimc; col++){
                    for (row=0; row<dimc; row++){
                        for (colC=0; colC<dimh; colC++){
                            for (rowC=0; rowC<dimh; rowC++){
                                ConvX[row + col * dimc + (num2 + scan * chan2) * dim2c]
                                += X[(row + rowC) + (col + colC) * dimx + (num + scan * chan) * dim2x]
                                * H[rowC + colC * dimh + num2 * dim2h];
                            }
                        }
                    }
                }
            }
        }
    }
}

void MaxPooling(double *A, double *poolA, int *WP, int pl, int st, int dimx, int chan, int batch){
    int dimpl = (int)((dimx-pl) / st) + 1;
    int dim2pl = dimpl * dimpl;
    int dim2x = dimx * dimx;
    int maxc, maxr;
    double bufp;
    for (scan=0; scan<batch; scan++){
        for (num=0; num<chan; num++){
            for (col=0; col<dimpl; col++){
                for (row=0; row<dimpl; row++){
                    poolA[row + col * dimpl + (num + scan * chan) * dim2pl] = 0.0;
                }
            }
        }
    }
    for (scan=0; scan<batch; scan++){
        for (num=0; num<chan; num++){
            for (col=0; col<dim2x; col++){
                for (row=0; row<dim2pl; row++){
                    WP[row + col * dim2pl +  (num + scan * chan) * dim2pl * dim2x] = 0;
                }
            }
        }
    }
    for (scan=0; scan<batch; scan++){
        for (num=0; num<chan; num++){
            for (col=0; col<dimpl; col++){
                for (row=0; row<dimpl; row++){
                    bufp = A[(row*st) + (col*st) * dimx + (num + scan * chan)  * dim2x];
                    maxc = 0;
                    maxr = 0;
                    for (colC=0; colC<pl; colC++){
                        for (rowC=0; rowC<pl; rowC++){
                            if(A[(row*st + rowC) + (col*st + colC) * dimx + (num + scan * chan)  * dim2x] > bufp){
                                bufp = A[(row*st + rowC) + (col*st + colC) * dimx + (num + scan * chan) * dim2x];
                                maxc = colC;
                                maxr = rowC;
                            }
                        }
                    }
                    if(bufp>0){
                        poolA[row + col * dimpl + (num + scan * chan) * dim2pl] = bufp;
                    }else{
                        poolA[row + col * dimpl + (num + scan * chan) * dim2pl] = 0.0;
                    }
                    WP[(row + col * dimpl) + ((row*st + maxr) + (col*st + maxc) * dimx) * dim2pl +  (num + scan * chan) * dim2pl * dim2x] = 1;
                }
            }
        }
    }
}

void AllConn(double *x_pool, double *w_all, double *b_all, double *y_out, int dimpl, int d_out, int Numfil, int Batch){
    //C[KxI]=B[KxJ]*A[JxI]+D[KxI]
    int dim2pl = dimpl * dimpl;
    for (row=0; row<d_out; row++){
        for (col=0; col<Batch; col++){
            y_out[col * d_out + row] = 0.0; //initialize
        }
    }
    for (col=0; col<Batch; col++){
        for (num=0; num<Numfil; num++){
            for (row=0; row<d_out; row++){
                y_out[row + col * d_out] += b_all[row];
                for (scan=0; scan<dim2pl; scan++){
                    y_out[row + col * d_out]
                    += w_all[row + scan * d_out] * x_pool[scan + num * dim2pl + col * FilterCount * dim2pl];
                }
            }
        }
    }
}

