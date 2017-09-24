//
//  main.c
//  AEC-CIFAR
//
//  Created by Nomura Ryoji on 2016/09/15.
//  Copyright © 2016年 Nomura Ryoji. All rights reserved.
// test git

#include <stdio.h>
#include <stdlib.h> /* rand, malloc */
#include <time.h>
#include <math.h>

#define BatchSize 10//batch size for each epoch
#define HeaderSize 54 //header size of bmp file
#define DataCount 10000 //data size of learning
#define TestCount 10000 //data size of learning
#define Size 32 //length of data 32x32=1024
#define ClassCount 10 //dimension of output

#define EPOCH 20000 //epoch of learning
#define DHid 700 //dimension of hidden layer
#define Chan 3
#define ETA 0.001 //learning rate
#define LAM 0.00001 //learning rate

int num, num2, row, col, scan, rowC, colC, ep; // for repeating

void ReadCifar(unsigned char *, int *, int, int); //(Image, Label, BatchSize, size)
void WriteBmp(unsigned char *, unsigned char *, int, int); //(header, data, size), filename:test.bmp
void WriteHidden(double *, int, int); //(header, data, size), filename:test.bmp
void AutoEncoder(double *, double *, double *, double *, double *, double *, int, int, int, int);
//(input, output, BatchSize * channel, imagesize, hiddensize)

void MulMatAddBias(double *, double *, double *, double *, int, int, int);
void SubMat(double *, double *, double *, int, int);
void ColNorm(double *, int , int);
void TANH(double *, int , int);
void dTANH(double *, double *, int , int);
void Softmax(double *, int, int);
void dSoftmax(double *, double *, int, int);
void MatT(double *, double *, int, int); //Trans W
void Delta(double *, double *, double *, double *, int, int, int); //f' o W*Delta
void UpdateW(double *, double *, double *, double, int, int, int, int);
void UpdateB(double *, double *, double, int, int, int);

int main(void){
    int ImageSize = Size * Size * Chan;
    unsigned char Image[ImageSize * BatchSize], Image2[ImageSize * BatchSize];
    double *X_In, *X_Out, *W_In, *B_In, *W_Out, *B_Out;
    X_In = (double*)malloc(sizeof(double) * ImageSize * BatchSize); //Input data matrix [dimxBS]
    X_Out = (double*)malloc(sizeof(double) * ImageSize * BatchSize); //Input data matrix [dimxBS]
    B_In = (double*)malloc(sizeof(double) * DHid);
    W_In = (double*)malloc(sizeof(double) * ImageSize * DHid);
    B_Out = (double*)malloc(sizeof(double) * ImageSize);
    W_Out = (double*)malloc(sizeof(double) * ImageSize * DHid);

    int Label[BatchSize];

    //initialize W
    for (row=0; row<DHid; row++){
        for (col=0; col<ImageSize; col++){
            W_In[col * DHid + row] = 0.01 * ((double)rand()/RAND_MAX - 0.5);
        }
    }
    for (row=0; row<DHid; row++){
        B_In[row] = 0.01 * ((double)rand()/RAND_MAX - 0.5);
    }
    for (row=0; row<ImageSize; row++){
        B_Out[row] = 0.01 * ((double)rand()/RAND_MAX - 0.5);
    }


    for(ep=0; ep<EPOCH; ep++){
        ReadCifar(Image, Label, BatchSize, Size);

        for(scan=0; scan<(ImageSize * BatchSize); scan++){
            X_In[scan] = (double)Image[scan] / 255;
        }

        AutoEncoder(X_In, X_Out, W_In, B_In, W_Out, B_Out, BatchSize, Size, DHid, Chan);
    }

    for(num=0; num<BatchSize; num++){
        for(scan=0; scan<(ImageSize); scan++){
            Image2[scan + num * ImageSize] = X_Out[scan + num * ImageSize] * 255;
        }
    }

    WriteBmp(Image, Image2, Size, BatchSize);
    WriteHidden(W_In, Size, 10);

    return 0;
}

void AutoEncoder(double *input, double *output, double *W_In, double *B_In, double *W_Out, double *B_Out, int batchsize, int size, int w_size, int chan){
    double *XT_In, *U_Hid, *dU_Hid, *UT_Hid, *dY_Out, *T_Out, *Dif_Out, *Dif_Hid; //variables
    double *WT_In, *dW_In, *WT_Out, *dW_Out; //parameters
    int imagesize = size * size * chan;
    XT_In = (double*)malloc(sizeof(double) * imagesize * batchsize); //Input data matrix [dimxBS]

    WT_In = (double*)malloc(sizeof(double) * imagesize * w_size);
    dW_In = (double*)malloc(sizeof(double) * imagesize * w_size);
    U_Hid = (double*)malloc(sizeof(double) * batchsize * w_size);
    UT_Hid = (double*)malloc(sizeof(double) * batchsize * w_size);
    dU_Hid = (double*)malloc(sizeof(double) * batchsize * w_size);
    WT_Out = (double*)malloc(sizeof(double) * imagesize * w_size);
    dW_Out = (double*)malloc(sizeof(double) * imagesize * w_size);
    dY_Out = (double*)malloc(sizeof(double) * imagesize * batchsize);
    T_Out = (double*)malloc(sizeof(double) * imagesize * batchsize);
    Dif_Out = (double*)malloc(sizeof(double) * imagesize * batchsize);
    Dif_Hid = (double*)malloc(sizeof(double) * batchsize * w_size);

    //learning
    //forward propagation
    MatT(W_In, W_Out, imagesize, w_size);
    MulMatAddBias(input, W_In, B_In, U_Hid, batchsize, imagesize, w_size); // linear connection In & Hidden, WX + B
    TANH(U_Hid, batchsize, w_size); //activation of hidden layer
    dTANH(U_Hid, dU_Hid, batchsize, w_size); //activation of hidden layer
    MulMatAddBias(U_Hid, W_Out, B_Out, output, batchsize, w_size, imagesize); // linear connection Hidden & Out, WX + B
    TANH(output, batchsize, imagesize); //activation of Out layer
    dTANH(output, dY_Out, batchsize, imagesize); //div of activation of Out layer

    //back propagation
    SubMat(output, input, Dif_Out, batchsize, imagesize); // diff of output
    MatT(U_Hid, UT_Hid, batchsize, w_size);
    UpdateB(Dif_Out, B_Out, ETA, batchsize, batchsize, imagesize);

    MatT(W_Out, WT_Out, w_size, imagesize);
    Delta(Dif_Out, WT_Out, dU_Hid, Dif_Hid, batchsize, imagesize, w_size); //delta of Hidden layer

    MatT(input, XT_In, batchsize, imagesize);
    UpdateW(XT_In, Dif_Hid, W_In, ETA, batchsize, imagesize, batchsize, w_size); //update W of Input layer
    UpdateB(Dif_Hid, B_In, ETA, batchsize, batchsize, w_size);

    free(XT_In);
    free(U_Hid);
    free(dU_Hid);
    free(UT_Hid);
    free(dY_Out);
    free(T_Out);
    free(Dif_Out);
    free(Dif_Hid);
    free(WT_In);
    free(dW_In);
    free(WT_Out);
    free(dW_Out);
}

void ReadCifar(unsigned char *image, int *label, int batch, int size){
    FILE *fp;

//    char *fname = "/Users/nomuraryoji/Documents/program/C/cifar-10-batches-bin/train-images-idx3-ubyte";
    char *fname = "cifar-10-batches-bin/data_batch_1.bin";
    int imagesize = size * size, pic_nmbr[batch];
    unsigned char buf[imagesize*3], bufL[1];
    long int offset;

    srand((unsigned int)time(NULL)); // initialize random

    //read image values
    fp = fopen(fname, "rb");
    if(fp == NULL){
        printf("%sファイルが開けません¥n", fname);
    }

    for(scan=0; scan<batch; scan++){
        pic_nmbr[scan] = rand() % DataCount + 1;
        offset = (pic_nmbr[scan] - 1) * (imagesize * 3 + 1);
        fseek(fp, offset, SEEK_SET);
        fread(bufL, sizeof(unsigned char), 1, fp);
        label[scan] = bufL[0];

        offset = (pic_nmbr[scan] - 1) * (imagesize * 3 + 1) + 1;
        fseek(fp, offset, SEEK_SET);
        fread(buf, sizeof(unsigned char), imagesize * 3, fp);

        for(num=0; num<(imagesize*3); num++){
            image[num + imagesize * 3 * scan] = buf[num];
        }
    }
    fclose(fp);
}

void WriteBmp(unsigned char *buf, unsigned char *out, int size, int cnt){ //for CIFAR image
    FILE *fph;
    FILE *fpw;
    char *fname_h = "cifar-10-batches-bin/Header";
    char *fname_w = "cifar-10-batches-bin/test.bmp";
    int offsetR, offsetG, offsetB;
    unsigned char buf2[size * size * 2 * 4];
    char bufHeader[HeaderSize];

    //BMP header values
    fph = fopen(fname_h, "rb");
    if(fph == NULL){
        printf("%sファイルが開けません¥n", fname_h);
    }

    fread(bufHeader, sizeof(unsigned char), HeaderSize, fph);
    fclose(fph);

    bufHeader[18] = size * 2;
    if(size*cnt >= 256){
        bufHeader[22] = -(size * cnt % 256);
        bufHeader[23] = -(size * cnt / 256) - 1;
    }else if(size*cnt < 256){
        bufHeader[22] = -size * cnt;
    }

    //write image values
    fpw = fopen(fname_w, "wb");
    if(fpw == NULL){
        printf("書込用 %sファイルが開けません¥n", fname_w);
    }

    fwrite(bufHeader, sizeof(unsigned char), HeaderSize, fpw);

    for(num=0; num<cnt; num++){
        for(col=0; col<size; col++){
            for(row=0; row<size; row++){
                offsetR = col + row * size + size * size * 0 + num * size * size * 3;
                offsetG = col + row * size + size * size * 1 + num * size * size * 3;
                offsetB = col + row * size + size * size * 2 + num * size * size * 3;

                buf2[(col + row * size * 2) * 4 + 0] = buf[offsetB];
                buf2[(col + row * size * 2) * 4 + 1] = buf[offsetG];
                buf2[(col + row * size * 2) * 4 + 2] = buf[offsetR];
                buf2[(col + row * size * 2) * 4 + 3] = 0;

                buf2[(col + size + row * size * 2) * 4 + 0] = out[offsetB];
                buf2[(col + size + row * size * 2) * 4 + 1] = out[offsetG];
                buf2[(col + size + row * size * 2) * 4 + 2] = out[offsetR];
                buf2[(col + size + row * size * 2) * 4 + 3] = 0;
            }
        }
        fwrite(buf2, sizeof(unsigned char), size * size * 2 * 4, fpw);
    }

    fclose(fpw);
}

void WriteHidden(double *hid, int size, int cnt){ //for CIFAR image
    FILE *fph;
    FILE *fpw;
    char *fname_h = "cifar-10-batches-bin/Header";
    char *fname_w = "cifar-10-batches-bin/hidden.bmp";
    int offsetR, offsetG, offsetB;
    unsigned char buf2[size * size * 4];
    char bufHeader[HeaderSize];

    //BMP header values
    fph = fopen(fname_h, "rb");
    if(fph == NULL){
        printf("%sファイルが開けません¥n", fname_h);
    }

    fread(bufHeader, sizeof(unsigned char), HeaderSize, fph);
    fclose(fph);

    if(size*cnt >= 256){
        bufHeader[22] = -(size * cnt % 256);
        bufHeader[23] = -(size * cnt / 256) - 1;
    }else if(size*cnt < 256){
        bufHeader[22] = -size * cnt;
    }

    //write image values
    fpw = fopen(fname_w, "wb");
    if(fpw == NULL){
        printf("書込用 %sファイルが開けません¥n", fname_w);
    }

    fwrite(bufHeader, sizeof(unsigned char), HeaderSize, fpw);

    for(num=0; num<cnt; num++){
        for(scan=0; scan<(size * size); scan++){
            offsetR = scan + size * size * 0 + num * size * size * 3;
            offsetG = scan + size * size * 1 + num * size * size * 3;
            offsetB = scan + size * size * 2 + num * size * size * 3;
            buf2[scan * 4 + 0] = hid[offsetB] * 255;
            buf2[scan * 4 + 1] = hid[offsetG] * 255;
            buf2[scan * 4 + 2] = hid[offsetR] * 255;
            buf2[scan * 4 + 3] = 0;
        }
        fwrite(buf2, sizeof(unsigned char), size * size * 4, fpw);
    }

    fclose(fpw);
}


void MulMatAddBias(double *A, double *B, double *Bi, double *BA_Bi, int I, int J, int K){
    //C[KxI]=B[KxJ]*A[JxI]+D[KxI]
    for (row=0; row<K; row++){
        for (col=0; col<I; col++){
            BA_Bi[col * K + row] = Bi[row]; //initialize
        }
    }
    for (row=0; row<K; row++){
        for (col=0; col<I; col++){
            for (scan=0; scan<J; scan++){
                BA_Bi[col * K + row] += B[scan * K + row] * A[col * J + scan];
            }
        }
    }
}

void MulMat(double *A, double *B, double *BA, int I, int J, int K){
    //C[KxI]=B[KxJ]*A[JxI]+D[KxI]
    for (row=0; row<K; row++){
        for (col=0; col<I; col++){
            BA[col * K + row] = 0.0; //initialize
        }
    }
    for (row=0; row<K; row++){
        for (col=0; col<I; col++){
            for (scan=0; scan<J; scan++){
                BA[col * K + row] += B[scan * K + row] * A[col * J + scan];
            }
        }
    }
}

void MatT(double *A, double *AT, int I, int J){
    //A[JxI] -> AT[IxJ]
    for (row=0; row<J; row++){
        for (col=0; col<I; col++){
            AT[col + I * row] = A[col * J + row];
        }
    }
}

void SubMat(double *Trn, double *Out, double *Del, int I, int K){
    //Del[KxI]=Out[KxI]-Trn[KxI]
    for (row=0; row<K; row++){
        for (col=0; col<I; col++){
            Del[col * K + row] = Trn[col * K + row] - Out[col * K + row];
        }
    }
}

void ColNorm(double *A, int I, int K){
    for (col=0; col<I; col++){
        double colmax = 1.0; //local variable???
        for (row=0; row<K; row++){
            if(fabs(A[col * K + row]) > colmax){
                colmax = fabs(A[col * K + row]);
            }
        }
        for (row=0; row<K; row++){
            A[col * K + row] /= colmax;
        }
    }
}

void TANH(double *A, int I, int K){
    for (col=0; col<I; col++){
        for (row=0; row<K; row++){
            A[col * K + row] = tanh(A[col * K + row]);
        }
    }
}

void dTANH(double *A, double *dA, int I, int K){
    for (col=0; col<I; col++){
        for (row=0; row<K; row++){
            dA[col * K + row] = 1 - tanh(A[col * K + row]) * tanh(A[col * K + row]);
        }
    }
}

void Softmax(double *A, int I, int K){
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

void Delta(double *D, double *Wt, double *df, double *Del, int I, int J, int K){
    double *WtD; //BA[KxI]
    WtD = (double*)malloc(sizeof(double) * K * I);
    MulMat(D, Wt, WtD, I, J, K); //WD[K][I] = W[KxJ] * Delta[JxI], W is Wt
    for (col=0; col<I; col++){
        for (row=0; row<K; row++){
            Del[col * K + row] = df[col * K + row] * WtD[col * K + row]; //mul by element
        }
    }
    free(WtD);
}

void UpdateW(double *Z, double *D, double *W, double eta, int bs, int I, int J, int K){
    double *DZ; //BA[KxI] = B[KxJ]*A[JxI]
    DZ = (double*)malloc(sizeof(double) * K * I);
    MulMat(Z, D, DZ, I, J, K); //BA[KxI] = B[KxJ]*A[JxI], W is Wt
    for (col=0; col<I; col++){
        for (row=0; row<K; row++){
            W[col * K + row] -= eta * (DZ[col * K + row] / bs + LAM * W[col * K + row]);
        }
    }
    free(DZ);
}

void UpdateB(double *A, double *Bi, double eta, int bs, int I, int J){
    for (col=0; col<I; col++){
        for (row=0; row<J; row++){
            Bi[row] -= eta * (A[col * J + row] / bs) ;
        }
    }
}
