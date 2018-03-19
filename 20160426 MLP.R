### Preparation for TrainData ###
# TrainData include variables in rows, last value of each row shows class (0,1,2,,,9)
# Including bias in first column
#20160427 H:1000 sum(RA)/10 = 0.84
#ユーザ   システム       経過  
#2666.208   1590.287   4356.039 

#20160427 H:500 sum(RA)/10 = 0.92
#ユーザ   システム       経過  
#1319.020    827.893   2211.599 


N <- 60000 # number of sample
#train_norm <- t(scale(t(train$x[1:N,])))
train_norm <-train$x[1:N,] / max(train$x[1:N,])
#train_norm <-train$x[1:N,]
TrainData <- cbind(c(rep(1,N)),matrix(c(train_norm,train$y[1:N]),nrow=N))

### functions ###
#SoftMax
Sofm <- function(x){
  y <- exp(x)
  return(t(t(y)/apply(y,2,sum)))
}
# derivation of Softmax function
DSofm <- function(x){
  return(Sofm(x)*(1-Sofm(x)))
}
# derivation of tanh function
Dtanh <- function(x){
  return(1-tanh(x)^2)
}
# KL divergence
KL <- function(x,y){
  return(x*log(x/y)+(1-x)*log((1-x)/(1-y)))
}
# derivation of KL div
DKL <- function(x,y){
  return(-x/y+(1-x)/(1-y))
}

### initial parameters ###
eta <- 0.01
lmd <- 10e-8
numX <- length(TrainData[1,]) - 1 # number of variables in one sample
numH <- 500 # dim of hidden layer
numC <- 10 # dim of class
M <- 1 # mini batch sampling
WH <- 0.01 * matrix(c(rnorm(numH*numX)),nrow=numH) # initial W for Hidden layer. include bias
WC <- 0.01 * matrix(c(rnorm(numC*(numH+1))),nrow=numC) # initial W for Classification. include bias

### arrange traindata ###
# arrenge variables, row shows one set
TrainX <- t(matrix(c(TrainData[,1:numX]),nrow=N))

# row shows one set of class (1-of-K)
TrainD <- matrix(0,ncol=N,nrow=10) 
for(i in 1:N) {
  TrainD[TrainData[i,numX+1]+1,i] <- 1
} 

### learning ###
t<-proc.time()
for(i in 1:(N/M)){
  p <- c(floor(runif(M, min=0, max=N+1)))
  UH <- rbind(1,tanh(WH%*%TrainX[,p[1:M]]))
#  UH <- rbind(1,UH) # add bias
  OutY <- Sofm(WC%*%UH)
  DeltaC <- OutY - TrainD[,p[1:M]] # [C,1]
  WC <- WC - eta * matrix(DeltaC,nrow=numC, ncol=numH+1)*(t(matrix(UH,nrow=numH+1,ncol=numC))+lmd * WC) / M
  # [C, H+1]
  DeltaH <- Dtanh(UH)*((t(WC)%*%DeltaC)) #[H+1,1]
  DeltaH <- DeltaH[-1,] #[H,1]
  WH <- WH - eta * matrix(DeltaH,nrow=numH,ncol=numX)*(t(matrix(TrainX[,p[1:M]],nrow=numX, ncol=numH)) + lmd*WH) / M
  #[H, X]
#  break
}
tt <- proc.time()-t

### testing ### 
count <- matrix(0,ncol=10, nrow=10)
for(i in 1:N){
#  break
  UH <- tanh(WH%*%TrainX[,i])
  UH <- rbind(1.0,UH)
  OutY <- Sofm(WC%*%UH)
  count[which.max(OutY),which.max(TrainD[,i])] <- count[which.max(OutY),which.max(TrainD[,i])] + 1
}

RA <- c(rep(0.0,10))
for(i in 1:10){
  RA[i] <- count[i,i] /sum(count[,i])
}

#image(WH)
#image(WC)
#show_digit(WH[10,2:785])
