set.seed(123)
pacman::p_load("dplyr")

# sigmoid

sigmoid <- function(x){
  
  1/(1+exp(-x))
  
}
# id
id <- function(x){x}

#softmax
softmax <- function(z){
  
  exp(z) / sum(exp(z))
  
}

AF <- list(id, sigmoid, tanh, softmax)

Did <- function(x){1}

Dsig <- function(x){exp(-x) / ((1+exp(-x))^2)}

Dtanh <- function(x){1-(tanh(x)^2)}
D.softmax <- function(z){
  
  n <- length(z)
  S <- softmax(z)
  DS <- matrix(NA, nrow = n, ncol = n)
  
  for(i in 1:n){
    for(j in 1:n){
      if(i == j){
        D.i.j <- S[i]*(1-S[j])
      }else{
        D.i.j <- -S[i]*S[j]
      }
      DS[i,j] <- D.i.j
    }}
  return(DS)
}

DAF <- list(Did,Dsig,Dtanh, D.softmax)

###Forward Pass

fwd.pass <- function(input, parameters, AF.chosen){
  
  W1 = parameters$W[[1]]
  W2 = parameters$W[[2]]
  
  B1 = parameters$B[[1]]
  B2 = parameters$B[[2]]
  
  AF1 = AF[[AF.chosen[1]]]
  AF2 = AF[[AF.chosen[2]]]
  AF3 = AF[[AF.chosen[3]]]
  
  Z1 = input
  A1 = AF1(Z1)
  Z2 = t(W1) %*% A1 + B1
  A2 = AF2(Z2)
  Z3 = t(W2) %*% A2 + B2
  A3 = AF3(Z3)
  
  return(list("Z" = list(Z1,Z2,Z3), "A" = list(A1,A2,A3)))
  
}


### Results
# *********************************************************************
# INITIALIZE PARAMETERS ************************************************
# *********************************************************************
initialize.parameters <- function(N,a){
  
  W1 <- matrix(runif(N[1]*N[2],-a,a),nrow = N[1])
  W2 <- matrix(runif(N[2]*N[3],-a,a),nrow = N[2])
  B1 <- matrix(runif(N[2],-a,a),nrow = N[2])
  B2 <- matrix(runif(N[3],-a,a),nrow = N[3])
  
  return(list("W" = list(W1,W2), "B" = list(B1,B2)))
  
}

# *********************************************************************
# LOSS FUNCTIONS ************************************************
# *********************************************************************
mse <- function(A,K){
  
  return( mean((A-K)^2) )
  
}
D.mse <- function(A,K){
  
  return(2/length(A)*(A-K))
  
}
cross.entropy <- function(A, K){
  
  return(sum( K * log(1/A) ))
  
}

D.cross.entropy <- function(A, K){
  
  return(-K/A)
  
}

LF <- list(mse, cross.entropy)
DLF <- list(D.mse, D.cross.entropy)

# *********************************************************************
# BACK PROPAGATION ************************************************
# *********************************************************************

back.propagation <- function(fwd.pass, 
                             parameters, 
                             AF.chosen, 
                             LF.chosen, 
                             known.input.result)
{
  # the loss function
  L  <- LF[[LF.chosen]]
  DL <- DLF[[LF.chosen]]
  
  # pre-active values
  Z3 <- fwd.pass$Z[[3]]
  Z2 <- fwd.pass$Z[[2]]
  Z1 <- fwd.pass$Z[[1]]
  
  # active values
  A3 <- fwd.pass$A[[3]]
  A2 <- fwd.pass$A[[2]]
  A1 <- fwd.pass$A[[1]]
  # net parameters
  W2 <- parameters$W[[2]]
  W1 <- parameters$W[[1]]
  B2 <- parameters$B[[2]]
  B1 <- parameters$B[[1]]
  
  # activation function derivatives
  D.AF3 <- DAF[[ AF.chosen[3] ]]
  D.AF2 <- DAF[[ AF.chosen[2] ]]
  
  # derivatives of loss w.r.t. neuron values Z:
  # if softmax is chosen:
  
  if(AF.chosen[[3]] == 4){
    DZ3 <- D.AF3( Z3 ) %*% DL(A3, known.input.result)
  }else{
    DZ3 <- DL(A3, known.input.result) * D.AF3( Z3 ) 
  }
  DZ2 <- (W2 %*% DZ3) * D.AF2( Z2 )
  
  # derivatives of loss w.r.t weights:
  DW2 <- A2 %*% t(DZ3)
  DW1 <- A1 %*% t(DZ2)
  
  # derivatives of loss w.r.t biases:
  DB2 <- DZ3
  DB1 <- DZ2
  
  return( list("DW" = list(DW1, DW2), "DB" = list(DB1, DB2) ) )
  
}

# *********************************************************************
# UPDATE PARAMETERS ************************************************
# *********************************************************************

updateParameters <- function(parameters, gradients, step){
  
  DW2 <- gradients$DW[[2]]  
  DW1 <- gradients$DW[[1]]
  DB2 <- gradients$DB[[2]]
  DB1 <- gradients$DB[[1]]
  
  W2 <- parameters$W[[2]]
  W1 <- parameters$W[[1]]
  B2 <- parameters$B[[2]]
  B1 <- parameters$B[[1]]
  
  W1 <- W1 - step*DW1
  W2 <- W2 - step*DW2
  B2 <- B2 - step*DB2
  B1 <- B1 - step*DB1
  
  return(list("W"=list(W1, W2), "B"=list(B1, B2)))
  
}

# *********************************************************************
# ONE HOT ************************************************
# *********************************************************************

one.hot.convert <- function(types, type){
  
  position <- which(types == type)
  oneHot   <- rep(0,length(types))
  oneHot[position] <- 1
  
  return(oneHot)
  
}
# *********************************************************************
# Test Net ************************************************
# *********************************************************************

testNet <- function(data){
  
  
  accuracy <- 0
  
  for(row in 1:nrow(data)){
    
    input      <- matrix(as.numeric(data[row,1:M]), nrow = M)
    known      <- matrix(one.hot.convert(types, pull(data, classCol)[row] ), nrow = numberOfTypes) 
    fPass      <- fwd.pass(input, parameters, AF.chosen)
    prediction <- round(fPass$A[[3]],0)
    prediction
    known
    if(all(known == prediction)){
      
      accuracy <- accuracy + 1
      
    }
  }
  
  return(accuracy/nrow(data)*100)
  
}

# *********************************************************************
# Running the NN ************************************************
# *********************************************************************

data <- read.csv('heart.csv')
data

sum(is.na(data))

str(data)

scale             <- 1
M                 <- ncol(data) - 1 
setDF(data)
if(scale == 1){
  
  data[,1:M] <- scale(data[,1:M])
  
}

#DATA PREP
trainingProportion <- 0.8
numberOfRows       <- nrow(data)
data               <- data[sample(1:numberOfRows),]
trainingRows       <- floor(trainingProportion*numberOfRows)
trainingData       <- data[1:trainingRows,]
testingData        <- data[((trainingRows+1):numberOfRows), ]

#NN PREP
classCol            <- M + 1
numberOfTypes       <- length( unique( pull(data,classCol) ) ) 
types               <- sort(unique(pull(data,classCol)))
N                   <- c(M,2*M,numberOfTypes)
AF.chosen           <- c(1,3,4)
lossFunctionChosen  <- 2
loss                <- LF[[lossFunctionChosen]]
step                <- 0.01
epochs              <- 100
N
M
# *********************************************************************
# RUN NN ************************************************
# *********************************************************************
plot(NA ,xlim = c(0,epochs), ylim = c(0,1), xlab="epochs", ylab="loss")

parameters    <- initialize.parameters(N,0.5)

for(e in 1:epochs){
  
  losses <- c()
  
  for(row in 1:nrow(trainingData)){
    
    
    input      <- matrix(as.numeric(trainingData[row,1:M]), nrow = M)
    known      <- matrix(one.hot.convert(types, pull(trainingData, classCol)[row] ), nrow = numberOfTypes) 
    fPass      <- fwd.pass(input, parameters, AF.chosen)
    grads      <- back.propagation(fPass, parameters, AF.chosen, lossFunctionChosen, known)
    parameters <- updateParameters(parameters, grads, step)
    
    losses <- c(losses,loss(fPass$A[[3]], known))
    
  }
  
  epochLoss <- mean(losses)
  
  points(e,epochLoss)
  
}
testNet(trainingData)
testNet(testingData)
