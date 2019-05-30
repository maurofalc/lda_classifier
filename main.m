%% Reconhecimento de Padr�es 2019.1
% Francisco Mauro Falc�o Matias Filho (374876)
% Trabalho III

clc;
close all;
clear;

%% Carregamento dos dados do dataset
data = csvread('wine.csv');

% normaliza��o dos dados de acordo com os requisitos do LDA
%data = zscore(data(:,2:13));

%% Defini��o de constantes e vari�veis auxiliares
TAM = length(data); % quantidade de amostras (178)
K = 10; % constante do K-fold
ACERTOS = 0; % quabtidade de acertos do k-fold

%% Separa��o dos dados da base entre classes
X1 = data(1:59,2:13);
X2 = data(60:130,2:13);
X3 = data(131:178,2:13);

N1 = length(X1); % tamanho das amostras da classe 1
N2 = length(X2); % tamanho das amostras da classe 2
N3 = length(X3); % tamanho das amostras da classe 3

%% C�lculo do vetor m�dia dos atributos para cada classe
u1 = mean(X1);
u2 = mean(X2);
u3 = mean(X3);
u = (u1 + u2 + u3)/3;

%% C�lculo das matrizes de covari�ncia para cada classe
S1 = cov(X1);
S2 = cov(X2);
S3 = cov(X3);

%% C�lculo da matriz de dispers�o intra-classe multiclasse
Sw = S1 + S2 + S3;

%% C�lculo das matrizes de dispers�o inter-classe multiclasse
SB1 = N1*(u1 - u)*(u1 - u).';
SB2 = N2*(u2 - u)*(u2 - u).';
SB3 = N3*(u3 - u)*(u3 - u).';
SB = SB1 + SB2 + SB3;

%% Desomposi��o em autovalores e autovetores 
[V,D] = eig(inv(Sw)*SB);

%% Vetores de proje��o
w1 = V(:,1);
w2 = V(:,3);

%% Amostras projetadas
y1(1,:) = w1.'*X1;
y2(1,:)  = w1.'*X2;
y3(1,:)  = w1.'*X3;
y1(2,:) = w2.'*X1;
y2(2,:)  = w2.'*X2;
y3(2,:)  = w2.'*X3;
%% Reconhecimento de Padr�es 2019.1
% Francisco Mauro Falc�o Matias Filho (374876)
% Trabalho III

clc;
close all;
clear;

%% Carregamento dos dados do dataset
data = csvread('wine.csv');

% normaliza��o dos dados de acordo com os requisitos do LDA
%data = zscore(data(:,2:13));

%% Defini��o de constantes e vari�veis auxiliares
TAM = length(data); % quantidade de amostras (178)
K = 10; % constante do K-fold
ACERTOS = 0; % quabtidade de acertos do k-fold

%% Separa��o dos dados da base entre classes
X1 = data(1:59,2:13);
X2 = data(60:130,2:13);
X3 = data(131:178,2:13);

N1 = length(X1);
N2 = length(X2);
N3 = length(X3);

%% Coisa
u1 = mean(X1);
u2 = mean(X2);
u3 = mean(X3);
u = (u1 + u2 + u3)/3;

%%
% Covariance matrices
S1 = cov(X1);
S2 = cov(X2);
S3 = cov(X3);

% Within-class scatter matrix
Sw = S1 + S2 + S3;

% Between-class scatter matrix
SB1 = N1*(u1 - u)*(u1 - u).';
SB2 = N2*(u2 - u)*(u2 - u).';
SB3 = N3*(u3 - u)*(u3 - u).';
SB = SB1 + SB2 + SB3;

% Eigendecomposition 
[V,D] = eig(inv(Sw)*SB);

w1 = V(:,1);
w2 = V(:,3);
% Amostras projetadas
y1(1,:) = w1.'*X1;
y2(1,:)  = w1.'*X2;
y3(1,:)  = w1.'*X3;
y1(2,:) = w2.'*X1;
y2(2,:)  = w2.'*X2;
y3(2,:)  = w2.'*X3;

% %% K-fold
% % �ndices l�gicos do kfold
% index = crossvalind('Kfold', TAM, K);
% 
% for i = 1:K
%     % separa��o dos �ndices de treino e teste
%     test = (index == i);
%     train = ~test;
%     
%     %x1_train = data_norm(train, );
% end