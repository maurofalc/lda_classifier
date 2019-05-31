%% Reconhecimento de Padr�es 2019.1
% Francisco Mauro Falc�o Matias Filho (374876)
% Trabalho III

clc;
close all;
clear;

%% Carregamento dos dados do dataset
data = csvread('wine.csv');

% normaliza��o dos dados
%data = zscore(data(:,2:13));

%% Defini��o de constantes e vari�veis auxiliares
TAM = length(data); % quantidade de amostras
K = 10; % constante do K-fold
ACERTOS = 0; % quantidade de acertos do k-fold

%% Separa��o dos dados da base entre classes
X1 = data(1:59,2:13).';
X2 = data(60:130,2:13).';
X3 = data(131:178,2:13).';

%% Separa��o dos dados para aplica��o do m�todo one-versus-all
X12 = [X1 X2];
X13 = [X1 X3];
X23 = [X2 X3];


%% C�lculo da quantidade de amostras de cada classe
N1 = length(X1); % tamanho das amostras da classe 1
N2 = length(X2); % tamanho das amostras da classe 2
N3 = length(X3); % tamanho das amostras da classe 3
N12 = N1 + N2; % tamanho das amostras da classe 12
N13 = N1 + N3; % tamanho das amostras da classe 13
N23 = N2 + N3; % tamanho das amostras da classe 23

%% C�lculo dos vetores m�dia dos atributos para cada classe
% 1 vs 2,3
u1 = mean(X1,2);
u23 = mean(X23,2);
U1_23 = (u1 + u23)/2;

% 2 vs 1,3
u2 = mean(X2,2);
u13 = mean(X13,2);
U2_13 = (u2 + u13)/2;

% 3 vs 1,2
u3 = mean(X3,2);
u12 = mean(X12,2);
U3_12 = (u3 + u12)/2;

%% C�lculo das matrizes de covari�ncia para cada classe
S1 = cov(X1.');
S2 = cov(X2.');
S3 = cov(X3.');
S12 = cov(X12.');
S13 = cov(X13.');
S23 = cov(X23.');

%% C�lculo das matrizes de dispers�o intra-classe multiclasse
% 1 vs 2,3
SW1_23 = S1 + S23;

% 2 vs 1,3
SW2_13 = S2 + S13;

% 3 vs 1,2
SW3_12 = S3 + S12;

%% C�lculo das matrizes de dispers�o inter-classe multiclasse
% 1 vs 2,3
SB1 = N1*(u1 - U1_23)*(u1 - U1_23).';
SB23 = N23*(u23 - U1_23)*(u23 - U1_23).';
SB1_23 = SB1 + SB23;

% 2 vs 1,3
SB2 = N1*(u2 - U2_13)*(u2 - U2_13).';
SB13 = N13*(u13 - U2_13)*(u13 - U2_13).';
SB2_13 = SB2 + SB13;

% 3 vs 1,2
SB3 = N3*(u3 - U3_12)*(u3 - U3_12).';
SB12 = N12*(u12 - U3_12)*(u12 - U3_12).';
SB3_12 = SB3 + SB12;

%% Desomposi��o em autovalores e autovetores 
% 1 vs 2,3
[V1_23,D1_23] = eig(inv(SW1_23)*SB1_23);
% 2 vs 1,3
[V2_13,D2_13] = eig(inv(SW2_13)*SB2_13);
% 3 vs 1,2
[V3_12,D3_12] = eig(inv(SW3_12)*SB3_12);

%% Vetores de proje��o
w1_23 = V1_23(:,1);
w2_13 = V2_13(:,1);
w3_12 = V3_12(:,1);

%% Amostras projetadas
% 1 vs 2,3
y1 = w1_23.'*X1;
y23 = w1_23.'*X23;

% 2 vs 1,3
y2 = w2_13.'*X2;
y13 = w2_13.'*X13;

% 3 vs 1,2
y3 = w3_12.'*X3;
y12 = w3_12.'*X12;
