%% Reconhecimento de Padrões 2019.1
% Francisco Mauro Falcão Matias Filho (374876)
% Trabalho III

clc;
close all;
clear;

fprintf("LDA multiclasse:\n");
fprintf("Classes:\n\t*1-vs-2,3\n\t*2-vs-1,3\n\t*3-vs-1,2\n");

%% Carregamento dos dados do dataset
data = csvread('wine.csv');
class = data(:,1);

% normalização dos dados
%  data = [zeros([178,1]) normalize(data(:,2:14))];

%% Definição de constantes e variáveis auxiliares
TAM = length(data); % quantidade de amostras
K = 10; % constante do K-fold
% quantidade de acertos do k-fold
acertos1_23 = 0; ACERTOS1_23 = zeros(K, 1); % 1 vs 2,3
acertos2_13 = 0; ACERTOS2_13 = zeros(K, 1); % 2 vs 1,3
acertos3_12 = 0; ACERTOS3_12 = zeros(K, 1); % 3 vs 1,2

%% Separação dos dados da base entre classes
X1 = data(1:59,2:14).';
X2 = data(60:130,2:14).';
X3 = data(131:178,2:14).';

%% Separação dos dados para aplicação do método one-versus-all
X12 = [X1 X2];
X13 = [X1 X3];
X23 = [X2 X3];

%% Cálculo da quantidade de amostras de cada classe
N1 = length(X1); % tamanho das amostras da classe 1
N2 = length(X2); % tamanho das amostras da classe 2
N3 = length(X3); % tamanho das amostras da classe 3
N12 = N1 + N2; % tamanho das amostras da classe 1,2
N13 = N1 + N3; % tamanho das amostras da classe 1,3
N23 = N2 + N3; % tamanho das amostras da classe 2,3

%% K-fold
% separação dos dados do kfold
index = cvpartition(class,'Kfold',K);

for i = 1:K
    %% Separação dos índices de treino e teste
    train = training(index,i);
    teste = test(index,i);
        
    data_train = data(train,2:14).'; % amostras de treino
    data_test = data(teste,2:14).'; % amostras de teste
    
    %% Separação dos dados de treino e teste
    % a função intersect retorna os elementos que estão na interseção dos
    % argumentos, no caso, retorna os componentes de X que foram separados
    % para treino e teste
    
    % 1 vs 2,3
    X1_train = intersect(X1',data_train', 'rows')';
    X1_test = intersect(X1',data_test', 'rows')';
    X23_train = intersect(X23',data_train', 'rows')';
    X23_test = intersect(X23',data_test', 'rows')';
    
    % 2 vs 1,3
    X2_train = intersect(X2',data_train', 'rows')';
    X2_test = intersect(X2',data_test', 'rows')';
    X13_train = intersect(X13',data_train', 'rows')';
    X13_test = intersect(X13',data_test', 'rows')';
    
    % 1 vs 2,3
    X3_train = intersect(X3',data_train', 'rows')';
    X3_test = intersect(X3',data_test', 'rows')';
    X12_train = intersect(X12',data_train', 'rows')';
    X12_test = intersect(X12',data_test', 'rows')';
    
    %% Cálculo dos vetores média dos atributos para cada classe
    % 1 vs 2,3
    u1 = mean(X1_train,2);
    u23 = mean(X23_train,2);
    u1_23 = (u1 + u23)/2;

    % 2 vs 1,3
    u2 = mean(X2_train,2);
    u13 = mean(X13_train,2);
    u2_13 = (u2 + u13)/2;

    % 3 vs 1,2
    u3 = mean(X3_train,2);
    u12 = mean(X12_train,2);
    u3_12 = (u3 + u12)/2;

    %% Cálculo das matrizes de covariância para cada classe
    S1 = cov(X1_train.'); % classe 1
    S2 = cov(X2_train.'); % classe 2
    S3 = cov(X3_train.'); % classe 3
    S12 = cov(X12_train.'); % classe 12
    S13 = cov(X13_train.'); % classe 13
    S23 = cov(X23_train.'); % classe 23

    %% Cálculo das matrizes de dispersão intra-classe multiclasse
    % 1 vs 2,3
    SW1_23 = S1 + S23;

    % 2 vs 1,3
    SW2_13 = S2 + S13;

    % 3 vs 1,2
    SW3_12 = S3 + S12;

    %% Cálculo das matrizes de dispersão inter-classe multiclasse
    % 1 vs 2,3
    SB1 = (u1 - u1_23)*(u1 - u1_23).';
    SB23 = (u23 - u1_23)*(u23 - u1_23).';
    
    SB1_23 = (u1 - u23)*(u1 - u23).';

    % 2 vs 1,3
    SB2 = (u2 - u2_13)*(u2 - u2_13).';
    SB13 = (u13 - u2_13)*(u13 - u2_13).';
    
    SB2_13 = (u2 - u13)*(u2 - u13).';

    % 3 vs 1,2
    SB3 = (u3 - u3_12)*(u3 - u3_12).';
    SB12 = (u12 - u3_12)*(u12 - u3_12).';
    
    SB3_12 = (u3 - u12)*(u3 - u12).';

    %% Desomposição em autovalores e autovetores 
    % 1 vs 2,3
    [V1_23,D1_23] = eig(inv(SW1_23)*SB1_23);
    V1_23 = real(V1_23);
    D1_23 = real(D1_23);
    
    % 2 vs 1,3
    [V2_13,D2_13] = eig(inv(SW2_13)*SB2_13);
    V2_13 = real(V2_13);
    D2_13 = real(D2_13);
    
    % 3 vs 1,2
    [V3_12,D3_12] = eig(inv(SW3_12)*SB3_12);
    V3_12 = real(V3_12);
    D3_12 = real(D3_12);

    %% Autovetores ótimos
    V1_23 = sort(V1_23,'descend');
    V2_13 = sort(V2_13,'descend');
    V3_12 = sort(V3_12,'descend');
   
    %% Vetores de projeção ótimos
    w1_23 = V1_23(:,1);
    w2_13 = V2_13(:,1);
    w3_12 = V3_12(:,1);

    %% Projeção das amostras de treino
    % 1 vs 2,3
    y1_train = w1_23.'*X1_train;
    y23_train = w1_23.'*X23_train;
       
    % 2 vs 1,3
    y2_train = w2_13.'*X2_train;
    y13_train = w2_13.'*X13_train;
       
    % 3 vs 1,2
    y3_train = w3_12.'*X3_train;
    y12_train = w3_12.'*X12_train;
    
    %% Projeção das amostras de teste    
    % 1 vs 2,3
    y1_test = w1_23.'*X1_test;
    y23_test = w1_23.'*X23_test;
    
    % 2 vs 1,3
    y2_test = w2_13.'*X2_test;
    y13_test = w2_13.'*X13_test;
        
    % 3 vs 1,2
    y3_test = w3_12.'*X3_test;
    y12_test = w3_12.'*X12_test;
    
     %% Verificação das classes das amostras de teste  
     % 1 vs 2,3
     [M1_23, main_class1_23] = verify_threshold(y1_train,y23_train);
    
     for z1 = 1:length(y1_test)
        if main_class1_23 == 0
            if y1_test(z1) < M1_23
                acertos1_23 = acertos1_23 + 1;
            end
        else
            if y1_test(z1) > M1_23
                acertos1_23 = acertos1_23 + 1;
            end
        end
     end
        
     for z23 = 1:length(y23_test)
        if main_class1_23 == 0
            if y23_test(z23) > M1_23
                acertos1_23 = acertos1_23 + 1;
            end
        else
            if y23_test(z23) < M1_23
                acertos1_23 = acertos1_23 + 1;
            end
        end
     end
     
    acertos1_23 = 100 * (acertos1_23/(length(y1_test)+ length(y23_test)));
    fprintf("\n%i-fold: acertos LDA 1-vs-2,3: %0.2f%%\n", i, acertos1_23);

    % 2 vs 1,3
    [M2_13, main_class2_13] = verify_threshold(y2_train,y13_train);
    
    for t2 = 1:length(y2_test)
        if main_class2_13 == 0
            if y2_test(t2) < M2_13
                acertos2_13 = acertos2_13 + 1;
            end
        else
            if y2_test(t2) > M2_13
                acertos2_13 = acertos2_13 + 1;
            end
        end
    end
        
    for t13 = 1:length(y13_test)
        if main_class2_13 == 0
            if y13_test(t13) > M2_13
                acertos2_13 = acertos2_13 + 1;
            end
        else
            if y13_test(t13) < M2_13
                acertos2_13 = acertos2_13 + 1;
            end
        end
    end
    
    acertos2_13 = 100 * acertos2_13/(length(y2_test)+ length(y13_test));
    fprintf("%i-fold: acertos LDA 2-vs-1,3: %0.2f%%\n", i, acertos2_13);
    
    % 3 vs 1,2
    [M3_12, main_class3_12] = verify_threshold(y3_train,y12_train);
    
    for t3 = 1:length(y3_test)
        if main_class3_12 == 0
            if y3_test(t3) < M3_12
                acertos3_12 = acertos3_12 + 1;
            end
        else
            if y3_test(t3) > M3_12
                acertos3_12 = acertos3_12 + 1;
            end
        end
    end
        
    for t12 = 1:length(y12_test)
        if main_class3_12 == 0
            if y12_test(t12) > M3_12
                acertos3_12 = acertos3_12 + 1;
            end
        else
            if y12_test(t12) < M3_12
                acertos3_12 = acertos3_12 + 1;
            end
        end
    end
    
    acertos3_12 = 100 * acertos3_12/(length(y3_test)+ length(y12_test));
    fprintf("%i-fold: acertos LDA 1-vs-2,3: %0.2f%%\n", i, acertos3_12);
     
    % acertos gerais
    ACERTOS1_23(i) = acertos1_23;
    ACERTOS2_13(i) = acertos2_13;
    ACERTOS3_12(i) = acertos3_12;
    
    acertos1_23 = 0;
    acertos2_13 = 0;
    acertos3_12 = 0;
    
    index = repartition(index); % novo sorteio de amostras
end

%% Plotagem dos resultados gerais
figure('Name','1 VS 2,3','NumberTitle','off');
hold on;
grid on;
ylabel('Porcentagem de Acerto');
xlabel('Valor de K');
title('Porcentagem de Acertos do K-Fold');
set(legend('show'), 'Location', 'best');
plot(ACERTOS1_23, 'DisplayName', 'Acertos', 'MarkerSize', 20, 'Marker', ...
    '.', 'LineWidth', 2);
plot(mean(ACERTOS1_23)*ones(1,K), 'DisplayName', 'Acerto Médio', ...
    'LineStyle', '--', 'LineWidth', 2);

figure('Name','2 VS 1,3','NumberTitle','off');
hold on;
grid on;
ylabel('Porcentagem de Acerto');
xlabel('Valor de K');
title('Porcentagem de Acertos do K-Fold');
set(legend('show'), 'Location', 'best');
plot(ACERTOS2_13, 'DisplayName', 'Acertos', 'MarkerSize', 20, 'Marker', ...
    '.', 'LineWidth', 2);
plot(mean(ACERTOS2_13)*ones(1,K), 'DisplayName', 'Acerto Médio', ...
    'LineStyle', '--', 'LineWidth', 2);

figure('Name','3 VS 1,2','NumberTitle','off');
hold on;
grid on;
ylabel('Porcentagem de Acerto');
xlabel('Valor de K');
title('Porcentagem de Acertos do K-Fold');
set(legend('show'), 'Location', 'best');
plot(ACERTOS3_12, 'DisplayName', 'Acertos', 'MarkerSize', 20, 'Marker', ...
    '.', 'LineWidth', 2);
plot(mean(ACERTOS3_12)*ones(1,K), 'DisplayName', 'Acerto Médio', ...
    'LineStyle', '--', 'LineWidth', 2);
