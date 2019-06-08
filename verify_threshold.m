% Determina o limiar para classificação entre as classes 'a' ou 'bc'
function [ threshold, main_class ] = verify_threshold(train_a,train_bc)
    mean_a = mean(train_a);
    mean_bc = mean(train_bc);
       
    if mean_a < mean_bc
        main_class = 0; % classe fica mais a esquerda
    else
        main_class = 1; % classe fica mais a direita
    end
    
    threshold = (mean_a + mean_bc)/2;
end