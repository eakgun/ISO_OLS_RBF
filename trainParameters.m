function [selected_rbfs, W, E_k, A_k, Q_k, B_k, centers, sigmas, G1, G2] =  trainParameters(X, val_in, y, G1, centers, sigmas, K)

    rbf_number = length(centers);


    D = y;
    Q = zeros(size(G1));
    Q_k = zeros(size(G1));
    B = zeros(1,rbf_number);
    B_k = zeros(1,rbf_number);
    E = zeros(1,rbf_number);
    E_k = zeros(1,K);
    selected_rbfs = zeros(1,rbf_number);      % indexes of selected rbfs ordered by decreasing energy 

    A = cell(1,rbf_number);
    A_k = eye(rbf_number);
    for i = 1:rbf_number
        A{i} = eye(rbf_number);
    end

    % ----- Subset Selection -----
    for k = 1:K
        % Gram-Schmidt orthogonalization
        for i = 1:rbf_number
           Q(:,i) = G1(:,i);

           % if rbf is already selected continue
           if ismember(i, selected_rbfs) == 1
               continue;
           end

           for j=1:k-1
                A{i}(j,k) = Q_k(:,j)'*G1(:,i) / (Q_k(:,j)'*Q_k(:,j));
                Q(:,i) = Q(:,i) - A{i}(j,k)*Q_k(:,j);
           end

           B(i) = Q(:,i)'*D / (Q(:,i)'*Q(:,i));
           E(i) = B(i)^2*Q(:,i)'*Q(:,i) / (D'*D);   
        end

        % find RBF with maximum energy (save index and copy to Q_k matrix)
        [E_k(k), selected_rbfs(k)] = max(E);
        B_k(k) = B(selected_rbfs(k));
        A_k(:,k) = A{selected_rbfs(k)}(:,k);
        W = A_k\B_k';
        % ------ Levenberg-Marquardt -----
        Theta = [W(k); sigmas(selected_rbfs(k))];
        for i = 1:size(centers,1)
        Theta = [Theta; centers(i, selected_rbfs(k))];
        end
        y_rbf = 0;
        for j = 1:k
            y_rbf = y_rbf + W(j) * RBFIO(X, sigmas(selected_rbfs(j)), centers(:,selected_rbfs(j))');
        end
        

        mu = 1;
        Theta_old = Theta;
        Nmax = 100; 
        err = zeros(Nmax,1);
        err(1) = abs(sum((y - y_rbf).^2)) / length(y);
        err_old = err(1);
        I = eye(length(Theta));
        for n = 2:Nmax
          
            dy_dw = RBFIO(X, Theta(2), [Theta(3:end)]);
%             dy_dsigma = Theta(1) * ((Theta(3) - X(:,1)).^2 + (Theta(4) - X(:,2)).^2) ./ (Theta(2)^3) .* RBFIO(X, Theta(2), [Theta(3:end)]);
%             dy_dc1 = (Theta(1)*(X(:,1) - Theta(3))) ./ (Theta(2)^2) .* RBFIO(X, Theta(2), [Theta(3:end)]);
%             dy_dc2 = (Theta(1)*(X(:,2) - Theta(4))) ./ (Theta(2)^2) .* RBFIO(X, Theta(2), [Theta(3:end)]);
            
            dy_dsigmaTemp = 0;
            for j = 1:size(centers,1)
                dy_dsigmaTemp = dy_dsigmaTemp + (Theta(2+j) - X(:,j)).^2;
            end
            dy_dsigma = Theta(1)*(dy_dsigmaTemp)./(Theta(2)^3).*RBFIO(X, Theta(2), [Theta(3:end)]);
            dy_dc = [];
            for i = 1:size(centers,1)
            dy_dc = [dy_dc, (Theta(1)*(X(:,i) - Theta(2+i))) ./ (Theta(2)^2) .* RBFIO(X, Theta(2), [Theta(3:end)])];
            end
            Z = [dy_dw dy_dsigma dy_dc];
            e = y - y_rbf;

            Theta = Theta + pinv(Z'*Z + mu*I)*Z'*e;             
            W(k) = Theta(1);
            sigmas(selected_rbfs(k)) = Theta(2);
            
            for i = 1:size(centers,1)
            centers(i,selected_rbfs(k)) = Theta(2+i);
            end
            
            y_rbf = 0;
            for j = 1:k
                y_rbf = y_rbf + W(j) * RBFIO(X, sigmas(selected_rbfs(j)), centers(:,selected_rbfs(j))');
            end

            err(n) = sum((y - y_rbf).^2) / length(y);
            if (err(n) >= err_old)
                Theta = Theta_old;
                mu = mu * 10;
            else
                Theta_old = Theta;
                err_old = err(n);
                mu = mu / 10;                
            end

            if (mu > 1e20 || mu < 1e-20)
                break;
            end
            
            if (err(n) < 1e-20)
                break;
            end
        end
        
        % Set optimized parameters
        W(k) = Theta_old(1);
        sigmas(selected_rbfs(k)) = Theta_old(2);
        for i = 1:size(centers,1)
        centers(i,selected_rbfs(k)) = Theta_old(2+i);
        end

        % Gram-Schmidt orthogonalization for optimized RBF
        i = selected_rbfs(k);
        G1(:,i) = RBFIO(X, Theta(2), [Theta(3:end)]);
        G2(:,i) = RBFIO(val_in, Theta(2), [Theta(3:end)]);

        Q(:,i) = G1(:,i);

        for j=1:k-1
            A{i}(j,k) = Q_k(:,j)'*G1(:,i) / (Q_k(:,j)'*Q_k(:,j));
            Q(:,i) = Q(:,i) - A{i}(j,k)*Q_k(:,j);
        end

        B(i) = Q(:,i)'*D / (Q(:,i)'*Q(:,i));
        B_k(k) = B(i);
        E_k(k) = B(i)^2*Q(:,i)'*Q(:,i) / (D'*D);
        A_k(:,k) = A{i}(:,k);
        Q_k(:,k) = Q(:,i);

        
%         W = A_k\B_k';

        E = zeros(size(E));
    end

%     W = A_k\B_k';

end