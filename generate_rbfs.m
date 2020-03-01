
function [G,centers,sigmas] = generate_rbfs(X, N)

    rbf_number = 0; % number of generated rbfs

    for i = 1:N
        rbf_number = rbf_number + (2^i + 1)^2;
    end

    G = zeros(length(X), rbf_number);
    centers = zeros(size(X,2), rbf_number);
    sigmas = zeros(1, rbf_number);
    k = 1;

    max_x = max(max(X));
    min_x = min(min(X));

    for i = 1:N

        iter_rbf_count = 2^i+1;  % number of RBFs in current iteration
        sigma = sqrt(max_x - min_x) / 2^(i-1);

%          figure(i+1)
%          title({sprintf('%d level of RBF functions',i);
%                 sprintf('sigma = %f',sigma);
%                 sprintf('Functions count = %d',iter_rbf_count^2)})
%          hold on
%           for j = 1:iter_rbf_count
%               for n = 1:size(X,2)
%                   
%                   centers(n, j) = min_x + (max_x - min_x) / (iter_rbf_count-1) * (nn-1);
%                   G(:,k) = RBFIO(X, sigma, centers(:,k)') ;
%        
%                 end
%               end
%           end
    % generating centers for rbfs
    for j = 1:iter_rbf_count
        for l = 1:iter_rbf_count
            for m = 1:iter_rbf_count
                for m2 = 1:iter_rbf_count
                    
                    centers(1, k) = min_x + (max_x - min_x) / (iter_rbf_count-1) * (j-1);  %Burayi generalize et.
                    centers(2, k) = min_x + (max_x - min_x) / (iter_rbf_count-1) * (l-1);
                    centers(3, k) = min_x + (max_x - min_x) / (iter_rbf_count-1) * (m-1);
                    centers(4, k) = min_x + (max_x - min_x) / (iter_rbf_count-1) * (m2-1);

                    G(:,k) = RBFIO(X, sigma, centers(:,k)') ;
                    
                    sigmas(k) = sigma;
                    k = k + 1;
                    
                end
            end
        end
    end
    
%         for j = 1:iter_rbf_count
%             for l = 1:iter_rbf_count
%                 for m = 1:iter_rbf_count
%                    for m2 = 1:iter_rbf_count 
%                        for m3 = 1:iter_rbf_count 
%                            for m4 = 1:iter_rbf_count 
%                                for m5 = 1:iter_rbf_count 
%                                    for m6 = 1:iter_rbf_count 
%                                        centers(1, k) = min_x + (max_x - min_x) / (iter_rbf_count-1) * (j-1);  %Burayi generalize et.
%                                        centers(2, k) = min_x + (max_x - min_x) / (iter_rbf_count-1) * (l-1);
%                                        centers(3, k) = min_x + (max_x - min_x) / (iter_rbf_count-1) * (m-1);
%                                        centers(4, k) = min_x + (max_x - min_x) / (iter_rbf_count-1) * (m2-1);
%                                        centers(5, k) = min_x + (max_x - min_x) / (iter_rbf_count-1) * (m3-1);
%                                        centers(6, k) = min_x + (max_x - min_x) / (iter_rbf_count-1) * (m4-1);
%                                        centers(7, k) = min_x + (max_x - min_x) / (iter_rbf_count-1) * (m5-1);
%                                        centers(8, k) = min_x + (max_x - min_x) / (iter_rbf_count-1) * (m6-1);
%                                        G(:,k) = RBFIO(X, sigma, centers(:,k)') ;
%                                        
%                                        sigmas(k) = sigma;
%                                        k = k + 1;
%                                    end
%                                end
%                            end
%                        end
%                    end
%                 end
%             end
%         end    

    end
