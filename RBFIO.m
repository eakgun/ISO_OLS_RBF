function y = RBFIO(x, sigma, centers)
   num_Input=(size(x,2));
   temp = 0;
   
   for i = 1:num_Input
        temp = temp + ((x(:,i)-centers(i)).^2);
   end 
   y = exp(-(temp)/(2*sigma^2));
%    y = exp(-((x1-c1).^2 + (x2-c2).^2 +(x3-c3).^2) / (2*sigma^2));

% for i = 1:size(x,2)
%     y = exp(-(x(:,i)-centers(i,K)).^2 / (2*sigma^2));
% end
end