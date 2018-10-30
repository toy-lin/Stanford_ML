function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
count_x = sum(y);
count_o = size(y)(1) - count_x;
coor_x = zeros(count_x,2);
coor_o = zeros(count_o,2);

i_x = 1;
i_o = 1;
for ii=1:size(y)(1),
  if y(ii) == 1,
    coor_x(i_x,:) = X(ii,:);
    i_x = i_x +1;
  else
    coor_o(i_o,:) = X(ii,:);
    i_o = i_o +1;
  endif
endfor


plot(coor_x(:,1),coor_x(:,2),'k+',coor_o(:,1),coor_o(:,2),'ko')

% =========================================================================



hold off;

end
